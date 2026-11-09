# Design: self-hosting GLM-5.3 on RunPod

This document explains every choice in this repo and the measurement behind it. The results are in
[`results/REPORT.md`](../results/REPORT.md) (answers to the eight design questions) and
[`results/FINDINGS.md`](../results/FINDINGS.md) (the step-by-step log with raw-data pointers).

## 1. The decision in one paragraph

We serve **GLM-5.3** for agentic coding on **one RunPod Secure Cloud pod with 4x H200 SXM** at $21.16/hr:

- **Checkpoint:** **`gpustack/GLM-5.3-W4A8`** (INT4 routed experts, FP8 elsewhere).
- **Engine:** the stock **`vllm/vllm-openai:v0.33.0`** image.
- **Parallelism:** tensor parallel 4 with expert parallelism, and **decode context parallel 4**, which shards the KV cache instead of copying it.
- **Cache:** an FP8 sparse-MLA KV cache with prefix caching, plus a **300 GiB host-memory KV tier** that keeps idle sessions warm.
- **Speed:** MTP speculative decoding with a draft count that falls as load rises.
- **Access:** an API key, behind RunPod's HTTPS proxy.

The config is [`deploy/configs/recommended.args`](../deploy/configs/recommended.args). Weights come from Hugging Face onto
the pod's local disk at first boot, so the pod is disposable and nothing bills after it is deleted. Native 1M context is
available as a separate profile of the same pod ([`L4d-ctx1m-bt2k.args`](../deploy/configs/L4d-ctx1m-bt2k.args)),
because it needs smaller prefill chunks that cost the agentic workload throughput.

## 2. Why this hardware, checkpoint and engine

**Model facts** (from [config.json](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) and the [model card](https://huggingface.co/zai-org/GLM-5.3)):

- 753B total parameters and about 40B active, in 78 layers. Each layer has 256 routed experts (8 used per token) plus 1 shared, and 64 attention heads.
- Attention is MLA (`kv_lora_rank` 512, `qk_rope_head_dim` 64) plus DeepSeek Sparse Attention. A "lightning indexer" picks the top 2,048 tokens per query, so decode cost stays flat as context grows. We measured a 23.9 ms decode step at 1M context, the same as at 80K.
- One MTP layer, used for speculative decoding.
- 1M-token native context.
- It is always a reasoning model. `reasoning_effort` accepts `low`/`high`/`max`; omitted or unknown values mean `max`.
- **The 64 heads and 256 experts mean TP, EP and DCP only work at 2, 4 or 8 GPUs.** A fifth GPU would sit idle.

**Checkpoint options and why W4A8 wins**

| Checkpoint | Weights | Fits on | Notes |
|---|---|---|---|
| `zai-org/GLM-5.3` (FP8) | 703.8 GiB | 8x H200 | Official default; $42.32/hr |
| `gpustack/GLM-5.3-W4A8` | 372.3 GiB | **4x H200** | Listed in the official [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3); Hopper (SM90) only |
| NVFP4 (Inferact / RadixArk / nvidia) | ~465 GB | Blackwell only | Not usable on H200 |
| BF16 | ~1.5 TB | Multi-node | Not considered |

- **Measured quality:** GSM8K (all 1,319) scored 96.7%, against 97.1-97.4% for FP8, and the 48-case tool-call suite and the needles at 128K, 200K, 512K and 1M all passed.
- **The card's figures:** GPQA-Diamond 90.28% against 89.65% for FP8, and NLL +0.001 nats per token.
- FP8 split over 4 GPUs would need 176 GiB per GPU and does not fit, so W4A8 is what makes the 4-GPU design possible.

**Why vLLM rather than SGLang.**
- vLLM loads this checkpoint natively (`CompressedTensorsW4A8Fp8MoEMethod`, SM90 CUTLASS); SGLang needs a shim and a dev image.
- vLLM ships the GLM-5.3 agentic features this design uses: DCP for sparse MLA, the CPU KV tier and per-batch-size MTP schedules ([blog Part 1](https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading), [Part 2](https://vllm.ai/blog/2026-10-23-glm53-part2-pcp-dcp-hisparse)).

**Why RunPod Secure Cloud, on-demand.**
- Spot pods are retired and savings plans are deprecated, so on-demand is the only self-serve option.
- Secure Cloud gives data-center hosts with a public IP. Direct TCP is needed for SSH and for long-context clients (section 5).
- An H200 SXM costs $5.29/GPU-hr there.

## 3. Memory math (the single most important thing to understand)

Per GPU, as measured on the recommended config (vLLM's own boot log, `results/raw/.../s6-final/boot_facts.txt`):

| Item | GiB per GPU |
|---|---|
| Usable memory (PyTorch "total capacity") | 139.8 |
| Budget at `--gpu-memory-utilization 0.92` | 128.6 |
| Weights, including the MTP layer ("Model loading took") | **107.3** (not the 93 GiB that 372.3 / 4 suggests) |
| Profiled activation peak at 8K batched tokens | 4.5 |
| CUDA graphs (counted inside the budget since v0.21) | 1.9 at a 128-token capture cap; 4.5 at the default 512 |
| **KV cache** | **14.9, which is 1,169,664 tokens** |
| Outside the budget (8%) | 11.2: CUDA context, NCCL and symmetric memory, and two buffers the profiler does not count |

**KV per token is about 54.8 KB for the whole node:**
- **MLA latent:** 656 B per layer x 78 layers.
- **Indexer keys:** about 2.8 KB.
- **MTP layer:** about 0.8 KB.

**MLA has a single KV head, so tensor parallelism cannot split it.** Without DCP, every GPU stores a full copy of every token: B0 had **about 131K tokens for the whole node, about one agent session**. DCP4 shards the KV by token across the 4 GPUs, so each holds about 13.7 KB per token and the node holds 1.17M tokens. That is about 13 OpenHands-style sessions of 86K tokens resident on GPU.
- **Cost:** about 26% of prefill time goes to DCP and TP collectives (profile in FINDINGS).
- **Side effect:** it removes a 16-to-64 head padding in the FP8 decode kernel.

**The two unprofiled buffers decide what fits:**
1. **The FlashMLA mixed-batch buffer** is about 256 KiB per batched token: 2 GiB at 8K, 4 GiB at 16K, 8 GiB at 32K. It caused the B1, B2 and L5p OOMs.
2. **An attention workspace sized by batched tokens x max-model-len x 4 B** is 16 GiB at 4K x 1M. It is why 1M context needs 2K chunks.

**Raising `--gpu-memory-utilization` does not help.** It shrinks exactly the headroom those buffers need, and at 1M the graph memory alone grew by more than a whole point of util.

**Beyond GPU KV:** the 300 GiB host tier holds about 5.5M more tokens, and the pod's cgroup allows 755 GB of RAM. When more sessions are active than GPU KV holds, LRU eviction on the cyclic agent access pattern evicts each session just before its next turn. GPU prefix hits fall to almost zero and every turn would re-prefill 80K tokens. The host tier turns those misses into PCIe reloads.

## 4. The recommended config: every flag and the evidence

Source of truth: [`deploy/configs/recommended.args`](../deploy/configs/recommended.args). Rung names refer to section 8.

| Flag | Why (measured) |
|---|---|
| `--tensor-parallel-size 4 --enable-expert-parallel` | 107 GiB of weights per GPU fits on 4. EP is required: without it each GPU holds slices of all 256 experts and the W4A8 grouped GEMM slows sharply (card). |
| `--decode-context-parallel-size 4` | KV per node goes from about 131K tokens (B0, does not even boot at 256K) to over 1M (L1). |
| `--kv-cache-dtype fp8_ds_mla` | The FlashMLA-sparse FP8 layout, the only DSA KV path on SM90. |
| `--gpu-memory-utilization 0.92` | The card's value. Headroom above the budget is what the unprofiled buffers use, and 0.93 bought nothing at 1M (L4). |
| `--max-model-len 262144` | With 8K chunks, the attention workspace grows with max-model-len, so 256K is the agentic ceiling. 1M is the L4d profile. |
| `--max-num-batched-tokens 8192` | 16K OOMs on a 128K prompt (L5p). 4K costs 19% of prefill speed and 11% of per-user speed at c=4 (L5c). A prefill step has a large fixed cost, so 8K is close to optimal. |
| `--max-cudagraph-capture-size 128` | A decode step under the MTP schedule is at most 64 tokens for 16 sessions. Capping graphs at 128 tokens frees 2.6 GiB per GPU (+21% KV) and cuts restarts from 357 to 257 s (L5c/L5d). |
| `--speculative-config` MTP, 5 drafts at 1-4 running requests, 3 at 5-16, 1 at 17 or more | 5 drafts lift real-prompt acceptance from 2.95 to 3.54 tokens per step at low load (+8% per-user speed at c=1). At higher load, fewer drafts cost less verify compute (L3). |
| `--enable-prefix-caching` | Each agent turn resends its whole history; 91.8% of prompt tokens hit the cache on this workload. |
| `--kv-offloading-size 300 --kv-offloading-backend native` | Without it, 16 sessions collapse (0 conversations finished in 20 min); with it, 16/16 finished in 5.5 min, with 91% of prefix hits served from host memory (L2). Cost: about 4% throughput at c=8 and 60 s at boot. |
| `--enable-prompt-tokens-details --enable-per-request-metrics` | Per-turn cached-token counts and per-request timings, used for the alerts in the report's runbook. |
| `--tool-call-parser glm47 --enable-auto-tool-choice` | GLM-5.3's `<tool_call>/<arg_key>/<arg_value>` format; 48/48 typed cases pass. |
| `--reasoning-parser glm45` | Splits `<think>` text into `reasoning_content`. |
| `--served-model-name glm-5.3 --trust-remote-code` | A stable client-facing name; the recipe's W4A8 variant sets `trust-remote-code`. |
| API key via env `VLLM_API_KEY` | Proxy URLs are public. It is injected from a RunPod secret, never stored in plaintext pod env. |

**Rejected after measurement:**
- **DSpark drafter (L7):** +27% per-user speed at c=1, but it halves the KV pool and collapses at 8 sessions.
- **HiSparse:** incompatible with DCP.
- **PCP:** doubles attention weights per GPU.
- **`expandable_segments`:** fails at load with DCP's symmetric memory.
- **`--dcp-q-replicate`:** costs about 2 GiB per GPU for a 9% collective.

**Client guidance:**
- **Stream every request.** Cloudflare in front of RunPod's proxy returns 524 if the first byte takes more than 100 s; an active stream may run longer (a 110 s stream survived). A prompt over about 400K tokens, or a cold turn queued behind several others, can exceed 100 s to the first token. Such clients must use the pod's direct TCP port.
- **Send a real User-Agent.** Cloudflare returns 403 to Python's default urllib User-Agent.
- **Sampling:** temperature 1.0 and top_p 0.95 (`generation_config`).
- **Reasoning history:** `clear_thinking=false`, so the history stays byte-stable and the prefix cache keeps hitting.
- **Effort:** pass `reasoning_effort` per request.
- **Declare a JSON-schema type for every tool parameter.** Untyped parameters come back as strings (`"null"`, `"3"`, stringified objects).
- **Keep each session on one pod.** The prefix cache and the host tier are per pod (report, question 8).

## 5. Pod shape

Created through the RunPod MCP `create-pod` tool, which is generated from REST v2, or through `deploy/pod.py create`. Both use the same body.

- **`image`:** `vllm/vllm-openai:v0.33.0@sha256:d825baf9…`, pinned by digest.
- **`gpu`:** `{"id": "NVIDIA H200", "count": 4, "minCudaVersion": "13.0"}`.
  - The image is CUDA 13 only.
  - "NVIDIA H200" is the SXM part, not "H200 NVL".
- **`cloud`:** `SECURE`.
- **`disk`:** 600 GB of local container disk, which holds 400 GB of weights plus the image, xet and compile caches.
  - It survives a container restart; a crash costs one vLLM boot, not a re-download.
  - **No network volume:** in EUR-IS-5 Hugging Face delivered 230-310 MB/s, as fast as a network volume reads (200-400 MB/s), so a volume would cost money without shortening a cold start.
- **`ports`:** `8000/http` (vLLM through the HTTPS proxy) and `22/tcp` (SSH and rsync). For long-context clients, also expose vLLM's port over TCP, which RunPod maps to the pod's public IP, to bypass the proxy's 100 s first-byte limit. Put TLS in front of it, because the TCP mapping is plain HTTP.
- **`startSsh`:** true. RunPod then injects `PUBLIC_KEY`, but this image does not start sshd by itself, so `start.sh` does.
- **`entrypoint`:** `["bash","-c"]`, with `cmd` holding the text of [`deploy/start.sh`](../deploy/start.sh).
- **`env`:**
  - `MODEL` and `MODEL_REV`, pinning the HF commit.
  - `SERVE_ARGS`, which seeds `serve.args`.
  - `MAX_HOURS`, the deadline.
  - `HF_XET_HIGH_PERFORMANCE=1`.
  - Secrets as `{{ RUNPOD_SECRET_vllm_api_key }}` and `{{ RUNPOD_SECRET_hf_token }}`.

## 6. How the system works

```
Mac (Cursor agent)                       Pod (4x H200)
  RunPod MCP / deploy/pod.py  ───create──▶ start.sh (PID 1)
                                            1. deadline watchdog: deletes its own pod at /root/deadline
                                            2. sshd (operator channel)
                                            3. host facts (GPU topology, /dev/shm, CPU quota)
                                            4. hf download → local disk (resumable)
                                            5. vLLM supervisor loop over /root/serve.args
  deploy/push.sh  ──rsync code + jobs──▶   deploy/runner.sh: runs /root/queue/*.sh one by one;
                                            idle → soak block on the current config
  deploy/sync.sh --loop ◀──rsync results── /root/results/** + /root/vllm.log  (every 5 min, then git commit)
  evals/smoke.py ──HTTPS proxy + key──▶    vLLM :8000
```

- **Experiments are config files.** `bench.sh apply` copies a config to `/root/serve.args` and kills vLLM. The supervisor restarts it from the weights on local disk, a warm restart of 4-6 minutes.
- **Bad configs are caught quickly.** `apply` gives up after a configurable number of crash-restarts and the job falls back, so a bad config cannot crash-loop the pod unattended.
- **The pod can die at any time.** Results reach git every 5 minutes and at the end of every step. The pod holds no state that git cannot recreate.
- **The pod never idles.** When the queue is empty, the runner keeps producing stability data.

## 7. Evaluation methodology

**Headline metric: max concurrent agent sessions at SLO.** It is the highest concurrency on the agentic workload that meets all of:

- TTFT p50 of 5 s or less.
- Median per-request interactivity of at least 50 tok/s per user, at real-prompt acceptance (below).
- Zero errors.

This is NVIDIA Dynamo's agentic SLO. Reference points:
- Dynamo's 8x H200 FP8 runs about 10-11 sessions at 57 tok/s per user, so the target for 4 GPUs is about 5.
- Z.ai's own API runs at about 66 tok/s.
- The fastest provider (Baseten FP8) runs at about 263 tok/s.

**Workloads**

- **P1, agentic.** The vLLM team's OpenHands workload (blog Part 1), run with EvalScope at the blog's pinned commit.
  - Each conversation has 13 turns: a cold 74K-token first turn, then about 1K new tokens and 220 output tokens per turn.
  - Each point runs 2c conversations from a disjoint dataset slice.
- **P2, reference shape.** `vllm bench serve`, random 8192 in / 1024 out at c = 1, 16, 64 (the SGLang cookbook's shape).
  - Random tokens give MTP almost nothing to accept (AL 1.4), so P2 understates decode speed by about 2x.
- **Long context.** Single-stream needles at 128K, 200K, 512K and 1M, and one 1M-token request for TTFT and decode step time.
- **MTP and drafter acceptance on real prompts.** SPEED-Bench coding at temperature 1.0, measured at the concurrency that selects the draft count.

**Two corrections the raw numbers need:**

1. **The agentic workload inflates speculative acceptance.**
   - **The artifact:** it forces `ignore_eos` at temperature 0, so outputs that would have ended degenerate into repetition. MTP predicts those loops almost perfectly, giving acceptance length 3.8-5.9 against 2.95-3.54 on real coding prompts.
   - **The fix:** `evals/report.py` derives the engine step time as TPOT x the run's acceptance, which is hardware time independent of acceptance. It then reports **`inter_real` = 1000 x real acceptance / step time.**
   - **The effect:** raw TPOT overstates per-user speed by about 29%.
2. **Median vs mean.** Steps are bimodal: a pure decode step takes 23-44 ms, and a step carrying another session's 8K prefill chunk takes about 1.4 s.
   - The SLO uses the per-request **median**, as Dynamo does.
   - The mean, which counts the stalls in full, is reported alongside because it is what a user feels during other sessions' cold starts.

**Quality gate** (`evals/quality.py`):
- **GSM8K:** 95% or more. The vLLM CI threshold for DSA+MTP models; the reference is 97.1-97.4%.
- **Tool calls:** a 24-case suite run streaming and non-streaming must pass 48/48. It covers quotes, JSON null, nested JSON, Windows paths, shell metacharacters, a 4 KB argument, and `<tool_call>` markers in prose.
- **Needles:** exact retrieval.
- **GPQA-Diamond:** runs once, on the final config.
- **Per rung:** a spot gate (tool calls, a 128K needle and 200 GSM8K questions).

**Benchmark hygiene:**
- **Clients run inside the pod**, with one smoke run through the proxy.
- **Fresh data per run:** a fresh seed or a disjoint dataset slice every time, and a warm restart between configs, so caches never carry over.
- **Warmups:** on every reference point.
- **p99 only with enough samples:** reported only where n is 400 or more; otherwise p90.

## 8. Optimization ladder and outcomes

Each rung changed one variable from the current best config. The keep rule:
- **Keep:** if max sessions at SLO rises, or holds while c=1 interactivity improves by at least 5%.
- **Revert:** if the spot gate fails or any errors appear.

| Rung | Change | Outcome |
|---|---|---|
| B0-B4 | Boot fixes | Without DCP the KV is replicated (about 131K tokens). DCP4 needs 8K chunks because of the unprofiled FlashMLA buffer; `expandable_segments` breaks DCP. |
| L1 | DCP4 | **Kept** (in B4): 1.04M KV tokens. |
| L3 | MTP 5/3/1-draft schedule | **Kept**: c=1 +8% at real acceptance; sessions level; KV -7%. |
| L5p | 16K chunks, util 0.91 | **Reverted**: OOM on a 128K prompt; cold-turn TTFT only -2.8%. |
| L2 | 300 GiB host KV tier | **Kept**: 16 sessions go from collapse to 16/16 in 5.5 min. |
| L7 | DSpark drafter (8 drafts in one pass) | **Reverted**: c=1 +27%, but the KV pool halves to 484K, the host tier barely hits (6%), and c=8 collapses. |
| L4 | 1M context | **Separate profile**: needs 2K chunks (330 s TTFT at 1M, decode step 23.9 ms). The agentic config stays at 256K. |
| L5c | 4K chunks + graph cap 128 | **4K reverted** (-19% prefill, -11% at c=4); **cap kept** as L5d. |
| L5d | B4 + L3 + L2 + graph cap 128 | **Final** (Step 6): 4 sessions at SLO with margin (77 tok/s median), 6 at the edge (50.0); 148 tok/s for one user; 0 errors through 16 sessions; GSM8K 96.9%, 48/48 tools; 257 s restart. |

## 9. Sources

- **Model:** [GLM-5.3 card](https://huggingface.co/zai-org/GLM-5.3), [config.json](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json), [W4A8 card](https://huggingface.co/gpustack/GLM-5.3-W4A8).
- **vLLM:** [GLM-5.3 recipe](https://recipes.vllm.ai/zai-org/GLM-5.3), [blog Part 1](https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading), [blog Part 2](https://vllm.ai/blog/2026-10-23-glm53-part2-pcp-dcp-hisparse), [DCP PR 46145](https://github.com/vllm-project/vllm/pull/46145), [FlashMLA-sparse backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/flashmla_sparse.py).
- **Other engines and benchmarks:** [SGLang GLM-5.3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3.mdx) and its [benchmark data](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/zai-org/glm-5.3-benchmarks.jsx), [NVIDIA Dynamo GLM-5.3 recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3), [InferenceX golden acceptance lengths](https://github.com/SemiAnalysisAI/InferenceX/tree/main/inferencex-e2e/infx/golden_al_distribution), [Artificial Analysis providers](https://artificialanalysis.ai/models/glm-5-3/providers).
- **RunPod:** [runpod-plugins-official skills](https://github.com/runpod/runpod-plugins-official), [REST v2 OpenAPI](https://api.runpod.io/v2/openapi.json), [pricing](https://www.runpod.io/pricing).
