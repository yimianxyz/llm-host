# Design: self-hosting GLM-5.3 on RunPod

This document explains every choice in this repo and the evidence behind it. Measured results are in
[`results/FINDINGS.md`](../results/FINDINGS.md) (live log) and `results/REPORT.md` (final).

## 1. The decision in one paragraph

We serve **GLM-5.3** for agentic coding on **one RunPod Secure Cloud pod with 4x H200 SXM**:

- **Checkpoint:** the **`gpustack/GLM-5.3-W4A8`** quantization, which keeps INT4 routed experts and FP8 everywhere else.
- **Engine:** the stock **`vllm/vllm-openai:v0.33.0`** image.
- **Parallelism:** tensor parallel 4 plus expert parallelism.
- **Cache:** an FP8 sparse-MLA KV cache, with prefix caching on.
- **Speed:** MTP speculative decoding.
- **Access:** an API key, behind RunPod's HTTPS proxy.

Weights come from Hugging Face onto the pod's local disk at boot, so the pod is disposable and nothing bills after it is
deleted. The design costs **$21.16/hr**, which is half the 8x H200 FP8 reference, with about the same KV capacity and
quality within noise of the official FP8 release.

## 2. Why this hardware, checkpoint and engine

**Model facts** (from [config.json](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) and the [model card](https://huggingface.co/zai-org/GLM-5.3)):

- 753B total parameters and about 40B active, in 78 layers. Each layer has 256 routed experts, with 8 used per token plus 1 shared.
- Attention is MLA (`kv_lora_rank` 512, `qk_rope_head_dim` 64) plus DeepSeek Sparse Attention. A "lightning indexer" picks the top 2048 tokens per query, so attention cost stays near-flat as context grows.
- One MTP layer (`num_nextn_predict_layers` 1), used for speculative decoding.
- 1M-token native context.
- It is always a reasoning model. `reasoning_effort` accepts `low`/`high`/`max`; omitted or unknown values mean `max`.

**Checkpoint options and why W4A8 wins**

| Checkpoint | Weights | Fits on | Notes |
|---|---|---|---|
| `zai-org/GLM-5.3` (FP8) | 703.8 GiB | 8x H200 (88 GiB/GPU) | Official default; $42.32/hr |
| `gpustack/GLM-5.3-W4A8` | 372.3 GiB | **4x H200 (93 GiB/GPU)** | Listed in the official [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3); Hopper (SM90) only |
| NVFP4 (Inferact / RadixArk / nvidia) | ~465 GB | Blackwell only | Not usable on H200 |
| BF16 | ~1.5 TB | Multi-node | Not considered |

The [W4A8 card](https://huggingface.co/gpustack/GLM-5.3-W4A8) measured the same TP4+EP shape on 141 GB Hopper GPUs:

- 638,336 KV tokens per 4-GPU replica.
- GPQA-Diamond pass@1 of 90.28%, against 89.65% for FP8.
- Teacher-forced NLL only +0.001 nats/token higher.
- MTP working.

FP8 split over 4 GPUs would need 176 GiB per GPU and does not fit, so W4A8 is what makes the 4-GPU, $21/hr design possible.

**Why vLLM rather than SGLang.** SGLang is the stronger choice for 8x H200 with FP8. For this checkpoint, though:

- vLLM loads it natively, through `CompressedTensorsW4A8Fp8MoEMethod` on the SM90 CUTLASS kernel.
- SGLang needs a `sitecustomize.py` shim and a dev image.
- vLLM also ships the GLM-5.3 agentic work this design builds on. The [Part 1](https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading) and [Part 2](https://vllm.ai/blog/2026-10-23-glm53-part2-pcp-dcp-hisparse) blog posts cover Hybrid HiSparse, CPU KV offload, and DCP for sparse MLA.

**Why RunPod Secure Cloud, on-demand.**

- Spot pods are retired and savings plans are deprecated, so on-demand is the only self-serve option.
- Secure Cloud is the RunPod skills' default: data-center-grade hosts and always a public IP, which direct SSH needs.
- An H200 SXM is $5.29/GPU-hr there.

## 3. Memory math (the single most important thing to understand)

**Weights:** 372.3 GiB / 4 = **93 GiB per GPU**. An H200 shows 140.4 GiB to software.

**KV cache per token** in vLLM's `fp8_ds_mla` layout (from the [FlashMLA-sparse backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/flashmla_sparse.py)):

| Part | Bytes |
|---|---|
| MLA latent: 512 FP8 values + 4 FP32 scales + 64 BF16 RoPE values = 656 B, times 78 layers | 51,168 |
| Indexer keys: 132 B on the ~21 layers that own an indexer (GLM-5.3 shares one indexer across layers) | ~2,770 |
| MTP layer | ~790 |
| **Total** | **~54.7 KB per token** |

**MLA has a single KV head, so tensor parallelism cannot split it.** Every GPU stores a *full copy* of every token's KV.
The node's KV capacity is therefore what one GPU can spare, not four times that.

Worked out for one GPU:

- Budget: 0.92 x 140.4 GiB = 129.2 GiB.
- Minus 93 GiB of weights and a few GiB of activations and CUDA graphs, about 31 GiB remains for KV.
- 31 GiB / 54.7 KB ≈ **0.6M tokens for the whole node.** The card measured 638K.

That is the binding constraint for agentic work:

- One OpenHands-style session grows to about 86K tokens (a 74K first turn, then 12 turns of 753 + 220 tokens).
- **About 7 sessions** therefore fit resident at once. Beyond that, prefixes get evicted and TTFT blows up.

The capacity levers, in order of expected value:

1. **DCP (`--decode-context-parallel-size 4`)** shards KV by token across the 4 GPUs instead of copying it.
   - [vLLM PR 46145](https://github.com/vllm-project/vllm/pull/46145) measured 1.96x more tokens at DCP2 and 4.78x at DCP4.
   - The cost is an all-gather plus a softmax merge in attention on every decode step.
2. **CPU KV offload** keeps evicted prefixes in host RAM (276 GB per GPU on RunPod H200, about 1.1 TB total). A returning session reloads its prefix instead of re-prefilling it.
3. **Hybrid HiSparse** offloads all but the indexer-selected top-k tokens to host memory under pressure, so requests keep decoding at partial residency.

## 4. Baseline B0: every flag and why

Source of truth: [`deploy/configs/B0.args`](../deploy/configs/B0.args).

| Flag | Why |
|---|---|
| `--tensor-parallel-size 4` | 4 GPUs; 93 GiB/GPU of weights fits 141 GB. |
| `--enable-expert-parallel` | Required for W4A8. Without it each GPU holds slices of all 256 experts and CUTLASS grouped GEMM slows sharply (card). |
| `--kv-cache-dtype fp8_ds_mla` | FlashMLA-sparse FP8 layout; the only DSA path on SM90. Plain `fp8` is an alias on Hopper. |
| `--gpu-memory-utilization 0.92` | The card's value. 0.95 hit an OOM during CUDA-graph capture in [vLLM #65312](https://github.com/vllm-project/vllm/issues/65312). |
| `--max-model-len 262144` | The recipe's H200 cap. The 1M default fails at startup on this much KV. Rung L4 tests raising it. |
| `--max-num-batched-tokens 32768` | Per-step token budget. The vLLM agentic blog uses 32768 so 74K-token first turns prefill in about 3 chunks. Cost: more decode jitter while a chunk runs. |
| `--speculative-config {"method":"mtp","num_speculative_tokens":3}` | The built-in MTP head drafts 3 tokens per step. This is the vLLM blog's and NVIDIA Dynamo's agentic setting, with measured acceptance length 2.69-2.94, so about 2.7-2.9 tokens per verify step. Speculation is lossless: the target model verifies every token. |
| `--enable-prefix-caching` | Agent turns resend the whole growing history; cached prefixes skip prefill. On by default, explicit for clarity. |
| `--enable-prompt-tokens-details` | Returns `usage.prompt_tokens_details.cached_tokens` per request, which is a per-turn cache-hit measurement. |
| `--tool-call-parser glm47 --enable-auto-tool-choice` | GLM-5.3 emits the `<tool_call>…<arg_key>…<arg_value>` format; the older `glm45` tool parser cannot read it. |
| `--reasoning-parser glm45` | Splits `<think>` text into `reasoning_content`, as used by the vLLM blog and the card. |
| `--served-model-name glm-5.3`, `--trust-remote-code` | A stable client-facing name; the recipe's W4A8 variant sets `trust-remote-code`. |
| API key via env `VLLM_API_KEY` | Proxy URLs are public. It is injected from a RunPod secret, never stored in plaintext pod env. |

**Client-side guidance:**

- Stream every request. RunPod's HTTP proxy is behind Cloudflare, which returns 524 after 100 s without a response.
- Use temperature 1.0 and top_p 0.95 (`generation_config`). The card notes that without top-p, thinking can loop.
- Keep `clear_thinking=false` for agents. Earlier reasoning then stays in the history and the prefix stays byte-stable, so the cache keeps hitting.
- Pass `reasoning_effort` per request: `low`/`high`; omitted means `max`.

## 5. Pod shape

Created through the RunPod MCP `create-pod` tool, which is generated from REST v2, or through `deploy/pod.py create`. Both use the same body.

- **`image`:** `vllm/vllm-openai:v0.33.0@sha256:d825baf9…`, pinned by digest because RunPod caches images per host.
- **`gpu`:** `{"id": "NVIDIA H200", "count": 4, "minCudaVersion": "13.0"}`.
  - The image is CUDA 13 only, so the floor keeps it off older hosts.
  - "NVIDIA H200" is the SXM part, not "H200 NVL".
- **`cloud`:** `SECURE`.
- **`disk`:** 600 GB of local container disk. That holds 400 GB of weights plus about 20 GB for the image, xet and compile caches.
  - Container disk measured 2.4-5.4 GB/s; network volumes read at 200-400 MB/s. Loading 400 GB from a volume would take 17-33 minutes at about $0.35/min.
  - Container disk is wiped when the pod stops, and that is acceptable because the pod is stateless.
- **`ports`:** `8000/http` (vLLM, through the HTTPS proxy) and `22/tcp` (direct SSH, used for rsync).
- **`startSsh`:** true. RunPod then injects `PUBLIC_KEY`, but this image does not start sshd by itself, so `start.sh` does.
- **`entrypoint`:** `["bash","-c"]`, with `cmd` holding the text of [`deploy/start.sh`](../deploy/start.sh). This overrides the image's `vllm serve` entrypoint.
- **`env`:**
  - `MODEL` and `MODEL_REV`, pinning the HF commit.
  - `VLLM_ARGS`, which seeds `serve.args`.
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
                                            idle → short soak block on the current config
  deploy/sync.sh --loop ◀──rsync results── /root/results/** + /root/vllm.log  (every 5 min, then git commit)
  evals/smoke.py ──HTTPS proxy + key──▶    vLLM :8000
```

- **Experiments are config files.** An experiment is a new `deploy/configs/*.args` file in git.
  - `bench.sh apply` copies it to `/root/serve.args` and kills vLLM.
  - The supervisor restarts vLLM with the weights already on disk: a warm restart of a few minutes, with no re-download.
- **The pod can die at any time.** Results reach git every 5 minutes and at the end of every step, so at most a few minutes of data are lost. The pod holds no state that git cannot recreate.
- **The pod never idles.** When the queue is empty, the runner keeps producing stability data.

## 7. Evaluation methodology

**Headline metric: max concurrent agent sessions at SLO.** It is the highest concurrency on the agentic workload that meets all of:

- TTFT p50 of 5 s or less.
- Median interactivity of at least 50 tok/s per user.
- Zero errors.

This is NVIDIA Dynamo's agentic SLO. Reference points:

- Dynamo's 8x H200 FP8 runs about 10-11 sessions at 57 tok/s per user, so the target for 4 GPUs is about 5.
- Z.ai's own API runs at about 66 tok/s.
- The fastest provider (Baseten FP8) runs at about 263 tok/s.

**Definitions**

| Metric | Definition | Why it matters |
|---|---|---|
| TTFT | Request sent until the first streamed token, reasoning included. Covers queue + prefill + network. | How long an agent turn waits before anything happens |
| Time to first answer token | Until the first non-reasoning token | What a user actually waits for. `vllm bench` cannot measure it; `evals/smoke.py` does |
| TPOT | (E2E - TTFT) / (output tokens - 1), per request | Per-token decode cost |
| Interactivity | 1000 / TPOT (tok/s per user) | Perceived speed. The SLO uses TPOT, not ITL, because MTP emits several tokens per chunk, which makes ITL bursty |
| Throughput per GPU | Total tok/s / 4 (InferenceX convention; logical tokens include cached prompt tokens), plus output tok/s | Capacity; cost derives from it |
| MTP acceptance length | 1 + d(accepted) / d(drafts) from `/metrics` | Explains real vs synthetic speed; a broken MTP shows up here first |
| Prefix-cache hit rate | d(hits) / d(queries) from `/metrics` | Whether agent sessions stay resident |
| Preemptions, KV usage, running/waiting | `/metrics` | The cause behind TTFT cliffs |
| $/session-hour, $/1M tokens | $21.16/hr / (sessions at SLO); $21.16 / (tokens/hr / 1e6) | The economics |
| Cold start, warm restart | Pod create to first generation; config swap to first generation | Recovery time for a long-running service |

**Workloads**

- **P1, agentic.** The vLLM team's OpenHands workload, from blog Part 1: 13-turn conversations, a 74,160-token first turn, 753-token later turns and 220-token outputs, run with EvalScope at the blog's pinned commit.
  - It runs at c = 1, 4, 8, 16, 24, 32.
  - Each point uses a disjoint slice of the 256-conversation dataset, so points never warm the cache for each other.
  - The dataset builder and deps list are vendored in [`evals/agentic/`](../evals/agentic/).
- **P2, reference shape.** `vllm bench serve` with random 8192 in / 1024 out at c = 1, 16, 64. This is the SGLang cookbook's shape.
  - SGLang's published H200 FP8 TP8 numbers: TTFT 780 ms and TPOT 3.71 ms at c=1; 5.5 s and 14.2 ms at c=16.
  - Caveat: they pinned MTP acceptance at 3.5 in simulation.
- **Long context.** Single-stream needle-in-a-haystack at 128K and 200K, extended to 512K and 1M if DCP is kept. Each run records prefill tok/s and decode tok/s.
- **MTP on real prompts.** `speed_bench` at temperature 1.0 and top_p 0.95. InferenceX's golden acceptance length for GLM-5.3 is 1.84, 2.94 and 3.48 at 1, 3 and 5 draft tokens. Random prompts understate it.

**Quality gate** (`evals/quality.py`):

- **GSM8K:** 95% or above. The vLLM CI threshold for DSA+MTP models; the reference is 97.1-97.4%.
- **Tool-call suite:** about 25 cases, both streaming and non-streaming, must all pass. It includes the open-bug edges: quotes, JSON null, nested JSON, Windows paths, shell metacharacters, a 4 KB argument, and `<tool_call>` markers in prose.
- **Needle retrieval** must be exact.
- **GPQA-Diamond:** optional. The card already measured it with 4x more samples.
- **Protocol note:** GSM8K runs zero-shot with a `####` answer line, a slightly different protocol from the cookbook's 1-shot `sgl-eval`. The gate is for detecting breakage.

**Benchmark hygiene:**

- **Run clients inside the pod**, so network noise stays out of engine numbers. One smoke run through the proxy measures the real client path.
- **Fresh data per run:** a fresh seed or a disjoint dataset slice every time, so the always-on prefix cache cannot inflate results.
- **Warmups and percentiles:** warmups on every point, and p99 reported only where n is 400 or more.

## 8. Optimization ladder

Each rung changes one variable from the current best config and is kept or reverted. Protocol and order are in the plan:

| Rung | Change | Expected effect |
|---|---|---|
| L1 | DCP4 | Primary: about 4x KV tokens, moving the session cliff from about 7 to far higher. Cost: c=1 TPOT. |
| L2 | CPU KV offload (`OffloadingConnector` / `TieringOffloadingSpec`, about 400 GiB) | Prefixes survive beyond HBM; TTFT for returning sessions past the cliff. |
| L3 | MTP per-batch schedule `[[1,4,5],[5,16,3],[17,1024,1]]` (v0.33) | Deep drafts at low load, cheap drafts at high load. #65312: k=5 107 tok/s vs schedule about 1,950 tok/s at c=256. |
| L4 | `--max-model-len 1048576` (with DCP) | Does 1M context work on 4x H200, and at what TTFT? |
| L5-L7 | HiSparse, Rust frontend, DSpark drafter | Only one is run, chosen from what L1-L4 show is still binding. |

**Keep rule:** a rung is kept if max sessions at SLO rises, or if it holds and c=1 interactivity improves by at least 5%. It must also pass the spot gate: tool calls, a 128K needle, and 200 GSM8K questions.

## 9. Sources

- **Model:** [GLM-5.3 card](https://huggingface.co/zai-org/GLM-5.3), [config.json](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json), [W4A8 card](https://huggingface.co/gpustack/GLM-5.3-W4A8).
- **vLLM:** [GLM-5.3 recipe](https://recipes.vllm.ai/zai-org/GLM-5.3), [blog Part 1](https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading), [blog Part 2](https://vllm.ai/blog/2026-10-23-glm53-part2-pcp-dcp-hisparse), [DCP PR 46145](https://github.com/vllm-project/vllm/pull/46145).
- **Other engines and benchmarks:** [SGLang GLM-5.3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3.mdx) and its [benchmark data](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/zai-org/glm-5.3-benchmarks.jsx), [NVIDIA Dynamo GLM-5.3 recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3), [InferenceX golden acceptance lengths](https://github.com/SemiAnalysisAI/InferenceX/tree/main/inferencex-e2e/infx/golden_al_distribution), [Artificial Analysis providers](https://artificialanalysis.ai/models/glm-5-3/providers).
- **RunPod:** [runpod-plugins-official skills](https://github.com/runpod/runpod-plugins-official), [REST v2 OpenAPI](https://api.runpod.io/v2/openapi.json), [pricing](https://www.runpod.io/pricing).
