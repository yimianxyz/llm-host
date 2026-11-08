# Findings log

This log is append-only, with one entry per step. Each entry records the config, the key numbers, the decision, what we learned, and the spend so far. Raw data is in `raw/<pod-name>/<step>/`.

## Rehearsal (2026-11-08 19:18-19:42Z, 1x A40, CA-MTL-1, Qwen3-0.6B, ~$0.24)

**Config:** `deploy/configs/R0-rehearsal.args` on the same `vllm/vllm-openai:v0.33.0` digest as production.

**Verified end to end:**
- **Entrypoint override:** REST v2 `entrypoint` + `cmd` run `start.sh` on the stock image, with no custom image.
- **Boot:** container start to `sshd` up took 11 s, and vLLM was serving 25 s after create, because the image was already cached on that host.
- **SSH and the runner:**
  - `startSsh` only injects `PUBLIC_KEY`, so the image must start sshd itself; `start.sh` does.
  - Direct SSH works.
  - The queue runner, `sync.sh`, and the report generator all work.
- **Pod environment:** RunPod injects a pod-scoped `RUNPOD_API_KEY`, so **the deadline watchdog deleted its own pod** via REST v2 `DELETE /v2/pods/{id}`. `get-pod` then returned 404, which proves the dead-man's switch works.
- **Secrets:** env shows `{{ RUNPOD_SECRET_* }}` placeholders in API reads; values are never returned.
- **Proxy findings:**
  - Cloudflare in front of the RunPod proxy returns **403 to Python's default urllib User-Agent**, so clients must send a real UA. The OpenAI SDK already does.
  - **A 110 s streamed response survived the proxy.** The 100 s cap applies to time-to-first-byte or idle time, not to an active stream, so streaming agents are safe.
- **All vLLM flags used by later rungs exist in v0.33.0:** `--decode-context-parallel-size`, `--kv-offloading-size`, `--kv-transfer-config`, `--attention-config`, `--enable-per-request-metrics`, `--tokenizer-mode`.

**Fixes made from the rehearsal:**
- The runner check matched its own `pgrep` and never started; it now uses a pidfile.
- The vLLM image has no `datasets` or `pandas`, so quality scripts run from the EvalScope client venv and pandas is installed for `speed_bench`.
- SPEED-Bench needs a prepared directory. It is built locally from NVIDIA's NeMo Skills `prepare.py`, patched to drop 208 rows from the gated HLE source; 672 rows remain, 80 per core category.
- `VLLM_*` env names trigger vLLM warnings, so the config env var was renamed to `SERVE_ARGS`.

**Host facts (production pod `fbsy2njw0xt5mp`, EUR-IS-5):**
- 4x H200 at 143,771 MiB each, driver 595.91.07, CUDA 13.2.
- All pairs are NV18 (full NVLink).
- `/dev/shm` is 352 GB.
- The cgroup memory limit is **755 GB** (`RUNPOD_MEM_GB`), not the 1.1 TB the marketing page implies; the CPU quota is 68 vCPUs.
- Cold start, measured so far: create to container start took 54 s (image cached on host), then 7 s to sshd.
- **Hugging Face download to EUR-IS-5:** one `hf download` process sustains about 220-230 MB/s, and 4 more parallel shard processes (`deploy/fastdl.sh`) raise it to about 270-310 MB/s. The link is mostly network-bound, so 400 GB takes about 25 minutes.
- **Design implication for question 7:** a standard network volume (200-400 MB/s) would not load faster than Hugging Face does here. Re-downloading on every cold start costs about 25 min, or about $9.

## Step 1: deploy on 4x H200 (pod fbsy2njw0xt5mp, EUR-IS-5, $21.16/hr), 19:19Z onward

**Cold start, measured:**
- 19:19:33: create.
- 19:20:26: container start (image already cached on the host).
- 19:20:33: sshd up.
- 19:48:31: **weights ready, after 1,678 s (28 min) for 373 GiB**, about 230 MB/s average from Hugging Face. Adding 4 parallel shards partway through lifted it to about 270-310 MB/s.
- After that: weights load from local disk in **56 s**, and total model loading takes 130 s. The first CUDA-graph capture takes 196 s, and warm boots drop to **29 s** because the compile cache lives on the container disk.

**A container restart keeps the container disk.** An accidental PID-1 kill restarted the container at 19:57:05, and `start.sh` reported `weights ready in 1s (373G)`. Recovering from a crash is therefore one vLLM boot, about 5 min, with no re-download.

**B0 (TP4+EP, MTP3, 32K batched tokens, no DCP) does not boot.**
- vLLM reports `Model loading took 111.3 GiB` per GPU, not the 93 GiB the W4A8 card implies. The MTP draft model loads a separate 12.64 GiB checkpoint, and runtime layouts add the rest.
- After 3.58 GiB of CUDA graphs and activations, `Available KV cache memory: 6.71 GiB` remains. Because MLA's KV is replicated on every TP rank, the node holds only **about 131K tokens**, so `--max-model-len 262144` cannot be served and vLLM refuses to start.
- The engine also logs `Padding num_heads from 16 to 64 for FP8 sparse decode kernel`. At TP4 each rank has 16 heads, but the kernel's tile is 64, so decode attention does 4x wasted work.
- **Conclusion: without DCP, 4x H200 + W4A8 + MTP holds about one agent session. DCP is mandatory, not optional.**

**B1 = B0 + DCP4 sizes KV fine but OOMs in warmup.**
- KV sizing: `GPU KV cache size: 494,080 tokens` (3.8x B0) and `Maximum concurrency for 262,144 tokens: 1.88x`. DCP's all-gather also removes the 16-to-64 head padding.
- The failure: the warmup `_dummy_run` then fails inside `flash_mla_with_kvcache` (FP8 mixed-batch path) on an 8 GiB allocation, with 7.67 GiB free. That buffer scales with `max_num_batched_tokens` x the 64 gathered heads and is not covered by the memory profile.
- The fix is B2, which uses 16K batched tokens.

**Operational lessons:**
- **Never `pkill -f "vllm serve"`.** It matches PID 1, because the start script contains that text, and kills the container. `bench.sh apply` now matches `bin/vllm serve`, which only the real vLLM process has.
- **Do not request 5 GPUs.** GLM-5.3 has 64 heads and 256 experts with an expert width of 2048, so TP, EP and DCP only work at 2, 4 or 8 GPUs. A fifth GPU would sit idle.

**B3 = B2 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fails at load.** Right after weight loading it logs `expandable_segments: memory mapping failed with OOM` with 22 MB free per GPU, and the workers die. It is not viable alongside DCP's symmetric-memory collectives here.

**B2 = B1 with 16K batched tokens boots with 895,232 KV tokens (11.41 GiB per GPU) but OOMs on its first real 16K prefill chunk.**
- The memory: a 4.00 GiB allocation in `flash_mla_with_kvcache` with 3.53 GiB free.
- **Root cause:** in the FP8 mixed-batch path, every token (prefill included) goes through the sparse-decode kernel with 64 DCP-gathered heads. Its transient output and split buffers are about **256 KiB per batched token**: 8 GiB at 32K and 4 GiB at 16K.
- vLLM's memory profiler does not count this buffer, so the KV cache expands into the space it needs. Earlier I misread both 8 GiB OOMs as B2's; they were B1's first boot and its automatic retry.
- **Fix, B4:** 8K batched tokens (vLLM's H200 default) gives a 2 GiB buffer, which fits the roughly 3.5 GiB that remains above `gpu-memory-utilization 0.92`.
- Smaller batched tokens also shrink the profiled activation peak, which is why B2's KV (895K) was much larger than B1's (494K).

**Hung-engine failure mode:** after an engine OOM, the vLLM API server and EngineCore can stay alive with the GPUs already freed. The supervisor then waits forever. `bench.sh apply` now escalates from SIGTERM to SIGKILL after 30 s, and a long-running deployment must watch for "process alive, engine dead". `ready.sh` runs a real generation for that reason.

**Step 1 outcome: B4 is the working baseline** (`deploy/configs/B4-dcp4-bt8k.args`: TP4, EP, DCP4, FP8 `fp8_ds_mla` KV, MTP3, 8K batched tokens, util 0.92, 262K max length).
- **KV capacity:** `GPU KV cache size: 1,042,944 tokens`, max concurrency 3.98x at 262K. That is about 12 full-length OpenHands sessions (86K each) resident, against about 1 for B0.
- **Restart time:** a warm config swap takes **277 s** to the first real generation.
- **Smoke, through the public proxy from a Mac in the US to EUR-IS-5:**
  - Auth is enforced: 401 without the key, 200 with it.
  - Reasoning is split from the answer.
  - Tool calls parse: `get_weather({"city": "Paris"})` with `finish_reason=tool_calls`.
  - A 110 s stream survives the proxy.
  - Time to first answer token under concurrent quality-gate load: 0.39 s (low effort), 1.36 s (high), 4.3 s (max).
- **The 128K needle was retrieved** in 24 s wall, on the request shape that crashed B2.
- **Spend:** balance $173.226279286 at 20:36Z.

## Step 2: quality gate on B4 (20:33-20:47Z): PASS

| Check | Result | Reference / gate |
|---|---|---|
| GSM8K (1,319, zero-shot, high effort) | **96.74%**, 144 mean completion tokens, 0 truncated, 185 s at c=64 | gate 95% or more; FP8 reference 97.1-97.4% (SGLang cookbook, sgl-eval 1-shot) |
| MTP acceptance on SPEED-Bench coding (64 prompts, c=8, T=1.0) | **AL 2.95**, acceptance rate 65%, TPOT p50 14.2 ms (about 70 tok/s per user) | InferenceX golden AL at 3 drafts: 2.94 (within 0.3%) |
| Needle-in-a-haystack, 128K x 3 depths | 3/3 found; TTFT 23.3 s (5,600 tok/s prefill); decode 120-157 tok/s | exact retrieval |
| Needle-in-a-haystack, 200K x 3 depths | 3/3 found; TTFT 37.3 s (5,480 tok/s); decode 130-138 tok/s | exact retrieval |
| Tool calls, typed schemas (24 cases x stream/non-stream) | **48/48**, no retries needed | all must pass |

**Tool-call findings, informational (`glm47` parser on vLLM v0.33.0):**
- **Untyped parameters (`"value": {}`) always come back as strings.** That gives `"null"`, `"3"`, `"false"`, and stringified JSON objects. GLM-5.3 emits XML-style `<arg_key>/<arg_value>` pairs, so only the declared JSON-schema type lets the parser restore numbers, booleans, null and objects. **Agent tool schemas must declare a type for every parameter.**
- **Empty `properties` combined with `oneOf`** gives the same string coercion (`{"id": "42"}`), which reproduces zai-org/GLM-5 #169.
- **Literal `<tool_call>` text in prose** is sometimes parsed as a call. It failed in the first run while streaming and in the second run while not streaming, a flaky reproduction of vLLM #64326.

**Single-stream speed at long context (no other load):** decode stays at 120-157 tok/s at 128K-200K context. That is about 2x Z.ai's own API at 66 tok/s, because sparse attention keeps attention cost flat and MTP3 contributes about 2.95 tokens per step. Cold prefill runs at about 5.5K tok/s, so a 74K-token cold first turn costs about 13 s.

**Gate A, quality half: W4A8 + FP8 sparse KV + DCP4 + MTP3 preserves quality.** No fallback to FP8 or 8x H200 is needed.

## Step 3: baseline performance on B4 (20:45-21:35Z): 4 sessions at SLO, $5.29 per session-hour

**Workload:** the vLLM GLM-5.3 blog's OpenHands multi-turn workload on EvalScope (commit acd09b44).
- Each conversation has 13 turns. The first turn is 74K tokens, unique per conversation. Each later turn adds about 1K tokens, and every turn emits 220 tokens.
- Each concurrency point runs 2c conversations, with disjoint dataset slices per point.

**The workload overstates MTP, and the report now corrects for it.**
- **The artifact:** the workload sends `ignore_eos: true` at temperature 0. Once the model would naturally stop, it is forced on and degenerates into loops ("Let me start by exploring the repository structure." repeated). Decoded samples are in `evalscope/agentic_c4/.../benchmark_data.db`.
- **The effect:** MTP predicts those loops almost perfectly, so the run's acceptance length is **3.8 of a possible 4**. On real coding prompts (SPEED-Bench) it is 2.95.
- **The correction:** `report.py` derives the engine step time as TPOT x the run's own acceptance length. That is the hardware quantity; verify cost does not depend on acceptance. It then reports `inter_real` = real acceptance / step time. Raw TPOT overstates per-user speed by about 29%.
- The SLO uses the median per-request TPOT, as `docs/DESIGN.md` specifies; until now the report had used EvalScope's mean.

| c | TTFT p50 / p90 / p99 | ITL p50 / p99 (one engine step) | Per-user tok/s at real AL, median (mean) | Logical tok/s per GPU | Prefix hit | SLO |
|---|---|---|---|---|---|---|
| 1 | 0.43 / 0.48 / 12.9 s | 23 / 29 ms | **135** (131) | 7,431 | 91.8% | pass |
| 4 | 0.50 / 0.75 / 39.4 s | 34 / 1,413 ms | **69** (39.5) | 11,481 | 91.8% | pass |
| 8 | 0.58 / 1.11 / 77.3 s | 43 / 1,421 ms | **44.6** (20.2) | 13,166 | 91.8% | fail |
| 16 | collapsed: 0 of 32 conversations finished in about 20 min | | | | 91.8% falling to 71% | fail |

**What limits per-user speed: other sessions' cold prefills, not decode.**
- **Two kinds of steps:** ITL is bimodal.
  - A pure decode step takes 23-43 ms for 1-8 users.
  - A step carrying an 8K prefill chunk takes about **1.41 s**, the ITL p99.
- **Cold first turns:** each new conversation's 74K-token first turn is 9 such chunks, about 13 s of compute. Every co-running session stalls through it.
- **Effect on speed:** the median request between stalls stays fast (69 tok/s at c=4), but the mean, which counts the stalls in full, is 39.5.
- **Effect on TTFT:** p99 TTFT (39 s at c=4) is a cold first turn queued behind another cold first turn.
- **Prefill to decode ratio:** prompt tokens computed per turn exceed decoded tokens by about 30:1.

**Prefill runs at about 5.5-5.8K tok/s whatever the context length.**
- The evidence:
  - an 8K random prompt takes 1.40 s to first token (reference c=1);
  - a 128K needle prefills at 5.6K tok/s;
  - a 200K needle at 5.5K tok/s.
- Sparse attention keeps attention cost flat, so prefill is bound by the MoE and linear layers (W4A8 CUTLASS grouped GEMM plus EP/DCP collectives). That is about 1.4K tok/s per GPU, or roughly 16% of FP8 peak FLOPs.
- **Prefill throughput is the lever for this workload.** Rung L5p tests bigger chunks, which give more tokens per expert per step.

**The c=16 collapse is a prefix-cache working-set cliff, not KV exhaustion.**
- **What the engine logged** (`s3-baseline/c16_engine_stats.txt`): KV usage stayed at 45-55%, with 7 running and 9 waiting. Prompt throughput ran at about 5.5K tok/s against 16 tok/s of generation, and the cumulative prefix hit fell from 91.8% to 71%.
- **Why the gauge looked healthy:** it counts only blocks held by running requests. The cached history of sessions waiting between turns is evictable.
- **Why it collapsed:** 16 sessions x about 85K tokens is a 1.36M-token working set against 1,042,944 tokens of KV. LRU eviction destroys waiting sessions' prefixes, they re-prefill 74-85K tokens, that steals the prefill budget, and fewer turns complete.
- **The cliff on B4** sits at about 12 sessions of this size: 12 x 85K is roughly equal to capacity.
- **This is what CPU KV offload (L2) addresses:** a 300 GB tier holds about 5.5M more tokens. It matters even more for long-running hosting, where many idle sessions should keep their context warm between turns.

**Reference shapes (random 8192 in / 1024 out, `vllm bench serve`):**

| c | TTFT p50 | TPOT p50 | tok/s per user | Output tok/s | Total tok/s per GPU | MTP AL |
|---|---|---|---|---|---|---|
| 1 | 1.40 s | 13.9 ms | 72 | 64 | 144 | 1.40 |
| 16 | 1.58 s | 51.2 ms | 19.5 | 257 | 579 | 1.36 |
| 64 | 5.18 s | 109.5 ms | 9.1 | 390 | 877 | 1.37 |

Random token IDs give MTP almost nothing to predict (acceptance length 1.37-1.40), so this shape understates real decode speed by about 2x. Use it only against published random-shape numbers. For example, the SGLang cookbook's GLM-5 figures come from FP8 on 8 GPUs, so they are not directly comparable.

**Economics at the SLO point (c=4, $21.16/hr):**
- **Per session-hour:** **$5.29**.
- **Per agent turn:** **about $0.010.** That is one turn with 80K tokens of context and 220 tokens out; 2,068 turns per hour.
- **Per logical token:** $0.128 per 1M, counting cached prompt tokens. Logical throughput is 45.9K tok/s, of which about 8% is computed.
- **Per output token:** $46.5 per 1M. This is meaningless for agents, because each turn emits only 220 tokens.
- **At c=8:** past the SLO, $0.0089 per turn. At c=1: $0.016 per turn.

**Gate A: PASS.** Quality holds (Step 2) and the baseline serves 4 concurrent agent sessions at SLO on 4x H200 + W4A8, so there is no fallback to 8 GPUs. The ladder continues on evidence:
- **L3 (MTP schedule):** more accepted tokens per step at c <= 4.
- **L5p (16K chunks):** prefill efficiency, the actual bottleneck.
- **L2 (CPU KV offload):** removes the working-set cliff; c=16 runs only time-capped.

**Spend:** balance $153.76 at 21:30Z ($46.24 spent; about 7.3 pod-hours left).

## Step 4, rung L3: MTP per-batch-size schedule (21:35-21:54Z): KEPT (plan's keep rule)

**Config:** `deploy/configs/L3-mtp-schedule.args`, which is B4 with `num_speculative_tokens_per_batch_size` [[1,4,5],[5,16,3],[17,1024,1]]. That gives 5 drafts at batch sizes up to 4 and 3 drafts from 5 to 16.

**Restart and quality:**
- Restart to ready took 297 s.
- The spot gate passed: tool calls 48/48, GSM8K-200 98.5%, 128K needle found.
- KV capacity is **965,376 tokens**, 7.4% below B4's 1,042,944, because deeper drafts need more buffers.

| | c=1 | c=4 | c=8 (3 drafts in both) |
|---|---|---|---|
| Real-prompt AL (SPEED-Bench coding; L3 measured at c=4) | 2.95 to **3.54** (+20%) | 2.95 to 3.54 | 2.95 |
| Engine step (ITL p50) | 23.1 to 26.3 ms | 33.7 to 40.0 ms | 43.2 to 44.0 ms |
| Per-user tok/s at real AL, median | 135 to **146** (+8%) | 69.0 to **71.6** (+4%) | 44.6 to 44.0 |
| Per-user tok/s, mean | 131 to 139 | 39.5 to 39.5 | 20.2 to 20.6 |
| TTFT p50 | 429 to 397 ms | 504 to 569 ms | 577 to 573 ms |
| Time per 220-token turn at real AL (TTFT + 219 x step / AL) | 2.05 to 1.90 s (-7%) | 3.68 to 3.63 s (-1%) | unchanged |

**Why the gain is small:**
- Verifying 6 tokens instead of 4 makes each step 11-16% slower, which eats most of the 20% acceptance gain.
- At c=4 the mean is set by other sessions' prefill stalls, which more drafts do not touch.
- The vLLM #65312 result (a schedule beats fixed k=5 at c=256) is about large batches. In our SLO region the schedule only adds depth.

**Decision: keep.**
- **The plan's rule:** keep a rung when max sessions at SLO stays level and c=1 interactivity rises by 5% or more.
- **L3 meets it:** sessions stay at 4, and c=1 rises 8% at real acceptance (28% on the raw, artifact-inflated numbers).
- **I first reverted it** on the costs below, but that overrode the agreed rule. Those costs go to the final report as the trade-off instead:
  - **KV:** 7.4% less, which moves the working-set cliff from about 12.3 to about 11.4 sessions. L2's CPU tier is meant to absorb this.
  - **Context:** a 1M context no longer fits in GPU KV (965K tokens). L4 measures the ceiling on the final config.
  - **TTFT:** 13% higher p50 at c=4, about 65 ms.
- **What it is worth:** +8% applies to every token of long single-user generations, such as writing whole files.
- **Already-queued rung:** L5p is measured on B4 (one variable). If both are kept, the combination is validated in Step 6.

**Lesson for the eval design:** acceptance must be measured at the concurrency that selects the draft count. A per-batch-size schedule measured at c=8 would have reported B4's 3-draft AL, so this rung measured it at c=4.

## Step 4, rung L5p: 16K prefill chunks at util 0.91 (22:00-22:12Z): REVERTED (spot gate failed)

**Config:** `deploy/configs/L5p-bt16k-util091.args`, which is B4 with `--max-num-batched-tokens 16384` and `--gpu-memory-utilization 0.91`. It is measured against B4 (one variable), started before L3 was kept.

**What happened:**
- **Boot:** KV capacity is **785,664 tokens**, 25% below B4.
- **Crash:** the tool-call suite passed 48/48. Then the 128K needle crashed the engine on its first 16K chunk: `CUDA out of memory. Tried to allocate 4.00 GiB ... 3.22 GiB free`, the same allocation that killed B2.
- **Lower utilization did not help:** at 0.91 there was *less* free memory at the crash than B2 had at 0.92 (3.53 GiB). That suggests the 1% of KV given up was absorbed elsewhere, probably by CUDA-graph memory and allocator fragmentation, not left as headroom.
- **Fitting 16K chunks** would need utilization of about 0.89 and about 33% less KV than B4.
- **After the supervisor's automatic restart,** agentic c=1 ran 52 requests with no crash: 74K cold turns at 16K chunks fit, and 128K context does not.

**The c=1 run answers the rung's real question: bigger chunks barely speed up prefill.**

| | 8K chunks (B4) | 16K chunks (L5p) |
|---|---|---|
| Cold 74K first turn, TTFT p95 / p99 | 12.88 / 12.91 s | 12.52 / 12.60 s (-2.8%) |
| Warm-turn TTFT p50 | 429 ms | 422 ms |
| TPOT p50 (raw) | 5.74 ms | 6.10 ms |

**Interpretation:**
- Prefill time is linear in tokens at about 5.8K tok/s and barely depends on tokens per step.
- Doubling tokens per expert per step did not help, so the bound is per-token compute in the W4A8 / attention kernels, not GEMM shape or launch overhead.
- Chunk size is therefore not a throughput lever on this stack. It only trades the length of each decode stall (the 1.41 s ITL p99 is one 8K chunk) against KV memory, because the unprofiled FlashMLA buffer is 256 KiB per batched token.
- **The opposite direction is the one worth a rung:** 4K chunks would roughly halve each stall, return about 1 GiB per GPU to KV, and should cost only a few percent of prefill speed.

**Decision: revert** (the plan's rule: the spot gate failed). The best config stays B4 + L3.
