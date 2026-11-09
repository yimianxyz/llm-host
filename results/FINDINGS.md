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

## Step 4, rung L2: CPU KV offload on B4 + L3 (22:12-22:37Z): KEPT

**Config:** `deploy/configs/L2-offload300.args`, which is B4 + L3 + `--kv-offloading-size 300 --kv-offloading-backend native`, plus an inert `--profiler-config`.
- **GPU KV is unchanged** at 965,376 tokens; the CPU tier uses only host RAM.
- **Restart to ready takes 357 s**, 60 s more than L3, for pinning 300 GiB of host memory.
- **The spot gate passed:** tool calls 48/48, the 128K needle, GSM8K-200.
- **The tier is write-through:** every computed block is copied to host memory, about 3.1 GB per 10 s in about 0.17 s of transfer time.

| | c=8 (fits in GPU KV) | c=16 (1.36M-token working set) |
|---|---|---|
| B4 / L3 without offload | 208 turns; prefix hit 91.8% (all GPU) | **collapse:** 0 of the first 16 conversations finished in about 20 min; hit fell from 91.8% to 71% |
| L2 | 208 turns; prefix hit 91.5% (all GPU) | **16 of 16 conversations (208 turns) in 5.5 min, 0 failures; hit 91.3%, served 91.25% from host memory and 0.5% from GPU** |
| Overhead vs L3 | median TPOT 17.75 to 20.23 ms; logical throughput 13,131 to 12,560 tok/s per GPU (-4.3%); ITL p50 unchanged (44 ms) | n/a |

**Why GPU hits drop to 0.5% at c=16:**
- LRU caches fail on a cyclic working set larger than the cache: each waiting session's blocks are evicted just before its next turn.
- The 300 GiB host tier holds about 5.5M tokens, so every one of those misses becomes a reload over PCIe instead of a recompute. An 85K-token history is about 4.4 GB, so it reloads in a fraction of a second, against about 13 s of prefill.
- c=16 is still far past the SLO (TTFT p50 3.9 s, mean TPOT 51 ms, 12,648 logical tok/s per GPU, which is prefill-saturated like c=8), but the engine degrades gracefully instead of collapsing.

**Decision: keep.**
- **The generic keep rule does not apply:** max sessions at SLO is unchanged and c=1 was not re-measured.
- **The plan's L2 test does:** "prefixes survive concurrency beyond HBM; test it past the cliff". That test is decisive.
- **Why it matters for long-running hosting:** the KV working set is every recently active session's history, not just the running ones. Without the tier, a returning user costs a 13 s cold prefill that also stalls everyone else.
- **The cost:** about 4% of throughput where offload is not needed, and 60 s on restart.

## Prefill profile (one cold 26,228-token prompt, B4 + L3 + L2; `s4c-l2-offload/profile/`)

The prompt took 4.71 s, which is 5.57K tok/s: three 8K chunks at about 830 ms of GPU time each, plus a remainder. The CUDA time on each rank is nearly identical:

| Kernel | Share of 4.57 s CUDA time |
|---|---|
| `_flashmla_C::sparse_decode_fwd` (FlashMLA **sparse decode**, split-KV, SM90) | **36%** |
| DCP and TP collectives: `all_gather` 9.4%, `all_reduce` 6.7%, DCP `all_to_all` 6.3%, pack/unpack 3.4% | **about 26%** |
| `cutlass_w4a8_moe_mm` (all routed experts) | **7.0%** |
| DeepGEMM FP8 (indexer and projections) | 4.5% |
| copies, `masked_fill`, elementwise, quantization | about 15% |

**Prefill is not MoE-bound; it is attention-path bound.** vLLM's FlashMLA-sparse backend has two FP8 modes (`flashmla_sparse.py`):
- **Mixed batch (the one in use):** every token, prefill included, goes through the FP8 *decode* kernel.
- **Separate:** KV is upconverted to BF16 and run through the sparse *prefill* kernel.

**Why this deployment cannot leave mixed-batch mode:**
- **TP4 forces it:** each rank has only 16 heads, below `MIN_HEADS_FOR_BF16_PREFILL = 32`, because the prefill kernel pads to 128 heads.
- **DCP requires it:** DCP is only supported on the mixed-batch path, since the DCP merge needs the LSE for every token. The one exception is PCP+DCP.
- **PCP does not fit here:** it means TP2 x PCP2, which doubles each GPU's attention and dense weights (about 8 GiB more per GPU) and leaves almost no KV.
- **HiSparse (plan rung L5) is ruled out the same way:** it disables mixed-batch mode, so it raises `NotImplementedError` with DCP, and without DCP the KV is replicated (B0: about 131K tokens).

**Implications:**
1. On 4x H200 with W4A8, prefill speed (about 5.6K tok/s) is set by this kernel mismatch and by DCP collectives, so it is structural.
2. On 8x H200 with FP8 and TP8, each rank has 8 heads, so it is still mixed-batch. The fast prefill path needs 32 or more heads per rank, i.e. TP2 or DP attention.
3. The cheap lever is `--dcp-q-replicate`: replicate the query projection so each step skips the query all-gather (the largest collective here, 9.4%). It costs about 2 GiB of weights per GPU, which is affordable only if DSpark frees the MTP layer's 12.64 GiB.

## Step 5, rung L7: DSpark drafter instead of MTP, on B4 + L2 (22:46-23:27Z): REVERTED (collapses at c=8)

**Config:** `deploy/configs/L7-dspark.args`, which is the L2 config with `RedHatAI/GLM-5.3-speculator.dspark` (a 1B BF16 drafter that proposes 8 tokens in one pass) replacing MTP.
- **The first boot failed:** `No valid attention backend ... head_size=64, kv_cache_dtype=fp8_ds_mla, use_mla=False`. The drafter is not MLA, so it needs its own KV dtype: `"kv_cache_dtype":"auto"` in the speculative config. It then uses FLASH_ATTN.
- **Memory:** model loading took 107.14 GiB per GPU against 107.27 GiB for MTP, so the expected 12.64 GiB MTP saving did not materialize. Available KV memory fell from 12.3 to 7.8 GiB per GPU, and the drafter's BF16 KV adds about 3.6 KB per token per GPU. **The GPU KV pool halves: 484,557 tokens, against 965,376.**
- **Restart to ready:** 316 s.
- **The spot gate passed:** tool calls 48/48, GSM8K-200 98.5%, 128K needle found. But that needle prefilled at 4,787 tok/s against 5,589 (-14%).

**Acceptance on real coding prompts (SPEED-Bench, c=8):** AL **3.92** with 8 drafts, against 2.95 for MTP3 and 3.54 for MTP5. Per-position acceptance is 0.76, 0.57, 0.44, 0.35, 0.27, 0.22, 0.17, 0.14. Median TPOT on that run is 13.07 ms against MTP3's 14.19 ms.

| | c=1 | c=4 | c=8 |
|---|---|---|---|
| Per-user tok/s at real AL, median: L3 MTP to DSpark | 145.9 to **185.0 (+27%)** | 71.6 to **80.2 (+12%)** | 44.0 to **collapse** |
| Per-user tok/s, mean (prefill stalls counted in full) | 139.1 to 156.4 (+12%) | 39.5 to **33.0 (-16%)** | 20.6 to collapse |
| Engine step, ITL p50 | 26.3 to 25.8 ms | 40.0 to 42.0 ms | |
| Prefill-chunk step, ITL p99 | 1.39 to 1.67 s (+20%) | 1.42 to 1.67 s | |
| TTFT p50 / p99 | 0.40 / 13.0 to 0.46 / 15.2 s | 0.57 / 39.5 to 0.55 / 46.4 s | |
| Logical tok/s per GPU | 8,342 to 7,911 (-5%) | 12,156 to **10,650 (-12%)** | |

**Why c=8 collapsed:**
- **The run:** after 21 minutes, 81 turns had finished and 0 of 16 conversations, so I killed it. L3 finished all 208 turns in about 9 minutes. The engine averaged 4.6 running and 3.0 waiting requests, 4.8K tok/s of prompt against 22 tok/s of generation.
- **GPU prefix hits fell to 0%** over the window (91.8% for L3 at c=8); all 6.6M prompt tokens of the 81 turns were recomputed. Eight sessions of 74-86K tokens (0.6-0.69M) exceed the 484K pool, and LRU on a cyclic working set evicts each session just before its next turn. This is B4's c=16 cliff, moved down to between 4 and 8 sessions.
- **The CPU tier did not rescue it:** only 6.4% of lookups hit host memory, against 91.25% for L2 at c=16. The likely cause is the drafter's second KV-cache group, since a prefix hit needs blocks for every group. I did not verify the mechanism.

**Decision: revert.**
- **The plan's rule:** sessions stay at 4 and c=1 rises 27%, which alone would keep it. But "revert if any errors appear" applies: the c=8 point never completed.
- **The deeper reason:** DSpark undoes the property L2 was kept for, graceful degradation past HBM. A long-running service does not control its load, and a collapse at 8 sessions is disqualifying.
- **It also rules out 1M context,** since the pool is 484K tokens.
- **Slower prefill costs throughput:** decode is faster, but slower prefill cuts throughput 12% at c=4 on this prefill-heavy workload.
- **When DSpark is the right choice:** a single-user or at most 4-session deployment (a personal coding agent) gets +27% / +12% decode speed. Untested fixes that could recover KV: FP8 drafter KV (FLASH_ATTN supports it on Hopper), or offload support for the drafter's KV group.
- **Correction to the prefill-profile note:** DSpark frees no weight memory, so `--dcp-q-replicate` (about 2 GiB per GPU) stays unaffordable.

The best config stays L2 (B4 + L3 + L2).

## Step 4, rung L4 (first attempt): 1M context on L2 at util 0.93 (23:26-23:31Z): DID NOT BOOT

**Config:** `deploy/configs/L4-ctx1m.args`, which is L2 with `--max-model-len 1048576` and util 0.93 instead of 0.92. At 0.92 the GPU KV pool is 965,376 tokens, short of one 1M request, and 0.93 should have added 1.4 GiB per GPU.

**What happened:** `ValueError: To serve at least one request with the model's max seq len (1048576), 13.36 GiB KV cache is needed, which is larger than the available KV cache memory (8.35 GiB) ... the estimated maximum model length is 655104.`
- **The cause is CUDA-graph memory, which grows with max-model-len.** vLLM counts graph memory inside the util budget ("0.93 is equivalent to 0.8657 without CUDA graph memory profiling"). That is **9.0 GiB per GPU at 1M, against 4.5 GiB at 256K.**
- **The likely mechanism:** the sparse indexer's decode buffers are sized as captured tokens x max-model-len, and the default captures up to 512 tokens per step.
- **The activation peak also grew** by about 0.85 GiB.
- **Raising util is the wrong lever:** one point of util is 1.4 GiB, and the graphs took 4.5 GiB more.
- **The fallback (960K at 0.92) would have failed the same way,** so I stopped the job instead of letting it crash-loop.

**Next:** L5c removes exactly those buffers. It caps graph capture at 128 tokens per step, which the SLO region never exceeds, and uses 4K prefill chunks. L4 is retried on top of it in the same boot.

## Step 4, rung L5c: 4K prefill chunks + CUDA graphs captured up to 128 tokens (23:41-23:58Z): 4K chunks REVERTED, capture cap KEPT (as L5d)

**Config:** `deploy/configs/L5c-bt4k-cg128.args`, which is L2 with `--max-num-batched-tokens 4096 --max-cudagraph-capture-size 128`, at 256K. The 1M variant on top of it failed (next entry), so the job fell back to 256K.

**Boot:** the capture cap is a clear win.
- **Graph memory:** 1.9 GiB per GPU, against 4.5 GiB.
- **Capture time:** 14 + 42 s, against 32 + 114 s, so **restart to ready takes 257 s, against 357 s**.
- **KV:** available memory is 16.46 GiB per GPU, against 12.3, so the pool is **1,292,032 tokens (+34%)**. Part of that gain comes from the smaller activation peak at 4K.
- **The spot gate passed:** tool calls 48/48, GSM8K-200 98.5%, 128K needle found.

**4K chunks are a loss:**

| | c=1: L3 to L5c | c=4: L3 to L5c |
|---|---|---|
| Per-user tok/s at real AL, median | 145.9 to 149.5 (+2.5%) | 71.6 to **63.5 (-11%)** |
| Per-user tok/s, mean | 139.1 to 135.9 | 39.5 to **31.5 (-20%)** |
| Prefill-chunk step, ITL p99 | | 1.42 to 0.89 s |
| TTFT p50 / p90 / p99 | 0.40 / 0.47 / 13.0 to 0.49 / 0.55 / 16.1 s | 0.57 / 1.16 / 39.5 to 0.57 / 2.04 / 48.4 s |
| Logical tok/s per GPU | 8,342 to 7,188 | 12,156 to **10,110 (-17%)** |

- **The cause:** the 128K needle prefilled at **4,534 tok/s against 5,589 (-19%)**. Each stall is shorter, but there are more than twice as many, so co-running sessions lose more time overall.
- **The arithmetic:** a prefill step has a large fixed part, about 0.34 s at 128K context, given an 8K chunk at 1.47 s and a 4K chunk at 0.90 s.
- **Combined with L5p,** 16K chunks gained only 2.8% on cold turns, so 8K is close to the optimum on this stack.
- **Better prefill speed must come from fewer fixed costs per step,** such as collectives and the decode-kernel path. Chunk size is not the lever.

**c=8 was skipped:** c=1 and c=4 already decide the rung.

**Decision:**
- **Revert the 4K chunks:** c=4 drops 11% and c=1 rises less than 5%.
- **Keep the capture cap:** under the MTP schedule no step at 16 or fewer sessions exceeds 64 decode tokens, so it should be runtime-neutral, and it gives back 2.6 GiB per GPU and 100 s per restart.
- **The combination is L5d** (`deploy/configs/L5d-cg128.args`, L2 + cap 128 at 8K chunks). Step 6 characterizes it, and its c=1/4/8 points are the rung test against L3/L2.
