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
