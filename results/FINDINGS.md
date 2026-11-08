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
