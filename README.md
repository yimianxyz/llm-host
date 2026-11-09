# llm-host: self-hosted GLM-5.3 on RunPod

GLM-5.3 (753B MoE) served for agentic coding on **one RunPod pod with 4x H200 SXM**:

- **Checkpoint:** `gpustack/GLM-5.3-W4A8`
- **Engine:** vLLM v0.33.0
- **Parallelism:** TP4 + EP + decode context parallel 4
- **KV cache:** FP8 sparse-MLA, plus a 300 GiB host-memory tier
- **Speculative decoding:** MTP with a per-batch-size draft schedule

It runs at $21.16/hr. The config is [`deploy/configs/recommended.args`](deploy/configs/recommended.args).
- **Results** and the answers to the eight design questions: [results/REPORT.md](results/REPORT.md).
- **Why each choice was made:** [docs/DESIGN.md](docs/DESIGN.md).
- **The step-by-step measurement log:** [results/FINDINGS.md](results/FINDINGS.md).

## Layout

| Path | What |
|---|---|
| `deploy/configs/*.args` | vLLM flags, one per line. `recommended.args` is the result; `B0.args` the starting point; every experiment is a new file |
| `deploy/start.sh` | Pod PID 1: deadline watchdog, sshd, weights download, vLLM restart loop |
| `deploy/pod.py` | Renders and creates the pod over RunPod REST v2 (same API as the RunPod MCP) |
| `deploy/push.sh`, `deploy/runner.sh` | Ship code and jobs to the pod; the on-pod job queue (idle = soak) |
| `deploy/sync.sh` | Pull results into `results/raw/` and git-commit (`--loop` every 5 min) |
| `evals/smoke.py` | Functional checks through the public proxy |
| `evals/quality.py` | GSM8K, tool-call suite, needle-in-a-haystack, GPQA-Diamond |
| `evals/bench.sh` | Agentic multi-turn (EvalScope, vLLM blog workload), reference shapes, MTP acceptance, soak, probes |
| `evals/metrics.py` | `/metrics` snapshots and diffs: MTP acceptance, prefix-cache hits, preemptions, KV usage |
| `clients/opencode.jsonc` | OpenCode v2 provider config for the pod (endpoint, limits, reasoning field, effort variants) |
| `results/` | `FINDINGS.md` (per-step log), `budget.md`, `raw/` (everything pulled from pods) |

## Run it

1. **Credentials.** Copy `.env.example` to `.env` and fill in `RUNPOD_API_KEY` and `HF_TOKEN`. `VLLM_API_KEY` is generated.
2. **One-time account setup.** Run `python3 deploy/pod.py ensure-ssh-key && python3 deploy/pod.py ensure-secrets`.
3. **Check stock.** Run `python3 deploy/pod.py stock --gpu-id "NVIDIA H200" --count 4`.
4. **Create the pod.** Run `python3 deploy/pod.py create --config deploy/configs/recommended.args`. Or use the RunPod MCP `create-pod` with the body from `pod.py render`.
5. **Start the session.**
   - Wait for SSH: `python3 deploy/pod.py ssh $POD_ID`, which writes `.env.pod`.
   - Then run `deploy/push.sh` and `deploy/sync.sh --loop &`.
6. **Wait for first boot, then smoke-test.**
   - Queue a job containing `bash evals/bench.sh wait`.
   - Then run `VLLM_API_KEY=... python evals/smoke.py --base-url "$PROXY_URL" --out results/smoke.json`.
7. **Clients.** Use the OpenAI-compatible API at `$PROXY_URL/v1` with `Authorization: Bearer $VLLM_API_KEY`, and always stream. For OpenCode, use [`clients/opencode.jsonc`](clients/opencode.jsonc); its header comment shows how to install the key file.
8. **Tear down.** Run `python3 deploy/pod.py delete $POD_ID`. Nothing keeps billing, because there is no volume.
