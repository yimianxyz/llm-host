# Rung L5p: 16K prefill chunks at util 0.91 (KV traded for prefill throughput; the agentic workload is ~30:1 prefill:decode).
# Includes a 128K needle first: that request shape crashed B2.
set -x
bash evals/bench.sh apply deploy/configs/L5p-bt16k-util091.args || exit 1
bash evals/bench.sh gate-spot
bash evals/bench.sh agentic 1 4 8 16
