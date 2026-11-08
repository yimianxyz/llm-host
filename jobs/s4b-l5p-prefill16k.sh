# Rung L5p: 16K prefill chunks at util 0.91 (KV traded for prefill throughput; the agentic workload is ~30:1 prefill:decode).
# The spot gate's 128K needle comes first: that request shape crashed B2. The profile (one cold 32K prefill) runs last
# so the profiler is idle during the measurements.
set -x
bash evals/bench.sh apply deploy/configs/L5p-bt16k-util091.args || exit 1
bash evals/bench.sh gate-spot
bash evals/bench.sh agentic 1 4 8
bash evals/bench.sh profile
