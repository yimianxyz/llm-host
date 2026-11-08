# Rung L2 on B4 + L3: CPU KV offload. c=8 fits in GPU KV (offload overhead check); c=16 is a 1.36M-token working set,
# past the ~965K-token GPU pool, where B4 collapsed. c=16 is capped at 10 minutes. Offsets match the B4 sweep's slices.
# The prefill profile (one cold 32K prompt, four 8K chunks) runs last.
set -x
bash evals/bench.sh apply deploy/configs/L2-offload300.args || exit 1
bash evals/bench.sh gate-spot
AGENTIC_OFFSET=12 bash evals/bench.sh agentic 8
AGENTIC_OFFSET=28 AGENTIC_DURATION=600 bash evals/bench.sh agentic 16
bash evals/bench.sh profile
