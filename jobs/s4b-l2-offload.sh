# Rung L2: CPU KV offload. Protocol: warm restart, spot gate, agentic subset c=1,8,16,32 (16 and 32 exceed GPU KV).
set -x
bash evals/bench.sh apply deploy/configs/L2-offload300.args || exit 1
bash evals/bench.sh gate-spot
bash evals/bench.sh agentic 1 4 8 16
