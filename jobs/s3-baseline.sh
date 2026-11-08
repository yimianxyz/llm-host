# Step 3: baseline numbers. P1 agentic sweep (vLLM blog OpenHands workload) then P2 reference shapes.
set -x
bash evals/bench.sh agentic 1 4 8 16 24 32
bash evals/bench.sh reference 1 16 64
