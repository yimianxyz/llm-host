# Step 6: full characterization of the final config: B4 + L2 CPU offload + L7 DSpark (replacing L3's MTP), 256K context.
# P1 at c=1-16 pins the SLO boundary (c=1 repeats s5a, a run-to-run noise check), then the SGLang reference
# shapes, then the full quality gate (tool calls, 128K/200K needles at 3 depths, all 1,319 GSM8K, DSpark acceptance).
set -x
bash evals/bench.sh apply deploy/configs/L7-dspark.args || exit 1
AGENTIC_OFFSET=100 AGENTIC_TIMEOUT=1800 bash evals/bench.sh agentic 1 2 4 6 8 12 16
bash evals/bench.sh reference 1 16 64
bash evals/bench.sh gate-full
