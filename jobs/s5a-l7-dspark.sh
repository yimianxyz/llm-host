# Step 5 rung L7: DSpark drafter (RedHatAI/GLM-5.3-speculator.dspark, 1B BF16, 8 drafts in one pass) instead of MTP.
# Two expected gains: higher acceptance (card: 4.0 tool calls, 4.8 HumanEval vs MTP 2.95/3.54 here) and the KV freed
# by dropping the 12.64 GiB MTP layer. Acceptance has no per-batch schedule, so one measurement at c=8 covers c<=8.
set -x
bash evals/bench.sh apply deploy/configs/L7-dspark.args || exit 1
bash evals/bench.sh gate-spot
AL_CONC=8 bash evals/bench.sh mtp-al
bash evals/bench.sh agentic 1 4 8
