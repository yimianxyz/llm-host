# Rung L3: MTP per-batch schedule (applied on top of the L2 decision, see FINDINGS). Spot gate, MTP acceptance, agentic subset.
set -x
bash evals/bench.sh apply deploy/configs/L3-mtp-schedule.args || exit 1
bash evals/bench.sh gate-spot
bash evals/bench.sh mtp-al
bash evals/bench.sh agentic 1 8 16 32
