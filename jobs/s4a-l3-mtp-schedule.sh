# Rung L3 (run first: at c=4 per-user speed is the SLO limit while KV has ~3x headroom). Spot gate, MTP acceptance, agentic subset.
set -x
bash evals/bench.sh apply deploy/configs/L3-mtp-schedule.args || exit 1
bash evals/bench.sh gate-spot
bash evals/bench.sh mtp-al
bash evals/bench.sh agentic 1 8 16 32
