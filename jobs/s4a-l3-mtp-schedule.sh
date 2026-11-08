# Rung L3 (run first: at c=4 per-user speed is the SLO limit while KV has ~3x headroom). Spot gate, MTP acceptance, agentic subset.
# Acceptance is measured at c=4 so the schedule uses 5 drafts; at c=8 it uses 3, the same as B4.
set -x
bash evals/bench.sh apply deploy/configs/L3-mtp-schedule.args || exit 1
bash evals/bench.sh gate-spot
AL_CONC=4 bash evals/bench.sh mtp-al
bash evals/bench.sh agentic 1 4 8
