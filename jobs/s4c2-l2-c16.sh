# Rung L2, c=16 point (rerun: EvalScope at the pinned commit has no --duration). 16 conversations on the B4 sweep's
# c=16 slice, capped at 25 min; B4 completed 0 of 32 in about 20 min at this concurrency.
set -x
AGENTIC_OFFSET=28 AGENTIC_CONVS=16 AGENTIC_TIMEOUT=1500 bash evals/bench.sh agentic 16
