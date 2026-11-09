# Step 6: full characterization of the final candidate, L5d (L2 + CUDA graphs captured up to 128 tokens).
# Idle smoke first (time to first answer token), then P1 at c=1-16 to pin the SLO boundary; its c=1/4/8 points are
# also L5d's rung test against L3/L2. Then the SGLang reference shapes and the full quality gate, with acceptance
# measured at c=4 (5 drafts under the schedule; 3 drafts were measured in Step 2).
set -x
bash evals/bench.sh apply deploy/configs/L5d-cg128.args || exit 1
bash evals/bench.sh smoke
AGENTIC_OFFSET=100 AGENTIC_TIMEOUT=1800 bash evals/bench.sh agentic 1 2 4 6 8 16
bash evals/bench.sh reference 1 16 64
AL_CONC=4 bash evals/bench.sh gate-full
