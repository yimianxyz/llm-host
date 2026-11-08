# Rung L4: context ceiling. Needle at 512K and about 1M (prompt + 2,048 output tokens must fit max-model-len), then
# one ~1M-token request for TTFT and TPOT at full context. Falls back to util 0.92 / 960K if util 0.93 does not boot,
# so a failed boot cannot leave vLLM in a crash loop.
set -x
L=1040000
if ! bash evals/bench.sh apply deploy/configs/L4-ctx1m.args; then
  bash evals/bench.sh apply deploy/configs/L4b-ctx960k.args || exit 1
  L=950000
fi
bash evals/bench.sh niah 524288 $L
REF_IN=$L REF_OUT=256 REF_N=1 REF_WARMUPS=0 bash evals/bench.sh reference 1
