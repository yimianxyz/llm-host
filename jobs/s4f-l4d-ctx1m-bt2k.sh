# Rung L4, third attempt: 1M context at 2K prefill chunks (see the config header). Needles at 128K, 512K and about
# 1M give prefill speed against context length at 2K chunks, then one ~1M-token request gives TTFT and TPOT at full
# context. On a failed boot the next queued job applies its own config.
set -x
APPLY_MAX_BOOTS=2 bash evals/bench.sh apply deploy/configs/L4d-ctx1m-bt2k.args || exit 1
bash evals/bench.sh niah 131072 524288 1040000
REF_IN=1040000 REF_OUT=256 REF_N=1 REF_WARMUPS=0 bash evals/bench.sh reference 1
