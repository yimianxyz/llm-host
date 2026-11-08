# Rungs L5c (4K chunks + graph capture cap 128) and L4 (context ceiling) in one boot. The perf comparison is L5c vs
# L2: max-model-len does not change decode or prefill. If 1M does not fit, retry once at vLLM's own estimate of the
# maximum length (that is the ceiling), and fall back to L5c at 256K only if that fails too.
set -x
L=1040000
n0=$(wc -l < /root/vllm.log)
if ! APPLY_MAX_BOOTS=2 bash evals/bench.sh apply deploy/configs/L4c-ctx1m.args; then
  est=$(tail -n +"$n0" /root/vllm.log | grep -o "estimated maximum model length is [0-9]*" | tail -1 | grep -o "[0-9]*$")
  M=$(( ${est:-0} / 4096 * 4096 )); L=$(( M - 4096 ))
  sed "s/^--max-model-len .*/--max-model-len $M/" deploy/configs/L4c-ctx1m.args > "$OUT/L4c-fit.args"
  if [ "$M" -le 262144 ] || ! APPLY_MAX_BOOTS=2 bash evals/bench.sh apply "$OUT/L4c-fit.args"; then
    bash evals/bench.sh apply deploy/configs/L5c-bt4k-cg128.args || exit 1
    L=0
  fi
fi
bash evals/bench.sh gate-spot
bash evals/bench.sh agentic 1 4 8
if [ "$L" -gt 0 ]; then
  bash evals/bench.sh niah 524288 "$L"
  REF_IN=$L REF_OUT=256 REF_N=1 REF_WARMUPS=0 bash evals/bench.sh reference 1
fi
