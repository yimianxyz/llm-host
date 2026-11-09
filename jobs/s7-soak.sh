# Step 7: soak the final config until the budget ends. For the whole soak, a liveness probe sends one real generation
# every 30 s and GPU memory plus total container RSS are sampled every minute (leak check). Load: one GPQA-Diamond run
# at c=32 (long max-effort reasoning: a decode-heavy shape past the SLO, and a quality check), one unplanned-crash
# drill, then agentic blocks cycling 50%, 100% and 150% of max sessions at SLO until the pod is deleted.
set -x
cp /root/serve.args "$OUT/applied.args"
setsid bash evals/bench.sh probe 21600 30 > /dev/null 2>&1 < /dev/null &
setsid bash -c 'while true; do
  echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,),$(ps -eo rss= | awk "{s+=\$1} END {print int(s/1024)}")" >> "$0/mem.csv"
  sleep 60; done' "$OUT" > /dev/null 2>&1 < /dev/null &
bash evals/bench.sh gpqa
bash evals/bench.sh drill
while true; do
  for c in ${SOAK_LEVELS:-2 4 6}; do bash evals/bench.sh soak-block "$c" 15; done
done
