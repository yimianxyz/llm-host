#!/usr/bin/env bash
# On-pod job runner. It runs /root/queue/*.sh one at a time, in name order; each job gets OUT=/root/results/<job>.
# When the queue is empty and /root/idle_soak holds a concurrency, it runs a short soak block on the
# current config, so time spent waiting on the operator still produces stability data.
# Start it detached: setsid bash /root/llm-host/deploy/runner.sh > /dev/null 2>&1 < /dev/null &
set -u
. /root/.pod_env
R=/root/results
cd /root/llm-host
while true; do
  job=$(ls /root/queue/*.sh 2>/dev/null | head -n1)
  if [ -n "$job" ]; then
    name=$(basename "$job" .sh)
    mkdir -p "$R/$name" && mv "$job" /root/done/
    echo "$(date -u +%FT%TZ) start $name" >> "$R/runner.log"
    OUT="$R/$name" bash "/root/done/$name.sh" > "$R/$name/job.log" 2>&1
    echo "$(date -u +%FT%TZ) end $name exit=$?" >> "$R/runner.log"
  elif [ -s /root/idle_soak ] && bash evals/ready.sh > /dev/null 2>&1; then
    OUT="$R/idle-soak" bash evals/bench.sh soak-block "$(cat /root/idle_soak)" 5 >> "$R/idle-soak.log" 2>&1
  else
    sleep 15
  fi
done
