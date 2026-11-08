#!/usr/bin/env bash
# Push the repo's deploy/, evals/ and the agentic dataset to the pod, then start the job runner if it is not running.
# Optional: push.sh JOB.sh [...] also enqueues jobs (copied to /root/queue in the order given).
set -eu
cd "$(dirname "$0")/.."
. "./${POD_ENV:-.env.pod}"
SSH="ssh -p $SSH_PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
$SSH "root@$SSH_HOST" 'mkdir -p /root/llm-host /root/data /root/queue'
rsync -az -e "$SSH" --exclude '__pycache__' deploy evals "root@$SSH_HOST:/root/llm-host/"
rsync -az -e "$SSH" --include "*.json" --include "speed-bench/***" --exclude "*" data/ "root@$SSH_HOST:/root/data/"
$SSH "root@$SSH_HOST" 'kill -0 "$(cat /root/runner.pid 2>/dev/null)" 2>/dev/null ||
  { setsid nohup bash /root/llm-host/deploy/runner.sh > /root/results/runner.out 2>&1 < /dev/null & echo "runner started"; }'
for job in "$@"; do
  rsync -az -e "$SSH" "$job" "root@$SSH_HOST:/root/queue/"
  echo "queued $(basename "$job")"
done
