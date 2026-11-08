#!/usr/bin/env bash
# Push the repo's deploy/, evals/ and the agentic dataset to the pod, then start the job runner if it is not running.
# Optional: push.sh JOB.sh [...] also enqueues jobs (copied to /root/queue in the order given).
set -eu
cd "$(dirname "$0")/.."
. ./.env.pod
SSH="ssh -p $SSH_PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
$SSH "root@$SSH_HOST" 'mkdir -p /root/llm-host /root/data /root/queue'
rsync -az -e "$SSH" --exclude '__pycache__' deploy evals "root@$SSH_HOST:/root/llm-host/"
[ -f data/openhand-zai-org-GLM-5.3-256.json ] && rsync -az -e "$SSH" data/openhand-zai-org-GLM-5.3-256.json "root@$SSH_HOST:/root/data/"
$SSH "root@$SSH_HOST" 'pgrep -f deploy/runner.sh > /dev/null || (setsid bash /root/llm-host/deploy/runner.sh > /dev/null 2>&1 < /dev/null &)'
for job in "$@"; do
  rsync -az -e "$SSH" "$job" "root@$SSH_HOST:/root/queue/"
  echo "queued $(basename "$job")"
done
