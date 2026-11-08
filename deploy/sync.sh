#!/usr/bin/env bash
# Pull everything the pod has produced into results/raw/<pod-name>/ and commit it.
#   deploy/sync.sh          one sync (run at the end of every step)
#   deploy/sync.sh --loop   every 5 minutes until killed (run in the background for the whole session)
# Safe to run anytime: when the pod is gone the rsync fails, and whatever was already pulled stays committed.
set -u
cd "$(dirname "$0")/.."
. ./.env.pod
SSH="ssh -p $SSH_PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15"
DEST="results/raw/$POD_NAME"

once() {
  mkdir -p "$DEST"
  rsync -az --timeout=120 -e "$SSH" --exclude '*.db' --exclude '*.sqlite' \
    "root@$SSH_HOST:/root/results/" "$DEST/" 2>> results/sync-errors.log
  rsync -az --timeout=120 -e "$SSH" "root@$SSH_HOST:/root/vllm.log" "$DEST/vllm.log" 2>> results/sync-errors.log
  git add results > /dev/null 2>&1
  git commit -qm "sync: $POD_NAME $(date -u +%FT%TZ)" > /dev/null 2>&1 && echo "synced $(date -u +%T)" || true
}

if [ "${1:-}" = "--loop" ]; then
  while true; do once; sleep 300; done
else
  once
fi
