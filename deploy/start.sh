#!/usr/bin/env bash
# PID 1 of the pod: entrypoint ["bash","-c"] with this file's text as cmd (see deploy/pod.py).
# Idempotent, so a container restart resumes instead of starting over.
# Order: deadline guard, SSH, host facts, weights, vLLM supervisor.
set -uo pipefail
R=/root/results
mkdir -p "$R" /root/queue /root/done
log() { echo "[start $(date -u +%FT%TZ)] $*" | tee -a "$R/boot.log"; }

selfdelete() {
  log "deadline reached, deleting pod ${RUNPOD_POD_ID:-unknown}"
  curl -fsS -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID:-}" >> "$R/boot.log" 2>&1 && return 0
  curl -fsS -H 'Content-Type: application/json' "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY:-}" \
    -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID:-}\\\"}) }\"}" >> "$R/boot.log" 2>&1
}

# 1) Deadline watchdog. /root/deadline holds an epoch second; the operator can move it over SSH.
[ -s /root/deadline ] || echo $(( $(date +%s) + $(awk -v h="${MAX_HOURS:-9}" 'BEGIN{printf "%d", h*3600}') )) > /root/deadline
( while sleep 60; do
    if [ "$(date +%s)" -ge "$(cat /root/deadline)" ]; then selfdelete; sleep 300; fi
  done ) &
log "deadline $(date -u -d @"$(cat /root/deadline)" +%FT%TZ)"

# 2) SSH. startSsh only injects PUBLIC_KEY; this image does not run sshd by itself.
if ! command -v sshd >/dev/null || ! command -v rsync >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync curl >/dev/null
fi
mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
printf '%s\n' "${PUBLIC_KEY:-}" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
# SSH sessions do not inherit PID 1's environment, so snapshot it for jobs and login shells.
export -p | grep -vE ' (PWD|OLDPWD|SHLVL|_)=' > /root/.pod_env
if ! grep -q pod_env /root/.bashrc 2>/dev/null; then
  { echo '[ -f /root/.pod_env ] && . /root/.pod_env'; cat /root/.bashrc 2>/dev/null; } > /tmp/.bashrc && mv /tmp/.bashrc /root/.bashrc
fi
pgrep -x sshd >/dev/null || /usr/sbin/sshd
log "sshd up"

# 3) Host facts, once per boot. Only variable names are recorded, never values.
{ date -u
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  nvidia-smi topo -m
  df -h / /dev/shm; free -g
  echo "nproc=$(nproc) cgroup_cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  echo "runpod_env: $(env | grep -oE '^RUNPOD_[A-Z_]+' | sort | tr '\n' ' ')"
} > "$R/host-$(date +%s).txt" 2>&1

# 4) Weights onto the local container disk. hf download is resumable, so a restart skips finished files.
t0=$(date +%s)
until hf download "$MODEL" --revision "$MODEL_REV" >> "$R/download.log" 2>&1; do
  log "hf download failed, retrying"; sleep 15
done
log "weights ready in $(( $(date +%s) - t0 ))s ($(du -sh "${HF_HOME:-/root/.cache/huggingface}" | cut -f1))"

# 5) vLLM supervisor. /root/serve.args is seeded from $SERVE_ARGS once. Experiments rewrite it and
#    kill vLLM; this loop restarts it with the new flags and the weights already on disk.
[ -s /root/serve.args ] || printf '%s\n' "$SERVE_ARGS" > /root/serve.args
n=0
while true; do
  n=$((n + 1)); args=(); envs=()
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ "$line" == ENV\ * ]]; then envs+=("${line#ENV }")
    elif [[ "$line" == *" "* ]]; then args+=("${line%% *}" "${line#* }")
    else args+=("$line"); fi
  done < /root/serve.args
  cp /root/serve.args "$R/serve-boot$n.args"
  echo "$n $(date +%s)" > /root/vllm.boot
  log "vllm boot $n: ${envs[*]} ${args[*]}"
  echo "===== vllm boot $n $(date -u +%FT%TZ) =====" >> /root/vllm.log
  env "${envs[@]}" vllm serve "$MODEL" --revision "$MODEL_REV" "${args[@]}" 2>&1 | tee -a /root/vllm.log
  log "vllm boot $n exited with ${PIPESTATUS[0]}"
  sleep 5
done
