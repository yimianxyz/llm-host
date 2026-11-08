#!/usr/bin/env bash
# Parallel weight download: N `hf download` processes over interleaved shards of the repo's file list.
# One process topped out near 230 MB/s from Hugging Face to EUR-IS-5. Shards share the cache safely through
# hf's per-file locks, and a final plain `hf download` in start.sh verifies that nothing is missing.
#   bash deploy/fastdl.sh [N]
set -u
[ -f /root/.pod_env ] && . /root/.pod_env
N=${1:-4}
python3 - "$N" <<'PY'
import os, sys
from huggingface_hub import list_repo_files
n = int(sys.argv[1])
files = sorted(list_repo_files(os.environ["MODEL"], revision=os.environ["MODEL_REV"]))
for i in range(n):
    with open(f"/root/dl_shard_{i}.txt", "w") as f:
        f.write("\n".join(files[i::n]))
PY
for i in $(seq 0 $((N - 1))); do
  # shellcheck disable=SC2046
  hf download "$MODEL" --revision "$MODEL_REV" $(cat "/root/dl_shard_$i.txt") >> "/root/results/download-shard$i.log" 2>&1 &
done
wait
