# Step 1f: apply B4 (8K batched tokens) so the unprofiled FlashMLA buffer (2 GiB) fits the ~3.5 GiB margin; verify with a 16K-token prompt.
set -x
bash evals/bench.sh apply deploy/configs/B4-dcp4-bt8k.args
/root/client-venv/bin/python evals/quality.py niah --lengths 131072 --depths 0.5 --out "$OUT/niah_probe.json"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$OUT/gpu_mem.csv"
