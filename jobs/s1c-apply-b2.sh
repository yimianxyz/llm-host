# Step 1c: apply B2 (B1 + 16K batched tokens) after B1's warmup OOM; record restart time and boot facts.
set -x
bash evals/bench.sh apply deploy/configs/B2-dcp4-bt16k.args
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$OUT/gpu_mem.csv"
