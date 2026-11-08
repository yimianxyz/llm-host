# Step 1d: apply B3 (B2 + expandable_segments) after the constant 8 GiB FlashMLA warmup OOM; record restart time and facts.
set -x
bash evals/bench.sh apply deploy/configs/B3-dcp4-bt16k-expseg.args
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$OUT/gpu_mem.csv"
