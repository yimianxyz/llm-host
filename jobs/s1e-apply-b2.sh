# Step 1e: B3 (expandable_segments) fails at load (mapping OOM with 22 MB free). Test B2 itself (16K batched tokens),
# which was never booted: both 8 GiB warmup OOMs came from B1's config (32K batched tokens).
set -x
bash evals/bench.sh apply deploy/configs/B2-dcp4-bt16k.args
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$OUT/gpu_mem.csv"
