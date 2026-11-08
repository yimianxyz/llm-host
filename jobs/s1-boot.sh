# Step 1: client setup during the weight download, then wait for the first boot and record cold-start facts.
set -x
bash evals/bench.sh setup-client
bash evals/bench.sh wait 5400
bash evals/bench.sh facts
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$OUT/gpu_mem.csv"
cp /root/results/boot.log "$OUT/boot_timeline.txt"
grep -E "Loading weights took|Model loading took|init engine|torch.compile|Graph capturing finished|KV cache size|Maximum concurrency|Starting vLLM API server" /root/vllm.log | cut -c1-300 > "$OUT/startup_lines.txt"
