#!/usr/bin/env bash
# Pod-side benchmark driver. Every subcommand writes raw outputs plus a /metrics diff into $OUT.
#
#   apply CONFIG.args          swap serve.args, restart vLLM, wait for a real generation, record restart time
#   wait [TIMEOUT_S]           wait until the server completes a generation (first boot)
#   facts                      KV capacity, load and compile times from the current boot's log
#   setup-client               EvalScope (vLLM blog's pinned commit) in /root/client-venv
#   agentic C1 [C2 ...]        vLLM GLM-5.3 blog OpenHands multi-turn workload, one point per concurrency
#   reference C1 [C2 ...]      random 8192 in / 1024 out, SGLang cookbook shape
#   mtp-al                     MTP acceptance on real prompts (speed_bench), temperature 1.0, top_p 0.95; AL_CONC sets
#                              concurrency, which picks the draft count under a per-batch-size MTP schedule
#   gate-full | gate-spot      quality gate (evals/quality.py)
#   soak-block C MINUTES       agentic load for a fixed time, used by the soak and the idle runner
#   probe SECONDS [INTERVAL]   liveness via a real generation; CSV of ts,http_code,seconds
set -uo pipefail
[ -f /root/.pod_env ] && . /root/.pod_env
cd "$(dirname "$0")/.."
OUT=${OUT:-/root/results/adhoc}
mkdir -p "$OUT"
BASE=http://127.0.0.1:8000
export OPENAI_API_KEY=${VLLM_API_KEY:-}
CLIENT=/root/client-venv/bin
DATA=${AGENTIC_DATA:-/root/data/openhand-zai-org-GLM-5.3-256.json}
DATA_N=${AGENTIC_N:-256}
EVALSCOPE_REF="https://github.com/modelscope/evalscope/archive/acd09b44384d53174768bb1063f675420f76fae9.zip"

tokenizer_dir() {
  python3 -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$MODEL_REV', allow_patterns=['*.json','*.jinja','*.py','*.txt','*.model','*.tiktoken']))"
}

metrics_start() {
  python3 evals/metrics.py snap "$OUT/$1.before.prom"
  python3 evals/metrics.py watch "$OUT/$1.gauges.csv" --interval 15 & WATCH=$!
}
metrics_end() {
  kill "$WATCH" 2> /dev/null
  python3 evals/metrics.py snap "$OUT/$1.after.prom"
  python3 evals/metrics.py diff "$OUT/$1.before.prom" "$OUT/$1.after.prom" > "$OUT/$1.metrics.json"
  cat "$OUT/$1.metrics.json"
}

boot_no() { cut -d' ' -f1 /root/vllm.boot 2> /dev/null || echo 0; }

wait_ready() {  # wait_ready TIMEOUT_S [BOOT_BEFORE]
  local t0 b0; t0=$(date +%s); b0=${2:-$(boot_no)}
  until bash evals/ready.sh > /dev/null 2>&1; do
    sleep 10
    if [ $(( $(date +%s) - t0 )) -gt "$1" ]; then echo "not ready after $1 s"; return 1; fi
    if [ $(( $(boot_no) - b0 )) -ge 3 ]; then echo "vLLM crashed 3 times; last log lines:"; tail -40 /root/vllm.log; return 1; fi
  done
  echo $(( $(date +%s) - t0 ))
}

cmd=${1:?usage: bench.sh CMD ...}; shift
case $cmd in
apply)
  cp "$1" /root/serve.args && cp "$1" "$OUT/applied.args"
  b0=$(boot_no); t0=$(date +%s)
  # Match only the vLLM process: PID 1 (start.sh) contains the text "vllm serve" and must survive.
  pkill -TERM -f "bin/vllm serve"
  # vLLM can ignore SIGTERM during engine init, and after an engine failure the API server can hang
  # with the GPUs already freed; escalate so the supervisor in start.sh always gets its restart.
  for _ in $(seq 15); do [ "$(boot_no)" -gt "$b0" ] && break; sleep 2; done
  if [ "$(boot_no)" -le "$b0" ]; then pkill -KILL -f "bin/vllm serve"; pkill -KILL -f "VLLM::"; fi
  until [ "$(boot_no)" -gt "$b0" ]; do sleep 2; done
  if secs=$(wait_ready 3600 "$b0"); then
    echo "{\"restart_to_ready_s\": $(( $(date +%s) - t0 ))}" | tee "$OUT/apply.json"
    bash evals/bench.sh facts
  else
    echo "$secs" | tee "$OUT/apply-failed.txt"; exit 1
  fi ;;

wait)
  wait_ready "${1:-5400}" | tee "$OUT/wait.txt" ;;

facts)
  awk '/===== vllm boot/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' /root/vllm.log \
    | grep -E "KV cache|[Cc]oncurrency|took [0-9.]+ ?s|seconds|GiB|max_model_len|[Gg]raph|DeepGEMM|backend|WARNING|ERROR" \
    | grep -vE "POST /v1|GET /metrics" | head -200 > "$OUT/boot_facts.txt"
  grep -E "KV cache size|Maximum concurrency" "$OUT/boot_facts.txt" ;;

setup-client)
  python3 -c "import pandas" 2> /dev/null || uv pip install -q --system pandas || python3 -m pip install -q pandas  # vllm[bench] extra, needed by speed_bench
  [ -x "$CLIENT/evalscope" ] && { echo "client ready"; exit 0; }
  rm -rf /root/client-venv  # a half-built venv from an interrupted install
  if command -v uv > /dev/null; then
    uv venv /root/client-venv -q --python 3.12 && VIRTUAL_ENV=/root/client-venv uv pip install -q "evalscope[perf] @ $EVALSCOPE_REF" openai datasets
  else
    python3 -m venv /root/client-venv && "$CLIENT/pip" install -q "evalscope[perf] @ $EVALSCOPE_REF" openai datasets
  fi
  "$CLIENT/evalscope" --help > /dev/null && echo "client ready" ;;

agentic)
  # Disjoint dataset slices per point, so one point never warms the prefix cache for the next.
  # Conversations per point = 2x concurrency (min 4), as in the vLLM blog's sweep.
  offset=${AGENTIC_OFFSET:-0}
  for c in "$@"; do
    n=$(( c * 2 < 4 ? 4 : c * 2 ))
    [ $(( offset + n )) -gt $DATA_N ] && offset=0
    name=agentic_c$c
    metrics_start "$name"
    "$CLIENT/evalscope" perf --model glm-5.3 --url "$BASE/v1/chat/completions" --api openai --api-key "$VLLM_API_KEY" \
      --dataset swe_smith --dataset-path "$DATA" --dataset-offset "$offset" --max-tokens 220 --multi-turn \
      --number "$n" --parallel "$c" --extra-args '{"ignore_eos":true}' ${AGENTIC_DURATION:+--duration $AGENTIC_DURATION} \
      --name "$name" --outputs-dir "$OUT/evalscope" --no-timestamp > "$OUT/$name.log" 2>&1
    echo "evalscope exit=$? c=$c n=$n offset=$offset"
    metrics_end "$name"
    offset=$(( offset + n ))
  done ;;

reference)
  TOK=$(tokenizer_dir)
  for c in "$@"; do
    case $c in 1) n=8 ;; 16) n=64 ;; 64) n=128 ;; *) n=$(( c * 4 )) ;; esac
    w=$(( c < 8 ? 2 : 16 ))
    name=ref_c$c
    metrics_start "$name"
    vllm bench serve --backend openai --base-url "$BASE" --endpoint /v1/completions \
      --model "$MODEL" --served-model-name glm-5.3 --tokenizer "$TOK" --trust-remote-code \
      --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 0 --ignore-eos \
      --num-prompts "$n" --max-concurrency "$c" --num-warmups "$w" --seed "$RANDOM" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --goodput ttft:2000 tpot:80 \
      --save-result --result-dir "$OUT" --result-filename "$name.json" > "$OUT/$name.log" 2>&1
    echo "bench exit=$? c=$c n=$n"
    metrics_end "$name"
  done ;;

mtp-al)
  TOK=$(tokenizer_dir)
  metrics_start mtp_al
  vllm bench serve --backend openai-chat --base-url "$BASE" --endpoint /v1/chat/completions \
    --model "$MODEL" --served-model-name glm-5.3 --tokenizer "$TOK" --trust-remote-code \
    --dataset-name speed_bench --dataset-path /root/data/speed-bench --speed-bench-category "${AL_CATEGORY:-coding}" \
    --speed-bench-output-len 2048 --num-prompts "${AL_PROMPTS:-64}" --max-concurrency "${AL_CONC:-8}" \
    --temperature 1.0 --top-p 0.95 --chat-template-kwargs '{"reasoning_effort":"high"}' --seed "$RANDOM" \
    --save-result --result-dir "$OUT" --result-filename mtp_al.json > "$OUT/mtp_al.log" 2>&1
  echo "bench exit=$?"
  metrics_end mtp_al ;;

gate-full)
  "$CLIENT/python" evals/quality.py toolcalls --out "$OUT/toolcalls.json"
  "$CLIENT/python" evals/quality.py niah --lengths 131072 204800 --out "$OUT/niah.json"
  "$CLIENT/python" evals/quality.py gsm8k --out "$OUT/gsm8k.json"
  bash evals/bench.sh mtp-al ;;

gate-spot)
  "$CLIENT/python" evals/quality.py toolcalls --out "$OUT/toolcalls.json"
  "$CLIENT/python" evals/quality.py niah --lengths 131072 --depths 0.5 --out "$OUT/niah.json"
  "$CLIENT/python" evals/quality.py gsm8k --limit 200 --out "$OUT/gsm8k.json" ;;

soak-block)
  c=$1; minutes=${2:-15}
  off_file=/root/soak_offset; offset=$(cat "$off_file" 2> /dev/null || echo 0)
  name=soak_$(date -u +%Y%m%dT%H%M%S)_c$c
  metrics_start "$name"
  timeout $(( minutes * 60 + 900 )) "$CLIENT/evalscope" perf --model glm-5.3 --url "$BASE/v1/chat/completions" --api openai \
    --api-key "$VLLM_API_KEY" --dataset swe_smith --dataset-path "$DATA" --dataset-offset "$offset" --max-tokens 220 \
    --multi-turn --number 100000 --duration $(( minutes * 60 )) --parallel "$c" --extra-args '{"ignore_eos":true}' \
    --name "$name" --outputs-dir "$OUT/evalscope" --no-timestamp > "$OUT/$name.log" 2>&1
  echo "soak exit=$? c=$c minutes=$minutes"
  metrics_end "$name"
  echo $(( (offset + 2 * c) % DATA_N )) > "$off_file" ;;

probe)
  end=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    r=$(curl -s -o /dev/null -w "%{http_code},%{time_total}" -m 120 "$BASE/v1/chat/completions" \
      -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16,"chat_template_kwargs":{"reasoning_effort":"low"}}')
    echo "$(date +%s),$r" >> "$OUT/probe.csv"
    sleep "${2:-30}"
  done ;;

*) echo "unknown command: $cmd"; exit 2 ;;
esac
