#!/usr/bin/env bash
# Pod-side benchmark driver. Every subcommand writes raw outputs plus a /metrics diff into $OUT.
#
#   apply CONFIG.args          swap serve.args, restart vLLM, wait for a real generation, record restart time
#   wait [TIMEOUT_S]           wait until the server completes a generation (first boot)
#   facts                      KV capacity, load and compile times from the current boot's log
#   setup-client               EvalScope (vLLM blog's pinned commit) in /root/client-venv
#   agentic C1 [C2 ...]        vLLM GLM-5.3 blog OpenHands multi-turn workload, one point per concurrency
#   reference C1 [C2 ...]      random 8192 in / 1024 out, SGLang cookbook shape (REF_IN, REF_OUT, REF_N, REF_WARMUPS
#                              override it, e.g. one long-context request)
#   mtp-al                     MTP acceptance on real prompts (speed_bench), temperature 1.0, top_p 0.95; AL_CONC sets
#                              concurrency, which picks the draft count under a per-batch-size MTP schedule
#   gate-full | gate-spot      quality gate (evals/quality.py)
#   niah L1 [L2 ...]           single-stream needle at each prompt length, depth 0.5 (context ceiling)
#   smoke                      evals/smoke.py on localhost: time to first answer token at low/high/max effort
#   gpqa                       GPQA-Diamond at max effort (198 long-reasoning requests at c=32)
#   drill                      unplanned crash: SIGKILL the engine, time until the supervisor's restart serves again
#   soak-block C MINUTES       agentic load for a fixed time, used by the soak and the idle runner
#   profile                    torch profile of one cold prefill (needs --profiler-config in serve.args)
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

wait_ready() {  # wait_ready TIMEOUT_S [BOOT_BEFORE]; APPLY_MAX_BOOTS (default 3) boots before giving up
  local t0 b0; t0=$(date +%s); b0=${2:-$(boot_no)}
  until bash evals/ready.sh > /dev/null 2>&1; do
    sleep 10
    if [ $(( $(date +%s) - t0 )) -gt "$1" ]; then echo "not ready after $1 s"; return 1; fi
    if [ $(( $(boot_no) - b0 )) -ge "${APPLY_MAX_BOOTS:-3}" ]; then
      echo "vLLM crashed $(( $(boot_no) - b0 - 1 )) times; last log lines:"; tail -40 /root/vllm.log; return 1
    fi
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
    n=${AGENTIC_CONVS:-$(( c * 2 < 4 ? 4 : c * 2 ))}
    [ $(( offset + n )) -gt $DATA_N ] && offset=0
    name=agentic_c$c
    metrics_start "$name"
    timeout "${AGENTIC_TIMEOUT:-14400}" "$CLIENT/evalscope" perf --model glm-5.3 --url "$BASE/v1/chat/completions" \
      --api openai --api-key "$VLLM_API_KEY" --dataset swe_smith --dataset-path "$DATA" --dataset-offset "$offset" \
      --max-tokens 220 --multi-turn --number "$n" --parallel "$c" --extra-args '{"ignore_eos":true}' \
      --name "$name" --outputs-dir "$OUT/evalscope" --no-timestamp > "$OUT/$name.log" 2>&1
    echo "evalscope exit=$? c=$c n=$n offset=$offset"
    metrics_end "$name"
    offset=$(( offset + n ))
  done ;;

reference)
  TOK=$(tokenizer_dir)
  for c in "$@"; do
    case $c in 1) n=8 ;; 16) n=64 ;; 64) n=128 ;; *) n=$(( c * 4 )) ;; esac
    n=${REF_N:-$n}; w=${REF_WARMUPS:-$(( c < 8 ? 2 : 16 ))}
    name=ref_c$c${REF_IN:+_in$REF_IN}
    metrics_start "$name"
    vllm bench serve --backend openai --base-url "$BASE" --endpoint /v1/completions \
      --model "$MODEL" --served-model-name glm-5.3 --tokenizer "$TOK" --trust-remote-code \
      --dataset-name random --random-input-len "${REF_IN:-8192}" --random-output-len "${REF_OUT:-1024}" \
      --random-range-ratio 0 --ignore-eos \
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

niah)
  "$CLIENT/python" evals/quality.py niah --lengths "$@" --depths 0.5 --out "$OUT/niah_long.json" ;;

smoke)
  "$CLIENT/python" evals/smoke.py --long-stream-seconds 20 --out "$OUT/smoke.json" ;;

gpqa)
  metrics_start gpqa
  "$CLIENT/python" evals/quality.py gpqa --out "$OUT/gpqa.json"
  metrics_end gpqa ;;

drill)
  b0=$(boot_no); t0=$(date +%s)
  pkill -KILL -f "bin/vllm serve"; pkill -KILL -f "VLLM::"
  until [ "$(boot_no)" -gt "$b0" ]; do sleep 2; done
  if secs=$(wait_ready 3600 "$b0"); then
    echo "{\"kill_to_ready_s\": $(( $(date +%s) - t0 ))}" | tee "$OUT/drill.json"
  else
    echo "$secs" | tee "$OUT/drill-failed.txt"; exit 1
  fi ;;

soak-block)
  c=$1; minutes=${2:-15}
  off_file=/root/soak_offset; offset=$(cat "$off_file" 2> /dev/null || echo 0)
  name=soak_$(date -u +%Y%m%dT%H%M%S)_c$c
  metrics_start "$name"
  # One agentic point (2c conversations; EvalScope at this commit has no --duration), so blocks are comparable;
  # MINUTES only caps a stuck block.
  timeout $(( minutes * 60 )) "$CLIENT/evalscope" perf --model glm-5.3 --url "$BASE/v1/chat/completions" --api openai \
    --api-key "$VLLM_API_KEY" --dataset swe_smith --dataset-path "$DATA" --dataset-offset "$offset" --max-tokens 220 \
    --multi-turn --number $(( c * 2 < 4 ? 4 : c * 2 )) --parallel "$c" --extra-args '{"ignore_eos":true}' \
    --name "$name" --outputs-dir "$OUT/evalscope" --no-timestamp > "$OUT/$name.log" 2>&1
  echo "soak exit=$? c=$c minutes=$minutes"
  metrics_end "$name"
  echo $(( (offset + 2 * c) % DATA_N )) > "$off_file" ;;

profile)
  # Needs --profiler-config with torch_profiler_dir=/root/results/profile in serve.args. Profiles one cold
  # prefill of PROFILE_TOKENS (default 32768) and keeps each rank's CUDA-time table.
  P=/root/results/profile; mkdir -p "$P" "$OUT/profile"; t0=$(date +%s)
  curl -s -X POST "$BASE/start_profile" -H "Authorization: Bearer $VLLM_API_KEY"; echo
  python3 - "${PROFILE_TOKENS:-32768}" <<'EOF'
import json, os, random, sys, time, urllib.request
n = int(sys.argv[1]); words = "alpha beta gamma delta kernel tensor expert router cache token stream".split()
prompt = f"{random.random()} " + " ".join(random.choice(words) for _ in range(int(n * 0.8)))
body = json.dumps({"model": "glm-5.3", "prompt": prompt, "max_tokens": 1}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", body, {"Content-Type": "application/json",
                             "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600))
print(f"prompt_tokens={r['usage']['prompt_tokens']} seconds={time.time() - t:.2f}")
EOF
  curl -s -X POST "$BASE/stop_profile" -H "Authorization: Bearer $VLLM_API_KEY"; echo
  sleep 60
  find "$P" -type f -newermt "@$t0" -exec mv {} "$OUT/profile/" \;
  ls -la "$OUT/profile" ;;

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
