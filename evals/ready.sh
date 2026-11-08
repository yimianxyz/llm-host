#!/usr/bin/env bash
# Ready means the server completes a real generation; a /health 200 can hide a hung engine (SGLang #45781).
set -u
[ -f /root/.pod_env ] && . /root/.pod_env
BASE=${BASE_URL:-http://127.0.0.1:8000}
curl -fsS -m "${READY_TIMEOUT:-60}" "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY:-}" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16,"chat_template_kwargs":{"reasoning_effort":"low"}}' > /dev/null
