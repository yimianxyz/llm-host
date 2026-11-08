# Rehearsal: re-run the two paths that failed in r01 (GSM8K via client venv, MTP acceptance via SPEED-Bench).
set -x
"$CLIENT/python" --version 2>/dev/null; CLIENT=/root/client-venv/bin
VIRTUAL_ENV=/root/client-venv uv pip install -q openai datasets
/root/client-venv/bin/python evals/quality.py gsm8k --limit 20 --effort low --out "$OUT/gsm8k.json"
AL_PROMPTS=8 bash evals/bench.sh mtp-al
echo REHEARSAL-FIXES-DONE
