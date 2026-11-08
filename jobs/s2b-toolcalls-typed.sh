# Step 2b: re-run the tool-call suite with typed-schema gated cases (untyped and known upstream bugs are informational).
set -x
/root/client-venv/bin/python evals/quality.py toolcalls --out "$OUT/toolcalls.json"
