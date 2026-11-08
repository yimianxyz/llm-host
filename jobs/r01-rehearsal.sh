# Rehearsal: every eval path end to end on a tiny model (plumbing check, not model quality).
set -x
export AGENTIC_DATA=/root/data/openhand-rehearsal-16.json AGENTIC_N=16
bash evals/bench.sh wait 900
bash evals/bench.sh facts
time bash evals/bench.sh setup-client
bash evals/bench.sh agentic 1 4
bash evals/bench.sh reference 1
AL_PROMPTS=8 bash evals/bench.sh mtp-al
python3 evals/quality.py toolcalls --out "$OUT/toolcalls.json"
python3 evals/quality.py niah --lengths 16384 --depths 0.5 --out "$OUT/niah.json"
python3 evals/quality.py gsm8k --limit 20 --effort low --out "$OUT/gsm8k.json"
python3 evals/smoke.py --long-stream-seconds 20 --out "$OUT/smoke_local.json"
bash evals/bench.sh probe 60 15
echo REHEARSAL-JOB-DONE
