# Hand the pod to interactive use: serve the recommended config (L5d without --profiler-config), smoke-test it,
# and leave the runner idle (no soak blocks) until the deadline watchdog deletes the pod.
set -x
rm -f /root/idle_soak
bash evals/bench.sh apply deploy/configs/recommended.args || exit 1
bash evals/bench.sh smoke
