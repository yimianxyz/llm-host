#!/usr/bin/env python3
"""Server-side metrics from vLLM's Prometheus /metrics endpoint (stdlib only).

  snap  OUT.prom                   save one raw snapshot
  diff  BEFORE.prom AFTER.prom     derived metrics for the interval between two snapshots, printed as JSON
  watch OUT.csv [--interval S]     sample gauges until killed

Counter definitions follow the vLLM GLM-5.3 blog: acceptance length = 1 + d(accepted) / d(drafts).
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")
GAUGES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"]


def fetch() -> str:
    req = urllib.request.Request(f"{BASE}/metrics")
    key = os.environ.get("VLLM_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def parse(text: str) -> dict:
    """Sum every sample over its label sets. Histogram _bucket series are dropped; _sum and _count stay."""
    out: dict = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if not m or m.group(1).endswith("_bucket"):
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def pick(d: dict, *names: str) -> float:
    for n in names:
        if n in d:
            return d[n]
    return 0.0


def derive(a: dict, b: dict) -> dict:
    def delta(*names):
        return pick(b, *names) - pick(a, *names)

    drafts = delta("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts")
    draft_tokens = delta("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens")
    queries = delta("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
    hits = delta("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
    ext_q = delta("vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_queries")
    ext_h = delta("vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_hits")
    return {
        "mtp_acceptance_length": round(1 + accepted / drafts, 3) if drafts else None,
        "mtp_acceptance_rate": round(accepted / draft_tokens, 3) if draft_tokens else None,
        "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None,
        "external_prefix_cache_hit_rate": round(ext_h / ext_q, 4) if ext_q else None,
        "preemptions": delta("vllm:num_preemptions_total", "vllm:num_preemptions"),
        "prompt_tokens": delta("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
        "generation_tokens": delta("vllm:generation_tokens_total", "vllm:generation_tokens"),
        "requests_finished": delta("vllm:request_success_total", "vllm:request_success"),
        "kv_cache_usage_perc_end": pick(b, "vllm:kv_cache_usage_perc"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap")
    s.add_argument("out")
    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    w = sub.add_parser("watch")
    w.add_argument("out")
    w.add_argument("--interval", type=float, default=30)
    args = ap.parse_args()

    if args.cmd == "snap":
        with open(args.out, "w") as f:
            f.write(fetch())
    elif args.cmd == "diff":
        a, b = (parse(open(p).read()) for p in (args.before, args.after))
        json.dump(derive(a, b), sys.stdout, indent=2)
        print()
    else:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["ts"] + GAUGES)
            while True:
                try:
                    m = parse(fetch())
                    wr.writerow([int(time.time())] + [m.get(g, "") for g in GAUGES])
                    f.flush()
                except Exception as e:  # a restart or overload must not kill the sampler
                    print(f"watch: {e}", file=sys.stderr)
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
