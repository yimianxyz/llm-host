#!/usr/bin/env python3
"""Aggregate everything synced into results/raw/ into results/REPORT_DATA.md (tables) and results/pareto.png.

  python evals/report.py [--price 21.16] [--gpus 4]

Conventions: interactivity = 1000 / TPOT (tok/s per user); throughput per GPU = total tok/s / GPUs, where total
includes prefix-cached prompt tokens ("logical" throughput, as in the vLLM GLM-5.3 blog). An agentic point
meets the SLO when TTFT p50 <= 5 s, interactivity >= 50 tok/s, and no request failed (NVIDIA Dynamo's SLO).
"""

import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")


def live(paths):
    """Skip quarantined runs (results/raw/<pod>/x-*), e.g. benchmarks that ran against a crashed server."""
    return [p for p in paths if not os.path.basename(os.path.dirname(p)).startswith("x-") and "/x-" not in p]


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def pct(rows, key, p):
    for r in rows or []:
        if r.get("Percentiles") == p:
            return r.get(key)
    return None


def agentic_rows(price, gpus):
    out = []
    for s in live(sorted(glob.glob(f"{RAW}/*/*/evalscope/*/parallel_*_number_*/benchmark_summary.json"))):
        parts = s.split(os.sep)
        pod, step, name = parts[-6], parts[-5], parts[-3]
        summ, perc = load(s), load(s.replace("benchmark_summary.json", "benchmark_percentile.json"))
        if not summ:
            continue
        m = load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(s)))), f"{name}.metrics.json")) or {}
        c = summ.get("Concurrency")
        tpot = summ.get("TPOT (ms)") or 0
        ttft50 = pct(perc, "TTFT (ms)", "50%")
        inter = 1000 / tpot if tpot else None
        failed = summ.get("Failed Requests", 0)
        ok = ttft50 is not None and ttft50 <= 5000 and inter is not None and inter >= 50 and failed == 0
        tot = summ.get("Total Throughput (tok/s)") or 0
        out.append({
            "pod": pod, "step": step, "name": name, "c": c, "requests": summ.get("Total Requests"), "failed": failed,
            "ttft_p50_ms": ttft50, "ttft_p90_ms": pct(perc, "TTFT (ms)", "90%"), "tpot_ms": tpot,
            "interactivity": round(inter, 1) if inter else None, "out_tok_s": summ.get("Output Throughput (tok/s)"),
            "total_tok_s_per_gpu": round(tot / gpus), "kv_hit_pct": summ.get("KV Cache Hit Rate (%)"),
            "decoded_per_iter": summ.get("Decoded Tok/Iter"), "mtp_al": m.get("mtp_acceptance_length"),
            "prefix_hit": m.get("prefix_cache_hit_rate"), "preemptions": m.get("preemptions"), "slo_ok": ok,
            "usd_per_session_hr": round(price / c, 2) if ok and c else None,
            "usd_per_m_out": round(price / (summ.get("Output Throughput (tok/s)") * 3600 / 1e6), 2) if summ.get("Output Throughput (tok/s)") else None,
        })
    return out


def reference_rows(price, gpus):
    out = []
    for p in live(sorted(glob.glob(f"{RAW}/*/*/ref_c*.json"))):
        r = load(p)
        if not r or "median_ttft_ms" not in r or not r.get("median_tpot_ms"):
            continue
        parts = p.split(os.sep)
        m = load(p.replace(".json", ".metrics.json")) or {}
        out.append({"pod": parts[-3], "step": parts[-2], "c": r.get("max_concurrency"), "n": r.get("completed"),
                    "failed": r.get("failed"), "ttft_p50_ms": round(r["median_ttft_ms"]), "ttft_p90_ms": round(r["p90_ttft_ms"]),
                    "tpot_p50_ms": round(r["median_tpot_ms"], 2), "tpot_p90_ms": round(r["p90_tpot_ms"], 2),
                    "interactivity": round(1000 / r["median_tpot_ms"], 1), "out_tok_s": round(r["output_throughput"]),
                    "total_tok_s_per_gpu": round(r["total_token_throughput"] / gpus), "goodput_rps": round(r.get("request_goodput") or 0, 3),
                    "mtp_al": m.get("mtp_acceptance_length"),
                    "usd_per_m_out": round(price / (r["output_throughput"] * 3600 / 1e6), 2)})
    return out


def table(rows, cols):
    if not rows:
        return "_none yet_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n" for r in rows)


def quality_rows():
    out = []
    for p in live(sorted(glob.glob(f"{RAW}/*/*/*.json"))):
        name = os.path.basename(p)
        if name not in ("gsm8k.json", "toolcalls.json", "niah.json", "gpqa.json", "mtp_al.json", "apply.json"):
            continue
        r = load(p) or {}
        parts = p.split(os.sep)
        row = {"pod": parts[-3], "step": parts[-2], "file": name}
        if name == "gsm8k.json":
            row["result"] = f"acc {r.get('accuracy')} n={r.get('n')} gate={r.get('gate_pass')} tokens={r.get('mean_completion_tokens')}"
        elif name == "toolcalls.json":
            row["result"] = f"{r.get('passed')}/{r.get('gated_cases')} gate={r.get('gate_pass')} info={r.get('informational')}"
        elif name == "niah.json":
            items = [i for i in r.get("items", []) if "found" in i]
            row["result"] = "; ".join(f"{i['prompt_tokens']} tok d={i['depth']} found={i['found']} ttft={i['ttft_s']}s "
                                      f"prefill={i['prefill_tok_s']} tok/s decode={i['decode_tok_s']} tok/s" for i in items)
        elif name == "gpqa.json":
            row["result"] = f"acc {r.get('accuracy')} n={r.get('n')} (card: W4A8 0.9028, FP8 0.8965)"
        elif name == "mtp_al.json":
            row["result"] = f"AL {r.get('spec_decode_acceptance_length')} rate {r.get('spec_decode_acceptance_rate')} tpot {round(r.get('median_tpot_ms') or 0, 2)} ms"
        elif name == "apply.json":
            row["result"] = f"restart to ready {r.get('restart_to_ready_s')} s"
        out.append(row)
    for p in live(sorted(glob.glob(f"{RAW}/*/*/boot_facts.txt"))):
        parts = p.split(os.sep)
        kv = [l.split("] ", 1)[-1].strip() for l in open(p) if "GPU KV cache size" in l]
        if kv:
            out.append({"pod": parts[-3], "step": parts[-2], "file": "boot_facts", "result": " / ".join(kv[-2:])})
    return out


def pareto(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    series = {}
    for r in rows:
        if r["interactivity"]:
            series.setdefault(f"{r['step']}", []).append((r["interactivity"], r["total_tok_s_per_gpu"], r["c"]))
    for label, pts in series.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
        for x, y, c in pts:
            ax.annotate(f"c{c}", (x, y), fontsize=7)
    ax.axvline(50, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("interactivity (tok/s per user, 1000/TPOT)")
    ax.set_ylabel("logical total tok/s per GPU")
    ax.set_title("GLM-5.3 W4A8 on 4x H200: agentic OpenHands workload")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=21.16)
    ap.add_argument("--gpus", type=int, default=4)
    a = ap.parse_args()
    ag, ref = agentic_rows(a.price, a.gpus), reference_rows(a.price, a.gpus)
    md = ["# Report data (generated by evals/report.py; do not edit)\n",
          f"Price ${a.price}/hr, {a.gpus} GPUs. SLO: TTFT p50 <= 5 s, interactivity >= 50 tok/s, zero failures.\n",
          "\n## Agentic (OpenHands multi-turn, EvalScope)\n",
          table(ag, ["step", "c", "requests", "failed", "ttft_p50_ms", "ttft_p90_ms", "tpot_ms", "interactivity", "out_tok_s",
                     "total_tok_s_per_gpu", "kv_hit_pct", "prefix_hit", "mtp_al", "decoded_per_iter", "preemptions",
                     "slo_ok", "usd_per_session_hr", "usd_per_m_out"]),
          "\n## Max sessions at SLO per step\n"]
    best = {}
    for r in ag:
        if r["slo_ok"]:
            best[r["step"]] = max(best.get(r["step"], 0), r["c"])
    md.append(table([{"step": k, "max_sessions_at_slo": v, "usd_per_session_hr": round(a.price / v, 2)} for k, v in best.items()],
                    ["step", "max_sessions_at_slo", "usd_per_session_hr"]))
    md += ["\n## Reference shape (random 8192/1024, vllm bench serve)\n",
           table(ref, ["step", "c", "n", "failed", "ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms", "tpot_p90_ms", "interactivity",
                       "out_tok_s", "total_tok_s_per_gpu", "goodput_rps", "mtp_al", "usd_per_m_out"]),
           "\n## Quality, MTP acceptance, restarts, KV capacity\n", table(quality_rows(), ["step", "file", "result"])]
    if pareto(ag, os.path.join(ROOT, "results", "pareto.png")):
        md.append("\n![pareto](pareto.png)\n")
    with open(os.path.join(ROOT, "results", "REPORT_DATA.md"), "w") as f:
        f.write("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
