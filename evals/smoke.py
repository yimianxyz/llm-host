#!/usr/bin/env python3
"""Functional smoke test of an OpenAI-compatible GLM-5.3 endpoint. Run it locally through the RunPod proxy
(the path real clients take) or on the pod against localhost.

  python evals/smoke.py --base-url https://<pod>-8000.proxy.runpod.net --out results/.../smoke.json

Checks: auth is enforced; reasoning is split from the answer; tool calls are parsed; time to first token
vs time to first *answer* token; a stream outlives Cloudflare's 100 s proxy cap.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from openai import OpenAI

MODEL = "glm-5.3"


def reasoning_of(obj) -> str:
    """vLLM has used both `reasoning_content` and `reasoning` for the thinking text."""
    extra = getattr(obj, "model_extra", None) or {}
    return getattr(obj, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning") or ""


def check_auth(base: str, key: str) -> dict:
    def status(headers):
        # Cloudflare (in front of the RunPod proxy) answers 403 to urllib's default User-Agent.
        req = urllib.request.Request(f"{base}/v1/models", headers={"User-Agent": "llm-host-smoke/1.0", **headers})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None

    no_key, _ = status({})
    ok, body = status({"Authorization": f"Bearer {key}"})
    ids = [m["id"] for m in (body or {}).get("data", [])]
    return {"pass": no_key == 401 and ok == 200 and MODEL in ids, "no_key_status": no_key, "key_status": ok, "models": ids}


def check_reasoning(c: OpenAI) -> dict:
    r = c.chat.completions.create(
        model=MODEL, max_tokens=4096,
        messages=[{"role": "user", "content": "What is 17*23? Reply with just the number."}],
        extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
    )
    m = r.choices[0].message
    content, reasoning = m.content or "", reasoning_of(m)
    ok = "391" in content and bool(reasoning) and "</think>" not in content and "<think>" not in content
    return {"pass": ok, "content": content[:200], "reasoning_chars": len(reasoning), "finish": r.choices[0].finish_reason}


def check_tool_call(c: OpenAI) -> dict:
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r = c.chat.completions.create(
        model=MODEL, max_tokens=4096, tools=tools,
        messages=[{"role": "user", "content": "What's the weather in Paris right now?"}],
        extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
    )
    ch = r.choices[0]
    calls = ch.message.tool_calls or []
    try:
        args = json.loads(calls[0].function.arguments) if calls else {}
    except json.JSONDecodeError:
        args = {}
    ok = ch.finish_reason == "tool_calls" and calls and calls[0].function.name == "get_weather" and "paris" in str(args.get("city", "")).lower()
    return {"pass": bool(ok), "finish": ch.finish_reason, "calls": [(t.function.name, t.function.arguments) for t in calls]}


def stream(c: OpenAI, messages, max_tokens, seconds_cap=None, extra=None) -> dict:
    t0 = time.perf_counter()
    first_any = first_answer = None
    chunks, usage, finish = 0, None, None
    s = c.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True}, extra_body=extra or {},
    )
    for ev in s:
        now = time.perf_counter() - t0
        if ev.usage:
            usage = ev.usage
        if not ev.choices:
            continue
        d = ev.choices[0].delta
        if (d.content or reasoning_of(d)) and first_any is None:
            first_any = now
        if d.content and first_answer is None:
            first_answer = now
        chunks += 1
        finish = ev.choices[0].finish_reason or finish
        if seconds_cap and now >= seconds_cap:
            s.close()
            break
    total = time.perf_counter() - t0
    out_tokens = usage.completion_tokens if usage else None
    return {"ttft_s": first_any, "time_to_first_answer_token_s": first_answer, "elapsed_s": round(total, 2),
            "chunks": chunks, "completion_tokens": out_tokens, "finish": finish,
            "decode_tok_s": round(out_tokens / (total - first_any), 1) if out_tokens and first_any else None}


def check_ttfat(c: OpenAI) -> dict:
    msgs = [{"role": "user", "content": "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Answer briefly."}]
    res = {}
    for effort in ("low", "high", "max"):
        res[effort] = stream(c, msgs, 16384, extra={"chat_template_kwargs": {"reasoning_effort": effort}})
    return {"pass": all(v["time_to_first_answer_token_s"] is not None for v in res.values()), **res}


def check_long_stream(c: OpenAI, seconds: float) -> dict:
    msgs = [{"role": "user", "content": "Write a very long, detailed technical history of compilers."}]
    r = stream(c, msgs, 30000, seconds_cap=seconds,
               extra={"ignore_eos": True, "chat_template_kwargs": {"reasoning_effort": "low"}})
    r["pass"] = r["elapsed_s"] >= min(seconds, 100) or r["finish"] == "length"
    r["survived_past_100s"] = r["elapsed_s"] >= 100
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--long-stream-seconds", type=float, default=110)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = os.environ["VLLM_API_KEY"]
    c = OpenAI(base_url=f"{a.base_url}/v1", api_key=key, timeout=600)

    results = {"base_url": a.base_url, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, fn in [("auth", lambda: check_auth(a.base_url, key)), ("reasoning", lambda: check_reasoning(c)),
                     ("tool_call", lambda: check_tool_call(c)), ("ttfat", lambda: check_ttfat(c)),
                     ("long_stream", lambda: check_long_stream(c, a.long_stream_seconds))]:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"pass": False, "error": repr(e)[:500]}
        print(f"{name}: {'PASS' if results[name]['pass'] else 'FAIL'}", flush=True)

    results["pass"] = all(v["pass"] for k, v in results.items() if isinstance(v, dict))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))
    sys.exit(0 if results["pass"] else 1)


if __name__ == "__main__":
    main()
