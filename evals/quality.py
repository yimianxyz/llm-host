#!/usr/bin/env python3
"""Quality gate for the serving stack. It checks that W4A8, the FP8 KV cache, MTP and the parsers do not
break the model; it is not a leaderboard run.

  gsm8k     [--limit N]                       accuracy; gate >= 95% (reference 97.1-97.4%, SGLang cookbook FP8)
  toolcalls                                   ~25 cases x {non-stream, stream}, including known parser-bug edges
  niah      --lengths 131072 204800 [--depths 0.1 0.5 0.9]  exact retrieval plus prefill/decode speed at long context
  gpqa      [--limit N]                       GPQA-Diamond pass@1 (reference 90.28% W4A8 / 89.65% FP8, model card)

All results go to --out as JSON. Requests use the server's generation defaults (temperature 1.0, top_p 0.95).
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time

from openai import OpenAI

MODEL = "glm-5.3"


def client(base: str) -> OpenAI:
    return OpenAI(base_url=f"{base}/v1", api_key=os.environ.get("VLLM_API_KEY", "EMPTY"), timeout=3600, max_retries=2)


def reasoning_of(obj) -> str:
    extra = getattr(obj, "model_extra", None) or {}
    return getattr(obj, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning") or ""


def effort(level: str) -> dict:
    return {"chat_template_kwargs": {"reasoning_effort": level}}


def save(out: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in obj.items() if k not in ("items", "failures", "cases")}, indent=2, default=str))


# ---------------------------------------------------------------- GSM8K
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm8k_answer(text: str):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    cand = m.group(1) if m else (NUM.findall(text) or [None])[-1]
    try:
        return float(cand.replace(",", "")) if cand else None
    except ValueError:
        return None


def run_gsm8k(c: OpenAI, a) -> dict:
    from datasets import load_dataset

    ds = list(load_dataset("openai/gsm8k", "main", split="test"))
    if a.limit:
        ds = random.Random(0).sample(ds, a.limit)

    def one(ex):
        prompt = ex["question"] + "\n\nSolve the problem. End with the final numeric answer on its own line as '#### <number>'."
        r = c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                      max_tokens=a.max_tokens, extra_body=effort(a.effort))
        text = r.choices[0].message.content or ""
        gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        pred = gsm8k_answer(text)
        return {"ok": pred is not None and abs(pred - gold) < 1e-6, "gold": gold, "pred": pred,
                "tokens": r.usage.completion_tokens, "finish": r.choices[0].finish_reason, "tail": text[-200:]}

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(one, ds))
    acc = sum(r["ok"] for r in res) / len(res)
    return {"task": "gsm8k", "n": len(res), "accuracy": round(acc, 4), "gate_pass": acc >= 0.95, "effort": a.effort,
            "seconds": round(time.time() - t0, 1), "mean_completion_tokens": round(sum(r["tokens"] for r in res) / len(res)),
            "truncated": sum(r["finish"] == "length" for r in res), "failures": [r for r in res if not r["ok"]][:30]}


# ---------------------------------------------------------------- tool calls
def fn(name, desc, props, required=None, **extra):
    params = {"type": "object", "properties": props, **extra}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


S = {"type": "string"}
WEATHER = fn("get_weather", "Get the current weather for a city", {"city": S, "unit": {"type": "string", "enum": ["c", "f"]}}, ["city"])
SEARCH = fn("search_files", "Search files by glob pattern", {"pattern": S, "max_results": {"type": "integer"},
                                                              "include_hidden": {"type": "boolean"}}, ["pattern"])
WRITE = fn("create_file", "Create a file with exact content", {"path": S, "content": S}, ["path", "content"])
SHELL = fn("run_shell", "Run a shell command", {"command": S, "timeout_s": {"type": "number"}}, ["command"])
CONFIG = fn("set_config", "Set a configuration key to a JSON value", {"key": S, "value": {}}, ["key", "value"])
MEETING = fn("book_meeting", "Book a meeting", {
    "attendees": {"type": "array", "items": {"type": "object", "properties": {"name": S, "email": S}, "required": ["name", "email"]}},
    "start": S, "duration_min": {"type": "integer"}}, ["attendees", "start", "duration_min"])
CLOCK = fn("get_time", "Get the current UTC time", {})
ONEOF = fn("lookup", "Look up a record by id or by email", {}, oneOf=[
    {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
    {"type": "object", "properties": {"email": S}, "required": ["email"]}])
ALL = [WEATHER, SEARCH, WRITE, SHELL, CONFIG, MEETING, CLOCK]

CODE = "\n".join(f"def f{i}(x):\n    return x * {i} + {i}  # step {i}" for i in range(120))


def u(text):
    return [{"role": "user", "content": text}]


def calls_named(m, name):
    return [t for t in m["calls"] if t["name"] == name]


def arg(m, name="", key=None):
    cs = calls_named(m, name) if name else m["calls"]
    return cs[0]["args"].get(key) if cs and isinstance(cs[0]["args"], dict) else None


CASES = [
    ("simple", u("What's the weather in Paris?"), [WEATHER], None,
     lambda m: "paris" in str(arg(m, "get_weather", "city")).lower()),
    ("enum", u("Weather in Tokyo, in fahrenheit please."), [WEATHER], None,
     lambda m: arg(m, "get_weather", "unit") == "f"),
    ("int_bool", u("Find at most 5 files matching *.py, including hidden ones."), [SEARCH], None,
     lambda m: arg(m, "search_files", "max_results") == 5 and arg(m, "search_files", "include_hidden") is True),
    ("quotes", u('Create a.txt whose content is exactly: He said "hello" and left'), [WRITE], None,
     lambda m: arg(m, "create_file", "content") == 'He said "hello" and left'),
    ("multiline_code", u("Create add.py containing a Python function add(a, b) that returns a + b."), [WRITE], None,
     lambda m: "def add" in str(arg(m, "create_file", "content")) and "\n" in str(arg(m, "create_file", "content"))),
    ("unicode", u("Create hi.txt with exactly this content: Grüße, 你好, 🚀"), [WRITE], None,
     lambda m: arg(m, "create_file", "content") == "Grüße, 你好, 🚀"),
    ("long_argument", u("Create big.py with exactly this content, verbatim:\n" + CODE), [WRITE], None,
     lambda m: str(arg(m, "create_file", "content")).strip() == CODE.strip()),
    ("json_null", u("Set config key proxy to null."), [CONFIG], None,
     lambda m: calls_named(m, "set_config") and arg(m, "set_config", "value") is None),
    ("json_number", u("Set config key retries to the number 3."), [CONFIG], None,
     lambda m: arg(m, "set_config", "value") == 3),
    ("json_bool", u("Set config key debug to false."), [CONFIG], None,
     lambda m: arg(m, "set_config", "value") is False),
    ("json_nested", u('Set config key layout to the JSON value {"a": {"b": [1, 2, {"c": null}]}}.'), [CONFIG], None,
     lambda m: arg(m, "set_config", "value") == {"a": {"b": [1, 2, {"c": None}]}}),
    ("array_of_objects", u("Book 30 minutes at 2026-11-10T15:00 with Alice (alice@x.com) and Bob (bob@x.com)."), [MEETING], None,
     lambda m: arg(m, "book_meeting", "duration_min") == 30 and len(arg(m, "book_meeting", "attendees") or []) == 2),
    ("parallel", u("What's the weather in Paris and in Berlin? Check both."), [WEATHER], None,
     lambda m: {str(t["args"].get("city", "")).lower() for t in calls_named(m, "get_weather")} >= {"paris", "berlin"}),
    ("shell_special_chars", u('Run exactly this command: grep -rn "TODO" src/ | wc -l'), [SHELL], None,
     lambda m: arg(m, "run_shell", "command") == 'grep -rn "TODO" src/ | wc -l'),
    ("windows_path", u(r"Create the file C:\Users\me\my file.txt with content: ok"), [WRITE], None,
     lambda m: arg(m, "create_file", "path") == r"C:\Users\me\my file.txt"),
    ("no_arg_tool", u("What time is it in UTC?"), [CLOCK], None,
     lambda m: calls_named(m, "get_time") and m["calls"][0]["args"] in ({}, None)),
    ("choice_required", u("Hello there."), ALL, "required", lambda m: len(m["calls"]) >= 1),
    ("choice_named", u("Tell me something."), ALL, {"type": "function", "function": {"name": "get_time"}},
     lambda m: calls_named(m, "get_time")),
    ("choice_none", u("What's the weather in Paris?"), [WEATHER], "none",
     lambda m: not m["calls"] and m["finish"] != "tool_calls"),
    ("should_not_call", u("What is 2+2? Answer directly."), ALL, None,
     lambda m: not m["calls"] and "4" in m["content"]),
    ("markers_in_prose", u("In one paragraph, explain what the literal text <tool_call> means in GLM chat templates. "
                           "Do not call any tools."), ALL, None, lambda m: not m["calls"] and len(m["content"]) > 20),
    ("first_step_only", u("Find all *.md files, then create SUMMARY.md listing them."), [SEARCH, WRITE], None,
     lambda m: calls_named(m, "search_files")),
    ("long_context_tool", [{"role": "system", "content": "Project notes: " + " ".join(f"note{i}" for i in range(6000))},
                           {"role": "user", "content": "What's the weather in Oslo?"}], ALL, None,
     lambda m: "oslo" in str(arg(m, "get_weather", "city")).lower()),
]
# Informational only: model-level or known-open engine issues, recorded but not gated.
INFO_CASES = [
    ("info_empty_props_oneof_glm5_169", u("Look up the record with id 42."), [ONEOF], None,
     lambda m: arg(m, "lookup", "id") == 42),
]


def run_case(c: OpenAI, case, stream: bool) -> dict:
    name, messages, tools, choice, check = case
    kw = dict(model=MODEL, messages=messages, tools=tools, max_tokens=12000, extra_body=effort("low"))
    if choice is not None:
        kw["tool_choice"] = choice
    t0 = time.time()
    if not stream:
        r = c.chat.completions.create(**kw)
        ch = r.choices[0]
        raw = [(t.function.name, t.function.arguments) for t in (ch.message.tool_calls or [])]
        content, finish = ch.message.content or "", ch.finish_reason
    else:
        acc, content, finish = {}, "", None
        for ev in c.chat.completions.create(stream=True, **kw):
            if not ev.choices:
                continue
            d = ev.choices[0].delta
            content += d.content or ""
            finish = ev.choices[0].finish_reason or finish
            for tc in d.tool_calls or []:
                slot = acc.setdefault(tc.index, {"name": "", "args": ""})
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        raw = [(v["name"], v["args"]) for _, v in sorted(acc.items())]
    calls = []
    for n, s in raw:
        try:
            calls.append({"name": n, "args": json.loads(s) if s else {}})
        except json.JSONDecodeError:
            calls.append({"name": n, "args": None, "raw": s[:500]})
    m = {"calls": calls, "content": content, "finish": finish}
    try:
        ok = bool(check(m))
    except Exception:
        ok = False
    if calls and finish != "tool_calls" and choice is None:  # forced choices may legitimately finish with "stop"
        ok = False
    return {"case": name, "stream": stream, "ok": ok, "finish": finish, "seconds": round(time.time() - t0, 2),
            "calls": calls, "content": content[:300]}


def run_toolcalls(c: OpenAI, a) -> dict:
    jobs = [(case, s, info) for info, cases in ((False, CASES), (True, INFO_CASES)) for case in cases for s in (False, True)]

    def one(j):
        case, s, info = j
        r = run_case(c, case, s)
        if not r["ok"] and not info:  # one retry separates a flaky sampling choice from a deterministic parser bug
            r2 = run_case(c, case, s)
            r = {**r2, "retried": True, "first_attempt": r}
        r["informational"] = info
        return r

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(one, jobs))
    gated = [r for r in res if not r["informational"]]
    passed = sum(r["ok"] for r in gated)
    return {"task": "toolcalls", "gated_cases": len(gated), "passed": passed, "pass_rate": round(passed / len(gated), 4),
            "gate_pass": passed == len(gated), "flaky_then_passed": sum(1 for r in gated if r.get("retried") and r["ok"]),
            "informational": {r["case"] + ("/stream" if r["stream"] else ""): r["ok"] for r in res if r["informational"]},
            "failures": [r for r in res if not r["ok"]], "cases": res}


# ---------------------------------------------------------------- needle in a haystack
WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand "
         "part place case week company system program question work government number night point home water room mother "
         "area money story fact month lot right study book eye job word business issue side kind head house service friend "
         "father power hour game line end member law car city community name president team minute idea kid body "
         "information back parent face others level office door health person art war history party result change morning "
         "reason research girl guy moment air teacher force education river mountain engine signal market garden").split()


def haystack(rng: random.Random, n_words: int) -> list:
    out, sent = [], []
    for _ in range(n_words):
        sent.append(rng.choice(WORDS))
        if len(sent) >= rng.randint(8, 18):
            out.append(" ".join(sent).capitalize() + ".")
            sent = []
    return out


def count_tokens(base: str, text: str) -> int:
    import urllib.request

    req = urllib.request.Request(f"{base}/tokenize", data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["count"]


def run_niah(c: OpenAI, a) -> dict:
    models = c.models.list().data
    max_len = getattr(models[0], "max_model_len", None) or (models[0].model_extra or {}).get("max_model_len")
    sample = " ".join(haystack(random.Random(0), 20000))
    words_per_token = 20000 / count_tokens(a.base_url, sample)  # calibrated on the served tokenizer
    items = []
    for L in a.lengths:
        if max_len and L > max_len:
            items.append({"length": L, "skipped": f"max_model_len={max_len}"})
            continue
        for d in a.depths:
            rng = random.Random(f"{L}-{d}-{time.time()}")  # unique haystack per request: no prefix-cache reuse
            vault, code = rng.randint(1000, 9999), f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(100, 999)}"
            sents = haystack(rng, int((L - 600) * words_per_token))
            sents.insert(int(len(sents) * d), f"The secret passphrase for vault {vault} is {code}.")
            prompt = " ".join(sents) + f"\n\nWhat is the secret passphrase for vault {vault}? Reply with only the passphrase."
            t0, first, text, usage = time.perf_counter(), None, "", None
            for ev in c.chat.completions.create(model=MODEL, messages=u(prompt), max_tokens=2048, stream=True,
                                                stream_options={"include_usage": True}, extra_body=effort("low")):
                if ev.usage:
                    usage = ev.usage
                if ev.choices:
                    dl = ev.choices[0].delta
                    if (dl.content or reasoning_of(dl)) and first is None:
                        first = time.perf_counter() - t0
                    text += dl.content or ""
            total = time.perf_counter() - t0
            pt = usage.prompt_tokens if usage else None
            ct = usage.completion_tokens if usage else None
            items.append({"length": L, "depth": d, "prompt_tokens": pt, "found": code in text, "ttft_s": round(first or 0, 2),
                          "prefill_tok_s": round(pt / first) if pt and first else None,
                          "decode_tok_s": round(ct / (total - first), 1) if ct and first and total > first else None,
                          "answer": text[:120]})
            print(json.dumps(items[-1]), flush=True)
    run = [i for i in items if "found" in i]
    return {"task": "niah", "max_model_len": max_len, "words_per_token": round(words_per_token, 3),
            "gate_pass": bool(run) and all(i["found"] for i in run), "items": items}


# ---------------------------------------------------------------- GPQA-Diamond
def run_gpqa(c: OpenAI, a) -> dict:
    from datasets import load_dataset

    ds = list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=os.environ.get("HF_TOKEN")))
    if a.limit:
        ds = ds[: a.limit]

    def one(i_ex):
        i, ex = i_ex
        opts = [ex["Correct Answer"], ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(i).shuffle(order)
        letters = "ABCD"
        gold = letters[order.index(0)]
        body = "\n".join(f"{letters[k]}) {opts[j].strip()}" for k, j in enumerate(order))
        prompt = f"{ex['Question'].strip()}\n\n{body}\n\nThink it through, then finish with 'ANSWER: <letter>'."
        r = c.chat.completions.create(model=MODEL, messages=u(prompt), max_tokens=a.max_tokens, extra_body=effort(a.effort))
        text = r.choices[0].message.content or ""
        m = re.findall(r"ANSWER:\s*\(?([ABCD])\)?", text)
        return {"ok": bool(m) and m[-1] == gold, "gold": gold, "pred": m[-1] if m else None,
                "tokens": r.usage.completion_tokens, "finish": r.choices[0].finish_reason}

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(one, enumerate(ds)))
    acc = sum(r["ok"] for r in res) / len(res)
    return {"task": "gpqa_diamond", "n": len(res), "accuracy": round(acc, 4), "effort": a.effort,
            "reference": {"w4a8_card": 0.9028, "fp8_card": 0.8965}, "seconds": round(time.time() - t0, 1),
            "mean_completion_tokens": round(sum(r["tokens"] for r in res) / len(res)),
            "truncated": sum(r["finish"] == "length" for r in res), "unparsed": sum(r["pred"] is None for r in res)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["gsm8k", "toolcalls", "niah", "gpqa"])
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--lengths", type=int, nargs="+", default=[131072, 204800])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    a = ap.parse_args()
    c = client(a.base_url)
    if a.task == "gpqa":
        a.effort = a.effort if a.effort != "high" else "max"
        a.max_tokens = max(a.max_tokens, 32768)
        a.concurrency = min(a.concurrency, 32)
    if a.task == "toolcalls":
        a.concurrency = min(a.concurrency, 16)
    t0 = time.time()
    res = {"gsm8k": run_gsm8k, "toolcalls": run_toolcalls, "niah": run_niah, "gpqa": run_gpqa}[a.task](c, a)
    res["wall_seconds"] = round(time.time() - t0, 1)
    save(a.out, res)


if __name__ == "__main__":
    main()
