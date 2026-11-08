#!/usr/bin/env python3
"""Deployment helper over RunPod REST v2, the same API the RunPod MCP `create-pod` tool is generated from.
The create body is rendered from git (deploy/start.sh plus a deploy/configs/*.args file), so a deployment
can be reproduced without MCP. Stdlib only; reads RUNPOD_API_KEY from .env.

  render  --config deploy/configs/B0.args [pod options]    print the create-pod JSON (secrets as placeholders)
  create  --config deploy/configs/B0.args [pod options]    POST /v2/pods, write .env.pod (POD_ID, SSH_HOST, SSH_PORT)
  get     POD_ID | ssh POD_ID | delete POD_ID
  stock   --gpu-id "NVIDIA H200" --count 4                 per-data-center stock and price for that shape
  ensure-secrets | ensure-ssh-key                          one-time account setup (idempotent)
"""

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = "https://api.runpod.io/v2"
IMAGE = "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e"
MODEL = "gpustack/GLM-5.3-W4A8"
MODEL_REV = "cb867a359aa39552f48f14f294a5f15418c97789"
SSH_PUB = pathlib.Path.home() / ".ssh/id_ed25519_runpod.pub"
SECRETS = {"vllm_api_key": "VLLM_API_KEY", "hf_token": "HF_TOKEN"}


def load_env() -> None:
    for name in (".env", ".env.pod"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def call(method: str, path: str, body=None, query=None):
    url = API + path + ("?" + urllib.parse.urlencode(query, doseq=True) if query else "")
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Authorization", f"Bearer {os.environ['RUNPOD_API_KEY']}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:2000]}")


def render(a) -> dict:
    env = {
        "MODEL": a.model, "MODEL_REV": a.model_rev, "MAX_HOURS": str(a.max_hours),
        "SERVE_ARGS": pathlib.Path(a.config).read_text(),
        "HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    env.update({var: "{{ RUNPOD_SECRET_%s }}" % name for name, var in SECRETS.items()})
    body = {
        "name": a.name, "image": a.image, "cloud": a.cloud,
        "gpu": {"id": a.gpu_id, "count": a.gpu_count, "minCudaVersion": a.min_cuda},
        "disk": a.disk, "ports": ["8000/http", "22/tcp"], "startSsh": True,
        "entrypoint": ["bash", "-c"], "cmd": [(ROOT / "deploy/start.sh").read_text()],
        "env": env,
    }
    if a.dc:
        body["dataCenterIds"] = a.dc
    return body


def ssh_endpoint(pod: dict):
    d = (pod.get("ssh") or {}).get("direct") or {}
    return d.get("host"), d.get("port")


def write_pod_env(pod: dict) -> None:
    host, port = ssh_endpoint(pod)
    lines = [f"POD_ID={pod['id']}", f"POD_NAME={pod['name']}", f"SSH_HOST={host or ''}", f"SSH_PORT={port or ''}",
             f"PROXY_URL=https://{pod['id']}-8000.proxy.runpod.net"]
    (ROOT / ".env.pod").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "create", "get", "ssh", "delete", "stock", "ensure-secrets", "ensure-ssh-key"])
    ap.add_argument("pod_id", nargs="?")
    ap.add_argument("--config", default=str(ROOT / "deploy/configs/B0.args"))
    ap.add_argument("--name", default="glm53-w4a8-4xh200")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-rev", default=MODEL_REV)
    ap.add_argument("--gpu-id", default="NVIDIA H200")
    ap.add_argument("--gpu-count", type=int, default=4)
    ap.add_argument("--min-cuda", default="13.0")
    ap.add_argument("--cloud", default="SECURE")
    ap.add_argument("--disk", type=int, default=600)
    ap.add_argument("--max-hours", type=float, default=9.1)
    ap.add_argument("--dc", nargs="*", help="data center ids from the stock read")
    ap.add_argument("--count", type=int, default=4)
    a = ap.parse_args()

    if a.cmd == "render":
        print(json.dumps(render(a), indent=2))
    elif a.cmd == "create":
        pod = call("POST", "/pods", render(a))
        print(json.dumps({k: pod.get(k) for k in ("id", "name", "status", "dataCenterId", "cudaVersion", "cost")}, indent=2))
        write_pod_env(pod)
    elif a.cmd in ("get", "ssh"):
        pod = call("GET", f"/pods/{a.pod_id}")
        pod.pop("env", None)
        if a.cmd == "ssh":
            write_pod_env(pod)
        else:
            print(json.dumps(pod, indent=2))
    elif a.cmd == "delete":
        call("DELETE", f"/pods/{a.pod_id}")
        time.sleep(3)
        print(f"deleted {a.pod_id}")
    elif a.cmd == "stock":
        res = call("GET", "/catalog/gpus", query={"include": "AVAILABILITY", "product": "POD", "count": a.count,
                                                  "minCudaVersion": a.min_cuda})
        for g in res.get("gpus", []):
            if a.gpu_id and g.get("id") != a.gpu_id:
                continue
            print(json.dumps(g, indent=2))
    elif a.cmd == "ensure-secrets":
        have = {s["name"] for s in call("GET", "/account/secrets").get("secrets", [])}
        for name, var in SECRETS.items():
            if name in have:
                print(f"secret {name}: exists")
            elif os.environ.get(var):
                call("POST", "/account/secrets", {"name": name, "value": os.environ[var], "description": f"llm-host {var}"})
                print(f"secret {name}: created")
            else:
                sys.exit(f"{var} missing from .env")
    elif a.cmd == "ensure-ssh-key":
        mine = SSH_PUB.read_text().strip()
        keys = call("GET", "/account/ssh-keys").get("keys", [])
        if any(k.split()[:2] == mine.split()[:2] for k in keys):
            print("ssh key: already registered")
        else:
            call("PUT", "/account/ssh-keys", {"keys": keys + [mine]})
            print(f"ssh key: registered (account now has {len(keys) + 1})")


if __name__ == "__main__":
    main()
