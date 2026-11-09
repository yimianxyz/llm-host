# GLM-5.3 on one RunPod 4x H200 pod: final report

This report answers the eight design questions for long-running, self-hosted GLM-5.3 serving agentic coding. It
draws on one $200 study on 2026-11-08/09: about 9 pod-hours on a 4x H200 SXM Secure Cloud pod in EUR-IS-5 at
$21.16/hr. The design and every flag are explained in [`docs/DESIGN.md`](../docs/DESIGN.md). The step-by-step evidence
is in [`FINDINGS.md`](FINDINGS.md), and every table regenerates from raw data with `python evals/report.py` into
[`REPORT_DATA.md`](REPORT_DATA.md).

## The answer in brief

- **It works, and quality holds.** One 4x H200 pod running the `gpustack/GLM-5.3-W4A8` checkpoint on stock vLLM v0.33.0
  scores 96.7% on GSM8K (the FP8 reference is 97.1-97.4%) and passes 48/48 tool-call cases. It found every
  needle from 128K to 1M tokens. GPQA-Diamond is consistent with the card's 90.3% (97.0% of the answers that fit a
  32K-token budget, 80.3% counting the 17% that did not).
- **Capacity: 4 concurrent agent sessions at the SLO with margin, 6 at the edge.** At 4 sessions each user gets 77 tok/s
  (median request) with a 0.7 s median time to first token. At 6 the median request is at 50.0 tok/s, exactly on the
  line.
- **Throughput is set by prefill, not decode.** The node completes 2,000-2,300 agent turns per hour whatever the
  concurrency. More sessions past 4-6 only slow everyone down.
- **Cost: about $0.01 per agent turn, or $5.29 per session-hour at 4 sessions ($3.53 at 6).** That is about what NVIDIA
  Dynamo's published 8x H200 FP8 setup achieves per session (about $4), at half the minimum spend.
- **Speed: 148 tok/s for a single user, about 2x Z.ai's own API.** Decode speed stays the same from 80K to 1M tokens of
  context.
- **Context: 256K in the agentic config; 1M works as a separate profile.** At 1M a cold prompt takes 5.5 minutes to
  prefill, and it must bypass RunPod's HTTPS proxy.
- **Stability:** PENDING (Step 7 soak).
- **Operations:** a cold start takes about 35 minutes, mostly the 373 GiB download, and a warm restart about 4.3
  minutes. No network volume is needed. Liveness must be checked with a real generation, because the API can stay up
  while the engine is dead.

## Recommended deployment

- **Pod:** 4x H200 SXM, Secure Cloud, 600 GB container disk, ports `8000/http` and `22/tcp`. Add vLLM's port as TCP
  as well if clients may send prompts over about 400K tokens. Section 5 of `DESIGN.md` has the exact REST v2 body;
  `deploy/pod.py create` sends it.
- **Server config:** [`deploy/configs/recommended.args`](../deploy/configs/recommended.args), passed to
  `vllm/vllm-openai:v0.33.0`. Its essentials:
  - tensor parallel 4 with expert parallelism, plus decode context parallel 4;
  - FP8 sparse-MLA KV cache with prefix caching, plus a 300 GiB host-memory KV tier;
  - 8K prefill chunks, CUDA graphs captured up to 128 tokens;
  - MTP speculative decoding with 5 drafts at 1-4 running requests, 3 at 5-16 and 1 above;
  - util 0.92, 256K max length;
  - the `glm47` tool parser and the `glm45` reasoning parser.
- **Long-context profile:** [`deploy/configs/L4d-ctx1m-bt2k.args`](../deploy/configs/L4d-ctx1m-bt2k.args), the same
  pod and weights with 2K chunks and 1M max length.
- **Client rules:**
  - Stream every request.
  - Send a real User-Agent.
  - Declare a JSON-schema type for every tool parameter.
  - Keep `clear_thinking=false`, so the history stays byte-stable and the prefix cache keeps hitting.
  - Give `reasoning_effort=max` a token budget well above 32K, or use `high`. Treat `finish_reason=length` as a retry
    signal.
  - Pin each session to one pod.

## The eight questions

### 1. Is this the right hardware, checkpoint and engine?

**Yes, for a team that needs up to about 6 concurrent coding agents.**

**Hardware.** 4x H200 is the smallest Hopper node that holds GLM-5.3 with room for KV.
- **Weights:** 107.3 GiB of the 139.8 GiB usable on each GPU.
- **Why not 5 GPUs:** the model's 64 heads and 256 experts only split over 2, 4 or 8 GPUs, so a fifth GPU would sit
  idle.
- **Price:** the pod costs $21.16/hr, against $42.32/hr for the 8x H200 that official FP8 needs.

**Checkpoint.** W4A8 (INT4 experts, FP8 elsewhere) is what makes 4 GPUs possible; FP8 would need 176 GiB per GPU. The
quality evidence:

| Check | W4A8 here (first working config B4 / final config L5d) | Reference |
|---|---|---|
| GSM8K, all 1,319, zero-shot | 96.74% / 96.89% | FP8 97.1-97.4% (SGLang cookbook, 1-shot) |
| Tool calls, 24 typed cases x stream/non-stream | 48/48 / 48/48 | all must pass |
| Needles at 128K and 200K (both configs); 512K and 1M (1M profile) | all found | exact retrieval |
| MTP acceptance on real coding prompts | 2.95 tokens per step at 3 drafts / 3.50 at 5 drafts | InferenceX golden value for GLM-5.3 at 3 drafts: 2.94 |
| GPQA-Diamond, one pass over 198 questions, max effort, 32K token budget | 80.3% raw; 97.0% of the 164 that finished; 34 hit the budget | card: W4A8 90.28%, FP8 89.65% |

**GPQA needs reading carefully.** The raw 80.3% is a token-budget artifact: 17% of the answers were still reasoning at
32,768 tokens and produced no answer. The true score lies between 80.3% (every truncated answer wrong) and 97.5% (every
one right), and the card's figures are inside that range. The practical lesson is for clients: max-effort reasoning
on hard problems averages 13.7K tokens and often needs more than 32K.

The MTP result is a weak quality signal as well as a speed one. Acceptance this close to the golden value is consistent
with W4A8's output distribution staying close to the reference model's on coding prompts.

**Engine.** vLLM v0.33.0 runs the W4A8 checkpoint natively. Each optimization that survived the ladder is a vLLM
feature: decode context parallelism, the host KV tier, and per-batch-size MTP schedules.

**Caveats:**
- **DCP is not optional.** Without it the KV cache is copied to every GPU and the node holds about 131K tokens, which
  is one agent session.
- **Prefill is structurally slow, about 5.6K tok/s for the node.** With 16 heads per GPU and DCP, vLLM must run prefill
  through the FP8 *decode* kernel: 36% of prefill time, plus 26% in collectives. Every lever we tried on prefill (chunk
  size, util) moved it by a few percent at most. This is the binding constraint on capacity, and the next thing to watch
  upstream.
- **The checkpoint is Hopper-only** (SM90 CUTLASS kernels). On Blackwell, use NVFP4.

### 2. What is the best config, and why each flag?

The final config is rung L5d, saved as [`recommended.args`](../deploy/configs/recommended.args). Each flag is
justified by measurement in [`DESIGN.md` section 4](../docs/DESIGN.md#4-the-recommended-config-every-flag-and-the-evidence).
The ladder that produced it changed one variable at a time:

| Rung | Change | Decision | Deciding evidence |
|---|---|---|---|
| L1 | DCP4 | kept | KV goes from about 131K to over 1M tokens |
| L3 | MTP 5/3/1-draft schedule | kept | +8% per-user speed for a single user; sessions level |
| L5p | 16K prefill chunks | reverted | OOM on a 128K prompt; cold turns only 2.8% faster |
| L2 | 300 GiB host KV tier | kept | 16 sessions: collapse (0 conversations finished) becomes 16/16 in 5.5 min |
| L7 | DSpark drafter | reverted | +27% for a single user, but the KV pool halves and 8 sessions collapse |
| L4 | 1M context | separate profile | needs 2K chunks, which cost the agentic workload throughput |
| L5c | 4K chunks + CUDA-graph cap | chunks reverted, cap kept | 4K: -11% per-user speed at 4 sessions. Cap: +21% KV, restart 357 s to 257 s |

**What the kept rungs bought over the first working config (B4), at the SLO point of 4 sessions:**
- **Single user:** 135 to 148 tok/s.
- **4 sessions:** 69 to 77 tok/s at the median.
- **KV pool:** 1.04M to 1.17M tokens on GPU, plus about 5.5M in host memory.
- **Overload:** 16 sessions now complete instead of collapsing.
- **Restart:** 277 to 257 s.

The capture cap is runtime-neutral: the single-user engine step is 24.0 ms with it and 24.3 ms without.

### 3. How many sessions fit at the SLO, and what does it cost?

**The SLO:** time to first token p50 of 5 s or less, median per-request speed of at least 50 tok/s at real coding
acceptance, and zero errors. This is NVIDIA Dynamo's agentic SLO.

**The workload:** the vLLM team's OpenHands trace. Each session runs 13 turns: a cold 74K-token first turn, then about
1K new tokens and 220 output tokens per turn, with 91.5% of prompt tokens served from the prefix cache.

| Sessions | Median tok/s per user | Mean tok/s per user | TTFT p50 / p90 / p99 | Turns per hour | $ per turn | $ per session-hour | Errors |
|---|---|---|---|---|---|---|---|
| 1 | 147.7 | 137.4 | 0.48 / 0.53 / 13.2 s | 1,396 | 0.0152 | 21.16 | 0 |
| 2 | 101.5 | 75.4 | 0.52 / 0.79 / 25.9 s | 1,784 | 0.0119 | 10.58 | 0 |
| **4** | **77.1** | 43.2 | 0.71 / 11.0 / 39.9 s | 2,003 | **0.0106** | **5.29** | 0 |
| 6 | 50.0 (SLO edge) | 26.7 | 0.76 / 1.3 / 64.8 s | 2,182 | 0.0097 | 3.53 at the edge | 0 |
| 8 | 39.5 | 19.7 | 0.60 / 8.4 / 77.4 s | 2,283 | 0.0093 | past SLO | 0 |
| 16 | 24.2 | 12.3 | 2.38 / 23.5 / 153 s | 2,340 | 0.0090 | past SLO | 0 |

**The answer: 4 sessions with margin, 6 at the edge.** At 6, the median request ran at 49.98 tok/s. The soak's
repeated blocks at 5 and 6 sessions settle the boundary: PENDING (Step 7).

**Why the node saturates at about 2,300 turns per hour.** Each turn computes about 6.8K new prompt tokens on average:
the cold 74K first turns amortized over 13 turns, plus about 1K new tokens per turn. 2,300 turns per hour is therefore
about 4.3K tok/s of prefill, close to the node's 5.6K ceiling. Past 4-6 sessions extra concurrency adds almost no
throughput and stretches every user's latency.

**Cost units, at 4 sessions:**
- **Per agent turn:** $0.0106. This is the useful unit for agents.
- **Per logical token:** $0.13 per 1M, counting cached prompt tokens; about 8% of them are computed.
- **Per output token:** $48 per 1M. This is misleading for agents, because each turn emits only 220 tokens against
  80K tokens of context.
- **Decode-heavy reasoning, for contrast:** GPQA at 32 concurrent max-effort requests averaged 920 output tok/s (about
  35 tok/s per stream), which is **$6.4 per 1M output tokens**.

**Utilization decides the real price.** A pod billed 24/7 costs $508 per day. Serving 4 sessions for 8 working hours a
day, the effective price is $15.87 per session-hour, three times the full-load figure. Section 7 covers deleting the pod
off-hours.

**Against 8 GPUs.** Dynamo's published 8x H200 FP8 recipe serves about 10-11 sessions at 57 tok/s on a similar agentic
SLO, about $4 per session-hour at $42.32/hr. The 4-GPU pod matches that per session at its edge ($3.53 at 6) and costs
30% more with margin ($5.29 at 4). It needs half the minimum spend, and a failure takes out half as much capacity. We
did not run the 8-GPU setup ourselves, and Dynamo's workload (64K in, 400 out, 90% cache hits) is close to ours but not
identical.

### 4. What speed and time to first token do users get?

**Decode speed.** Steps come in two kinds:
- A pure decode step takes 24-46 ms for 1-4 sessions.
- A step that also carries another session's 8K prefill chunk takes about 1.42 s.

The median request mostly sees the first kind; the mean includes the stalls in full.

| | 1 session | 4 sessions | 6 sessions |
|---|---|---|---|
| Median request | 148 tok/s | 77 tok/s | 50 tok/s |
| Mean, including other sessions' prefill stalls | 137 tok/s | 43 tok/s | 27 tok/s |
| A typical 220-token turn, median (TTFT + decode) | 2.0 s | 3.6 s | 5.1 s |

**Comparison:** Z.ai's own API runs at about 66 tok/s, and the fastest provider Artificial Analysis lists (Baseten,
FP8) at about 263 tok/s.

**Time to first token:**
- **Warm turn** (history cached, about 1K new tokens): 0.48 s for one session, 0.71 s at 4 sessions. The host tier adds
  about 80 ms to the median.
- **Cold first turn** of 74K tokens: about 13 s of prefill, during which every co-running session stalls. Waiting
  behind other cold turns gives the p99 of 40 s at 4 sessions.
- **Short prompt at idle:** the first answer token arrives after 0.15 s at low reasoning effort, 0.31 s at high and
  0.36 s at max.

**Long context:** the decode step is 23.9 ms at 1M tokens of context, the same as at 80K. Sparse attention reads only
the indexer's top 2,048 tokens, so a single user decodes at about 148 tok/s at any context length.

**Unpredictable output is the exception.** All these speeds assume coding-like text, where MTP accepts 3.5 tokens per
step. On random-token prompts acceptance falls to 1.18, and a single user gets 48 tok/s, against 72 tok/s on the
3-draft baseline. Five drafts cost more to verify than they return there. Output dominated by hashes, base64 or
random IDs would see the same slowdown.

**What users feel most:** other people's cold starts. A new session, or a returning session whose history has to be
recomputed, costs everyone else about 13 s of stall. Keeping histories warm (the host tier, session affinity) matters
more for perceived speed than any decode optimization.

### 5. What is the context ceiling: 256K or 1M?

**256K for the agentic config; 1M as a separate profile.**

**Why the agentic config stops at 256K.** Sparse attention allocates a workspace of batched tokens x max-model-len x 4
bytes, and vLLM's memory profiler does not count it.
- At 8K chunks: 8 GiB at 256K, 16 GiB at 512K, against about 11.8 GiB of headroom per GPU.
- Raising `--gpu-memory-utilization` does not help, because it shrinks that same headroom.

**1M works with 2K chunks** ([`L4d-ctx1m-bt2k.args`](../deploy/configs/L4d-ctx1m-bt2k.args)):

| Prompt, single stream | Needle found | Time to first token | Prefill tok/s |
|---|---|---|---|
| 130,542 tokens | yes | 30.8 s | 4,235 |
| 523,973 tokens | yes | 135.9 s | 3,857 |
| 1,039,828 tokens | yes | 330.5 s | 3,146 |

The 1M profile's costs:
- **Memory:** the GPU KV pool is 1,230,848 tokens, so one 1M request occupies 85% of it and the whole engine for 5.5
  minutes.
- **Throughput:** 2K chunks cost 24% of prefill speed at 128K against 8K chunks. That is why it is a separate profile
  rather than the default.
- **Network:** RunPod's HTTPS proxy sits behind Cloudflare, which returns 524 when the first byte takes more than 100 s.
  Prompts over about 400K tokens must use the pod's direct TCP port, behind your own TLS.

**Recommendation:** route the rare prompts over 256K to a pod running the 1M profile; a restart switches profiles in
about 4.5 minutes. Keep agent sessions under 256K with context compaction, which most coding agents already do.

### 6. Is it stable for hours, and what fails?

**Soak (Step 7):** PENDING. The soak runs the final config under cycling load at 2, 4, 5 and 6 sessions, with:
- a real-generation liveness probe every 30 s;
- GPU memory and container RSS every minute;
- one GPQA-Diamond run at 32 concurrent max-effort requests;
- one deliberate unplanned kill of vLLM.

**Failure modes seen during the study, and what each one requires of operations:**

| Failure | When it happens | Symptom | What prevents or catches it |
|---|---|---|---|
| Unprofiled GPU buffers overflow | The first large request after a config change (B1, B2, L5p, L4c) | CUDA OOM in FlashMLA or the attention workspace | Change config only with a gate: a 128K needle plus the tool suite before taking traffic |
| Engine dies but the API stays up | After an engine OOM | `/health` can pass while every request hangs | Liveness through a real generation (`evals/ready.sh`); escalate SIGTERM to SIGKILL after 30 s |
| Prefix working-set cliff | Active sessions x history exceed GPU KV (about 13 sessions of 86K) | GPU prefix hits collapse, everything re-prefills, turns stop completing | The 300 GiB host tier (kept); admission cap; alert on prefix hits |
| Proxy timeout | First byte later than 100 s: long prompts, or cold turns queued past the SLO (TTFT p99 77 s at 8 sessions) | HTTP 524 from Cloudflare | Stream; admission cap; direct TCP for long context |
| Overload | Sessions far past the SLO | No errors and no preemptions even at 16 sessions, but TTFT p99 reaches 153 s | Admission control; alert on queue and TTFT, not on errors |
| Odd max-model-len | 651,264 (not a power of two) | `illegal memory access` at init | Use only lengths validated by a needle run |

**Failures seen and survived during the study** (not caused by the model or engine):
- **A container restart keeps the container disk.** An accidental kill of PID 1 restarted the container, and the
  weights were still present, so recovery was one vLLM boot.
- **A pattern-matching kill took out the container.** `pkill -f "vllm serve"` matches the start script and kills PID 1.
  Supervisors must match the real process (`bin/vllm serve`).

### 7. How long do cold start and warm restart take, and is a network volume needed?

| Phase | Time | Notes |
|---|---|---|
| Create to container start | 54 s | The image was already cached on the host; a cold image pull adds more |
| Container to sshd | 7 s | |
| Weights: 373 GiB from Hugging Face | 28 min | 230 MB/s with one process, 270-310 MB/s with 4 parallel shard downloads (`deploy/fastdl.sh`) |
| First vLLM boot | about 5-6 min | 56 s to read weights, plus compile and CUDA-graph capture, plus pinning 300 GiB of host memory |
| **Cold start total** | **about 35 min, about $12** | |
| **Warm restart** (config change or crash; weights and compile cache on disk) | **257 s** to the first real generation | 357 s before the capture cap |
| Unplanned crash to ready (SIGKILL drill) | **257 s**; clients saw about 270 s of refused connections | The supervisor restarted vLLM 5 s after the kill |

**No network volume.**
- **It would not be faster.** A standard RunPod network volume reads at about 200-400 MB/s. Hugging Face delivered
  230-310 MB/s in EUR-IS-5, so a volume would not shorten the cold start.
- **It would cost money and capacity.** It costs about $0.07 per GB-month, about $31/month for the weights, and pins
  every pod to the volume's data center. H200 capacity in one data center is the scarcer resource.
- **The high-performance tier** (up to 3x throughput, in select data centers) might cut the download to about 10 min.
  We did not measure it; it is the one storage option worth testing if cold starts become frequent.

**What survives what:**

| Event | Container disk (weights, compile cache) | Recovery |
|---|---|---|
| vLLM crash | kept | one boot, 4-5 min |
| Container crash (PID 1 exits) | kept (observed) | one boot, 4-5 min |
| RunPod stop/start, restart action, or pod edit | **wiped** (RunPod docs) | full cold start, 35 min |
| Delete | gone | full cold start on a new pod |

**Off-hours:** delete the pod and recreate it about 35 minutes before it is needed, rather than stopping it.
- **Why not stop:** a stop wipes the container disk anyway. The stopped pod also stays bound to its host and resumes only
  if that host's GPUs are still free, while still billing for storage.
- **What deleting saves:** each night off saves $21.16 per idle hour, against about $12 of cold start.
- **The risk is capacity:** a 4x H200 Secure pod may not be available at recreate time. Check `get-capacity` first and
  keep a second data center as a fallback.

### 8. When should a second pod be added, and what routing does it need?

**Add a second pod when any of these holds:**
- **Demand exceeds about 2,000 agent turns per hour, or 4-6 concurrently active sessions.** The node's throughput is
  flat past that point (question 3), so only a second pod adds capacity.
- **TTFT p50 stays above 2 s, or requests wait in the queue for most of a 15-minute window.**
- **Uptime matters.** One pod is a single point of failure: a crash costs about 4-5 minutes of outage, and losing the
  host costs about 35 minutes plus finding capacity. Two pods from day one is the high-availability design.

**Routing requirements, in order of importance:**
1. **Session affinity.** The prefix cache and the host KV tier live inside one pod. A turn sent to the wrong pod
   recomputes its whole history, about 13 s at 74K tokens, and stalls that pod's other sessions. Hash a stable session
   id (a conversation header, or the user) to a pod, and re-pin only when a pod leaves.
2. **Admission cap per pod.** Allow about 6 active sessions per pod and queue the rest at the router, so overload shows
   up as queueing rather than as everyone's turn slowing to 25 tok/s and cold-turn TTFT passing the proxy's 100 s
   limit.
3. **Health-aware membership.** Drop a pod when its real-generation probe fails twice. Sessions moved to the survivor pay
   one cold prefill each, so spread them out.
4. **A long-context lane.** Send prompts over 256K tokens to a pod running the 1M profile, over its direct TCP port.

**Tools:** a consistent-hash proxy covers items 1-3 at this scale. KV-aware routers (vLLM's production-stack router, or
Dynamo's KV router) do the same with real cache state and are worth it beyond a handful of pods.

## Ops runbook

**Deploy:**
1. Check capacity: `get-capacity` or `list-gpu-types` for 4x H200 Secure Cloud.
2. Create: `deploy/pod.py create` with `recommended.args`, the API key and the HF token as RunPod secrets.
3. Wait for the `ready.sh` probe to pass (about 35 minutes cold).
4. Run the smoke test through the public URL: `evals/smoke.py`.

**Health checks:**
- **Liveness, every 30 s:** `evals/ready.sh`, a real 16-token generation. Restart vLLM after 2 consecutive failures. A
  `/health` check alone does not catch a dead engine.
- **Readiness after any restart:** the same probe, then put the pod back into routing.
- **Config changes:** only through a gate (tool suite, 128K needle, 200 GSM8K questions), because OOMs from unprofiled
  buffers appear on the first large request, not at boot.

**Alerts** (vLLM Prometheus metrics at `/metrics`, plus the probe):

| Alert | Signal | Threshold | Meaning and action |
|---|---|---|---|
| Engine dead | probe | 2 consecutive failures (60 s) | Restart vLLM; page if 3 restarts in an hour |
| Users waiting | `vllm:num_requests_waiting` | > 0 for most of 15 min | Over capacity: admission cap or second pod |
| Slow first token | per-request TTFT p50 | > 2 s over 15 min (warn), > 5 s (SLO breach) | Same; check for long-context traffic |
| Cache cliff | GPU prefix-hit rate (`vllm:prefix_cache_hits` / `queries`) | below 50% while load is normal | Working set exceeds GPU KV; check the host tier is hitting |
| Host tier not helping | `vllm:external_prefix_cache_hits` / `queries` | low while GPU hits are low | Cold re-prefills: affinity broken, or the tier is full |
| Restart loop | restart counter in the supervisor log | 3 per hour | Bad config or hardware; roll back the config |
| Memory growth | GPU memory used, container RSS | PENDING (soak trend) | Leak; schedule a warm restart |
| 524s at the edge | client or proxy logs | any | A prompt or queue is past 100 s to the first byte |

**Restart policy:**
- **Automatic:** the supervisor restarts vLLM on exit, and `bench.sh apply` escalates SIGTERM to SIGKILL after 30 s.
- **Crash loops:** give up after 3 boots and roll back to the last good config.
- **Planned restarts:** drain the pod (stop routing new sessions, let turns finish), restart, probe, and readmit. A
  restart drops the GPU and host KV caches, so every session's next turn is cold.
- **Scheduled restarts:** only if the soak shows memory growth: PENDING.

**Weights storage:** the container disk. No network volume (question 7).

**Capacity management:** watch turns per hour and active sessions against the 2,000-turn and 4-6-session lines. Add the
second pod before them, not after: a cold start takes 35 minutes.

## What the study cost

| Step | Balance after | Spent in step |
|---|---|---|
| Start | $200.00 | |
| Rehearsal (1x A40) and Step 1 deploy, including 4 failed boot configs | $173.23 | $26.77 |
| Steps 2-3: quality gate, baseline, reference shapes | $153.76 | $19.46 |
| Step 4: MTP schedule, 16K chunks, host tier | $137.83 | $15.93 |
| Step 5 and L4: DSpark, 1M attempts | $111.30 | $26.53 |
| L5c, L4d (1M at 2K chunks), Gate B | $93.58 | $17.72 |
| Step 6: final characterization | $70.54 | $23.04 |
| Step 7: soak | PENDING | |

## Limits of this study and what to try next

**Limits:**
- **One synthetic workload.** The OpenHands trace has short sessions: one cold 74K turn per 13 turns, with fixed
  220-token outputs. Real agent sessions are usually longer, so cold turns are rarer, and both per-user speed and turns
  per hour should be better.
- **The trace forces `ignore_eos` at temperature 0.** That inflates speculative acceptance, which `report.py` corrects
  using acceptance measured on real coding prompts.
- **One pod on one host, and one region for the network measurements.**
- **No head-to-head runs** against 8x H200 FP8 or SGLang.

**Next experiments, by expected value:**
1. **DSpark with an FP8 drafter KV cache**, so it stops halving the KV pool. It gave +27% single-user speed.
2. **The prefill path.** A BF16 sparse-prefill kernel that works at 16 heads per GPU with DCP would lift the binding
   constraint. Watch vLLM upstream.
3. **The MTP schedule above 16 running requests.** It drops to 1 draft there (acceptance 1.74 on GPQA at c=32). A
   step at that load verifies only 64 tokens, so 3 drafts may be faster for decode-heavy batch work. This is cheap to
   test.
4. **A real agent trace** (your own sessions) to replace the synthetic one in the capacity numbers.
5. **The high-performance network-volume tier**, if cold starts become frequent.
