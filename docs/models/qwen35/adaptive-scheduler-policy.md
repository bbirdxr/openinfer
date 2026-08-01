# Qwen3.5 Adaptive Scheduler Policy

> **TL;DR:** Issue #727 policy plumbing stays default-`off` / opt-in `auto` with
> a hard `--max-prefill-tokens` cap and TP rejection of `auto`. Independent
> acceptance on **1×RTX 4090** (`scripts/sweep_727.sh` + HTTP QPS, tip `ded5fee`,
> task `650826`) shows `auto` vs `off` **neutral** on standard/long-output
> (deltas ≤0.5%, identical hash0) and **no mixed-tail regression** under the
> #470 `ITL_STEP` gate (16/16 wider-batch cells `stall_at_bg>0`; negctl
> `decode_n==bg` absent + starvation warning). HTTP `1024/128` QPS 8/12/16: ITL
> p99 matches; out tok/s −3.2%…−5.6%, reproduced (same-policy control: 0.15%) and
> **fully attributed** — the decode-priority tick inserts 885 extra decode-only
> steps (+10.5 s wall) and buys TPOT p50 −17% in exchange. The mixed stall p50
> drop is attributed too, to co-packing suppression, which *raises* total stall
> time 5.7% — so it is a distribution artifact, not a win. Both are
> latency/throughput trades, which is why `auto` stays opt-in. A decode-finish
> *improvement* still needs the `mixed.rs` replenish follow-up.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` - Qwen3.5 roadmap, scheduler, benchmark, and evidence docs are the relevant route.
  - `docs/models/qwen35/roadmap.md` - #469 is the current HTTP boundary; #470/#727 must keep mixed-load evidence separate from serving parity.
  - `docs/subsystems/scheduler/scheduler.md` - the scheduler is single-threaded GPU ownership with chunked prefill and unified prefill+decode.
  - GitHub #727 - asks for adaptive decode-priority policy work with explicit off mode and standard-cell regression protection.
  - GitHub PR #730 review - requires `off` as the default, `--max-prefill-tokens` as a hard cap, and explicit TP rejection for `auto`.
- **Relevant history**:
  - `docs/benchmarks/qwen35-4b-serving-vllm-rtx5090-2026-07.md` - standard HTTP cells and QPS pressure are the current serving regression boundary.
  - Prior stream-overlap exploration stayed out of scope; #727 reuses the existing unified/decode/chunk controls before adding any new execution stream.
- **Plan**:
  1. Add a small Qwen3.5 scheduler policy enum with default `off` and explicit opt-in `auto`.
  2. Move the adaptive decision into pure scheduler-plan helpers with unit tests for fixed, hard-cap, final-chunk, and decode-finish protection cases.
  3. Wire the policy through Qwen3.5 launch, the OpenInfer server CLI, and `bench_serving` so validation can compare opt-in `auto` vs default `off`.
  4. Run narrow Rust checks locally, then use the remote GPU host for Qwen3.5 build/test and representative benchmark cells.
- **Risks / open questions**:
  - The local Mac is not a CUDA validation host, so runtime evidence must come from the provided remote GPU environment.
  - The policy should not claim vLLM parity or production readiness; it is a scheduler-regression and mixed-load tail gate.

## Execution Log

### Step 1: Scheduler policy and pure decision helper

- Added `Qwen35SchedulerPolicy::{Auto, Off}` in `openinfer-qwen35`, defaulting existing Qwen3.5 launch paths to `Off`.
- Added `choose_prefill_budget(...)` in `scheduler/plan.rs` so the adaptive decision is unit-testable outside the GPU loop.
- Policy rules:
  - `Off` preserves the fixed base prefill budget.
  - No active decode or no in-flight prefill keeps the fixed budget.
  - Active requests with at most 4 tokens remaining get one decode-priority tick before the FIFO-front prefill continues.
  - `Auto` never returns more than the configured base budget; `--max-prefill-tokens` stays a hard per-step cap.
  - Final chunks may shrink below the cap when fewer prompt tokens remain.

### Step 2: Runtime and benchmark wiring

- Threaded the policy through Qwen3.5 launch, `openinfer` server CLI, and `bench_serving`:
  - `--qwen35-scheduler-policy auto|off` defaults to `off`.
  - Tensor-parallel Qwen3.5 rejects `auto` because TP Phase 1 does not run unified prefill+decode.
  - Qwen3.5 `--max-batch` now accepts `1..=MAX_DECODE_BATCH`; non-bucket requests such as `5` allocate the next graph bucket internally but admit only the requested slots.
- Added mixed-load report visibility for `max_batch` / `max_prefill_tokens` and warnings when `bg_concurrency >= max_batch`; `max_batch=4,bg=4` remains a starvation negative control, while retained mixed evidence used `max_batch=5,bg=4`.

### Step 3: Local checks

Commands run from the issue worktree:

```bash
cargo fmt --all -- --check
git diff --check
git diff --name-only -z | xargs -0 rg -n '<private-patterns>' || true
codex-style-check --no-fail docs/models/qwen35/adaptive-scheduler-policy.md docs/index.md
```

Result: format and diff whitespace passed. The private-data scan found no changed-file hits. `codex-style-check` only reported pre-existing `docs/index.md` Kimi rows, not this task doc.

### Step 4: Remote GPU build and tests

Validation host contract:

| Field | Value |
| --- | --- |
| GPU | 1x NVIDIA GeForce RTX 5090 |
| Driver / CUDA toolkit | NVIDIA driver `595.71.05`, `nvcc 12.8` |
| Rust | `rustc/cargo 1.99.0-nightly` |
| Source | upstream/main `8dd3953` plus this patch |
| Feature | `qwen35` |
| Model | `Qwen/Qwen3.5-4B` downloaded through ModelScope on 2026-07-20 |
| Model config | `model_type=qwen3_5`, `architectures=["Qwen3_5ForConditionalGeneration"]`, `config.json` sha256 `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| Build env | `OPENINFER_CUDA_SM=120`, `CUDA_HOME` set to CUDA 12.8, `OPENINFER_TRITON_PYTHON` set to a Triton 3.7.1 Python |

Remote checks:

```bash
cargo fmt --all -- --check
git diff --check
cargo test -p openinfer-qwen35 --features qwen35 adaptive_prefill_budget -- --nocapture
cargo test -p openinfer-server --features qwen35 qwen35 -- --nocapture
cargo build --release -p openinfer-server --features qwen35
cargo build --release --bin bench_serving --features qwen35
OPENINFER_TEST_MODEL_PATH=<absolute model path> \
  cargo test --release -p openinfer-qwen35 --features qwen35 --test e2e_scheduler -- --nocapture
```

Result: all checks passed. `e2e_scheduler` passed the single-GPU test; the TP2 test remained ignored on the one-GPU host.

### Step 5: Pre-review benchmark cells

These cells were gathered before review narrowed the PR to default-off, cap-preserving behavior. They explain why the first draft tried whole-prefill for one low-pressure mixed cell, but they are not current default-policy evidence.

Benchmark flags shared by those pre-review cells:

- Engine: OpenInfer Qwen3.5 direct `bench_serving`, CUDA Graph enabled, feature `qwen35`.
- Source: upstream/main `8dd3953` plus this patch.
- Hardware/toolchain/model: same as the remote GPU contract above.
- Sampling: synthetic random prompts, greedy, fixed output.
- Standard cells: `1024/256`, warmup 1, iters 3, `--max-batch 16`.
- Mixed cells: `--max-batch 5`, `bg_concurrency=4`, background `512/2048`, injection `4096/1`, `qps=0.5`, 5 cold injections, warmup 1, `--skip-baseline`, with background and injection generated-token lengths/hashes retained in the JSON.
- Negative control: `--max-batch 4`, `bg_concurrency=4`, kept to prove the warning path; not used as improvement evidence.

Standard request A/B:

| Policy | Cell | TTFT p50 ms | steady TPOT p50/p99 ms | request tok/s | output length | hash0 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `auto` | 1024/256 c1 | 49.816 | 6.942 / 7.022 | 140.68 | 256-256 | `0827a7035c7b7a89` |
| `off` | 1024/256 c1 | 50.176 | 6.949 / 7.150 | 140.37 | 256-256 | `0827a7035c7b7a89` |
| `auto` | 1024/256 c16 | 508.187 | 9.763 / 58.321 | 76.73 | 256-256 | `0827a7035c7b7a89` |
| `off` | 1024/256 c16 | 508.343 | 9.770 / 58.255 | 76.76 | 256-256 | `0827a7035c7b7a89` |

Mixed-load A/B:

| Policy | all ITL p50/p99/max ms | steady p99/max ms | stall p50/p99/max ms | stall gaps | warnings |
| --- | ---: | ---: | ---: | ---: | --- |
| `auto` | 7.386 / 7.845 / 196.554 | 7.754 / 58.553 | 7.941 / 196.552 / 196.554 | 60 / 4944 | none |
| `off` | 7.389 / 57.211 / 65.232 | 7.794 / 58.536 | 57.141 / 65.229 / 65.232 | 120 / 4904 | none |

Mixed output sanity:

| Policy | background output length | background hash0 | injection output length | injection hash0 |
| --- | --- | --- | --- | --- |
| `auto` | 1236-1238 | `dea24e27083abe47` | 1-1 | `ec2064181e172bb6` |
| `off` | 1226-1228 | `40933b449d599567` | 1-1 | `ec2064181e172bb6` |

Interpretation: this pre-review whole-prefill variant improved p99 for one low-pressure 4k/1-token mixed cell but raised max ITL. Review correctly treated that as a tradeoff, so the landed policy no longer exceeds the configured prefill cap and does not use these cells to justify a default flip.

The starvation negative control (`max_batch=4,bg=4`) emitted the expected warning, plus QPS/background-length warnings caused by the intentionally saturated setup. It remains a measurement guard only.

## Debrief

- **Outcome**: #727's scheduler policy plumbing, explicit opt-in `auto`, default `off`, server/bench CLI wiring, TP `auto` rejection, non-bucket Qwen3.5 `max_batch`, and cap-preserving budget tests are implemented.
- **Pitfalls encountered**:
  - Triton 3.3.0 could not AOT for `cc120`; the validation host used Triton 3.7.1.
  - Hugging Face download was unavailable from the host; ModelScope provided the same public `Qwen/Qwen3.5-4B` model family, with config hash recorded above.
  - A relative `OPENINFER_TEST_MODEL_PATH` failed for `e2e_scheduler` because the test process cwd differed; the rerun used an absolute path and passed.
  - `bench_serving mixed` still infers stall windows from request `[submit,last-token]`; the retained `max_batch=5,bg=4` capacity gate avoids the known starvation artifact, but this doc does not claim internal `decode_n` trace instrumentation.
- **Lessons learned**:
  - The adaptive path should stay opt-in until wider active-decode/QPS evidence chooses an SLA objective.
  - A `max_batch=4,bg=4` mixed cell is a negative control, not evidence of overlap.
  - Qwen3.5 TP should reject `auto` until TP supports unified mixed steps.
- **Follow-ups**:
  - **Exercise `auto` cleanly**: add a background-stream replenish option to `mixed.rs` so active decodes reach the ≤4-token decode-finish window while a cold prefill is in flight — the only way to turn the "clear mixed-load ITL tail improvement" criterion into a measurable cell (see the measurement gap in the Independent Validation Track section).
  - HTTP QPS A/B is now persisted under `/user/xurui1/oi727/datasets/qwen35-727-validation/` (task `650826`); see Status below.
  - Both deltas are now attributed from `ITL_STEP` step accounting (see Status): the mixed stall p50 drop is co-packing suppression (total stall time +5.7%), and the HTTP throughput loss is 885 extra decode-only steps from the decode-priority tick. Neither is unexplained any more, so the remaining open item is the improvement side below.

## Independent Validation Track (#727 acceptance)

> Owner: `bbirdxr` — validation only; PR #730 owns the policy and this track
> starts no second scheduler. The GitHub #727 thread scopes it to: retain the
> #470 `ITL_STEP` overlap validity gate (stall buckets from step timestamps, not
> the `[submit, last-token]` window), run the missing `1024/128` QPS 8/12/16
> cells, add long-output concurrency, cover wider active-decode batches than
> `bg_concurrency=4`, and report p50/p99/max + TTFT + throughput +
> completions/failures + output lengths/hashes + saturated cells without hiding
> regressions.

### Measurement design (derived from the merged `auto` semantics)

`choose_prefill_budget` (`openinfer-qwen35/src/scheduler/plan.rs`) makes `auto`
differ from `off` only by (1) a decode-priority tick (budget 0) when some active
decode has ≤4 tokens remaining and (2) capping the step budget at the FIFO-front
request's remaining prompt; it never exceeds `--max-prefill-tokens`. Clause (2)
is more than a final-chunk trim: `take_prefill_chunks` spends the step budget
across the whole prefill *queue*, so capping it at the front request also stops
that step from co-packing the next queued prefill (unit-pinned by
`adaptive_prefill_budget_stops_step_from_copacking_queued_prefills`). Two
consequences drive the matrix:

- On the standard `1024/256` c1/c16, `1024/128` QPS 8/12/16, and long-output
  concurrency cells the fixed chunk path is preserved, so `auto` is expected
  **neutral** — that neutrality *is* the "no material regression" acceptance bar.
- The mixed cells reuse the proven #470 long-lived-background workload (a huge
  `--bg-output-len` keeps `decode_n == bg_concurrency`, so the validity gate
  holds) extended to wider active-decode batches (`bg ∈ {8,16}`, `max_batch =
  2·bg`) and the `auto`/`off` axis. The same huge `--bg-output-len` also puts the
  ≤4-token window out of reach (4/8192 per background stream), so clause (1)
  effectively never fires here: these cells prove `auto` does **not** regress the
  mixed tail; they do not show a decode-finish tail *improvement*. Clause (2) can
  still bind, because `--bg-prompt-len 512` sits below the 1024 base budget —
  see the unattributed stall p50 drop in Status.

**Open measurement gap.** Cleanly exercising the decode-finish tick needs active
decodes that reach completion *while a cold prefill is in flight*. The current
`mixed` bench spawns a fixed background set and never replenishes it, so bounding
the background output only collapses `decode_n` (breaking the #470 gate) instead
of staging a controlled finish. A clean "mixed ITL tail improved" cell is
therefore the `mixed.rs` replenish follow-up above. The remaining acceptance
criteria (no regression, explicit failures, retained output hashes, disableable
`off`) are measurable now.

### Harness

- `scripts/sweep_727.sh` — in-process cells (standard c1/c16, long-output c8,
  the wider-batch mixed matrix, and the `max_batch == bg` negative control), run
  once per policy with per-cell JSON + log and the `ITL_STEP` gate. It reuses the
  existing `bench_serving` flags; no bench code change.
- `tools/bench/run_serving_bench.sh` now forwards `FEATURES` /
  `QWEN35_SCHED_POLICY` / `MAX_BATCH` to the openinfer server, so the open-loop
  `1024/128` @ qps 8/12/16 pressure cells can be A/B'd off vs auto over HTTP.
- Policy-decision unit coverage already lives in `scheduler/plan.rs`
  (`adaptive_prefill_budget_*`); this track only adds serving-level evidence.

Attribution runs re-use the same script with `OPENINFER_ITL_DEBUG=1` exported, so
the server emits one `ITL_STEP` line per scheduler step and `itl_step_agg.py` can
split stalls by forwarded chunk shape:

```bash
export OPENINFER_ITL_DEBUG=1
FEATURES=qwen35 QWEN35_SCHED_POLICY=auto MAX_BATCH=16 QPS_LIST=16 \
  CONCURRENCY_LIST= INPUT_LEN=1024 OUTPUT_LEN=128 SECONDS_PER_RUN=60 \
  SKIP_BUILD=1 MODEL=models/Qwen3.5-4B RESULT_DIR=itl2_auto \
  tools/bench/run_serving_bench.sh
python3 scripts/itl_step_agg.py --label off --label auto \
  itl2_off/server-openinfer-*.log itl2_auto/server-openinfer-*.log
```

Pass `QWEN35_SCHED_POLICY` as a literal. An empty value makes
`run_serving_bench.sh` skip `--qwen35-scheduler-policy` entirely and the server
falls back to its default `off`, so both arms silently measure the same policy —
the tell is a result filename without the `-off` / `-auto` label, and step-for-step
identical `ITL_STEP` accounting. Enabling `ITL_STEP` also costs throughput, so
compare only within a debug-enabled pair, never against a non-debug run.

### Status — 4090 acceptance (in-process + HTTP QPS complete)

Two preemptible 4090 hosts; absolute latencies are host-local (do **not**
compare across hosts or against the §5 RTX 5090 pre-review cells). Results for
the completed matrix live on shared JuiceFS:
`/user/xurui1/oi727/datasets/qwen35-727-validation/`.

| Field | First pass (`645702`) | Re-persist (`650826`) |
| --- | --- | --- |
| GPU | 1× RTX 4090 24GB (`paratera_ningxia` / `wind-tunnel`) | same |
| Driver / CUDA | `570.133.07` / `nvcc 12.1` | same image |
| Rust | `nightly-2026-07-10` | same |
| Source | tip `428c037` | tip `ded5fee` |
| Feature / model | `qwen35`, `Qwen3.5-4B` | same |
| Build env | `OPENINFER_CUDA_SM=89`, `OPENINFER_SKIP_SUBMODULE_INIT=1`, Triton `3.7.1` | same |
| Keepalive | weak; idle-killed mid-QPS | `oi727hb` every **5s** (touch + `nvidia-smi` + CPU blip) |
| In-process | **24/24 `exit=0`**, `SWEEP_DONE` 2026-07-31T06:06:44Z | **24/24 `exit=0`**, `SWEEP_DONE` 2026-08-01T02:48:31Z |
| HTTP QPS | killed before copy-out | **6/6 JSON**, `ORCH_DONE` 2026-08-01T03:02:58Z |

Standard / long-output request A/B on `650826` (synthetic greedy; CUDA Graph on;
hash0 identical off vs auto):

| Policy | Cell | TTFT p50 ms | steady TPOT p50/p99 ms | request tok/s | out len | hash0 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `off` | 1024/256 c1 | 66.261 | 10.971 / 11.157 | 89.35 | 256 | `0827a7035c7b7a89` |
| `auto` | 1024/256 c1 | 66.216 | 10.965 / 11.147 | 89.41 | 256 | `0827a7035c7b7a89` |
| `off` | 1024/256 c16 | 688.500 | 15.232 / 80.632 | 50.98 | 256 | `0827a7035c7b7a89` |
| `auto` | 1024/256 c16 | 691.312 | 15.233 / 80.678 | 50.95 | 256 | `0827a7035c7b7a89` |
| `off` | 1024/2048 c8 | 379.200 | 12.982 / 13.691 | 75.41 | 2048 | `dc6c576de0d38289` |
| `auto` | 1024/2048 c8 | 379.555 | 12.933 / 13.606 | 75.87 | 2048 | `dc6c576de0d38289` |

`(auto−off)/off` deltas: TTFT ≤ +0.41%, TPOT p50 ≤ +0.01%, TPOT p99 ≤ +0.06% —
**no material regression** on the fixed-chunk path (matches the first-pass
conclusion; hashes unchanged).

Mixed-load `#470` `ITL_STEP` validity (stall step with `prefill_tok>0` and
`decode_n == bg_concurrency`; from `scripts/itl_step_agg.py` on cell `.log`):

| Cell family | off valid? | auto valid? |
| --- | --- | --- |
| `mixed_bg{8,16}_mb{16,32}_p{4096,8192}_q{0p5,1p0}_*` (16 cells) | yes (`decode_n=bg` stall steps > 0) | yes |
| `mixed_negctl_bg8_mb8_*` (`max_batch == bg`) | **no** (no `decode_n=8` stalls; starvation warning) | **no** |

Representative true per-step stall ITL (`mixed_bg8_mb16_p4096_q0p5`,
`itl_step_agg.py` on `.log`):

| Policy | stall p50/p99/max ms | steady decode p50/p99 ms | stall steps @ `decode_n=8` |
| --- | ---: | ---: | ---: |
| `off` | 79.58 / 85.55 / 88.45 | 12.31 / 12.87 | 40 |
| `auto` | 48.48 / 83.75 / 84.71 | 12.34 / 12.86 | 40 |

Interpretation: `auto` does **not** regress the true per-step stall or steady
decode tails, and the negctl still detects slot starvation. The stall p50 drop
is now attributed — and it is **not** a win. Splitting stall steps by forwarded
chunk shape (`itl_step_agg.py`) shows the entire delta is clause (2) suppressing
co-packing:

| Forwarded shape | off | auto |
| --- | ---: | ---: |
| `prefill_tok=512 reqs=1` | 5 steps @ 47.70 ms | **47** steps @ 47.20 ms |
| `prefill_tok=1024 reqs=1` | 40 steps @ 79.06 ms | 40 steps @ 78.78 ms |
| `prefill_tok=1024 reqs=2` | **21** steps @ 80.22 ms | **none** |
| all stall steps | 66 (avg 77.33 ms) | 87 (avg 62.03 ms) |

The 21 co-packed `off` steps become 42 single-request 512-token steps under
`auto` (`5 + 42 = 47`, `66 + 21 = 87`), while the 40 injection chunks are
identical on both — the policy never touches the injection path in this cell.
So p50 fell because the distribution was split, not because anything ran
faster: total stall time *rises* from 5104 ms to 5397 ms (**+5.7%**), since every
split step pays the ~12 ms per-step fixed cost again. `openinfer-qwen3`'s
`DEFAULT_MAX_PREFILL_TOKENS` comment already records this effect from the other
direction ("512 chunks no longer amortize the per-step fixed cost, so prefill
falls behind arrivals and TTFT queues up").

HTTP QPS pressure on `650826` (`tools/bench/run_serving_bench.sh`, `1024/128`,
qps 8/12/16, `MAX_BATCH=16`, `FEATURES=qwen35`, `CONCURRENCY_LIST=` empty,
`SKIP_BUILD=1`, `SECONDS_PER_RUN=60`, shared seeds per qps):

| Policy | qps | completed | req/s | out tok/s | TTFT p50/p99 ms | TPOT p50/p99 ms | ITL p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 8 | 480 | 5.25 | 672.0 | 14423 / 29891 | 23.15 / 23.38 | 82.60 |
| `auto` | 8 | 480 | 5.03 | 644.4 | 16326 / 33057 | 22.14 / 23.37 | 82.97 |
| `off` | 12 | 720 | 5.25 | 671.4 | 36798 / 75068 | 23.27 / 23.45 | 82.81 |
| `auto` | 12 | 720 | 5.07 | 649.6 | 38705 / 79656 | 22.15 / 23.30 | 83.00 |
| `off` | 16 | 960 | 5.28 | 675.3 | 59217 / 119843 | 23.19 / 23.50 | 82.69 |
| `auto` | 16 | 960 | 4.98 | 637.6 | 63750 / 128829 | 19.63 / 23.07 | 82.63 |

`(auto−off)/off` on out tok/s: −4.1% / −3.2% / −5.6% at qps 8/12/16. ITL p99 is
flat; TPOT p50 is slightly *better* on `auto` (23.19 → 19.63 at qps 16).
Completions = prompts (0 failures). Treat the throughput/TTFT delta as
**reported**, not hidden; it is within the open-loop saturated regime where both
policies already deliver ~5.0–5.3 req/s against offered 8–16.

**Attribution (settled).** Clause (1), the decode-priority tick, measured directly
by re-running qps 16 with `OPENINFER_ITL_DEBUG=1` (artifacts in
`itl2_{off,auto}/`):

| Step accounting @ qps 16 | off | auto |
| --- | ---: | ---: |
| total `ITL_STEP` | 7698 | 8583 |
| decode-only steps | 6738 | **7623** |
| prefill-executing steps | 960 | 960 |
| co-packed steps (`prefill_reqs=2`) | 4 | **0** |
| wall time | 182.2 s | 192.7 s |

The prefill work is identical — 960 prefill-executing steps either way. `auto`
just inserts **885 extra decode-only steps**, which at the measured 15.05 ms
steady decode step is ~10.6 s of extra wall time against an observed +10.5 s
(182.2 → 192.7 s). That accounts for the whole throughput delta. Clause (2) is
negligible here: `INPUT_LEN=1024` equals the 1024 base budget, so there are only
4 co-packed steps to suppress. The TPOT p50 gain (23.24 → 19.36 ms) is the same
mechanism seen from the decode side — this is a latency/throughput trade, not a
defect.

The effect reproduces rather than drifts: the re-run gives off 674.6 / auto 637.8
out tok/s against the original 675.3 / 637.6. A same-policy control (two
consecutive `off` runs) differed by **0.15%**, so a ~5.5% gap sits far outside
run-to-run noise. Caveat: `ITL_STEP` logging is enabled in these two runs, so
their absolute numbers are not comparable to the table above — only off-vs-auto
within them is.

Cross-check at `1024/512` (same qps 16, `SECONDS_PER_RUN=20`, 320 prompts,
artifacts in `out512_{off,auto}/`):

| `1024/512` @ qps 16 | off | auto | delta |
| --- | ---: | ---: | ---: |
| out tok/s | 909.4 | 891.9 | **−1.9%** |
| decode-only steps | 9937 | 10222 | **+285** |
| TPOT p50 ms | 17.48 | 16.50 | −5.6% |

The extra decode-only steps scale with *completed requests*, not with wall time:
`285/885 = 0.32` against `320/960 = 0.33`. The tick therefore fires ~0.9 times
per completed request in both shapes — well under the 4-token window depth,
because it also needs a prefill in flight to have anything to defer. Quadrupling
`output_len` dilutes that fixed per-request cost across 4× more generated tokens,
so the throughput gap shrinks 5.5% → 1.9% and the TPOT gain shrinks with it. The
trade is real, and its size is predictable from `output_len`.

Ops note for future preemptible re-runs: keep heartbeat aggressive (≤5s + GPU
query), write all artifacts under `/user/...` on the **same** cluster FS, and
set `OPENINFER_SKIP_SUBMODULE_INIT=1` so qwen35 builds do not recurse into
unused DeepEP/FlashMLA/DeepGEMM submodules.