# Qwen3.5 Adaptive Scheduler Policy

> **TL;DR:** Issue #727 now lands Qwen3.5 scheduler policy plumbing with
> conservative defaults: `off` remains the default, `auto` is explicit opt-in,
> `--max-prefill-tokens` remains a hard per-step cap, and TP rejects `auto`
> instead of silently downgrading to `off`. The independent validation track
> (`scripts/sweep_727.sh` + `run_serving_bench.sh` QPS A/B) is prepared but its
> acceptance numbers are **pending a reachable GPU**. Because `auto` is
> cap-preserving and only fires a decode-priority tick when an active decode has
> ≤4 tokens remaining, it is expected neutral on the standard/QPS/long-output
> guards, and the current `mixed` harness cannot cleanly stage that tick — see
> the Independent Validation Track section.
>
> **Last touched:** 2026-07

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
  - **Run the prepared #727 acceptance matrix** (`scripts/sweep_727.sh` plus the two `run_serving_bench.sh` QPS A/B lines it prints) once a GPU is reachable, then replace the pre-review §5 cells with the real off-vs-auto evidence.
  - **Exercise `auto` cleanly**: add a background-stream replenish option to `mixed.rs` so active decodes reach the ≤4-token decode-finish window while a cold prefill is in flight — the only way to turn the "clear mixed-load ITL tail improvement" criterion into a measurable cell (see the measurement gap in the Independent Validation Track section).

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
decode has ≤4 tokens remaining and (2) trimming the final chunk to the remaining
prompt; it never exceeds `--max-prefill-tokens`. Two consequences drive the
matrix:

- On the standard `1024/256` c1/c16, `1024/128` QPS 8/12/16, and long-output
  concurrency cells the fixed chunk path is preserved, so `auto` is expected
  **neutral** — that neutrality *is* the "no material regression" acceptance bar.
- The mixed cells reuse the proven #470 long-lived-background workload (a huge
  `--bg-output-len` keeps `decode_n == bg_concurrency`, so the validity gate
  holds) extended to wider active-decode batches (`bg ∈ {8,16}`, `max_batch =
  2·bg`) and the `auto`/`off` axis. Under this workload the ≤4-token window is
  never hit, so `auto == off` by construction: these cells prove `auto` does
  **not** regress the mixed tail; they do not show a tail *improvement*.

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

### Status

Harness landed and `bash -n`-checked on the Mac authoring host. **The acceptance
run is blocked on a reachable GPU** — the RTX 5090/4090 hosts in `~/.ssh/config`
require VPN and were unreachable at authoring time, and no `prime`/`gh`
credentials are configured locally. Resume once a GPU is reachable:

```bash
cargo build --release -p openinfer-server --bin bench_serving --features qwen35
MODEL=<absolute Qwen3.5-4B path> scripts/sweep_727.sh
# then run the two run_serving_bench.sh QPS A/B lines printed by the sweep footer
```
