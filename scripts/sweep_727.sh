#!/usr/bin/env bash
# sweep_727.sh — issue #727 Qwen3.5 adaptive-scheduler validation (independent
# track). The policy itself is already merged (PR #730: `off` default, `auto`
# opt-in); this harness only *validates* it. It never starts a second scheduler.
#
# It runs the acceptance cells that a single in-process `bench_serving` binary
# can express, once per policy (off, auto), writing <name>.json + <name>.log per
# cell and printing the #470 ITL_STEP validity gate. The open-loop HTTP QPS
# pressure cells (1024/128 @ qps 8/12/16) are NOT here — they need an HTTP
# client; run them with tools/bench/run_serving_bench.sh (see the footer).
#
# Policy semantics that shape this matrix (openinfer-qwen35 scheduler/plan.rs
# `choose_prefill_budget`): `auto` differs from `off` ONLY by (1) yielding a
# decode-priority tick when some active decode has <= 4 tokens left, and (2)
# trimming the final prefill chunk to the remaining prompt. It NEVER exceeds
# --max-prefill-tokens. Consequence for measurement:
#   * The standard / long-output / QPS cells keep the fixed chunk path, so `auto`
#     is expected NEUTRAL there — that is exactly the "no material regression"
#     acceptance bar, not a null result.
#   * The mixed cells below reuse the PROVEN #470 long-lived-background workload
#     (bg-output-len huge -> decode_n stays == bg-concurrency, so the #470
#     validity gate holds) extended to WIDER active-decode batches + the auto/off
#     axis. Under this workload the <=4-token decode-finish window is never hit,
#     so `auto` == `off` by construction: the mixed cells prove auto does NOT
#     regress the tail, they do NOT demonstrate a tail *improvement*.
#   * Cleanly exercising the decode-finish tick needs active decodes that reach
#     completion *while a cold prefill is in flight*. The current `mixed` bench
#     spawns a fixed background set and never replenishes it, so bounding the
#     background output just collapses decode_n (breaking the #470 gate) instead
#     of staging a controlled finish. Producing a clean "mixed ITL tail improved"
#     cell is therefore a follow-up that needs a background-stream replenish
#     option in mixed.rs (or a staged-completion workload generator).
#
# Prereq (build the qwen35 bench binary; run from the repo root):
#   cargo build --release -p openinfer-server --bin bench_serving --features qwen35
# GPU note: acceptance thresholds are host-relative. #727's prior doc §5 evidence
# is on RTX 5090; a 4090 / other-GPU run is a SEPARATE baseline, not comparable
# to the 5090 numbers.
set -u

BIN=${BIN:-target/release/bench_serving}
MODEL=${MODEL:-models/Qwen3.5-4B}
DATA=${DATA:-datasets/qwen35-727-validation}
COOLDOWN=${COOLDOWN:-25}
POLICIES=${POLICIES:-"off auto"}
mkdir -p "$DATA"
export RUST_LOG=${RUST_LOG:-info}

gpu_used() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
  else
    echo "n/a"
  fi
}

# run_cell NAME GLOBAL_FLAGS... SUBCOMMAND SUB_FLAGS...
# Global flags (--model-path/--format/--out plus --max-batch/--qwen35-scheduler-policy)
# MUST precede the subcommand; callers pass everything from --max-batch onward.
run_cell() {
  local name=$1
  shift
  echo "[$(date +%T)] cell: $name  (gpu used $(gpu_used) MiB)"
  # shellcheck disable=SC2086
  OPENINFER_ITL_DEBUG=1 "$BIN" \
    --model-path "$MODEL" --format json --out "$DATA/${name}.json" \
    "$@" > "$DATA/${name}.log" 2>&1
  local rc=$?
  local gate
  gate=$(python3 scripts/itl_step_agg.py "$DATA/${name}.log" 2>/dev/null \
    | grep -E 'decode_n=' | tr '\n' ' ')
  echo "  exit=$rc  itl_step: ${gate:-<none>}"
  sleep "$COOLDOWN"
}

echo "=== #727 in-process acceptance sweep start $(date +%T) ==="
for pol in $POLICIES; do
  # A. Standard serving regression guard (expect NEUTRAL vs off).
  run_cell "std_1024x256_c1_${pol}" \
    --max-batch 16 --qwen35-scheduler-policy "$pol" \
    request --prompt-len 1024 --output-len 256 --concurrency 1 --warmup 1 --iters 3
  run_cell "std_1024x256_c16_${pol}" \
    --max-batch 16 --qwen35-scheduler-policy "$pol" \
    request --prompt-len 1024 --output-len 256 --concurrency 16 --warmup 1 --iters 3

  # B. Long-output concurrency (acceptance: no large TTFT regression).
  run_cell "longout_1024x2048_c8_${pol}" \
    --max-batch 8 --qwen35-scheduler-policy "$pol" \
    request --prompt-len 1024 --output-len 2048 --concurrency 8 --warmup 1 --iters 3

  # C. Mixed-load ITL: WIDER active-decode batch (bg 8/16, max_batch = 2*bg so
  # the injector always has a free slot) + long-lived background (valid #470
  # gate). auto vs off expected neutral here (see header).
  for bg in 8 16; do
    mb=$((bg * 2))
    for p in 4096 8192; do
      for q in 0.5 1.0; do
        qtag=$(echo "q$q" | tr '.' 'p')
        run_cell "mixed_bg${bg}_mb${mb}_p${p}_${qtag}_${pol}" \
          --max-batch "$mb" --qwen35-scheduler-policy "$pol" \
          mixed --bg-concurrency "$bg" --bg-prompt-len 512 --bg-output-len 8192 \
          --inj-prompt-len "$p" --inj-output-len 1 --qps "$q" \
          --num-injections 10 --inj-warm-frac 0.0 --warmup 5
      done
    done
  done

  # D. Negative control: max_batch == bg_concurrency must be INVALID (#470 gate),
  # proving the slot-starvation artifact is detected, not silently reported.
  run_cell "mixed_negctl_bg8_mb8_${pol}" \
    --max-batch 8 --qwen35-scheduler-policy "$pol" \
    mixed --bg-concurrency 8 --bg-prompt-len 512 --bg-output-len 8192 \
    --inj-prompt-len 4096 --inj-output-len 1 --qps 0.5 \
    --num-injections 10 --inj-warm-frac 0.0 --warmup 5
done
echo "=== SWEEP_DONE $(date +%T) ==="

cat <<EOF

Aggregate the true per-step ITL_STEP stall + validity gate per mixed cell:
  python3 scripts/itl_step_agg.py $DATA/mixed_*.log
  # valid  = a stall step with prefill_tok>0 AND decode_n == bg_concurrency
  # invalid= negative-control cells (decode_n never reaches bg_concurrency)

Open-loop HTTP QPS pressure cells (1024/128 @ qps 8/12/16), run per policy
(needs a qwen35 server build + vllm-bench; FEATURES/QWEN35_SCHED_POLICY are
forwarded to the openinfer server by run_serving_bench.sh):
  FEATURES=qwen35 QWEN35_SCHED_POLICY=off  MAX_BATCH=16 QPS_LIST='8 12 16' \\
    INPUT_LEN=1024 OUTPUT_LEN=128 MODEL=$MODEL tools/bench/run_serving_bench.sh
  FEATURES=qwen35 QWEN35_SCHED_POLICY=auto MAX_BATCH=16 QPS_LIST='8 12 16' \\
    INPUT_LEN=1024 OUTPUT_LEN=128 MODEL=$MODEL tools/bench/run_serving_bench.sh
EOF
