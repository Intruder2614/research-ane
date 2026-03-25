#!/usr/bin/env bash
# collection/run_experiment.sh
# =============================
# Master orchestration script for the full measurement campaign.
# Runs all (model × compute_unit × precision) conditions systematically,
# captures bandwidth traces in parallel, and writes structured JSON output.
#
# Prerequisites:
#   - ANEBenchmark binary built from harness/ (Xcode)
#   - xctrace (comes with Xcode CLI tools)
#   - powermetrics (built-in on macOS)
#   - Python 3.9+ with requirements installed
#
# Usage:
#   bash collection/run_experiment.sh
#   bash collection/run_experiment.sh --models data/models/ --output data/raw/
#   bash collection/run_experiment.sh --sweep                 (working-set sweep mode)
#   bash collection/run_experiment.sh --dry-run               (print plan without running)

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
HARNESS="./harness/.build/release/ANEBenchmark"
MODELS_DIR="data/models"
SWEEP_DIR="data/models/sweep"
OUTPUT_DIR="data/raw"
ITERATIONS=500
WARMUP=200
THERMAL_CUTOFF=45.0
SWEEP_MODE=false
DRY_RUN=false
LOG_FILE="data/raw/experiment_run.log"

COMPUTE_UNITS=("cpuOnly" "cpuAndNeuralEngine" "all")

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --models)    MODELS_DIR="$2"; shift 2;;
    --output)    OUTPUT_DIR="$2"; shift 2;;
    --iterations) ITERATIONS="$2"; shift 2;;
    --sweep)     SWEEP_MODE=true; shift;;
    --dry-run)   DRY_RUN=true; shift;;
    --help)
      echo "Usage: $0 [--models DIR] [--output DIR] [--iterations N] [--sweep] [--dry-run]"
      exit 0;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

# ── setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
mkdir -p "data/raw/traces"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "ANE Cache Research — Experiment Run"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(uname -n)  Chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
echo "============================================================"

# Verify harness binary exists
if [[ ! -f "$HARNESS" ]]; then
  echo "ERROR: Harness binary not found at $HARNESS"
  echo "Build it first: open harness/ in Xcode → Product → Build"
  exit 1
fi

# ── helper: run one benchmark condition ───────────────────────────────────────
run_condition() {
  local model_path="$1"
  local compute_unit="$2"
  local model_name
  model_name=$(basename "$model_path" .mlpackage)
  local output_file="${OUTPUT_DIR}/${model_name}_${compute_unit}_$(date +%s).json"

  echo ""
  echo "--- Running: $model_name | compute_unit=$compute_unit ---"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] $HARNESS --model $model_path --compute-unit $compute_unit ..."
    return 0
  fi

  # Check current thermal state before starting
  local skin_temp
  skin_temp=$(python3 -c "
import subprocess, re
try:
    out = subprocess.check_output(['powermetrics', '-n', '1', '-i', '500',
                                   '--samplers', 'thermal'], text=True, timeout=5)
    m = re.search(r'CPU die temperature: ([\d.]+)', out)
    print(m.group(1) if m else '0')
except Exception:
    print('0')
" 2>/dev/null || echo "0")

  if (( $(echo "$skin_temp >= $THERMAL_CUTOFF" | bc -l) )); then
    echo "  SKIP: Device too hot ($skin_temp °C >= $THERMAL_CUTOFF °C). Waiting 60s..."
    sleep 60
  fi

  # Launch bandwidth capture in background (uses xctrace)
  local trace_file="data/raw/traces/${model_name}_${compute_unit}_$(date +%s).trace"
  bash collection/capture_bandwidth.sh "$trace_file" &
  local trace_pid=$!

  # Run the benchmark
  "$HARNESS" \
    --model "$model_path" \
    --compute-unit "$compute_unit" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --thermal-cutoff "$THERMAL_CUTOFF" \
    --output "$output_file" \
    --verbose

  # Stop bandwidth capture
  kill "$trace_pid" 2>/dev/null || true
  wait "$trace_pid" 2>/dev/null || true

  # Parse bandwidth from trace and inject into the JSON output
  if [[ -f "$output_file" && -f "$trace_file" ]]; then
    python3 collection/parse_xctrace.py \
      --trace "$trace_file" \
      --json "$output_file" \
      --inject \
      2>/dev/null || echo "  [WARN] Bandwidth parse failed for $trace_file"
  fi

  echo "  Written: $output_file"
}

# ── main run logic ─────────────────────────────────────────────────────────────

if [[ "$SWEEP_MODE" == "true" ]]; then
  echo "MODE: Working-set sweep"
  echo "Sweep directory: $SWEEP_DIR"

  # For the sweep, only use cpuAndNeuralEngine (the primary condition)
  # and all (for full ANE+GPU comparison)
  SWEEP_COMPUTE_UNITS=("cpuAndNeuralEngine" "all")
  model_count=0

  for cu in "${SWEEP_COMPUTE_UNITS[@]}"; do
    while IFS= read -r -d '' pkg; do
      run_condition "$pkg" "$cu"
      model_count=$((model_count + 1))
      # Cool-down between models to avoid thermal accumulation
      if [[ "$DRY_RUN" != "true" ]]; then
        echo "  Cooling down (15s)..."
        sleep 15
      fi
    done < <(find "$SWEEP_DIR" -name "*.mlpackage" -print0 | sort -z)
  done

  echo ""
  echo "Sweep complete. Total conditions run: $model_count"

else
  echo "MODE: Full experiment matrix"
  echo "Models directory: $MODELS_DIR"
  echo "Compute units: ${COMPUTE_UNITS[*]}"

  total=0
  while IFS= read -r -d '' pkg; do
    for cu in "${COMPUTE_UNITS[@]}"; do
      run_condition "$pkg" "$cu"
      total=$((total + 1))
      if [[ "$DRY_RUN" != "true" ]]; then
        echo "  Cooling down (20s)..."
        sleep 20
      fi
    done
  done < <(find "$MODELS_DIR" -maxdepth 1 -name "*.mlpackage" -print0 | sort -z)

  echo ""
  echo "Full experiment complete. Total conditions run: $total"
fi

echo ""
echo "============================================================"
echo "Run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Raw data in: $OUTPUT_DIR"
echo "Next step: python analysis/preprocessing.py --input $OUTPUT_DIR"
echo "============================================================"
