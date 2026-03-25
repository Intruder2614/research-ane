#!/usr/bin/env bash
# scripts/full_pipeline.sh
# =========================
# Runs the complete statistical analysis pipeline end-to-end.
# Execute this after all benchmark data has been collected.
#
# Usage:
#   bash scripts/full_pipeline.sh
#   bash scripts/full_pipeline.sh --input data/raw/ --skip-plots

set -euo pipefail

INPUT_DIR="data/raw"
PROCESSED_DIR="data/processed"
RESULTS_DIR="data/results"
SKIP_PLOTS=false
PYTHON="${PYTHON:-python3}"

for arg in "$@"; do
  case "$arg" in
    --input)    INPUT_DIR="$2"; shift 2;;
    --skip-plots) SKIP_PLOTS=true; shift;;
  esac
done 2>/dev/null || true

LOG="$RESULTS_DIR/pipeline_run.log"
mkdir -p "$RESULTS_DIR/figures"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "ANE Cache Research — Full Analysis Pipeline"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

run_step() {
  local name="$1"; shift
  echo ""
  echo "── $name ──────────────────────────────────────────────────"
  if "$@"; then
    echo "  [OK] $name"
  else
    echo "  [FAIL] $name exited with code $?"
    exit 1
  fi
}

# Step 1: Preprocessing
run_step "Preprocessing + outlier removal" \
  "$PYTHON" analysis/preprocessing.py \
    --input "$INPUT_DIR" \
    --output "$PROCESSED_DIR"

# Step 2: Piecewise regression + Chow breakpoint test
run_step "Piecewise regression + Chow test" \
  "$PYTHON" analysis/piecewise_regression.py \
    --input "$PROCESSED_DIR/master.csv" \
    --output "$RESULTS_DIR/breakpoint_results.csv"

# Step 3: Multi-factor ANOVA
run_step "ANOVA variance decomposition" \
  "$PYTHON" analysis/anova_decomposition.py \
    --input "$PROCESSED_DIR/master.csv" \
    --output "$RESULTS_DIR/anova_table.csv"

# Step 4: Bandwidth–speedup correlation
run_step "Bandwidth–speedup correlation analysis" \
  "$PYTHON" analysis/correlation_analysis.py \
    --input "$PROCESSED_DIR/master.csv" \
    --output "$RESULTS_DIR/correlation_results.csv"

# Step 5: Visualisations (skippable)
if [[ "$SKIP_PLOTS" == "false" ]]; then
  run_step "Scaling curve plots" \
    "$PYTHON" visualization/plot_scaling_curves.py \
      --input "$PROCESSED_DIR/master.csv" \
      --breakpoints "$RESULTS_DIR/breakpoint_results.csv" \
      --output-dir "$RESULTS_DIR/figures"

  run_step "Bandwidth–speedup scatter" \
    "$PYTHON" visualization/plot_bandwidth_speedup.py \
      --ratios "$RESULTS_DIR/speedup_bw_ratios.csv" \
      --breakpoints "$RESULTS_DIR/breakpoint_results.csv" \
      --output-dir "$RESULTS_DIR/figures"

  run_step "Thermal degradation plots" \
    "$PYTHON" visualization/plot_thermal_degradation.py \
      --input "$INPUT_DIR" \
      --output-dir "$RESULTS_DIR/figures"
fi

echo ""
echo "============================================================"
echo "Pipeline complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Results:"
ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null | awk '{print "  "$NF" ("$5")"}' || true
echo ""
echo "Figures:"
ls -lh "$RESULTS_DIR/figures/"*.png 2>/dev/null | awk '{print "  "$NF}' || true
echo "============================================================"
