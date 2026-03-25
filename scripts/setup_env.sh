#!/usr/bin/env bash
# scripts/setup_env.sh
# =====================
# One-command environment setup for the research project.
# Creates a Python virtual environment and installs all dependencies.
#
# Usage:
#   bash scripts/setup_env.sh
#   bash scripts/setup_env.sh --no-torch   (skip heavy PyTorch install — analysis only)

set -euo pipefail

NO_TORCH=false
for arg in "$@"; do
  [[ "$arg" == "--no-torch" ]] && NO_TORCH=true
done

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo "ANE Cache Research — Environment Setup"
echo "============================================================"

# Check Python version
PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON_VERSION"
if [[ "$(echo "$PYTHON_VERSION >= 3.9" | bc -l 2>/dev/null || echo 1)" -ne 1 ]]; then
  echo "Python 3.9+ required. Current: $PYTHON_VERSION"
  exit 1
fi

# Create virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies
if [[ "$NO_TORCH" == "true" ]]; then
  echo "Installing analysis dependencies (no PyTorch)..."
  pip install numpy pandas scipy pyyaml pingouin statsmodels matplotlib seaborn \
              jupyter ipykernel tqdm rich --quiet
  pip install coremltools --quiet || echo "  [WARN] coremltools install failed (macOS with Xcode required)"
else
  echo "Installing all dependencies (including PyTorch — this may take a few minutes)..."
  pip install -r requirements.txt --quiet
fi

# Create data directory structure
echo "Creating data directory structure..."
mkdir -p data/{raw,processed,results/figures} data/models/{sweep} data/raw/traces

# Create .gitignore for data directories
cat > data/.gitignore << 'EOF'
# Raw benchmark outputs (large binary/JSON files)
raw/*.json
raw/traces/
# Processed CSVs (regenerated from raw data)
processed/
# Results (regenerate from scripts — commit only final figures)
results/*.csv
EOF

echo ""
echo "============================================================"
echo "Setup complete."
echo ""
echo "Activate the environment:  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Convert models:   python models/convert_models.py --output data/models/"
echo "  2. Build Swift harness in Xcode (harness/ANEBenchmark/)"
echo "  3. Run experiments:  bash collection/run_experiment.sh"
echo "  4. Analyse results:  bash scripts/full_pipeline.sh"
echo "============================================================"
