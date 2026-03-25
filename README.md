# Cache-Performance Decomposition Analysis
## Quantifying Memory Hierarchy vs Arithmetic Contributions in INT8 Quantized Inference on Apple Neural Engine



---

## Overview

This repository contains the complete research codebase for empirically decomposing
INT8 quantization speedups on Apple's Neural Engine (ANE) into cache-locality and
arithmetic contributions — using behavioural proxy measurements since ANE hardware
performance counters are not publicly exposed.

### Core hypothesis
> When a quantized model's working set fits within the ANE's estimated on-chip SRAM,
> cache residency effects produce a disproportionate speedup relative to the arithmetic
> benefit alone. This breakpoint is detectable via latency-vs-footprint scaling curves
> and shifts proportionally to L2 cache size across Apple Silicon generations.

---

## Repository structure

```
research-ane-cache/
├── config/
│   └── experiment_config.yaml      # All tunable parameters in one place
│
├── models/                         # Phase 1 — model preparation
│   ├── convert_models.py           # FP32 → FP16 → INT8 via coremltools
│   ├── working_set_sweep.py        # Generate 8 model-size variants
│   └── verify_models.py            # Sanity-check all converted models
│
├── harness/ANEBenchmark/           # Phase 2 — Swift measurement harness
│   ├── main.swift                  # CLI entry point
│   ├── BenchmarkRunner.swift       # Core inference + timing loop
│   ├── ModelLoader.swift           # CoreML model loading + compute unit control
│   ├── ThermalMonitor.swift        # IOKit skin-temperature polling
│   └── MetricsLogger.swift         # Structured JSON output writer
│
├── collection/                     # Phase 2 — data collection automation
│   ├── run_experiment.sh           # Master orchestration script
│   ├── capture_bandwidth.sh        # xctrace System Trace wrapper
│   ├── parse_xctrace.py            # Extract GB/s from .trace exports
│   └── pressure_test.py            # Background memory pressure injector
│
├── analysis/                       # Phase 3+4 — statistical pipeline
│   ├── preprocessing.py            # Outlier removal, thermal filtering, normalise
│   ├── piecewise_regression.py     # Two-segment fit with unknown breakpoint
│   ├── chow_breakpoint.py          # Chow structural break test
│   ├── anova_decomposition.py      # Multi-factor ANOVA + partial eta²
│   ├── correlation_analysis.py     # Pearson r: bandwidth reduction vs speedup
│   └── cross_device_comparison.py  # Align breakpoints across M1/M2/M3
│
├── visualization/
│   ├── plot_scaling_curves.py      # Latency vs footprint — the key figure
│   ├── plot_bandwidth_speedup.py   # Scatter: bandwidth reduction vs speedup
│   └── plot_thermal_degradation.py # Sustained inference thermal curves
│
├── notebooks/
│   ├── 01_baseline_exploration.ipynb
│   ├── 02_working_set_analysis.ipynb
│   ├── 03_statistical_tests.ipynb
│   └── 04_final_figures.ipynb
│
├── data/
│   ├── raw/        # .json files from Swift harness (gitignored)
│   ├── processed/  # Cleaned CSVs (gitignored)
│   └── results/    # Final tables and figures
│
├── scripts/
│   ├── setup_env.sh                # One-command environment setup
│   └── full_pipeline.sh            # Run entire analysis end-to-end
│
└── requirements.txt
```

---

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| macOS | 13.0+ Ventura | powermetrics, xctrace CLI |
| Xcode | 15.0+ | Swift harness, Instruments |
| Python | 3.9+ | All analysis scripts |
| coremltools | 7.x | Model conversion |
| Apple Silicon Mac | M1/M2/M3 | ANE access |
| iOS device (optional) | iPhone 12+ | Cross-device validation |

---

## Quick start

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd research-ane-cache
bash scripts/setup_env.sh

# 2. Convert models (requires PyTorch + torchvision)
python models/convert_models.py --output data/models/

# 3. Generate model-size sweep variants
python models/working_set_sweep.py --output data/models/sweep/

# 4. Build the Swift harness in Xcode, then run the master experiment
bash collection/run_experiment.sh --device macOS --iterations 500

# 5. Run the full analysis pipeline
bash scripts/full_pipeline.sh

# 6. Open results
open data/results/
```

---

## Key experimental parameters (see config/experiment_config.yaml)

- **Warm-up iterations**: 200 (discarded before recording)
- **Measurement iterations**: 500 per condition
- **Thermal cutoff**: 45°C skin temperature
- **Model family**: MobileNetV3-Small (primary), MobileNetV3-Large (secondary)
- **Precision variants**: FP32, FP16, INT8 (linear quantization)
- **Compute unit conditions**: `.cpuOnly`, `.cpuAndNeuralEngine`, `.all`
- **Working-set sweep**: 8 sizes from 0.5 MB to 40 MB
- **Pressure sweep**: 0%, 25%, 50%, 65%, 75%, 85% DRAM occupancy

---

## Output artefacts

After running the full pipeline, `data/results/` contains:

- `latency_matrix.csv` — median + P95 + P99 latency per condition
- `bandwidth_matrix.csv` — GB/s per condition from xctrace
- `breakpoint_results.csv` — Chow test p-values and breakpoint MB estimates
- `anova_table.csv` — partial η² per factor
- `correlation_results.csv` — Pearson r subgroup analysis
- `figures/` — all publication-ready plots (PDF + PNG)

---
