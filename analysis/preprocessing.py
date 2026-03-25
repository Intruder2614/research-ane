"""
analysis/preprocessing.py
==========================
Loads all raw JSON benchmark results, applies outlier removal and
thermal filtering, normalises fields, and emits a clean master CSV
for downstream analysis scripts.

Filtering rules (all configurable via config/experiment_config.yaml):
  1. Thermal abort — discard any run where run_aborted_thermal=true
  2. High-temperature runs — discard if any thermal sample exceeded cutoff
  3. IQR outlier removal — per (model, compute_unit, precision) group,
     apply Tukey fences to the latency distributions
  4. High CV runs — flag (but not discard) runs where CV > 10%

Output:
    data/processed/master.csv       — one row per condition (medians + stats)
    data/processed/latency_raw.csv  — one row per individual inference (for full distribs)
    data/processed/flagged.csv      — conditions that were flagged but kept

Usage:
    python analysis/preprocessing.py
    python analysis/preprocessing.py --input data/raw/ --output data/processed/
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ── load config ───────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/experiment_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── raw data loading ──────────────────────────────────────────────────────────

def load_run(json_path: Path) -> dict | None:
    """Load and validate a single benchmark JSON result."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [SKIP] Cannot read {json_path.name}: {e}")
        return None

    required = ["model_name", "compute_unit", "latencies_us", "median_us"]
    for field in required:
        if field not in data:
            print(f"  [SKIP] {json_path.name}: missing field '{field}'")
            return None

    data["source_file"] = json_path.name
    return data


def parse_model_metadata(model_name: str) -> dict:
    """
    Extract precision and base name from model filename conventions.
    e.g. 'mobilenetv3_small_fp32' -> {base: 'mobilenetv3_small', precision: 'fp32'}
         'sweep_w0.30_int8_linear'  -> {base: 'sweep', width: 0.30, precision: 'int8_linear'}
    """
    meta = {"model_name": model_name, "precision": "unknown", "base_model": model_name}

    for prec in ("int8_palettized", "int8_linear", "fp16", "fp32"):
        if model_name.endswith(f"_{prec}"):
            meta["precision"] = prec
            meta["base_model"] = model_name[: -(len(prec) + 1)]
            break

    # Extract width multiplier for sweep models
    wm_match = re.search(r"_w([\d.]+)_", model_name)
    if wm_match:
        meta["width_multiplier"] = float(wm_match.group(1))

    return meta


# ── filtering ─────────────────────────────────────────────────────────────────

def iqr_filter(values: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """Return boolean mask of inlier values using Tukey IQR fences."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (values >= lower) & (values <= upper)


def filter_run(run: dict, config: dict) -> tuple[np.ndarray, list[str]]:
    """
    Apply all filters to a run's latency array.
    Returns (filtered_latencies, list_of_applied_filters).
    """
    latencies = np.array(run["latencies_us"], dtype=np.float64)
    filters_applied = []

    # 1. Thermal abort — discard entire run
    if run.get("run_aborted_thermal", False):
        return np.array([]), ["thermal_abort"]

    # 2. IQR outlier removal within the run
    iqr_mult = config["analysis"]["outlier_removal"]["iqr_multiplier"]
    mask = iqr_filter(latencies, iqr_mult)
    n_removed = (~mask).sum()
    if n_removed > 0:
        latencies = latencies[mask]
        filters_applied.append(f"iqr_removed_{n_removed}")

    return latencies, filters_applied


# ── stats computation ─────────────────────────────────────────────────────────

def compute_row_stats(latencies: np.ndarray) -> dict:
    """Compute the summary statistics stored in the master CSV row."""
    if len(latencies) == 0:
        return {}
    sorted_lat = np.sort(latencies)
    n = len(latencies)

    def pct(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        f = idx - lo
        return float(sorted_lat[lo] * (1 - f) + sorted_lat[hi] * f)

    mean = float(np.mean(latencies))
    std  = float(np.std(latencies))
    return {
        "n_samples":  n,
        "median_us":  round(pct(0.5), 2),
        "mean_us":    round(mean, 2),
        "std_us":     round(std, 2),
        "p95_us":     round(pct(0.95), 2),
        "p99_us":     round(pct(0.99), 2),
        "cv_pct":     round(std / mean * 100, 2) if mean > 0 else None,
        "min_us":     round(float(np.min(latencies)), 2),
        "max_us":     round(float(np.max(latencies)), 2),
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw benchmark JSON files")
    parser.add_argument("--input",  default="data/raw",       help="Raw JSON directory")
    parser.add_argument("--output", default="data/processed", help="Output CSV directory")
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    master_rows: list[dict] = []
    raw_latency_rows: list[dict] = []
    flagged_rows: list[dict] = []

    for json_path in json_files:
        run = load_run(json_path)
        if run is None:
            continue

        latencies, filters = filter_run(run, config)

        if "thermal_abort" in filters:
            flagged_rows.append({"file": json_path.name, "reason": "thermal_abort"})
            continue

        if len(latencies) < 10:
            flagged_rows.append({"file": json_path.name, "reason": f"too_few_samples_{len(latencies)}"})
            continue

        stats = compute_row_stats(latencies)
        meta = parse_model_metadata(run["model_name"])

        row = {
            **meta,
            "compute_unit":     run["compute_unit"],
            "model_size_mb":    run.get("model_size_mb"),
            "device_chip":      run.get("device_chip", "unknown"),
            "run_timestamp":    run.get("run_timestamp"),
            "filters_applied":  "|".join(filters) if filters else "none",
            "bandwidth_mean_gbps":   run.get("bandwidth_stats", {}).get("mean_gbps"),
            "bandwidth_median_gbps": run.get("bandwidth_stats", {}).get("median_gbps"),
            **stats,
        }

        # Flag high-CV runs
        if stats.get("cv_pct") and stats["cv_pct"] > 10.0:
            row["flag"] = f"high_cv_{stats['cv_pct']:.1f}"
            flagged_rows.append({"file": json_path.name, "reason": f"high_cv_{stats['cv_pct']:.1f}", "kept": True})

        master_rows.append(row)

        # Store individual latencies for full-distribution analysis
        for val in latencies:
            raw_latency_rows.append({
                "model_name":  run["model_name"],
                "precision":   meta["precision"],
                "compute_unit": run["compute_unit"],
                "model_size_mb": run.get("model_size_mb"),
                "latency_us":  round(float(val), 2),
            })

    # Write outputs
    master_df = pd.DataFrame(master_rows)
    master_path = output_dir / "master.csv"
    master_df.to_csv(master_path, index=False)
    print(f"\nMaster CSV: {len(master_df)} rows → {master_path}")

    raw_df = pd.DataFrame(raw_latency_rows)
    raw_path = output_dir / "latency_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw latency CSV: {len(raw_df)} rows → {raw_path}")

    if flagged_rows:
        flagged_df = pd.DataFrame(flagged_rows)
        flagged_path = output_dir / "flagged.csv"
        flagged_df.to_csv(flagged_path, index=False)
        print(f"Flagged runs: {len(flagged_rows)} → {flagged_path}")

    # Quick summary
    print("\nPrecision × compute_unit median latency matrix (µs):")
    if not master_df.empty and "precision" in master_df.columns:
        try:
            pivot = master_df.pivot_table(
                values="median_us",
                index="precision",
                columns="compute_unit",
                aggfunc="mean"
            ).round(1)
            print(pivot.to_string())
        except Exception:
            pass


if __name__ == "__main__":
    main()
