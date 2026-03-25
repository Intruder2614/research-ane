"""
analysis/correlation_analysis.py
==================================
Pearson correlation between memory bandwidth reduction (FP32→INT8) and
inference speedup, analysed separately for:
  - cache-resident subgroup (small models, footprint < threshold_mb)
  - cache-evicting subgroup (large models, footprint > threshold_mb)

The key insight: if bandwidth reduction and speedup are highly correlated
for LARGE models (which already strain the cache/DRAM interface) but
weakly correlated for SMALL models (which fit in cache regardless of
precision), it provides indirect evidence that memory bandwidth is the
performance-limiting factor — i.e., cache effects dominate over arithmetic.

Usage:
    python analysis/correlation_analysis.py
    python analysis/correlation_analysis.py --input data/processed/master.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats


# ── compute speedup and bandwidth reduction ratios ────────────────────────────

def compute_ratios(df: pd.DataFrame, compute_unit: str = "cpuAndNeuralEngine") -> pd.DataFrame:
    """
    For each (base_model, model_size_mb) pair, compute:
      - speedup_fp16_vs_fp32   = median_fp32 / median_fp16
      - speedup_int8_vs_fp32   = median_fp32 / median_int8
      - bw_reduction_fp16_vs_fp32  = bw_fp32 / bw_fp16  (lower = less bandwidth used)
      - bw_reduction_int8_vs_fp32  = bw_fp32 / bw_int8
    """
    df_cu = df[df["compute_unit"] == compute_unit].copy()

    # Pivot so each row has FP32, FP16, INT8 columns
    pivot_latency = df_cu.pivot_table(
        index=["base_model", "model_size_mb"],
        columns="precision",
        values="median_us",
        aggfunc="mean",
    )
    pivot_bw = df_cu.pivot_table(
        index=["base_model", "model_size_mb"],
        columns="precision",
        values="bandwidth_mean_gbps",
        aggfunc="mean",
    )

    rows = []
    for idx in pivot_latency.index:
        lat = pivot_latency.loc[idx]
        bw  = pivot_bw.loc[idx] if idx in pivot_bw.index else pd.Series(dtype=float)

        fp32_lat = lat.get("fp32")
        fp32_bw  = bw.get("fp32") if len(bw) else None

        if fp32_lat is None or fp32_lat == 0:
            continue

        row = {
            "base_model":   idx[0],
            "model_size_mb": idx[1],
            "lat_fp32":     fp32_lat,
        }

        for prec in ("fp16", "int8_linear"):
            prec_lat = lat.get(prec)
            prec_bw  = bw.get(prec) if len(bw) else None

            if prec_lat is not None and prec_lat > 0:
                row[f"speedup_{prec}"]   = round(fp32_lat / prec_lat, 4)
            if prec_bw is not None and prec_bw > 0 and fp32_bw is not None:
                row[f"bw_ratio_{prec}"] = round(fp32_bw / prec_bw, 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ── Pearson correlation with subgroup analysis ────────────────────────────────

def pearson_subgroup(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    cache_threshold_mb: float,
) -> dict:
    """
    Compute Pearson r for:
      - full dataset
      - small-model subgroup (likely cache-resident)
      - large-model subgroup (likely cache-evicting)
    """
    df_clean = df[[x_col, y_col, size_col]].dropna()
    if len(df_clean) < 4:
        return {"error": "insufficient_data"}

    def pearson(subset):
        if len(subset) < 4:
            return {"r": None, "p": None, "n": len(subset)}
        r, p = stats.pearsonr(subset[x_col], subset[y_col])
        return {"r": round(float(r), 4), "p": round(float(p), 6), "n": len(subset)}

    full = pearson(df_clean)
    small = pearson(df_clean[df_clean[size_col] <= cache_threshold_mb])
    large = pearson(df_clean[df_clean[size_col] >  cache_threshold_mb])

    return {
        "x_col": x_col,
        "y_col": y_col,
        "cache_threshold_mb": cache_threshold_mb,
        "full": full,
        "cache_resident": small,
        "cache_evicting": large,
    }


# ── cross-device alignment ────────────────────────────────────────────────────

def align_devices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise latency by device baseline so results from M1 and M3
    can be plotted together. Normalisation: divide each run's latency
    by that device's mean FP32 latency on the smallest model.
    """
    df = df.copy()
    if "device_chip" not in df.columns:
        return df

    for chip, group in df.groupby("device_chip"):
        baseline = group[
            (group["precision"] == "fp32") &
            (group["model_size_mb"] == group["model_size_mb"].min())
        ]["median_us"].mean()

        if baseline > 0:
            df.loc[df["device_chip"] == chip, "median_us_normalised"] = (
                df.loc[df["device_chip"] == chip, "median_us"] / baseline
            )

    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bandwidth–speedup correlation analysis")
    parser.add_argument("--input",  default="data/processed/master.csv")
    parser.add_argument("--output", default="data/results/correlation_results.csv")
    parser.add_argument("--cache-threshold-mb", type=float, default=10.0,
                        help="Model footprint threshold separating resident from evicting subgroups")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Compute speedup and bandwidth reduction ratios
    ratios = compute_ratios(df)
    print(f"Computed {len(ratios)} model-condition pairs with ratio data")

    if ratios.empty:
        print("No ratio data — check that bandwidth_mean_gbps is present in master.csv")
        print("Run collection/parse_xctrace.py --inject to add bandwidth to JSON files")
        return

    all_results = []

    for prec in ("fp16", "int8_linear"):
        speedup_col = f"speedup_{prec}"
        bw_col      = f"bw_ratio_{prec}"

        if speedup_col not in ratios.columns:
            continue

        print(f"\n{'='*60}")
        print(f"Correlation: bandwidth reduction vs speedup [{prec}]")

        result = pearson_subgroup(
            ratios, bw_col, speedup_col, "model_size_mb",
            args.cache_threshold_mb
        )
        result["precision"] = prec
        all_results.append(result)

        if "error" not in result:
            print(f"  Full dataset:       r={result['full']['r']}  "
                  f"p={result['full']['p']}  n={result['full']['n']}")
            print(f"  Cache-resident (≤{args.cache_threshold_mb}MB):  "
                  f"r={result['cache_resident']['r']}  p={result['cache_resident']['p']}")
            print(f"  Cache-evicting (>{args.cache_threshold_mb}MB):  "
                  f"r={result['cache_evicting']['r']}  p={result['cache_evicting']['p']}")

    # Interpretation guide
    print("\n" + "="*60)
    print("Interpretation guide:")
    print("  If r(cache_evicting) >> r(cache_resident) for INT8 vs FP32:")
    print("    → Large models are memory-bandwidth bound → cache hypothesis supported")
    print("  If r values are similar across subgroups:")
    print("    → Arithmetic effect is uniform → conventional narrative holds")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.output, index=False)
    print(f"\nCorrelation results saved to {args.output}")

    # Also save the ratios dataframe for use by visualisation scripts
    ratios_path = Path(args.output).parent / "speedup_bw_ratios.csv"
    ratios.to_csv(ratios_path, index=False)
    print(f"Ratio data saved to {ratios_path}")


if __name__ == "__main__":
    main()
