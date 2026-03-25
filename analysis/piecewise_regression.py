"""
analysis/piecewise_regression.py / chow_breakpoint.py
=======================================================
The single most important analysis in the research.

Goal: Fit a two-segment piecewise linear regression to the
latency-vs-model-footprint curve for each (precision × compute_unit)
condition. The breakpoint location (in MB) is the estimated effective
cache capacity for that precision.

If cache-residency hypothesis is correct:
  - FP32 breakpoint ≈ estimated on-chip SRAM capacity
  - INT8 breakpoint ≈ 2× FP32 breakpoint
    (because INT8 tensors are ~half the size, so twice as many fit)

Chow structural break test:
  - H0: the slope is uniform across the entire footprint range
  - H1: there is a structural break — two different slopes
  - A significant p-value (< 0.05) supports the cache-effect hypothesis

Usage:
    python analysis/piecewise_regression.py
    python analysis/piecewise_regression.py --input data/processed/master.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize_scalar


# ── piecewise regression ──────────────────────────────────────────────────────

def fit_piecewise(x: np.ndarray, y: np.ndarray, breakpoint: float) -> tuple[np.ndarray, float]:
    """
    Fit a two-segment piecewise linear regression with a fixed breakpoint.
    Returns (residuals, rss).
    """
    mask_left  = x <= breakpoint
    mask_right = x >  breakpoint

    if mask_left.sum() < 2 or mask_right.sum() < 2:
        return np.full_like(y, np.inf, dtype=float), np.inf

    # Segment 1: x ≤ breakpoint
    X1 = np.column_stack([np.ones(mask_left.sum()), x[mask_left]])
    b1, _, _, _ = np.linalg.lstsq(X1, y[mask_left], rcond=None)

    # Segment 2: x > breakpoint
    X2 = np.column_stack([np.ones(mask_right.sum()), x[mask_right]])
    b2, _, _, _ = np.linalg.lstsq(X2, y[mask_right], rcond=None)

    # Residuals
    res1 = y[mask_left]  - (b1[0] + b1[1] * x[mask_left])
    res2 = y[mask_right] - (b2[0] + b2[1] * x[mask_right])
    residuals = np.concatenate([res1, res2])
    rss = float(np.sum(residuals ** 2))

    return residuals, rss


def find_breakpoint(x: np.ndarray, y: np.ndarray, min_frac: float = 0.2) -> dict:
    """
    Grid search + Brent optimisation to find the breakpoint minimising RSS.
    min_frac: each segment must contain at least this fraction of data points.
    """
    n = len(x)
    x_min = np.percentile(x, min_frac * 100)
    x_max = np.percentile(x, (1 - min_frac) * 100)

    # Coarse grid search
    grid = np.linspace(x_min, x_max, 50)
    rss_grid = [fit_piecewise(x, y, bp)[1] for bp in grid]
    best_grid_bp = grid[np.argmin(rss_grid)]

    # Fine optimisation around the best grid point
    search_range = (max(x_min, best_grid_bp - (x_max - x_min) * 0.1),
                    min(x_max, best_grid_bp + (x_max - x_min) * 0.1))
    try:
        result = minimize_scalar(
            lambda bp: fit_piecewise(x, y, bp)[1],
            bounds=search_range,
            method="bounded"
        )
        optimal_bp = result.x
    except Exception:
        optimal_bp = best_grid_bp

    # Fit the final piecewise model
    _, rss_piecewise = fit_piecewise(x, y, optimal_bp)

    # Fit the null model (single slope)
    X_full = np.column_stack([np.ones(n), x])
    b_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    y_pred_full = b_full[0] + b_full[1] * x
    rss_full = float(np.sum((y - y_pred_full) ** 2))
    slope_full = float(b_full[1])

    # Slopes on each segment
    mask_l = x <= optimal_bp
    mask_r = x >  optimal_bp
    Xl = np.column_stack([np.ones(mask_l.sum()), x[mask_l]])
    Xr = np.column_stack([np.ones(mask_r.sum()), x[mask_r]])
    bl, _, _, _ = np.linalg.lstsq(Xl, y[mask_l], rcond=None)
    br, _, _, _ = np.linalg.lstsq(Xr, y[mask_r], rcond=None)

    return {
        "breakpoint_mb":   round(float(optimal_bp), 3),
        "slope_left":      round(float(bl[1]), 4),   # µs per MB before breakpoint
        "slope_right":     round(float(br[1]), 4),   # µs per MB after breakpoint
        "slope_ratio":     round(float(br[1]) / float(bl[1]), 3) if bl[1] != 0 else None,
        "slope_null":      round(slope_full, 4),
        "rss_piecewise":   round(rss_piecewise, 2),
        "rss_null":        round(rss_full, 2),
        "rss_reduction":   round(1 - rss_piecewise / rss_full, 4) if rss_full > 0 else None,
        "n_left":          int(mask_l.sum()),
        "n_right":         int(mask_r.sum()),
    }


# ── Chow structural break test ────────────────────────────────────────────────

def chow_test(
    x: np.ndarray,
    y: np.ndarray,
    breakpoint: float,
) -> dict:
    """
    Chow test for structural break at a given breakpoint.

    H0: no structural break (single regression)
    H1: two separate regression lines

    Test statistic follows F-distribution under H0.
    p < 0.05 → reject H0 → structural break is significant.

    References:
        Chow, G.C. (1960). Tests of Equality Between Sets of Coefficients
        in Two Linear Regressions. Econometrica, 28(3), 591–605.
    """
    n = len(x)
    mask_l = x <= breakpoint
    mask_r = x >  breakpoint
    n1, n2 = mask_l.sum(), mask_r.sum()

    if n1 < 3 or n2 < 3:
        return {"chow_f": None, "chow_p": None, "df1": None, "df2": None,
                "note": "insufficient_samples_in_segment"}

    def rss_segment(xs, ys):
        X = np.column_stack([np.ones(len(xs)), xs])
        b, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
        return float(np.sum((ys - (b[0] + b[1] * xs)) ** 2))

    # Restricted model (single regression)
    X_full = np.column_stack([np.ones(n), x])
    b_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    rss_r = float(np.sum((y - (b_full[0] + b_full[1] * x)) ** 2))

    # Unrestricted model (two separate regressions)
    rss_u = rss_segment(x[mask_l], y[mask_l]) + rss_segment(x[mask_r], y[mask_r])

    # F statistic: ((RSS_R - RSS_U) / k) / (RSS_U / (n - 2k))
    k = 2   # parameters per regression (intercept + slope)
    df1 = k
    df2 = n - 2 * k

    if rss_u < 1e-12 or df2 < 1:
        return {"chow_f": None, "chow_p": None, "df1": df1, "df2": df2,
                "note": "degenerate_residuals"}

    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    return {
        "chow_f":    round(float(f_stat), 4),
        "chow_p":    round(float(p_value), 6),
        "df1":       df1,
        "df2":       df2,
        "rss_r":     round(rss_r, 2),
        "rss_u":     round(rss_u, 2),
        "significant_005": bool(p_value < 0.05),
    }


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse_group(df_group: pd.DataFrame, label: str) -> dict:
    """Run breakpoint + Chow analysis on one (precision, compute_unit) group."""
    df = df_group.dropna(subset=["model_size_mb", "median_us"]).copy()
    df = df.sort_values("model_size_mb")

    x = df["model_size_mb"].values.astype(float)
    y = df["median_us"].values.astype(float)

    if len(x) < 6:
        return {"label": label, "n": len(x), "error": "insufficient_points"}

    bp_result = find_breakpoint(x, y)
    chow_result = chow_test(x, y, bp_result["breakpoint_mb"])

    return {"label": label, "n": len(x), **bp_result, **chow_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Piecewise regression + Chow test")
    parser.add_argument("--input",  default="data/processed/master.csv")
    parser.add_argument("--output", default="data/results/breakpoint_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Filter to sweep models (they have model_size_mb ranging across the sweep)
    sweep_df = df[df["model_name"].str.startswith("sweep_")].copy() if "model_name" in df.columns else df

    if sweep_df.empty:
        print("[WARN] No sweep models found — using all models for breakpoint analysis")
        sweep_df = df.copy()

    results = []
    groups = sweep_df.groupby(["precision", "compute_unit"])

    for (precision, cu), group in groups:
        label = f"{precision}_{cu}"
        print(f"\nAnalysing: {label}  (n={len(group)})")
        result = analyse_group(group, label)
        result["precision"]    = precision
        result["compute_unit"] = cu
        results.append(result)

        if "breakpoint_mb" in result:
            sig = "SIGNIFICANT" if result.get("significant_005") else "not significant"
            print(f"  Breakpoint: {result['breakpoint_mb']:.2f} MB  "
                  f"slope ratio: {result.get('slope_ratio')}  "
                  f"Chow p={result.get('chow_p')}  [{sig}]")

    # Cross-precision breakpoint comparison
    print("\n" + "="*60)
    print("Breakpoint comparison across precisions (cpuAndNeuralEngine):")
    print("Expected: INT8 breakpoint ≈ 2× FP32 breakpoint if cache dominates")
    ane_results = [r for r in results if r.get("compute_unit") == "cpuAndNeuralEngine"]
    fp32_bp = next((r["breakpoint_mb"] for r in ane_results if r.get("precision") == "fp32"), None)
    for r in ane_results:
        bp = r.get("breakpoint_mb")
        ratio = round(bp / fp32_bp, 2) if fp32_bp and bp else "?"
        print(f"  {r.get('precision'):20s}  bp={bp} MB  ratio vs fp32: {ratio}×")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)
    print(f"\nBreakpoint results saved to {args.output}")


if __name__ == "__main__":
    main()
