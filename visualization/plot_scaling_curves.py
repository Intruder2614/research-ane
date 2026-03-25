"""
visualization/plot_scaling_curves.py
=====================================
Generates the PRIMARY FIGURE of the research:
  Latency (µs) vs Model Footprint (MB) for FP32, FP16, INT8
  with vertical lines showing detected cache breakpoints.

This is the plot that makes or breaks the cache-residency hypothesis.
If INT8's breakpoint is shifted ~2× rightward relative to FP32's,
and the post-breakpoint slope steepens significantly, the hypothesis
is visually and statistically confirmed.

Output: data/results/figures/scaling_curves.pdf + .png

Usage:
    python visualization/plot_scaling_curves.py
    python visualization/plot_scaling_curves.py --compute-unit cpuAndNeuralEngine
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── style ─────────────────────────────────────────────────────────────────────

PRECISION_COLORS = {
    "fp32":           "#185FA5",   # blue
    "fp16":           "#1D9E75",   # teal
    "int8_linear":    "#D85A30",   # coral
    "int8_palettized": "#7F77DD",  # purple
}

PRECISION_LABELS = {
    "fp32":           "FP32",
    "fp16":           "FP16",
    "int8_linear":    "INT8 (linear)",
    "int8_palettized": "INT8 (palettized)",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(master_csv: str, breakpoints_csv: str | None) -> tuple:
    df = pd.read_csv(master_csv)
    df = df.dropna(subset=["model_size_mb", "median_us"])

    # Filter to sweep models
    if "model_name" in df.columns:
        sweep = df[df["model_name"].str.contains("sweep_", na=False)]
        if not sweep.empty:
            df = sweep

    bp_df = None
    if breakpoints_csv and Path(breakpoints_csv).exists():
        bp_df = pd.read_csv(breakpoints_csv)

    return df, bp_df


# ── plot functions ────────────────────────────────────────────────────────────

def plot_scaling_curves(
    df: pd.DataFrame,
    bp_df: pd.DataFrame | None,
    compute_unit: str,
    output_dir: Path,
) -> None:
    """Main scaling curve plot — one line per precision variant."""
    df_cu = df[df["compute_unit"] == compute_unit].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: raw latency ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_title(f"Inference latency vs model footprint\n({compute_unit})")
    ax.set_xlabel("Model footprint (MB)")
    ax.set_ylabel("Median inference latency (µs)")

    precisions_present = df_cu["precision"].dropna().unique()
    for prec in ["fp32", "fp16", "int8_linear", "int8_palettized"]:
        if prec not in precisions_present:
            continue
        sub = df_cu[df_cu["precision"] == prec].sort_values("model_size_mb")
        x = sub["model_size_mb"].values
        y = sub["median_us"].values
        p95 = sub.get("p95_us", pd.Series(dtype=float)).values if "p95_us" in sub else None

        color = PRECISION_COLORS.get(prec, "gray")
        label = PRECISION_LABELS.get(prec, prec)

        ax.plot(x, y, "o-", color=color, label=label, linewidth=2, markersize=5)
        if p95 is not None and len(p95) == len(y):
            ax.fill_between(x, y, p95, alpha=0.12, color=color)

        # Annotate breakpoint vertical line
        if bp_df is not None:
            bp_rows = bp_df[(bp_df["precision"] == prec) & (bp_df["compute_unit"] == compute_unit)]
            if not bp_rows.empty:
                bp_x = bp_rows.iloc[0]["breakpoint_mb"]
                ax.axvline(bp_x, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
                ax.text(bp_x + 0.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1000,
                        f"bp={bp_x:.1f}", color=color, fontsize=8, va="top")

    ax.legend(frameon=False, fontsize=9)

    # ── Right panel: speedup relative to FP32 ──────────────────────────────
    ax2 = axes[1]
    ax2.set_title(f"Speedup vs FP32 by footprint\n({compute_unit})")
    ax2.set_xlabel("Model footprint (MB)")
    ax2.set_ylabel("Speedup vs FP32 (×)")
    ax2.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax2.text(0.5, 1.02, "baseline (FP32)", transform=ax2.transAxes + plt.matplotlib.transforms.ScaledTranslation(0, 0.05, fig.dpi_scale_trans),
             ha="center", fontsize=8, color="gray") if False else None

    # Compute speedup per size
    fp32_lats = df_cu[df_cu["precision"] == "fp32"].set_index("model_size_mb")["median_us"]

    for prec in ["fp16", "int8_linear"]:
        if prec not in precisions_present:
            continue
        sub = df_cu[df_cu["precision"] == prec].sort_values("model_size_mb")
        speedups, sizes = [], []
        for _, row in sub.iterrows():
            fp32_lat = fp32_lats.get(row["model_size_mb"])
            if fp32_lat and fp32_lat > 0 and row["median_us"] > 0:
                speedups.append(fp32_lat / row["median_us"])
                sizes.append(row["model_size_mb"])

        if sizes:
            color = PRECISION_COLORS.get(prec, "gray")
            ax2.plot(sizes, speedups, "s-", color=color,
                     label=PRECISION_LABELS.get(prec, prec), linewidth=2, markersize=5)

    ax2.legend(frameon=False, fontsize=9)

    # Annotation box explaining the hypothesis
    ax2.text(0.97, 0.05,
             "Cache hypothesis:\nSpeedup should\nincrease post-\nbreakpoint (↑)",
             transform=ax2.transAxes,
             ha="right", va="bottom",
             fontsize=8, color="#444441",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1EFE8", edgecolor="#D3D1C7", linewidth=0.5))

    fig.tight_layout(pad=2.0)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = output_dir / f"scaling_curves_{compute_unit}.{fmt}"
        fig.savefig(out)
        print(f"  Saved: {out}")

    plt.close(fig)


def plot_all_compute_units_overlay(
    df: pd.DataFrame,
    output_dir: Path,
    precision: str = "int8_linear",
) -> None:
    """Overlay all compute unit conditions for one precision — shows ANE benefit clearly."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Compute unit comparison — {PRECISION_LABELS.get(precision, precision)}")
    ax.set_xlabel("Model footprint (MB)")
    ax.set_ylabel("Median latency (µs)")

    cu_colors = {"cpuOnly": "#888780", "cpuAndNeuralEngine": "#185FA5", "all": "#1D9E75"}

    df_prec = df[df["precision"] == precision].copy()
    for cu, group in df_prec.groupby("compute_unit"):
        sub = group.sort_values("model_size_mb")
        ax.plot(sub["model_size_mb"], sub["median_us"], "o-",
                color=cu_colors.get(cu, "gray"), label=cu, linewidth=2, markersize=4)

    ax.legend(frameon=False, title="Compute unit")
    fig.tight_layout()

    out = output_dir / f"compute_unit_comparison_{precision}.png"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latency vs footprint scaling curves")
    parser.add_argument("--input",       default="data/processed/master.csv")
    parser.add_argument("--breakpoints", default="data/results/breakpoint_results.csv")
    parser.add_argument("--output-dir",  default="data/results/figures")
    parser.add_argument("--compute-unit", default="cpuAndNeuralEngine")
    args = parser.parse_args()

    setup_style()
    output_dir = Path(args.output_dir)
    df, bp_df = load_data(args.input, args.breakpoints)

    if df.empty:
        print("No data found. Run the benchmark pipeline first.")
        return

    # Main figure per compute unit
    for cu in ["cpuOnly", "cpuAndNeuralEngine", "all"]:
        if cu in df["compute_unit"].values:
            plot_scaling_curves(df, bp_df, cu, output_dir)

    # Compute unit overlay
    for prec in ["int8_linear", "fp32"]:
        if prec in df.get("precision", pd.Series()).values:
            plot_all_compute_units_overlay(df, output_dir, prec)

    print(f"\nAll figures written to {output_dir}")


if __name__ == "__main__":
    main()
