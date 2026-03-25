"""
visualization/plot_bandwidth_speedup.py
========================================
Scatter plot: memory bandwidth reduction ratio vs inference speedup ratio.
One point per (model_size, precision) condition.
Colour-codes points by whether the model is estimated to be cache-resident
or cache-evicting relative to the detected breakpoint.

The shape of this scatter tells the story:
  - Points in the top-right (high bandwidth reduction + high speedup)
    = memory-bandwidth-bound regime (cache effect dominates)
  - Points along a flat horizontal line (high bandwidth reduction, low speedup)
    = compute-bound regime (arithmetic effect dominates)

Usage:
    python visualization/plot_bandwidth_speedup.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


CACHE_COLOR    = "#D85A30"   # coral — cache-evicting (large models)
RESIDENT_COLOR = "#185FA5"   # blue  — cache-resident (small models)
FP16_MARKER    = "^"
INT8_MARKER    = "o"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def plot_scatter(ratios_csv: str, breakpoints_csv: str, output_dir: Path, threshold_mb: float = 10.0):
    ratios = pd.read_csv(ratios_csv)

    # Try to get FP32 breakpoint from breakpoints CSV
    bp_fp32 = threshold_mb
    if Path(breakpoints_csv).exists():
        bp_df = pd.read_csv(breakpoints_csv)
        fp32_rows = bp_df[
            (bp_df["precision"] == "fp32") &
            (bp_df["compute_unit"] == "cpuAndNeuralEngine")
        ]
        if not fp32_rows.empty:
            bp_fp32 = float(fp32_rows.iloc[0]["breakpoint_mb"])
    print(f"  Using cache threshold: {bp_fp32:.1f} MB (fp32 breakpoint)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (prec, marker) in enumerate([("fp16", FP16_MARKER), ("int8_linear", INT8_MARKER)]):
        ax = axes[ax_idx]
        speedup_col = f"speedup_{prec}"
        bw_col      = f"bw_ratio_{prec}"

        sub = ratios[[speedup_col, bw_col, "model_size_mb"]].dropna()
        if sub.empty:
            ax.set_title(f"No data for {prec}")
            continue

        is_evicting = sub["model_size_mb"] > bp_fp32
        colors = [CACHE_COLOR if e else RESIDENT_COLOR for e in is_evicting]

        ax.scatter(sub[bw_col], sub[speedup_col],
                   c=colors, marker=marker, s=70, alpha=0.8, linewidths=0.5,
                   edgecolors="white", zorder=3)

        # Annotate size labels
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['model_size_mb']:.0f}MB",
                xy=(row[bw_col], row[speedup_col]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="#5F5E5A",
            )

        # Best-fit line
        if len(sub) >= 4:
            m, b, r, p, _ = st.linregress(sub[bw_col].values, sub[speedup_col].values)
            xs = np.linspace(sub[bw_col].min(), sub[bw_col].max(), 100)
            ax.plot(xs, m * xs + b, "--", color="#888780", linewidth=1.2, alpha=0.7,
                    label=f"r={r:.2f}, p={p:.3f}")
            ax.legend(frameon=False, fontsize=9)

        # Reference diagonal (perfect 1:1 — bandwidth reduction = speedup ratio)
        lim = max(sub[bw_col].max(), sub[speedup_col].max()) * 1.1
        ax.plot([1, lim], [1, lim], ":", color="#B4B2A9", linewidth=1, alpha=0.6)
        ax.text(lim * 0.95, lim * 0.92, "1:1 line", fontsize=7, color="#888780", ha="right")

        prec_label = "FP16" if prec == "fp16" else "INT8"
        ax.set_title(f"Bandwidth reduction vs speedup — {prec_label}")
        ax.set_xlabel("Bandwidth reduction ratio (FP32/prec GB/s)")
        ax.set_ylabel(f"Speedup ratio (FP32/{ prec_label} latency)")

        # Legend for colours
        handles = [
            mpatches.Patch(color=RESIDENT_COLOR, label=f"Cache-resident (≤{bp_fp32:.0f} MB)"),
            mpatches.Patch(color=CACHE_COLOR,    label=f"Cache-evicting  (>{bp_fp32:.0f} MB)"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout(pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = output_dir / f"bandwidth_speedup_scatter.{fmt}"
        fig.savefig(p)
        print(f"  Saved: {p}")
    plt.close(fig)


def main():
    import matplotlib.patches as _mp
    global mpatches
    mpatches = _mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios",      default="data/results/speedup_bw_ratios.csv")
    parser.add_argument("--breakpoints", default="data/results/breakpoint_results.csv")
    parser.add_argument("--output-dir",  default="data/results/figures")
    parser.add_argument("--threshold",   type=float, default=10.0)
    args = parser.parse_args()

    setup_style()
    plot_scatter(args.ratios, args.breakpoints, Path(args.output_dir), args.threshold)


if __name__ == "__main__":
    main()
