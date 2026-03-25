"""
analysis/cross_device_comparison.py
=====================================
Aligns and compares breakpoint locations across Apple Silicon generations
(M1, M2, M3) to test whether the detected cache threshold scales
proportionally to the published L2 cache sizes.

This is the strongest validation of the cache-residency hypothesis:
if breakpoint(M3) / breakpoint(M1) ≈ L2_M3 / L2_M1, then the
behavioural proxy methodology is internally consistent.

Published L2 cache sizes (die analyses + Apple tech briefs):
  M1:  8 MB  (shared L2 per cluster — ANE has dedicated bandwidth)
  M2: 12 MB
  M3: 18 MB  (approximate — Apple does not publish exact ANE SRAM)

Note: these values are researcher estimates, not official Apple specs.
Document this as a limitation in the paper.

Usage:
    python analysis/cross_device_comparison.py \
        --m1 data/results/breakpoints_m1.csv \
        --m2 data/results/breakpoints_m2.csv \
        --m3 data/results/breakpoints_m3.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Published/estimated on-chip SRAM capacities in MB (researcher estimates)
SRAM_ESTIMATES = {"M1": 8.0, "M2": 12.0, "M3": 18.0}


def load_breakpoints(csv_path: str, device_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["device"] = device_label
    df["estimated_sram_mb"] = SRAM_ESTIMATES.get(device_label, float("nan"))
    return df


def compute_scaling_ratio(df_all: pd.DataFrame, reference_device: str = "M1") -> pd.DataFrame:
    """
    For each (precision, compute_unit), compute the breakpoint ratio
    relative to the reference device.
    """
    rows = []
    ref = df_all[df_all["device"] == reference_device]

    for (prec, cu), group in df_all.groupby(["precision", "compute_unit"]):
        ref_bp = ref[
            (ref["precision"] == prec) & (ref["compute_unit"] == cu)
        ]["breakpoint_mb"].values

        if len(ref_bp) == 0 or ref_bp[0] == 0:
            continue
        ref_bp = float(ref_bp[0])

        for _, row in group.iterrows():
            ratio = row["breakpoint_mb"] / ref_bp if ref_bp > 0 else float("nan")
            sram_ratio = (
                row["estimated_sram_mb"] / SRAM_ESTIMATES.get(reference_device, 1)
                if not pd.isna(row["estimated_sram_mb"])
                else float("nan")
            )
            rows.append({
                "precision":         prec,
                "compute_unit":      cu,
                "device":            row["device"],
                "breakpoint_mb":     row["breakpoint_mb"],
                "breakpoint_ratio":  round(ratio, 3),
                "sram_mb":           row["estimated_sram_mb"],
                "sram_ratio":        round(sram_ratio, 3),
                "ratio_matches_sram": abs(ratio - sram_ratio) < 0.25 if not pd.isna(sram_ratio) else None,
            })

    return pd.DataFrame(rows)


def plot_cross_device(df_ratios: pd.DataFrame, output_dir: Path):
    """
    Two-panel plot:
      Left:  Breakpoint MB vs device (grouped bars by precision)
      Right: Breakpoint ratio vs SRAM ratio scatter (should be on y=x if hypothesis holds)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    PREC_COLORS = {
        "fp32": "#185FA5", "fp16": "#1D9E75",
        "int8_linear": "#D85A30", "int8_palettized": "#7F77DD",
    }

    cu = "cpuAndNeuralEngine"
    df_cu = df_ratios[df_ratios["compute_unit"] == cu].copy()

    # ── Left: breakpoint by device ─────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Detected cache breakpoint by device\n(cpuAndNeuralEngine)")
    ax.set_xlabel("Device")
    ax.set_ylabel("Breakpoint location (MB)")

    devices  = sorted(df_cu["device"].unique())
    precs    = [p for p in ["fp32", "fp16", "int8_linear"] if p in df_cu["precision"].unique()]
    n_precs  = len(precs)
    x        = np.arange(len(devices))
    bar_w    = 0.25

    for i, prec in enumerate(precs):
        sub = df_cu[df_cu["precision"] == prec].set_index("device")
        heights = [sub.loc[d, "breakpoint_mb"] if d in sub.index else 0 for d in devices]
        offset  = (i - n_precs / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, heights, bar_w,
                      color=PREC_COLORS.get(prec, "gray"),
                      label=prec.replace("_", " ").upper(),
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        f"{h:.1f}", ha="center", fontsize=7)

    # Overlay SRAM estimate points
    for j, dev in enumerate(devices):
        sram = SRAM_ESTIMATES.get(dev, None)
        if sram:
            ax.scatter(j, sram, marker="D", color="black", s=40, zorder=5,
                       label="Est. SRAM" if j == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(devices)
    ax.legend(frameon=False, fontsize=8)

    # ── Right: ratio scatter ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Breakpoint ratio vs SRAM ratio\n(relative to M1 baseline)")
    ax2.set_xlabel("SRAM size ratio (vs M1)")
    ax2.set_ylabel("Breakpoint ratio (vs M1)")

    # y = x reference line
    lim = df_ratios[["breakpoint_ratio", "sram_ratio"]].max().max() * 1.15
    ax2.plot([0.8, lim], [0.8, lim], "--", color="#888780", linewidth=1, alpha=0.6,
             label="Perfect 1:1 (hypothesis)")

    for prec in precs:
        sub = df_cu[df_cu["precision"] == prec].dropna(subset=["breakpoint_ratio", "sram_ratio"])
        if sub.empty:
            continue
        ax2.scatter(sub["sram_ratio"], sub["breakpoint_ratio"],
                    color=PREC_COLORS.get(prec, "gray"),
                    label=prec.replace("_", " ").upper(),
                    s=80, alpha=0.85, edgecolors="white", linewidths=0.5)
        for _, row in sub.iterrows():
            ax2.annotate(row["device"],
                         xy=(row["sram_ratio"], row["breakpoint_ratio"]),
                         xytext=(4, 3), textcoords="offset points", fontsize=8)

    ax2.legend(frameon=False, fontsize=8)
    ax2.text(0.05, 0.92,
             "Points on the 1:1 line →\ncache hypothesis confirmed",
             transform=ax2.transAxes, fontsize=8, color="#5F5E5A",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1EFE8",
                       edgecolor="#D3D1C7", linewidth=0.5))

    fig.tight_layout(pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = output_dir / f"cross_device_comparison.{fmt}"
        fig.savefig(p)
        print(f"  Saved: {p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cross-device breakpoint comparison")
    parser.add_argument("--m1",         default=None, help="Breakpoint CSV from M1 device")
    parser.add_argument("--m2",         default=None, help="Breakpoint CSV from M2 device")
    parser.add_argument("--m3",         default=None, help="Breakpoint CSV from M3 device")
    parser.add_argument("--combined",   default=None, help="Single CSV with device column")
    parser.add_argument("--output-dir", default="data/results/figures")
    parser.add_argument("--save-csv",   default="data/results/cross_device_ratios.csv")
    args = parser.parse_args()

    frames = []
    if args.combined:
        frames.append(pd.read_csv(args.combined))
    else:
        for label, path in [("M1", args.m1), ("M2", args.m2), ("M3", args.m3)]:
            if path and Path(path).exists():
                frames.append(load_breakpoints(path, label))
            elif path:
                print(f"  [WARN] {label} file not found: {path}")

    if not frames:
        print("No breakpoint data provided.")
        print("Run analysis/piecewise_regression.py for each device first,")
        print("then pass the resulting CSVs to this script.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df_all)} breakpoint records across {df_all['device'].nunique()} devices")

    df_ratios = compute_scaling_ratio(df_all)

    # Print comparison table
    print("\nBreakpoint scaling vs SRAM scaling (reference: M1):")
    cu = "cpuAndNeuralEngine"
    for prec in ["fp32", "fp16", "int8_linear"]:
        sub = df_ratios[(df_ratios["compute_unit"] == cu) & (df_ratios["precision"] == prec)]
        if sub.empty:
            continue
        print(f"\n  {prec}:")
        for _, row in sub.sort_values("device").iterrows():
            match = "OK" if row.get("ratio_matches_sram") else "MISMATCH"
            print(f"    {row['device']}:  bp={row['breakpoint_mb']:.1f}MB  "
                  f"bp_ratio={row['breakpoint_ratio']:.2f}×  "
                  f"sram_ratio={row['sram_ratio']:.2f}×  [{match}]")

    # Save and plot
    Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
    df_ratios.to_csv(args.save_csv, index=False)
    print(f"\nRatio CSV saved to {args.save_csv}")

    plot_cross_device(df_ratios, Path(args.output_dir))


if __name__ == "__main__":
    main()
