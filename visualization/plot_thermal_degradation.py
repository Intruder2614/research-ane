"""
visualization/plot_thermal_degradation.py
==========================================
Plots sustained-inference latency over time alongside skin temperature.
Reveals how ANE throttling degrades performance during extended runs,
and whether INT8 is more thermally stable than FP32 due to lower power draw.

Also generates a power-vs-precision bar chart (mJ per inference).

Usage:
    python visualization/plot_thermal_degradation.py
    python visualization/plot_thermal_degradation.py --input data/raw/sustained/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


PRECISION_COLORS = {
    "fp32": "#185FA5", "fp16": "#1D9E75",
    "int8_linear": "#D85A30", "int8_palettized": "#7F77DD",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def load_sustained_runs(input_dir: str) -> list[dict]:
    """Load all JSON benchmark files from a directory."""
    runs = []
    for path in Path(input_dir).rglob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            if "latencies_us" in data and len(data["latencies_us"]) > 50:
                runs.append(data)
        except Exception:
            continue
    return runs


def plot_thermal_degradation(runs: list[dict], output_dir: Path):
    """
    For each run, plot latency over iteration number (proxy for time)
    alongside thermal samples. A rising latency curve indicates throttling.
    """
    if not runs:
        print("  No sustained-run data found — skipping thermal plot")
        return

    # Group by precision
    by_precision: dict[str, list] = {}
    for run in runs:
        prec = None
        name = run.get("model_name", "")
        for p in ("fp32", "fp16", "int8_linear", "int8_palettized"):
            if p in name:
                prec = p
                break
        if prec is None:
            continue
        by_precision.setdefault(prec, []).append(run)

    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_lat    = fig.add_subplot(gs[0, :])   # top: latency over time
    ax_temp   = fig.add_subplot(gs[1, 0])   # bottom-left: temperature
    ax_energy = fig.add_subplot(gs[1, 1])   # bottom-right: energy per inference

    # ── Latency over iterations ──────────────────────────────────────────────
    ax_lat.set_title("Latency over sustained run — thermal throttling visible as rising curve")
    ax_lat.set_xlabel("Inference iteration")
    ax_lat.set_ylabel("Latency (µs)")

    for prec, run_list in sorted(by_precision.items()):
        # Use the run with most iterations
        run = max(run_list, key=lambda r: len(r["latencies_us"]))
        lats = np.array(run["latencies_us"])

        # Smooth with rolling median (window=20) for readability
        window = min(20, len(lats) // 5)
        smoothed = pd.Series(lats).rolling(window, center=True).median().values

        x = np.arange(len(smoothed))
        color = PRECISION_COLORS.get(prec, "gray")
        ax_lat.plot(x, smoothed, color=color,
                    label=prec.replace("_", " ").upper(), linewidth=1.8, alpha=0.9)
        ax_lat.fill_between(x, smoothed * 0.97, smoothed * 1.03, alpha=0.08, color=color)

    ax_lat.legend(frameon=False, fontsize=9)

    # ── Temperature over time ────────────────────────────────────────────────
    ax_temp.set_title("Skin temperature during run")
    ax_temp.set_xlabel("Time (s from run start)")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.axhline(45, color="#E24B4A", linestyle="--", linewidth=1,
                    alpha=0.7, label="Thermal cutoff (45°C)")

    for prec, run_list in sorted(by_precision.items()):
        run = max(run_list, key=lambda r: len(r["latencies_us"]))
        samples = run.get("thermal_samples", [])
        if not samples:
            continue
        t0 = samples[0]["timestamp_s"]
        ts = [s["timestamp_s"] - t0 for s in samples]
        temps = [s["celsius"] for s in samples]
        ax_temp.plot(ts, temps, color=PRECISION_COLORS.get(prec, "gray"),
                     label=prec.replace("_", " ").upper(), linewidth=1.5)

    ax_temp.legend(frameon=False, fontsize=8)

    # ── Energy per inference ─────────────────────────────────────────────────
    ax_energy.set_title("Energy per inference (mJ) by precision")
    ax_energy.set_xlabel("Precision")
    ax_energy.set_ylabel("Energy (mJ)")

    energy_data = {}
    for prec, run_list in sorted(by_precision.items()):
        mJ_values = []
        for run in run_list:
            # energy_mj may be injected by the harness if powermetrics was running
            if "energy_mj" in run:
                mJ_values.append(run["energy_mj"])
        if mJ_values:
            energy_data[prec] = np.mean(mJ_values)

    if energy_data:
        labels = list(energy_data.keys())
        values = [energy_data[k] for k in labels]
        colors = [PRECISION_COLORS.get(k, "gray") for k in labels]
        bars = ax_energy.bar(
            [l.replace("_", "\n") for l in labels], values,
            color=colors, alpha=0.85, width=0.5, edgecolor="white", linewidth=0.5
        )
        for bar, val in zip(bars, values):
            ax_energy.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.02 * max(values),
                           f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        ax_energy.text(0.5, 0.5, "No energy data\n(requires powermetrics during run)",
                       transform=ax_energy.transAxes, ha="center", va="center",
                       fontsize=9, color="#888780")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = output_dir / f"thermal_degradation.{fmt}"
        fig.savefig(p)
        print(f"  Saved: {p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/raw")
    parser.add_argument("--output-dir", default="data/results/figures")
    args = parser.parse_args()

    setup_style()
    runs = load_sustained_runs(args.input)
    print(f"Loaded {len(runs)} sustained runs")
    plot_thermal_degradation(runs, Path(args.output_dir))


if __name__ == "__main__":
    main()
