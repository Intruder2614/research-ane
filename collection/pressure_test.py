"""
collection/pressure_test.py
============================
Injects controlled memory pressure by allocating and locking a configurable
amount of physical RAM, then triggers inference runs at each pressure level.

This is Experiment 2 from the research plan — it tests whether inference
latency degrades non-linearly as available memory shrinks, which would
provide behavioural evidence for cache/memory-hierarchy effects.

If the cache-residency hypothesis is correct:
  - Small models (fitting in estimated on-chip SRAM) should show minimal
    latency increase even under heavy DRAM pressure.
  - Large models (already causing DRAM accesses) should show proportional
    latency increases with pressure.

Usage:
    python collection/pressure_test.py \
        --model data/models/mobilenetv3_small_fp32.mlpackage \
        --output data/raw/pressure_test/ \
        --pressures 0.0 0.25 0.40 0.55 0.65 0.75

    python collection/pressure_test.py --sweep-models data/models/sweep/
"""

import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ── memory allocation helpers ─────────────────────────────────────────────────

def get_total_ram_mb() -> float:
    """Get total physical RAM in MB using sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) / (1024 ** 2)
    except Exception:
        return 8192.0  # Default fallback: 8 GB


def allocate_memory_mb(target_mb: float) -> bytearray:
    """
    Allocate and touch `target_mb` MB of RAM to force physical memory use.
    Returns the bytearray — caller MUST hold a reference to keep it allocated.
    Touching every page (every 4096 bytes) is required to actually commit
    physical pages (not just virtual address space).
    """
    n_bytes = int(target_mb * 1024 * 1024)
    buf = bytearray(n_bytes)
    # Touch every page to force physical allocation (prevent lazy paging)
    page_size = 4096
    for i in range(0, n_bytes, page_size):
        buf[i] = 0xFF
    return buf


# ── benchmark invocation ──────────────────────────────────────────────────────

def run_benchmark(
    model_path: str,
    compute_unit: str,
    output_path: str,
    harness_binary: str,
    iterations: int = 300,
    warmup: int = 100,
) -> dict | None:
    """
    Invoke the Swift harness for a single benchmark run.
    Returns the parsed JSON result dict or None on failure.
    """
    cmd = [
        harness_binary,
        "--model", model_path,
        "--compute-unit", compute_unit,
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--output", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  [ERROR] Harness exited with code {result.returncode}", file=sys.stderr)
        print(f"  {result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        with open(output_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [ERROR] Cannot read output JSON: {e}", file=sys.stderr)
        return None


# ── pressure sweep ────────────────────────────────────────────────────────────

def run_pressure_sweep(
    model_path: str,
    pressure_fractions: list[float],
    output_dir: Path,
    compute_unit: str,
    harness_binary: str,
    stabilise_wait_sec: float = 3.0,
) -> list[dict]:
    """
    Run the benchmark at each pressure level, building up pressure
    incrementally. Returns a list of result dicts with pressure metadata.
    """
    total_ram_mb = get_total_ram_mb()
    model_name = Path(model_path).stem
    results = []
    allocated_buffers: list[bytearray] = []  # hold refs to keep memory allocated

    print(f"\nTotal RAM: {total_ram_mb:.0f} MB")
    print(f"Model: {model_name}  |  Compute unit: {compute_unit}")
    print(f"Pressure levels: {[f'{p*100:.0f}%' for p in pressure_fractions]}")

    current_allocated_mb = 0.0

    for frac in sorted(pressure_fractions):
        target_mb = total_ram_mb * frac
        extra_mb = target_mb - current_allocated_mb

        if extra_mb > 100:
            print(f"\n  Allocating {extra_mb:.0f} MB to reach {frac*100:.0f}% occupancy...")
            try:
                buf = allocate_memory_mb(extra_mb)
                allocated_buffers.append(buf)
                current_allocated_mb = target_mb
            except MemoryError:
                print(f"  [WARN] Cannot allocate {extra_mb:.0f} MB — skipping this pressure level")
                continue
        elif extra_mb < 0:
            # Decreasing pressure — release some buffers (not typical in our protocol)
            print(f"  Releasing memory to reach {frac*100:.0f}% occupancy...")
            while current_allocated_mb > target_mb and allocated_buffers:
                buf = allocated_buffers.pop()
                release_mb = len(buf) / (1024 ** 2)
                del buf
                current_allocated_mb -= release_mb
            gc.collect()

        print(f"\n  Pressure: {frac*100:.0f}%  ({current_allocated_mb:.0f}/{total_ram_mb:.0f} MB allocated)")
        print(f"  Waiting {stabilise_wait_sec}s for OS to settle...")
        time.sleep(stabilise_wait_sec)

        output_file = output_dir / f"{model_name}_{compute_unit}_pressure_{int(frac*100):03d}.json"
        result = run_benchmark(
            model_path=model_path,
            compute_unit=compute_unit,
            output_path=str(output_file),
            harness_binary=harness_binary,
        )

        if result:
            result["pressure_fraction"] = frac
            result["allocated_mb"] = round(current_allocated_mb, 1)
            result["total_ram_mb"] = round(total_ram_mb, 1)
            result["available_ram_mb"] = round(total_ram_mb - current_allocated_mb, 1)
            results.append(result)
            print(f"  Median latency: {result.get('median_us', '?'):.1f} µs")

    # Release all allocated memory
    print("\nReleasing pressure allocations...")
    del allocated_buffers
    gc.collect()
    time.sleep(2.0)

    return results


# ── main ──────────────────────────────────────────────────────────────────────

DEFAULT_PRESSURES = [0.0, 0.25, 0.40, 0.55, 0.65, 0.75, 0.85]
DEFAULT_HARNESS = "./harness/.build/release/ANEBenchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory pressure sweep for ANE cache research")
    parser.add_argument("--model", help="Single .mlpackage path")
    parser.add_argument("--sweep-models", help="Directory with .mlpackage files to sweep")
    parser.add_argument("--output", default="data/raw/pressure", help="Output directory")
    parser.add_argument("--pressures", nargs="+", type=float, default=DEFAULT_PRESSURES)
    parser.add_argument("--compute-unit", default="cpuAndNeuralEngine")
    parser.add_argument("--harness", default=DEFAULT_HARNESS)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.harness).exists():
        print(f"Harness binary not found: {args.harness}", file=sys.stderr)
        print("Build it first in Xcode: Product > Build", file=sys.stderr)
        sys.exit(1)

    model_paths = []
    if args.model:
        model_paths.append(args.model)
    elif args.sweep_models:
        model_paths = sorted(Path(args.sweep_models).glob("*.mlpackage"),
                             key=lambda p: p.stat().st_size)  # sort by size ascending
    else:
        print("Provide --model or --sweep-models", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for model_path in model_paths:
        results = run_pressure_sweep(
            model_path=str(model_path),
            pressure_fractions=args.pressures,
            output_dir=output_dir,
            compute_unit=args.compute_unit,
            harness_binary=args.harness,
        )
        all_results.extend(results)

    # Save combined summary
    summary_path = output_dir / "pressure_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPressure summary written to {summary_path}")


if __name__ == "__main__":
    main()
