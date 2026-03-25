"""
collection/parse_xctrace.py
============================
Parses an xctrace .trace file exported as XML/JSON to extract
memory bandwidth readings (GB/s) and optionally injects them
into an existing benchmark result JSON.

xctrace export command:
    xctrace export --input run.trace --xpath '/trace-toc/run/data/table[@schema="mem-bandwidth"]' --output - > bandwidth.xml

The exported XML has this structure:
    <trace-toc>
      <run number="1">
        <data>
          <table schema="mem-bandwidth">
            <col id="start" .../>
            <col id="duration" .../>
            <col id="bandwidth" .../>
            <row>
              <td id="start">1000000000</td>   (nanoseconds from trace start)
              <td id="duration">1000000</td>
              <td id="bandwidth">12.4</td>       (GB/s)
            </row>
            ...
          </table>
        </data>
      </run>
    </trace-toc>

Usage:
    # Export the trace first, then parse:
    xctrace export --input run.trace \
        --xpath '/trace-toc/run/data/table[@schema="mem-bandwidth"]' \
        --output bandwidth.xml
    python collection/parse_xctrace.py --xml bandwidth.xml --output bandwidth.csv

    # Or inject bandwidth stats directly into a benchmark JSON:
    python collection/parse_xctrace.py --trace run.trace --json run.json --inject
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


# ── xctrace XML export ────────────────────────────────────────────────────────

def export_bandwidth_xml(trace_path: str) -> str | None:
    """
    Export the mem-bandwidth table from a .trace file as XML.
    Returns the XML string or None on failure.
    """
    cmd = [
        "xctrace", "export",
        "--input", trace_path,
        "--xpath", "/trace-toc/run/data/table[@schema='mem-bandwidth']",
        "--output", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  [WARN] xctrace export failed: {result.stderr.strip()}", file=sys.stderr)
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        print("  [WARN] xctrace export timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("  [WARN] xctrace not found — is Xcode installed?", file=sys.stderr)
        return None


# ── XML parsing ───────────────────────────────────────────────────────────────

def parse_bandwidth_xml(xml_string: str) -> list[dict]:
    """
    Parse the exported XML and return a list of bandwidth samples:
    [{"timestamp_ns": int, "duration_ns": int, "bandwidth_gbps": float}, ...]
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"  [WARN] XML parse error: {e}", file=sys.stderr)
        return []

    # Find column indices
    table = root.find(".//table[@schema='mem-bandwidth']")
    if table is None:
        # Try without schema if the trace uses a different export format
        table = root.find(".//table")
    if table is None:
        print("  [WARN] No mem-bandwidth table found in XML", file=sys.stderr)
        return []

    col_index = {}
    for col in table.findall("col"):
        col_id = col.get("id", "")
        col_index[col_id] = len(col_index)

    # Parse rows
    samples = []
    for row in table.findall("row"):
        cells = [td.text or "" for td in row.findall("td")]
        try:
            sample = {
                "timestamp_ns": int(cells[col_index.get("start", 0)]),
                "duration_ns":  int(cells[col_index.get("duration", 1)]),
                "bandwidth_gbps": float(cells[col_index.get("bandwidth", 2)]),
            }
            samples.append(sample)
        except (IndexError, ValueError):
            continue

    return samples


# ── statistics ────────────────────────────────────────────────────────────────

def bandwidth_stats(samples: list[dict]) -> dict:
    """Compute summary statistics from bandwidth sample list."""
    if not samples:
        return {}
    values = [s["bandwidth_gbps"] for s in samples]
    values.sort()
    n = len(values)

    def pct(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return values[lo] * (1 - (idx - lo)) + values[hi] * (idx - lo)

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "sample_count":    n,
        "mean_gbps":       round(mean, 3),
        "median_gbps":     round(pct(0.5), 3),
        "p95_gbps":        round(pct(0.95), 3),
        "max_gbps":        round(max(values), 3),
        "min_gbps":        round(min(values), 3),
        "std_gbps":        round(variance ** 0.5, 3),
        "total_duration_s": round(sum(s["duration_ns"] for s in samples) / 1e9, 3),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Parse xctrace bandwidth output")
    parser.add_argument("--trace", help="Path to .trace file (requires xctrace)")
    parser.add_argument("--xml", help="Pre-exported XML file")
    parser.add_argument("--json", help="Benchmark result JSON to inject bandwidth into")
    parser.add_argument("--inject", action="store_true", help="Inject stats into --json")
    parser.add_argument("--output", help="CSV output file for raw samples")
    args = parser.parse_args()

    # Get XML content
    xml_content = None
    if args.xml:
        xml_content = Path(args.xml).read_text()
    elif args.trace:
        print(f"Exporting bandwidth table from {args.trace}...")
        xml_content = export_bandwidth_xml(args.trace)
    else:
        print("Provide --trace or --xml", file=sys.stderr)
        sys.exit(1)

    if not xml_content:
        print("No bandwidth data available", file=sys.stderr)
        sys.exit(1)

    samples = parse_bandwidth_xml(xml_content)
    print(f"Parsed {len(samples)} bandwidth samples")

    if not samples:
        print("No samples extracted — check trace format", file=sys.stderr)
        sys.exit(1)

    stats = bandwidth_stats(samples)
    print(f"Mean: {stats['mean_gbps']} GB/s  Median: {stats['median_gbps']} GB/s  "
          f"P95: {stats['p95_gbps']} GB/s")

    # Write CSV of raw samples
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp_ns", "duration_ns", "bandwidth_gbps"])
            writer.writeheader()
            writer.writerows(samples)
        print(f"Raw samples written to {args.output}")

    # Inject into existing benchmark JSON
    if args.inject and args.json:
        json_path = Path(args.json)
        with open(json_path) as f:
            bench_data = json.load(f)
        bench_data["bandwidth_stats"] = stats
        bench_data["bandwidth_samples"] = samples[:200]  # cap to 200 samples to keep file small
        with open(json_path, "w") as f:
            json.dump(bench_data, f, indent=2)
        print(f"Injected bandwidth stats into {json_path}")


if __name__ == "__main__":
    main()
