#!/usr/bin/env bash
# collection/capture_bandwidth.sh
# =================================
# Wraps xctrace to capture a System Trace .trace file during a benchmark run.
# Called as a background process by run_experiment.sh.
# Killed via SIGTERM when the benchmark run finishes.
#
# The captured trace is later parsed by parse_xctrace.py to extract
# memory bandwidth (GB/s) readings aligned with the inference window.
#
# Prerequisites:
#   xctrace is installed with Xcode CLI tools:
#   xcode-select --install
#
# Usage (internal — called by run_experiment.sh):
#   bash collection/capture_bandwidth.sh output.trace
#   (runs until killed, then saves the trace)

set -euo pipefail

OUTPUT_TRACE="${1:-data/raw/traces/bandwidth_$(date +%s).trace}"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_TRACE")"

echo "  [BANDWIDTH] Starting xctrace capture → $OUTPUT_TRACE"

# Clean up any existing trace at this path
if [[ -d "$OUTPUT_TRACE" ]]; then
  rm -rf "$OUTPUT_TRACE"
fi

# xctrace record with System Trace template captures:
#   - Memory: Memory Bandwidth (GB/s)
#   - Memory: Physical Memory Footprint
#   - CPU: CPU Usage
#   - Energy: Energy Impact
#
# We run it against the whole system (--all-processes) so we capture
# activity from the ANEBenchmark process regardless of its PID.
#
# NOTE: xctrace requires sudo OR screen recording permission granted
# to Terminal in System Preferences > Privacy & Security.
xctrace record \
  --template "System Trace" \
  --all-processes \
  --output "$OUTPUT_TRACE" \
  --time-limit 600  # 10 minute maximum safety limit

# This line is only reached if xctrace exits naturally (time limit hit).
# Normally it is killed by run_experiment.sh via kill $trace_pid.
echo "  [BANDWIDTH] Trace saved: $OUTPUT_TRACE"
