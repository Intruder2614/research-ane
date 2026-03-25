// harness/ANEBenchmark/BenchmarkRunner.swift
// ==========================================
// Core inference + timing loop.
// Uses mach_absolute_time for nanosecond-resolution wall-clock timing.
// Thermal monitoring runs on a background thread and can abort a run mid-way.

import Foundation
import CoreML

class BenchmarkRunner {

    let modelPath: String
    let computeUnit: String
    let warmup: Int
    let iterations: Int
    let outputPath: String
    let thermalCutoff: Double
    let verbose: Bool

    // Shared state between thermal monitor and timing loop
    private var thermalSamples: [[String: Any]] = []
    private var thermalAbort = false

    init(
        modelPath: String,
        computeUnit: String,
        warmup: Int,
        iterations: Int,
        outputPath: String,
        thermalCutoff: Double,
        verbose: Bool
    ) {
        self.modelPath   = modelPath
        self.computeUnit = computeUnit
        self.warmup      = warmup
        self.iterations  = iterations
        self.outputPath  = outputPath
        self.thermalCutoff = thermalCutoff
        self.verbose     = verbose
    }

    // ── public entry point ────────────────────────────────────────────────

    @discardableResult
    func run() -> Bool {
        // 1. Load the model
        guard let model = ModelLoader.load(
            path: modelPath,
            computeUnit: computeUnit
        ) else {
            fputs("Failed to load model from \(modelPath)\n", stderr)
            return false
        }

        let modelName = URL(fileURLWithPath: modelPath)
            .deletingPathExtension().lastPathComponent
        let modelSizeMB = diskSizeMB(path: modelPath)

        if verbose {
            print("  Model: \(modelName)")
            print("  Size: \(String(format: "%.2f", modelSizeMB)) MB")
            print("  Compute unit: \(computeUnit)")
        }

        // 2. Prepare a dummy input that matches the model's expected shape
        guard let inputFeatureName = model.modelDescription.inputDescriptionsByName.keys.first,
              let inputConstraint  = model.modelDescription.inputDescriptionsByName[inputFeatureName],
              case let .multiArray(shape) = inputConstraint.type else {
            fputs("Cannot determine input shape from model spec\n", stderr)
            return false
        }

        let inputShape = shape.shapeConstraint.enumeratedShapes.first?.map { $0.intValue }
            ?? [1, 3, 224, 224]
        let inputArray = try! MLMultiArray(shape: inputShape.map { NSNumber(value: $0) },
                                           dataType: .float32)
        // Fill with zeros — we want consistent inference, not accuracy
        for i in 0 ..< inputArray.count { inputArray[i] = 0.0 }
        let inputFeature = try! MLFeatureValue(multiArray: inputArray)
        let inputProvider = try! MLDictionaryFeatureProvider(
            dictionary: [inputFeatureName: inputFeature]
        )

        // 3. Start thermal monitor on background thread
        let thermalMonitor = ThermalMonitor(cutoffCelsius: thermalCutoff, pollIntervalMs: 2000)
        thermalMonitor.onReading = { [weak self] celsius, timestamp in
            self?.thermalSamples.append(["timestamp_s": timestamp, "celsius": celsius])
            if celsius >= (self?.thermalCutoff ?? 45.0) {
                self?.thermalAbort = true
            }
        }
        thermalMonitor.start()

        // 4. Warm-up loop (discarded)
        if verbose { print("  Warming up (\(warmup) iterations)...") }
        for i in 0 ..< warmup {
            _ = try? model.prediction(from: inputProvider)
            if thermalAbort {
                fputs("  Thermal abort during warm-up at iteration \(i)\n", stderr)
                break
            }
        }

        // 5. Timed recording loop
        if verbose { print("  Recording (\(iterations) iterations)...") }
        var latenciesUS: [Double] = []
        latenciesUS.reserveCapacity(iterations)

        var timebaseInfo = mach_timebase_info_data_t(numer: 0, denom: 0)
        mach_timebase_info(&timebaseInfo)
        let nsPerTick = Double(timebaseInfo.numer) / Double(timebaseInfo.denom)

        for i in 0 ..< iterations {
            if thermalAbort {
                fputs("  Thermal abort at iteration \(i)\n", stderr)
                break
            }
            let start = mach_absolute_time()
            _ = try? model.prediction(from: inputProvider)
            let end   = mach_absolute_time()

            let elapsedNS = Double(end - start) * nsPerTick
            latenciesUS.append(elapsedNS / 1_000.0)
        }

        thermalMonitor.stop()

        // 6. Compute statistics
        guard !latenciesUS.isEmpty else {
            fputs("No latency measurements recorded\n", stderr)
            return false
        }

        let stats = computeStats(latenciesUS)

        if verbose {
            print(String(format: "  Median: %.1f µs  P95: %.1f µs  P99: %.1f µs  CV: %.1f%%",
                         stats.median, stats.p95, stats.p99, stats.cv))
        }

        // 7. Build output JSON
        let output: [String: Any] = [
            "model_name":           modelName,
            "model_path":           modelPath,
            "compute_unit":         computeUnit,
            "device_chip":          deviceChip(),
            "model_size_mb":        modelSizeMB,
            "warmup_iterations":    warmup,
            "record_iterations":    latenciesUS.count,
            "latencies_us":         latenciesUS,
            "median_us":            stats.median,
            "p95_us":               stats.p95,
            "p99_us":               stats.p99,
            "mean_us":              stats.mean,
            "std_us":               stats.std,
            "cv_pct":               stats.cv,
            "thermal_samples":      thermalSamples,
            "run_aborted_thermal":  thermalAbort,
            "run_timestamp":        ISO8601DateFormatter().string(from: Date()),
        ]

        return MetricsLogger.write(output, to: outputPath)
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private struct Stats {
        let median: Double
        let p95: Double
        let p99: Double
        let mean: Double
        let std: Double
        let cv: Double
    }

    private func computeStats(_ values: [Double]) -> Stats {
        let sorted = values.sorted()
        let n = sorted.count

        func percentile(_ p: Double) -> Double {
            let idx = p * Double(n - 1)
            let lo  = Int(idx)
            let hi  = min(lo + 1, n - 1)
            let frac = idx - Double(lo)
            return sorted[lo] * (1 - frac) + sorted[hi] * frac
        }

        let mean = values.reduce(0, +) / Double(n)
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(n)
        let std = variance.squareRoot()

        return Stats(
            median: percentile(0.50),
            p95:    percentile(0.95),
            p99:    percentile(0.99),
            mean:   mean,
            std:    std,
            cv:     std / mean * 100.0
        )
    }

    private func diskSizeMB(path: String) -> Double {
        let url = URL(fileURLWithPath: path)
        if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            var total = 0
            for case let fileURL as URL in enumerator {
                total += (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) ?? 0
            }
            return Double(total) / (1024 * 1024)
        }
        return 0
    }

    private func deviceChip() -> String {
        // sysctlbyname("machdep.cpu.brand_string") gives the chip identifier on macOS
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        return String(cString: buffer)
    }
}
