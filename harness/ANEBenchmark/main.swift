// harness/ANEBenchmark/main.swift
// ================================
// CLI entry point for the ANE benchmark harness.
// Parses arguments, orchestrates the benchmark run, and exits with
// a non-zero code on fatal errors so shell scripts can detect failures.
//
// Build: Open in Xcode 15+, select macOS target, Product > Build.
// Run:   ./.build/release/ANEBenchmark --help
//
// Usage examples:
//   ./ANEBenchmark --model data/models/mobilenetv3_small_fp32.mlpackage \
//                  --compute-unit cpuAndNeuralEngine \
//                  --iterations 500 --warmup 200 \
//                  --output data/raw/run_001.json
//
//   ./ANEBenchmark --model data/models/sweep/ \
//                  --batch   (sweeps all .mlpackage files in a directory)

import Foundation

// ── CLI argument parsing ──────────────────────────────────────────────────────

struct CLIConfig {
    var modelPath: String = ""
    var computeUnit: String = "all"           // cpuOnly | cpuAndNeuralEngine | all
    var iterations: Int    = 500
    var warmup: Int        = 200
    var outputPath: String = "data/raw/run_\(Int(Date().timeIntervalSince1970)).json"
    var batchMode: Bool    = false            // sweep all .mlpackage in a directory
    var thermalCutoff: Double = 45.0          // Celsius
    var verbose: Bool      = false

    static func parse() -> CLIConfig {
        var config = CLIConfig()
        var args = CommandLine.arguments.dropFirst()

        while !args.isEmpty {
            let arg = args.removeFirst()
            switch arg {
            case "--model":
                config.modelPath = args.removeFirst()
            case "--compute-unit":
                config.computeUnit = args.removeFirst()
            case "--iterations":
                config.iterations = Int(args.removeFirst()) ?? 500
            case "--warmup":
                config.warmup = Int(args.removeFirst()) ?? 200
            case "--output":
                config.outputPath = args.removeFirst()
            case "--batch":
                config.batchMode = true
            case "--thermal-cutoff":
                config.thermalCutoff = Double(args.removeFirst()) ?? 45.0
            case "--verbose":
                config.verbose = true
            case "--help", "-h":
                printHelp()
                exit(0)
            default:
                fputs("Unknown argument: \(arg)\n", stderr)
                exit(1)
            }
        }

        if config.modelPath.isEmpty {
            fputs("Error: --model is required\n", stderr)
            printHelp()
            exit(1)
        }
        return config
    }

    static func printHelp() {
        print("""
        ANEBenchmark — CoreML inference latency measurement harness
        
        Usage:
          ANEBenchmark --model <path> [options]
        
        Options:
          --model <path>           Path to .mlpackage file or directory (with --batch)
          --compute-unit <unit>    cpuOnly | cpuAndNeuralEngine | all  [default: all]
          --iterations <n>         Number of timed inferences           [default: 500]
          --warmup <n>             Warm-up inferences (discarded)       [default: 200]
          --output <path>          JSON output file path
          --batch                  Run all .mlpackage files in --model directory
          --thermal-cutoff <°C>    Abort if skin temp exceeds this      [default: 45.0]
          --verbose                Print progress during run
          --help                   Show this help
        
        Output JSON schema:
          {
            "model_name": "mobilenetv3_small_fp32",
            "compute_unit": "cpuAndNeuralEngine",
            "device_chip": "M2",
            "model_size_mb": 12.4,
            "warmup_iterations": 200,
            "record_iterations": 500,
            "latencies_us": [...],           // microseconds, one per iteration
            "median_us": 4521.3,
            "p95_us": 5102.0,
            "p99_us": 5876.0,
            "mean_us": 4589.1,
            "std_us": 234.5,
            "cv_pct": 5.1,
            "thermal_samples": [...],        // {timestamp_s, celsius} pairs
            "run_aborted_thermal": false
          }
        """)
    }
}

// ── entry point ───────────────────────────────────────────────────────────────

let config = CLIConfig.parse()

if config.batchMode {
    // Sweep all .mlpackage files in the given directory
    let dirURL = URL(fileURLWithPath: config.modelPath)
    guard let contents = try? FileManager.default.contentsOfDirectory(
        at: dirURL,
        includingPropertiesForKeys: nil
    ) else {
        fputs("Cannot read directory: \(config.modelPath)\n", stderr)
        exit(1)
    }

    let packages = contents.filter { $0.pathExtension == "mlpackage" }.sorted { $0.path < $1.path }
    print("Batch mode: found \(packages.count) models")

    var anyFailed = false
    for (i, packageURL) in packages.enumerated() {
        let outputName = packageURL.deletingPathExtension().lastPathComponent
            + "_\(config.computeUnit).json"
        let outputPath = URL(fileURLWithPath: config.outputPath)
            .deletingLastPathComponent()
            .appendingPathComponent(outputName)
            .path

        print("\n[\(i+1)/\(packages.count)] \(packageURL.lastPathComponent)")
        let runner = BenchmarkRunner(
            modelPath: packageURL.path,
            computeUnit: config.computeUnit,
            warmup: config.warmup,
            iterations: config.iterations,
            outputPath: outputPath,
            thermalCutoff: config.thermalCutoff,
            verbose: config.verbose
        )
        if !runner.run() {
            anyFailed = true
        }
    }
    exit(anyFailed ? 1 : 0)

} else {
    let runner = BenchmarkRunner(
        modelPath: config.modelPath,
        computeUnit: config.computeUnit,
        warmup: config.warmup,
        iterations: config.iterations,
        outputPath: config.outputPath,
        thermalCutoff: config.thermalCutoff,
        verbose: config.verbose
    )
    exit(runner.run() ? 0 : 1)
}
