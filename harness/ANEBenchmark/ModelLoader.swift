// harness/ANEBenchmark/ModelLoader.swift
// =======================================
// Handles CoreML model loading with explicit compute unit configuration.
//
// .mlpackage files produced by coremltools are SOURCE packages — the Swift
// CoreML runtime requires a compiled .mlmodelc bundle. This file handles
// compilation transparently:
//   1. Check if a cached .mlmodelc already exists next to the .mlpackage
//   2. If not, compile via MLModel.compileModel(at:) and cache it
//   3. Load from the compiled .mlmodelc

import Foundation
import CoreML

enum ComputeUnitOption: String {
    case cpuOnly            = "cpuOnly"
    case cpuAndNeuralEngine = "cpuAndNeuralEngine"
    case cpuAndGPU          = "cpuAndGPU"
    case all                = "all"

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly:            return .cpuOnly
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuAndGPU:          return .cpuAndGPU
        case .all:                return .all
        }
    }
}

class ModelLoader {

    /// Load an MLModel, compiling the .mlpackage to .mlmodelc if needed.
    /// The compiled bundle is cached alongside the source package so
    /// subsequent runs skip the compilation step.
    static func load(path: String, computeUnit: String) -> MLModel? {
        guard let unitOption = ComputeUnitOption(rawValue: computeUnit) else {
            fputs("Unknown compute unit: '\(computeUnit)'. " +
                  "Valid: cpuOnly, cpuAndNeuralEngine, cpuAndGPU, all\n", stderr)
            return nil
        }

        let sourceURL = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            fputs("Model not found at path: \(path)\n", stderr)
            return nil
        }

        // Derive the cached .mlmodelc path:
        //   data/models/mobilenetv3_small_fp32.mlpackage
        //   → data/models/mobilenetv3_small_fp32.mlmodelc
        let compiledURL = sourceURL
            .deletingPathExtension()
            .appendingPathExtension("mlmodelc")

        // Compile if the .mlmodelc does not exist yet
        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            print("  Compiling \(sourceURL.lastPathComponent) → \(compiledURL.lastPathComponent) ...")
            do {
                let tempURL = try MLModel.compileModel(at: sourceURL)
                // Move from the temp location to our permanent cache location
                // (compileModel places the result in a system temp directory)
                if FileManager.default.fileExists(atPath: compiledURL.path) {
                    try FileManager.default.removeItem(at: compiledURL)
                }
                try FileManager.default.moveItem(at: tempURL, to: compiledURL)
                print("  Compiled and cached at \(compiledURL.lastPathComponent)")
            } catch {
                fputs("Compilation failed for \(path): \(error)\n", stderr)
                return nil
            }
        } else {
            print("  Using cached \(compiledURL.lastPathComponent)")
        }

        // Load from the compiled bundle
        let config = MLModelConfiguration()
        config.computeUnits = unitOption.mlComputeUnits
        config.allowLowPrecisionAccumulationOnGPU = true

        do {
            return try MLModel(contentsOf: compiledURL, configuration: config)
        } catch {
            fputs("MLModel load error: \(error)\n", stderr)
            // If the cached .mlmodelc is stale/corrupt, delete it so next
            // run recompiles from scratch
            try? FileManager.default.removeItem(at: compiledURL)
            fputs("Deleted stale cache — re-run to recompile.\n", stderr)
            return nil
        }
    }
}
