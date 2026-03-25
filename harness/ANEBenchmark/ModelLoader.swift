// harness/ANEBenchmark/ModelLoader.swift
// =======================================
// Handles CoreML model loading with explicit compute unit configuration.
//
// Critical note on compute unit isolation:
//   MLComputeUnits is a *hint* to the scheduler, not a guarantee.
//   When you specify .cpuAndNeuralEngine, CoreML CAN dispatch some
//   ops to the GPU if it deems it more efficient. True isolation
//   requires profiling with Instruments to confirm which units
//   actually ran — compare "Neural Engine Utilization %" in
//   Instruments > Energy Log alongside your latency measurements.
//
//   For the research, document this limitation explicitly.

import Foundation
import CoreML

enum ComputeUnitOption: String {
    case cpuOnly             = "cpuOnly"
    case cpuAndNeuralEngine  = "cpuAndNeuralEngine"
    case cpuAndGPU           = "cpuAndGPU"
    case all                 = "all"

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

    /// Load an MLModel from an .mlpackage path with specified compute unit.
    /// Returns nil on failure — caller should handle the nil case explicitly.
    static func load(path: String, computeUnit: String) -> MLModel? {
        guard let unitOption = ComputeUnitOption(rawValue: computeUnit) else {
            fputs("Unknown compute unit: '\(computeUnit)'. " +
                  "Valid options: cpuOnly, cpuAndNeuralEngine, cpuAndGPU, all\n", stderr)
            return nil
        }

        let modelURL = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            fputs("Model not found at path: \(path)\n", stderr)
            return nil
        }

        let config = MLModelConfiguration()
        config.computeUnits = unitOption.mlComputeUnits

        // Allow function specialization — this is what triggers ANE compilation
        // on the first load. After warm-up, the compiled graph is resident.
        config.allowLowPrecisionAccumulationOnGPU = true

        do {
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            return model
        } catch {
            fputs("MLModel load error: \(error)\n", stderr)
            return nil
        }
    }

    /// Describe model inputs/outputs — useful for debugging
    static func describeModel(_ model: MLModel) -> String {
        var lines: [String] = []
        lines.append("Inputs:")
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            lines.append("  \(name): \(desc.type)")
        }
        lines.append("Outputs:")
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
            lines.append("  \(name): \(desc.type)")
        }
        return lines.joined(separator: "\n")
    }
}
