// harness/ANEBenchmark/MetricsLogger.swift
// =========================================
// Serialises benchmark results to a JSON file.
// Ensures the output directory exists and uses atomic writes
// to avoid partial files if the process is interrupted.

import Foundation

class MetricsLogger {

    @discardableResult
    static func write(_ data: [String: Any], to path: String) -> Bool {
        let outputURL = URL(fileURLWithPath: path)

        // Create parent directory if needed
        let parentDir = outputURL.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(
                at: parentDir,
                withIntermediateDirectories: true
            )
        } catch {
            fputs("Cannot create output directory \(parentDir.path): \(error)\n", stderr)
            return false
        }

        do {
            let jsonData = try JSONSerialization.data(
                withJSONObject: data,
                options: [.prettyPrinted, .sortedKeys]
            )
            // Atomic write — avoids partial file on crash/interrupt
            try jsonData.write(to: outputURL, options: .atomic)
            print("  Output written to \(path)")
            return true
        } catch {
            fputs("JSON write error: \(error)\n", stderr)
            return false
        }
    }

    /// Read and merge multiple JSON run files into a single combined record.
    /// Useful for batch post-processing.
    static func readRuns(from directory: String) -> [[String: Any]] {
        let dirURL = URL(fileURLWithPath: directory)
        guard let enumerator = FileManager.default.enumerator(
            at: dirURL, includingPropertiesForKeys: nil
        ) else { return [] }

        var runs: [[String: Any]] = []
        for case let fileURL as URL in enumerator {
            guard fileURL.pathExtension == "json" else { continue }
            guard let data = try? Data(contentsOf: fileURL),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            runs.append(json)
        }
        return runs
    }
}
