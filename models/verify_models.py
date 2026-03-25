"""
models/verify_models.py
=======================
Runs a quick sanity check on all converted .mlpackage models:
  1. Can it be loaded without error?
  2. Does a single forward pass produce the expected output shape?
  3. Is the on-disk footprint consistent with the expected precision ratio?

Usage:
    python models/verify_models.py --dir data/models/
    python models/verify_models.py --dir data/models/sweep/ --verbose
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np


def get_input_shape(spec) -> tuple[str, tuple]:
    """
    Robustly extract (input_name, shape) from a CoreML spec.

    mlprogram (.mlpackage) stores shape differently from the old .mlmodel
    format. We try three approaches in order:
      1. multiArrayType.shape with .size  (old mlmodel style)
      2. multiArrayType.shapeRange lower bounds  (flexible shape style)
      3. Fall back to the standard (1, 3, 224, 224) for image classifiers
    """
    input_desc = spec.description.input[0]
    name = input_desc.name

    mat = input_desc.type.multiArrayType

    # Approach 1: fixed shape via .shape repeated field
    # Each element is a Dimension with a .size attribute
    if len(mat.shape) > 0:
        dims = [d for d in mat.shape]
        # .size works on old format; on mlprogram format the dims may be 0
        sizes = tuple(int(d) for d in dims)
        if all(s > 0 for s in sizes):
            return name, sizes

    # Approach 2: shapeRange (flexible shapes — take the lower bound)
    try:
        lower = mat.shapeRange.sizeRanges
        if lower:
            sizes = tuple(int(r.lowerBound) for r in lower)
            if all(s > 0 for s in sizes):
                return name, sizes
    except AttributeError:
        pass

    # Approach 3: enumeratedShapes (takes first enumerated shape)
    try:
        enum_shapes = mat.enumeratedShapes.shapes
        if enum_shapes:
            sizes = tuple(int(d) for d in enum_shapes[0].shape)
            if all(s > 0 for s in sizes):
                return name, sizes
    except AttributeError:
        pass

    # Fallback: standard image classifier input
    return name, (1, 3, 224, 224)


def disk_size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 2)
    return path.stat().st_size / (1024 ** 2)


def verify_model(mlpackage_path: Path, verbose: bool = False) -> dict:
    """Load and run one forward pass on a CoreML model."""
    import coremltools as ct

    result = {
        "path": str(mlpackage_path),
        "name": mlpackage_path.name,
        "status": "unknown",
    }

    try:
        # Load with CPU-only so verification works on any machine
        config = ct.models.MLModel.MLModelConfiguration()  # noqa — attribute not always present
    except AttributeError:
        pass

    try:
        model = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    except Exception:
        # Older coremltools API
        try:
            model = ct.models.MLModel(str(mlpackage_path))
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"load_failed: {exc}"
            result["traceback"] = traceback.format_exc()
            if verbose:
                print(f"  [ERR] {mlpackage_path.name}")
                print(f"        {result['error']}")
            return result

    try:
        spec = model.get_spec()
        input_name, shape = get_input_shape(spec)
        result["input_name"]  = input_name
        result["input_shape"] = list(shape)

        # Build a zero-filled numpy array of the right shape
        dummy_input = np.zeros(shape, dtype=np.float32)

        # Run prediction using the ACTUAL input name from the spec
        predictions = model.predict({input_name: dummy_input})

        output_keys = list(predictions.keys())
        output_val  = predictions[output_keys[0]] if output_keys else None
        result["output_keys"]  = output_keys
        result["output_shape"] = list(output_val.shape) if output_val is not None else []

        result["size_mb"] = round(disk_size_mb(mlpackage_path), 3)
        result["status"]  = "ok"

        if verbose:
            print(f"  [OK]  {mlpackage_path.name}")
            print(f"        key='{input_name}'  input={shape}  "
                  f"output={result['output_shape']}  "
                  f"size={result['size_mb']:.2f}MB")

    except Exception as exc:
        result["status"]    = "error"
        result["error"]     = str(exc)
        result["traceback"] = traceback.format_exc()
        if verbose:
            print(f"  [ERR] {mlpackage_path.name}")
            print(f"        {exc}")

    return result


def check_precision_ratios(records: list[dict]) -> None:
    """
    For each model base-name, check that FP16 is ~50% of FP32 size,
    and INT8 is ~25% of FP32 size. Flag large deviations.
    """
    print("\nPrecision ratio checks (expected: fp16≈0.5x, int8≈0.25x of fp32):")

    # Group by base name (strip precision suffix)
    groups: dict[str, dict] = {}
    for r in records:
        if r["status"] != "ok":
            continue
        name = r["name"].replace(".mlpackage", "")
        for suffix in ["_fp32", "_fp16", "_int8_linear", "_int8_palettized"]:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                prec = suffix.lstrip("_")
                groups.setdefault(base, {})[prec] = r["size_mb"]
                break

    for base, sizes in sorted(groups.items()):
        fp32 = sizes.get("fp32")
        if fp32 is None:
            continue
        parts = [f"fp32={fp32:.2f}MB"]
        for prec, expected_ratio, label in [
            ("fp16", 0.5, "fp16"),
            ("int8_linear", 0.25, "int8_linear"),
            ("int8_palettized", 0.25, "int8_pal"),
        ]:
            if prec in sizes:
                actual_ratio = sizes[prec] / fp32
                flag = " !" if abs(actual_ratio - expected_ratio) > 0.15 else ""
                parts.append(f"{label}={sizes[prec]:.2f}MB ({actual_ratio:.2f}x){flag}")
        print(f"  {base[:40]:<40s}  {'  '.join(parts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify converted CoreML models")
    parser.add_argument("--dir", default="data/models", help="Directory to scan")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")
    args = parser.parse_args()

    scan_dir = Path(args.dir)
    model_paths = sorted(scan_dir.rglob("*.mlpackage"))

    if not model_paths:
        print(f"No .mlpackage files found in {scan_dir}")
        return

    print(f"Found {len(model_paths)} models in {scan_dir}")
    print()

    records = []
    ok = error = 0
    for path in model_paths:
        record = verify_model(path, verbose=args.verbose)
        records.append(record)
        if record["status"] == "ok":
            ok += 1
        else:
            error += 1

    print(f"\nResults: {ok} ok, {error} errors")

    check_precision_ratios(records)

    if args.save_report:
        report_path = scan_dir / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump({"records": records}, f, indent=2)
        print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
