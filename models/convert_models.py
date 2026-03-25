"""
models/convert_models.py
========================
Converts MobileNetV3 (Small + Large) from PyTorch to CoreML in
FP32, FP16, and INT8 precision variants.

Usage:
    python models/convert_models.py --output data/models/
    python models/convert_models.py --output data/models/ --model large
    python models/convert_models.py --output data/models/ --dry-run

Requirements:
    pip install coremltools torch torchvision

Notes on INT8 on ANE:
    Apple's ANE does not expose raw INT8 operations in the same way
    NVIDIA TensorRT does. When you specify INT8 via coremltools, the
    CoreML compiler applies weight palettization or linear quantization,
    and the scheduler decides how to dispatch ops. This script produces
    both linear-quantized and palettized variants so you can compare
    how the scheduler treats each.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torchvision.models as tv_models
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
    palettize_weights,
    OpPalettizerConfig,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_pytorch_model(model_name: str) -> torch.nn.Module:
    """Load a pretrained MobileNetV3 variant from torchvision."""
    print(f"  Loading PyTorch {model_name}...")
    if model_name == "mobilenetv3_small":
        model = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
    elif model_name == "mobilenetv3_large":
        model = tv_models.mobilenet_v3_large(
            weights=tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.eval()
    return model


def trace_to_mlpackage(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: Path,
    precision: str,
) -> ct.models.MLModel:
    """
    Trace PyTorch model and convert to CoreML MLPackage.

    precision must be one of: 'fp32', 'fp16', 'int8_linear', 'int8_palettized'
    """
    dummy_input = torch.zeros(*input_shape)
    traced = torch.jit.trace(model, dummy_input)

    # CoreML requires a compute precision at conversion time for the compute graph.
    # NOTE: This sets the *graph* compute type, not the weight storage type.
    # INT8 weight quantization is applied POST-conversion as a separate pass.
    if precision in ("fp32",):
        compute_precision = ct.precision.FLOAT32
    else:
        # FP16 graph for fp16, int8_linear, int8_palettized
        # ANE prefers FP16 compute graph — all three non-fp32 variants use it
        compute_precision = ct.precision.FLOAT16

    print(f"    Tracing with compute_precision={compute_precision}...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=input_shape, name="input_image")],
        outputs=[ct.TensorType(name="class_logits")],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.macOS13,
        convert_to="mlprogram",   # .mlpackage format — required for ANE dispatch
    )

    # Apply weight quantization passes for INT8 variants
    if precision == "int8_linear":
        print("    Applying linear weight quantization (INT8)...")
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=config)

    elif precision == "int8_palettized":
        print("    Applying 4-bit palettization...")
        op_config = OpPalettizerConfig(mode="kmeans", nbits=4)
        config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return mlmodel


def get_model_file_size_mb(path: Path) -> float:
    """Recursively sum all files in an .mlpackage directory."""
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return round(total / (1024 * 1024), 3)


def write_manifest(output_dir: Path, records: list[dict]) -> None:
    """Write a JSON manifest of all converted models with metadata."""
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coremltools_version": ct.__version__,
        "models": records,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written to {manifest_path}")


# ── main conversion logic ─────────────────────────────────────────────────────

PRECISION_VARIANTS = [
    "fp32",
    "fp16",
    "int8_linear",
    "int8_palettized",
]

MODEL_NAMES = {
    "small": "mobilenetv3_small",
    "large": "mobilenetv3_large",
    "both": None,  # handled below
}

INPUT_SHAPE = (1, 3, 224, 224)


def convert_one_model(
    model_name: str,
    output_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    """Convert a single PyTorch model to all precision variants."""
    records = []
    print(f"\n{'='*60}")
    print(f"Converting: {model_name}")
    print(f"{'='*60}")

    pytorch_model = load_pytorch_model(model_name)

    for precision in PRECISION_VARIANTS:
        output_name = f"{model_name}_{precision}.mlpackage"
        output_path = output_dir / output_name

        if output_path.exists():
            size_mb = get_model_file_size_mb(output_path)
            print(f"  [SKIP] {output_name} already exists ({size_mb} MB)")
            records.append({
                "name": output_name,
                "model": model_name,
                "precision": precision,
                "size_mb": size_mb,
                "status": "skipped",
            })
            continue

        print(f"\n  Converting to {precision}...")
        if dry_run:
            print(f"  [DRY RUN] Would write to {output_path}")
            records.append({
                "name": output_name,
                "model": model_name,
                "precision": precision,
                "size_mb": None,
                "status": "dry_run",
            })
            continue

        start = time.time()
        try:
            trace_to_mlpackage(pytorch_model, INPUT_SHAPE, output_path, precision)
            elapsed = round(time.time() - start, 1)
            size_mb = get_model_file_size_mb(output_path)
            print(f"  Done: {output_name}  ({size_mb} MB, {elapsed}s)")
            records.append({
                "name": output_name,
                "model": model_name,
                "precision": precision,
                "size_mb": size_mb,
                "elapsed_sec": elapsed,
                "status": "ok",
            })
        except Exception as exc:
            print(f"  ERROR converting {output_name}: {exc}")
            records.append({
                "name": output_name,
                "model": model_name,
                "precision": precision,
                "size_mb": None,
                "status": "error",
                "error": str(exc),
            })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert models to CoreML variants")
    parser.add_argument("--output", default="data/models", help="Output directory")
    parser.add_argument(
        "--model",
        choices=["small", "large", "both"],
        default="both",
        help="Which MobileNetV3 variant to convert",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without converting",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "both":
        names = ["mobilenetv3_small", "mobilenetv3_large"]
    else:
        names = [MODEL_NAMES[args.model]]

    all_records = []
    for name in names:
        records = convert_one_model(name, output_dir, dry_run=args.dry_run)
        all_records.extend(records)

    if not args.dry_run:
        write_manifest(output_dir, all_records)

    print("\nSummary:")
    for r in all_records:
        status_str = r["status"].upper()
        size_str = f"{r['size_mb']} MB" if r["size_mb"] else "—"
        print(f"  [{status_str:8s}] {r['name']:<45s} {size_str}")


if __name__ == "__main__":
    main()
