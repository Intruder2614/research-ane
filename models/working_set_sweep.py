"""
models/working_set_sweep.py
===========================
Generates MobileNetV3-style models at 8 different width-multiplier
settings to create a sweep of working-set sizes from ~0.5 MB to ~38 MB.

This is the MOST IMPORTANT experiment in the research — the latency-vs-
footprint scaling curve across these models (comparing FP32 vs INT8) is
the primary figure used to detect the cache-residency breakpoint.

Usage:
    python models/working_set_sweep.py --output data/models/sweep/
    python models/working_set_sweep.py --output data/models/sweep/ --precisions fp32 int8_linear
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)


# ── scaled MobileNetV3-style architecture ────────────────────────────────────

class InvertedResidual(nn.Module):
    """Depthwise-separable bottleneck block — core of MobileNet architectures."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: float):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.Hardswish()]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class ScaledMobileNet(nn.Module):
    """
    Width-scalable MobileNetV3-inspired architecture.

    width_multiplier controls total parameter count (and thus weight
    memory footprint). This is the primary knob for the sweep.
    """

    # Base channel configuration — scaled by width_multiplier
    BASE_CHANNELS = [16, 16, 24, 24, 40, 40, 80, 80, 112, 112, 160, 160, 960]
    BASE_STRIDES  = [2,  1,  2,  1,  2,  1,  2,  1,  1,   1,   2,   1,  1 ]
    EXPAND_RATIOS = [1,  4,  4,  3,  3,  3,  6,  2.5, 2.3, 2.3, 6,  6,  1 ]

    def __init__(self, width_multiplier: float = 1.0, num_classes: int = 1000):
        super().__init__()
        self.width_multiplier = width_multiplier

        def scale(ch: int) -> int:
            return max(8, int(ch * width_multiplier))

        # First conv
        first_ch = scale(16)
        layers: list[nn.Module] = [
            nn.Conv2d(3, first_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_ch),
            nn.Hardswish(),
        ]

        # Inverted residual blocks
        in_ch = first_ch
        for i, (base_out, stride, expand) in enumerate(
            zip(self.BASE_CHANNELS, self.BASE_STRIDES, self.EXPAND_RATIOS)
        ):
            out_ch = scale(base_out)
            layers.append(InvertedResidual(in_ch, out_ch, stride, expand))
            in_ch = out_ch

        # Head
        head_ch = scale(self.BASE_CHANNELS[-1])
        layers += [
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_ch, scale(1280)),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(scale(1280), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_footprint_mb(model: nn.Module, dtype: str = "fp32") -> float:
    """Estimate model weight memory footprint in MB."""
    params = count_parameters(model)
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8_linear": 1}[dtype]
    return round(params * bytes_per_param / (1024 * 1024), 3)


# ── conversion ────────────────────────────────────────────────────────────────

def convert_scaled_model(
    width_multiplier: float,
    precision: str,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
) -> dict:
    """Create, trace, and convert one model variant."""
    model = ScaledMobileNet(width_multiplier=width_multiplier)
    model.eval()

    estimated_fp32_mb = estimate_footprint_mb(model, "fp32")
    param_count = count_parameters(model)

    dummy = torch.zeros(*input_shape)
    traced = torch.jit.trace(model, dummy)

    compute_precision = ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=input_shape, name="input_image")],
        outputs=[ct.TensorType(name="class_logits")],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.macOS13,
        convert_to="mlprogram",
    )

    if precision == "int8_linear":
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
        mlmodel = linear_quantize_weights(mlmodel, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # Measure actual disk size after conversion
    if output_path.is_dir():
        actual_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024**2)
    else:
        actual_mb = output_path.stat().st_size / (1024**2)

    return {
        "width_multiplier": round(width_multiplier, 3),
        "precision": precision,
        "param_count": param_count,
        "estimated_fp32_mb": estimated_fp32_mb,
        "actual_disk_mb": round(actual_mb, 3),
        "output_path": str(output_path),
    }


# ── main ──────────────────────────────────────────────────────────────────────

# Width multipliers designed to hit approximately the target footprint sizes:
# [0.5, 1.0, 2.0, 4.0, 8.0, 14.0, 22.0, 38.0] MB (FP32)
SWEEP_WIDTHS = [0.05, 0.10, 0.18, 0.30, 0.50, 0.75, 1.00, 1.40]

ALL_PRECISIONS = ["fp32", "fp16", "int8_linear"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate working-set size sweep models")
    parser.add_argument("--output", default="data/models/sweep", help="Output directory")
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=ALL_PRECISIONS,
        default=ALL_PRECISIONS,
        help="Which precisions to generate",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Working-set sweep model generation")
    print(f"Width multipliers: {SWEEP_WIDTHS}")
    print(f"Precisions: {args.precisions}")
    print(f"Total models to generate: {len(SWEEP_WIDTHS) * len(args.precisions)}")
    print()

    # Preview estimated sizes before converting
    print("Estimated footprints (FP32 reference):")
    for wm in SWEEP_WIDTHS:
        m = ScaledMobileNet(width_multiplier=wm)
        fp32_mb = estimate_footprint_mb(m, "fp32")
        int8_mb = estimate_footprint_mb(m, "int8_linear")
        print(f"  w={wm:.2f}  FP32≈{fp32_mb:6.2f} MB  INT8≈{int8_mb:5.2f} MB  params={count_parameters(m):,}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without converting.")
        return

    print()
    records = []
    for i, wm in enumerate(SWEEP_WIDTHS):
        for precision in args.precisions:
            name = f"sweep_w{wm:.2f}_{precision}.mlpackage"
            out_path = output_dir / name

            if out_path.exists():
                print(f"[SKIP] {name}")
                continue

            print(f"[{i+1}/{len(SWEEP_WIDTHS)}] Generating {name}...")
            start = time.time()
            try:
                record = convert_scaled_model(wm, precision, out_path)
                record["elapsed_sec"] = round(time.time() - start, 1)
                record["status"] = "ok"
                records.append(record)
                print(
                    f"  Done: {record['actual_disk_mb']:.2f} MB disk  "
                    f"({record['param_count']:,} params)  "
                    f"{record['elapsed_sec']}s"
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
                records.append({"width_multiplier": wm, "precision": precision, "status": "error", "error": str(exc)})

    # Save sweep manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sweep_widths": SWEEP_WIDTHS,
        "precisions": args.precisions,
        "records": records,
    }
    manifest_path = output_dir / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSweep manifest saved to {manifest_path}")
    print(f"Total generated: {sum(1 for r in records if r.get('status') == 'ok')}")


if __name__ == "__main__":
    main()
