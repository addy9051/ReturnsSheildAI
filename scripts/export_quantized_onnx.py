#!/usr/bin/env python3
"""CLI utility to export and quantize DINOv2 vision models to ONNX FP32 and INT8.

Usage:
    python scripts/export_quantized_onnx.py
    python scripts/export_quantized_onnx.py --model-name dinov2_vits14 --verify
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np

from core.quantization import (
    DEFAULT_MODELS_DIR,
    export_dinov2_to_onnx,
    quantize_onnx_model_int8,
    verify_numerical_parity,
)
from core.edge_inference import EdgeEmbeddingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_quantized_onnx")


def main():
    parser = argparse.ArgumentParser(
        description="Export and quantize ReturnShield AI DINOv2 vision models to ONNX FP32 and INT8."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vits14",
        help="Model identifier (default: dinov2_vits14)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help=f"Target directory for exported models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify numerical parity against PyTorch baseline (default: True)",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Skip INT8 quantization step",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / f"{args.model_name}.onnx"
    int8_path = output_dir / f"{args.model_name}_int8.onnx"

    print("=" * 70)
    print("ReturnShield AI - Model Quantization & Edge ONNX Compiler")
    print("=" * 70)
    print(f"Model:      {args.model_name}")
    print(f"Output Dir: {output_dir}")
    print(f"ONNX Opset: {args.opset}")
    print("-" * 70)

    # 1. Export FP32 ONNX
    t0 = time.perf_counter()
    export_dinov2_to_onnx(
        output_path=fp32_path,
        model_name=args.model_name,
        opset_version=args.opset,
    )
    fp32_time = time.perf_counter() - t0
    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    print(f"[OK] FP32 Export Complete in {fp32_time:.2f}s -> {fp32_path.name} ({fp32_size:.2f} MB)")

    # 2. Quantize to INT8
    if not args.no_int8:
        t0 = time.perf_counter()
        quantize_onnx_model_int8(
            input_path=fp32_path,
            output_path=int8_path,
            per_channel=True,
        )
        int8_time = time.perf_counter() - t0
        int8_size = int8_path.stat().st_size / (1024 * 1024)
        compression = fp32_size / max(int8_size, 1e-6)
        print(f"[OK] INT8 Quantization Complete in {int8_time:.2f}s -> {int8_path.name} ({int8_size:.2f} MB, {compression:.2f}x compression)")

    # 3. Verify Parity
    if args.verify and not args.no_int8:
        print("-" * 70)
        print("Running Mathematical Parity Audit against PyTorch Baseline...")
        from core.catalog_manager import DINOv2EmbeddingEngine

        # Find sample image or create synthetic test pattern
        sample_img_paths = (
            list(PROJECT_ROOT.glob("storage/catalog_images/*/*.jpg"))
            + list(PROJECT_ROOT.glob("storage/catalog_images/*/*.png"))
            + list(PROJECT_ROOT.glob("storage/real_catalog_images/*/*.png"))
        )
        if sample_img_paths:
            test_img = Image.open(sample_img_paths[0]).convert("RGB")
            test_source = str(sample_img_paths[0].relative_to(PROJECT_ROOT))
        else:
            # Generate deterministic procedural gradient test image
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            arr[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)
            arr[:, :, 1] = np.linspace(255, 0, 224, dtype=np.uint8)[:, None]
            arr[:, :, 2] = 128
            test_img = Image.fromarray(arr)
            test_source = "synthetic procedural pattern"

        print(f"Audit Image: {test_source}")

        # PyTorch Baseline
        engine_torch = DINOv2EmbeddingEngine(device="cpu")
        torch_cls, torch_patches = engine_torch.extract_features(test_img)

        # ONNX INT8 Engine
        engine_int8 = EdgeEmbeddingEngine(model_path=int8_path, use_int8=True, auto_export=False)
        int8_cls, int8_patches = engine_int8.extract_features(test_img)

        parity = verify_numerical_parity(
            fp32_output=(torch_cls, torch_patches),
            quantized_output=(int8_cls, int8_patches),
            min_cls_cosine=0.98,
            min_patch_cosine=0.95,
            max_patch_mae=0.60,
        )

        print(f"CLS Global Cosine Similarity: {parity['cls_cosine_similarity']:.5f} (Threshold >= {parity['min_cls_cosine_threshold']})")
        print(f"Patch Token Cosine:           {parity['patch_cosine_similarity']:.5f} (Threshold >= {parity['min_patch_cosine_threshold']})")
        print(f"Patch Mean Absolute Error:    {parity['patch_mean_absolute_error']:.5f} (Threshold <= {parity['max_patch_mae_threshold']})")
        print(f"Parity Verdict:               {'PASSED' if parity['parity_passed'] else 'FAILED'}")

    print("=" * 70)


if __name__ == "__main__":
    main()

