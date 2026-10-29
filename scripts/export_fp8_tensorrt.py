#!/usr/bin/env python3
"""NVIDIA GPU TensorRT and FP8 (E4M3) Export Tool for ReturnShield AI.

Exports DINOv2 vision models to FP8 ONNX (Opset 19+, QFLOAT8E4M3FN) for
NVIDIA Ada Lovelace (RTX 4090 / L40 / L4), Hopper (H100 / H800), and
Blackwell (B200) Tensor Cores, and generates TensorRT compilation recipes.

Usage:
    python scripts/export_fp8_tensorrt.py
    python scripts/export_fp8_tensorrt.py --precision fp8 --verify
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import onnx
from core.quantization import (
    DEFAULT_MODELS_DIR,
    export_dinov2_to_onnx,
    generate_tensorrt_build_command,
    quantize_onnx_model_fp8,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_fp8_tensorrt")


def inspect_fp8_onnx_graph(model_path: Path) -> dict:
    """Inspects ONNX graph to verify FP8 initializers and operator coverage."""
    model = onnx.load(str(model_path))
    graph = model.graph

    num_nodes = len(graph.node)
    num_initializers = len(graph.initializer)

    # Count FP8 initializers (FLOAT8E4M3FN has enum value 17 in ONNX)
    fp8_type_val = getattr(onnx.TensorProto, "FLOAT8E4M3FN", 17)
    fp8_init_count = sum(1 for init in graph.initializer if init.data_type == fp8_type_val)

    op_types = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    return {
        "opset_version": model.opset_import[0].version if model.opset_import else None,
        "total_nodes": num_nodes,
        "total_initializers": num_initializers,
        "fp8_initializers": fp8_init_count,
        "dequantize_nodes": op_types.get("DequantizeLinear", 0),
        "quantize_nodes": op_types.get("QuantizeLinear", 0),
        "matmul_nodes": op_types.get("MatMul", 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export DINOv2 to FP8 ONNX and generate NVIDIA TensorRT compilation profiles."
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
        "--precision",
        type=str,
        choices=["fp8", "fp16", "int8"],
        default="fp8",
        help="Target TensorRT precision mode (default: fp8)",
    )
    parser.add_argument(
        "--batch-profile",
        type=str,
        default="1,4,16",
        help="Min,Opt,Max batch sizes (default: 1,4,16)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify exported ONNX graph schema and FP8 initializers (default: True)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / f"{args.model_name}.onnx"
    fp8_path = output_dir / f"{args.model_name}_fp8.onnx"
    engine_path = output_dir / f"{args.model_name}_{args.precision}.engine"

    batches = [int(b.strip()) for b in args.batch_profile.split(",")]
    if len(batches) != 3:
        batches = [1, 4, 16]

    print("=" * 75)
    print("ReturnShield AI - NVIDIA GPU FP8 & TensorRT Acceleration Compiler")
    print("=" * 75)
    print(f"Model:          {args.model_name}")
    print(f"Target Precision: {args.precision.upper()} (E4M3 for Forward Vision Inference)")
    print(f"Batch Profile:    min={batches[0]}, opt={batches[1]}, max={batches[2]}")
    print("-" * 75)

    # 1. Ensure FP32 ONNX Base Model Exists
    if not fp32_path.exists():
        print("[*] Generating base FP32 ONNX model...")
        export_dinov2_to_onnx(output_path=fp32_path, model_name=args.model_name)

    fp32_size = fp32_path.stat().st_size / (1024 * 1024)

    # 2. Export / Quantize to FP8 (QFLOAT8E4M3FN)
    print("[*] Compiling FP8 dynamic quantization graph (Opset 19+)...")
    t0 = time.perf_counter()
    quantize_onnx_model_fp8(input_path=fp32_path, output_path=fp8_path)
    fp8_time = time.perf_counter() - t0
    fp8_size = fp8_path.stat().st_size / (1024 * 1024)
    compression = fp32_size / max(fp8_size, 1e-6)

    print(f"[OK] FP8 Graph Compiled in {fp8_time:.2f}s -> {fp8_path.name}")
    print(f"     FP32 Size:    {fp32_size:.2f} MB")
    print(f"     FP8 Size:     {fp8_size:.2f} MB ({compression:.2f}x compression)")

    # 3. Verify Graph Structure
    if args.verify:
        print("-" * 75)
        print("Auditing FP8 ONNX Graph Architecture...")
        graph_stats = inspect_fp8_onnx_graph(fp8_path)
        print(f"  ONNX Opset:              {graph_stats['opset_version']}")
        print(f"  Total Computation Nodes: {graph_stats['total_nodes']}")
        print(f"  Total Initializers:      {graph_stats['total_initializers']}")
        print(f"  FP8 (E4M3) Tensors:      {graph_stats['fp8_initializers']}")
        print(f"  DequantizeLinear Nodes:  {graph_stats['dequantize_nodes']}")
        print(f"  MatMul Blocks:           {graph_stats['matmul_nodes']}")

    # 4. Generate TensorRT Engine Command & Script
    print("-" * 75)
    print("Generating Production NVIDIA TensorRT Compilation Spec...")
    trt_spec = generate_tensorrt_build_command(
        onnx_path=fp8_path if args.precision == "fp8" else fp32_path,
        engine_path=engine_path,
        precision=args.precision,
        min_batch=batches[0],
        opt_batch=batches[1],
        max_batch=batches[2],
    )

    print(f"CLI Command:  {trt_spec['command']}")
    print(f"Target GPU:   {trt_spec['recommended_hardware'][args.precision]}")

    # Write Windows .bat and Linux .sh builder scripts
    bat_script = output_dir / f"build_tensorrt_{args.precision}.bat"
    sh_script = output_dir / f"build_tensorrt_{args.precision}.sh"

    with open(bat_script, "w") as f:
        f.write("@echo off\n")
        f.write("rem ReturnShield AI - TensorRT Engine Compilation Script\n")
        f.write(f"echo Building TensorRT {args.precision.upper()} Engine from {fp8_path.name}...\n")
        f.write(f"{trt_spec['command']}\n")
        f.write("echo TensorRT build complete.\n")

    with open(sh_script, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# ReturnShield AI - TensorRT Engine Compilation Script\n")
        f.write(f"echo 'Building TensorRT {args.precision.upper()} Engine from {fp8_path.name}...'\n")
        f.write(f"{trt_spec['command']}\n")
        f.write("echo 'TensorRT build complete.'\n")

    print(f"[OK] Generated Windows batch builder: {bat_script.name}")
    print(f"[OK] Generated Linux shell builder:   {sh_script.name}")

    print("=" * 75)
    print("Summary of Acceleration Tiers:")
    print("  Tier 1 (Edge / Courier Phones / Dock PCs): INT8 Dynamic (AVX-512 / ARM NEON / Wasm)")
    print("  Tier 2 (NVIDIA GPU Hubs: Ada / Hopper / Blackwell): FP8 E4M3 + TensorRT Plan")
    print("=" * 75)


if __name__ == "__main__":
    main()

