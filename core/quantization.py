"""Model Quantization and Edge ONNX Toolchain for ReturnShield AI.

Provides utilities to:
1. Export PyTorch DINOv2 vision transformer to ONNX graph with dual outputs:
   - L2-normalized global CLS embedding: (B, 384)
   - Dense spatial patch tokens: (B, 256, 384)
2. Quantize the ONNX model to INT8 dynamic quantization (QInt8).
3. Mathematically verify numerical parity between PyTorch FP32 and ONNX FP32/INT8 representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("returnshield.quantization")

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "storage" / "models"


class DINOv2ExportWrapper:
    """Wraps DINOv2 PyTorch model for clean, deterministic ONNX graph tracing."""

    def __init__(self, dinov2_model: Any):
        import torch.nn as nn

        super().__init__()
        self.m = dinov2_model

    def __call__(self, x: Any) -> Tuple[Any, Any]:
        return self.forward(x)

    def forward(self, x: Any) -> Tuple[Any, Any]:
        """Forward pass extracting normalized CLS token and spatial patch tokens."""
        import torch

        features = self.m.forward_features(x)
        
        # 1. Global CLS token
        if isinstance(features, dict) and "x_norm_clstoken" in features:
            cls_token = features["x_norm_clstoken"]
        elif hasattr(features, "x_norm_clstoken"):
            cls_token = features.x_norm_clstoken
        else:
            cls_token = self.m(x)

        # L2-normalization for unit hypersphere cosine comparison
        cls_norm = cls_token / torch.norm(cls_token, dim=-1, keepdim=True)

        # 2. Patch tokens
        if isinstance(features, dict) and "x_norm_patchtokens" in features:
            patch_tokens = features["x_norm_patchtokens"]
        elif hasattr(features, "x_norm_patchtokens"):
            patch_tokens = features.x_norm_patchtokens
        else:
            patch_tokens = features

        return cls_norm, patch_tokens


def export_dinov2_to_onnx(
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "dinov2_vits14",
    opset_version: int = 17,
    device: str = "cpu",
) -> Path:
    """Exports DINOv2 from PyTorch to standard ONNX FP32 format.

    Args:
        output_path: Destination path for the `.onnx` file.
        model_name: Torch Hub model identifier (default: `dinov2_vits14`).
        opset_version: Target ONNX opset (default: 17).
        device: Device to use during export tracing (`cpu` or `cuda`).

    Returns:
        Path to the exported ONNX model.
    """
    import torch
    import torch.nn as nn
    from core.catalog_manager import DINOv2EmbeddingEngine

    output_path = Path(output_path or DEFAULT_MODELS_DIR / f"{model_name}.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing PyTorch %s engine for ONNX export on %s...", model_name, device)
    engine = DINOv2EmbeddingEngine(model_name=model_name, device=device)
    raw_model = engine.model

    class _ExportModule(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            feats = self.m.forward_features(x)
            cls_token = feats["x_norm_clstoken"]
            cls_norm = cls_token / torch.norm(cls_token, dim=-1, keepdim=True)
            patch_tokens = feats["x_norm_patchtokens"]
            return cls_norm, patch_tokens

    wrapper = _ExportModule(raw_model)
    wrapper.to(device)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    logger.info("Tracing ONNX graph to %s (opset %d)...", output_path, opset_version)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["global_cls", "patch_tokens"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "global_cls": {0: "batch_size"},
                "patch_tokens": {0: "batch_size"},
            },
        )

    logger.info(
        "Successfully exported %s to ONNX (size: %.2f MB)",
        model_name,
        output_path.stat().st_size / (1024 * 1024),
    )
    return output_path


def quantize_onnx_model_int8(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    per_channel: bool = True,
    reduce_range: bool = False,
) -> Path:
    """Quantizes an ONNX FP32 model to INT8 using dynamic activation & weight quantization.

    Args:
        input_path: Path to the FP32 ONNX model.
        output_path: Target path for the INT8 ONNX model.
        per_channel: Quantize weights per-channel for higher fidelity.
        reduce_range: Reduce quant range (default False for standard VNNI/ARM).

    Returns:
        Path to the quantized INT8 model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_int8.onnx"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Executing INT8 dynamic quantization on %s -> %s...", input_path.name, output_path.name)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )

    fp32_size_mb = input_path.stat().st_size / (1024 * 1024)
    int8_size_mb = output_path.stat().st_size / (1024 * 1024)
    compression = fp32_size_mb / max(int8_size_mb, 1e-6)

    logger.info(
        "INT8 quantization complete: %.2f MB -> %.2f MB (%.2fx compression)",
        fp32_size_mb,
        int8_size_mb,
        compression,
    )
    return output_path


def quantize_onnx_model_fp8(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    per_channel: bool = True,
) -> Path:
    """Quantizes an ONNX FP32 model to FP8 (QFLOAT8E4M3FN) for NVIDIA Ada/Hopper/Blackwell Tensor Cores.

    Args:
        input_path: Path to the FP32 ONNX model.
        output_path: Target path for the FP8 ONNX model.
        per_channel: Quantize weights per-channel for higher fidelity.

    Returns:
        Path to the quantized FP8 model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fp8.onnx"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Executing FP8 (E4M3) dynamic quantization on %s -> %s...", input_path.name, output_path.name)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QFLOAT8E4M3FN,
        per_channel=per_channel,
    )

    fp32_size_mb = input_path.stat().st_size / (1024 * 1024)
    fp8_size_mb = output_path.stat().st_size / (1024 * 1024)
    compression = fp32_size_mb / max(fp8_size_mb, 1e-6)

    logger.info(
        "FP8 quantization complete: %.2f MB -> %.2f MB (%.2fx compression)",
        fp32_size_mb,
        fp8_size_mb,
        compression,
    )
    return output_path


def generate_tensorrt_build_command(
    onnx_path: Union[str, Path],
    engine_path: Optional[Union[str, Path]] = None,
    precision: str = "fp8",
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 16,
) -> Dict[str, Any]:
    """Generates the production NVIDIA TensorRT (trtexec) compilation command and profile spec.

    Args:
        onnx_path: Path to the ONNX model (FP32 or FP8).
        engine_path: Destination path for the `.engine` / `.plan` binary.
        precision: Precision target ('fp8', 'fp16', or 'best').
        min_batch: Minimum batch size profile.
        opt_batch: Optimal batch size profile.
        max_batch: Maximum batch size profile.

    Returns:
        Dict with trtexec CLI command, profile definitions, and target architecture notes.
    """
    onnx_p = Path(onnx_path)
    if engine_path is None:
        engine_p = onnx_p.with_suffix(".engine")
    else:
        engine_p = Path(engine_path)

    # trtexec flags
    flags = [
        "trtexec",
        f"--onnx={onnx_p}",
        f"--saveEngine={engine_p}",
        f"--minShapes=input:{min_batch}x3x224x224",
        f"--optShapes=input:{opt_batch}x3x224x224",
        f"--maxShapes=input:{max_batch}x3x224x224",
    ]

    if precision == "fp8":
        flags.extend(["--fp8", "--fp16"])
    elif precision == "fp16":
        flags.append("--fp16")
    elif precision == "int8":
        flags.append("--int8")

    cli_command = " ".join(flags)

    return {
        "command": cli_command,
        "onnx_path": str(onnx_p),
        "engine_path": str(engine_p),
        "precision": precision,
        "min_shapes": f"input:{min_batch}x3x224x224",
        "opt_shapes": f"input:{opt_batch}x3x224x224",
        "max_shapes": f"input:{max_batch}x3x224x224",
        "recommended_hardware": {
            "fp8": "NVIDIA Ada Lovelace (RTX 4090 / L40 / L4) or Hopper (H100 / H800 / B200)",
            "fp16": "NVIDIA Turing / Ampere (T4 / A100 / RTX 30-series)",
            "int8": "Universal (x86 AVX-512 VNNI, ARM NEON, NVIDIA Tensor Cores)",
        },
    }


def verify_numerical_parity(
    fp32_output: Tuple[np.ndarray, np.ndarray],
    quantized_output: Tuple[np.ndarray, np.ndarray],
    min_cls_cosine: float = 0.98,
    min_patch_cosine: float = 0.95,
    max_patch_mae: float = 0.60,
) -> Dict[str, Any]:
    """Evaluates mathematical parity between baseline FP32 and quantized INT8 embeddings.

    Args:
        fp32_output: (global_cls, patch_tokens) from FP32 engine.
        quantized_output: (global_cls, patch_tokens) from INT8 engine.
        min_cls_cosine: Minimum acceptable cosine similarity for global vector (default: 0.98).
        min_patch_cosine: Minimum acceptable cosine similarity for spatial patch tokens (default: 0.95).
        max_patch_mae: Maximum acceptable Mean Absolute Error for unnormalized patch tokens (default: 0.60).

    Returns:
        Dict containing parity metrics and boolean verification status.
    """
    cls_fp32, patch_fp32 = fp32_output
    cls_q, patch_q = quantized_output

    # Flatten to 1D vectors
    cls_fp32_flat = cls_fp32.reshape(-1)
    cls_q_flat = cls_q.reshape(-1)

    # 1. Cosine similarity
    norm_fp32 = np.linalg.norm(cls_fp32_flat)
    norm_q = np.linalg.norm(cls_q_flat)
    if norm_fp32 > 1e-9 and norm_q > 1e-9:
        cls_cosine = float(np.dot(cls_fp32_flat, cls_q_flat) / (norm_fp32 * norm_q))
    else:
        cls_cosine = 0.0

    # 2. Patch tokens Mean Absolute Error and Cosine
    patch_fp32_flat = patch_fp32.reshape(-1)
    patch_q_flat = patch_q.reshape(-1)
    patch_mae = float(np.mean(np.abs(patch_fp32_flat - patch_q_flat)))

    norm_p_fp32 = np.linalg.norm(patch_fp32_flat)
    norm_p_q = np.linalg.norm(patch_q_flat)
    if norm_p_fp32 > 1e-9 and norm_p_q > 1e-9:
        patch_cosine = float(np.dot(patch_fp32_flat, patch_q_flat) / (norm_p_fp32 * norm_p_q))
    else:
        patch_cosine = 0.0

    passed = (cls_cosine >= min_cls_cosine) and (patch_cosine >= min_patch_cosine) and (patch_mae <= max_patch_mae)

    metrics = {
        "cls_cosine_similarity": round(cls_cosine, 5),
        "patch_cosine_similarity": round(patch_cosine, 5),
        "patch_mean_absolute_error": round(patch_mae, 5),
        "min_cls_cosine_threshold": min_cls_cosine,
        "min_patch_cosine_threshold": min_patch_cosine,
        "max_patch_mae_threshold": max_patch_mae,
        "parity_passed": passed,
    }
    logger.info("Numerical parity check: %s (CLS Cosine: %.4f, Patch Cosine: %.4f, Patch MAE: %.4f)",
                "PASSED" if passed else "FAILED", cls_cosine, patch_cosine, patch_mae)
    return metrics

