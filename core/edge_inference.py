"""Lightweight Edge ONNX Inference Engine for ReturnShield AI.

Provides a drop-in replacement for DINOv2EmbeddingEngine that executes on
ONNX Runtime (CPU / CUDA / TensorRT) with pure NumPy image pre-processing.
Zero PyTorch or Torchvision imports are required at inference time, enabling
deployments on low-power edge gateways, warehouse inspection kiosks, and
mobile web-assembly environments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("returnshield.edge_inference")

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "storage" / "models"


class EdgeEmbeddingEngine:
    """High-throughput, lightweight ONNX Runtime embedding engine for DINOv2.

    Features:
    - Pure NumPy image pre-processing (Bicubic 224x224 resize, ImageNet normalization).
    - Zero runtime dependency on PyTorch / CUDA toolkit.
    - Native support for both FP32 and Quantized INT8 graphs.
    - Automatic hardware acceleration provider selection (TensorRT -> CUDA -> CPU).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_int8: bool = True,
        precision: str = "int8",
        model_name: str = "dinov2_vits14",
        providers: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        auto_export: bool = True,
    ):
        """Initializes the Edge ONNX Embedding Engine.

        Args:
            model_path: Explicit path to the `.onnx` model. If None, resolves from default storage.
            use_int8: If True and precision is 'int8', resolves the INT8 quantized model.
            precision: Precision target ('int8', 'fp8', or 'fp32'). Defaults to 'int8'.
            model_name: Model base name (default: `dinov2_vits14`).
            providers: List of ONNX execution providers to prioritize.
            num_threads: Intra-op thread count for CPU execution.
            auto_export: If True, exports and quantizes from PyTorch if model does not exist.
        """
        import onnxruntime as ort

        self.model_name = model_name
        self.precision = precision.lower()
        if not use_int8 and precision == "int8":
            self.precision = "fp32"
        self.use_int8 = (self.precision == "int8")
        self.auto_export = auto_export

        if model_path is not None:
            self.model_path = Path(model_path)
            if "_fp8" in self.model_path.name:
                self.precision = "fp8"
            elif "_int8" in self.model_path.name:
                self.precision = "int8"
            else:
                self.precision = "fp32"
        else:
            if self.precision == "fp8":
                suffix = "_fp8.onnx"
            elif self.precision == "fp32":
                suffix = ".onnx"
            else:
                suffix = "_int8.onnx"
            self.model_path = DEFAULT_MODELS_DIR / f"{model_name}{suffix}"

        # Check model existence; auto-export if requested
        if not self.model_path.exists():
            if self.auto_export:
                logger.info("Edge ONNX model not found at %s. Triggering auto-export...", self.model_path)
                self._run_auto_export()
            else:
                raise FileNotFoundError(
                    f"Edge model not found at {self.model_path}. "
                    "Run 'python scripts/export_quantized_onnx.py' or set auto_export=True."
                )

        # Configure Execution Providers
        available_providers = ort.get_available_providers()
        if providers is None:
            desired_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            self.providers = [p for p in desired_providers if p in available_providers]
        else:
            self.providers = [p for p in providers if p in available_providers]

        if not self.providers:
            self.providers = ["CPUExecutionProvider"]

        # Configure Session Options
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads is not None and num_threads > 0:
            opts.intra_op_num_threads = num_threads
        elif "CPUExecutionProvider" in self.providers:
            # Use sensible thread allocation for edge cores
            cores = max(1, os.cpu_count() or 4)
            opts.intra_op_num_threads = min(4, cores)

        logger.info(
            "Initializing ONNX InferenceSession for %s (size: %.2f MB) with providers %s",
            self.model_path.name,
            self.model_path.stat().st_size / (1024 * 1024),
            self.providers,
        )
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=self.providers)

        # Resolve model input and output metadata
        self.input_name = self.session.get_inputs()[0].name
        output_names = [out.name for out in self.session.get_outputs()]
        self.cls_output_name = output_names[0]
        self.patch_output_name = output_names[1] if len(output_names) > 1 else output_names[0]

        # Standard ImageNet statistics for pure NumPy pre-processing
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _run_auto_export(self) -> None:
        """Invokes quantization toolchain to generate ONNX FP32, INT8, or FP8 models."""
        from core.quantization import (
            export_dinov2_to_onnx,
            quantize_onnx_model_fp8,
            quantize_onnx_model_int8,
        )

        fp32_path = DEFAULT_MODELS_DIR / f"{self.model_name}.onnx"
        if not fp32_path.exists():
            export_dinov2_to_onnx(output_path=fp32_path, model_name=self.model_name)

        if not self.model_path.exists():
            if self.precision == "fp8":
                quantize_onnx_model_fp8(input_path=fp32_path, output_path=self.model_path)
            elif self.precision == "int8":
                quantize_onnx_model_int8(input_path=fp32_path, output_path=self.model_path)

    def preprocess_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Preprocesses input image to normalized NCHW (1, 3, 224, 224) float32 tensor using NumPy.

        Does NOT depend on torch or torchvision.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Input image not found: {path}")
            pil_img = Image.open(path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input, mode="L").convert("RGB")
            elif image_input.shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 1. Resize to 224x224 with Bicubic resampling (matching PyTorch torchvision DINOv2 transform)
        resized_pil = pil_img.resize((224, 224), resample=Image.BICUBIC)

        # 2. Convert to float32 [0.0, 1.0]
        img_np = np.array(resized_pil, dtype=np.float32) / 255.0  # shape: (224, 224, 3)

        # 3. Standardize by ImageNet mean and std
        img_norm = (img_np - self.mean) / self.std

        # 4. Transpose HWC (224, 224, 3) -> CHW (3, 224, 224)
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # 5. Add batch dimension: NCHW (1, 3, 224, 224)
        return np.expand_dims(img_chw, axis=0).astype(np.float32)

    def extract_features(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts L2-normalized global embedding and patch token embeddings.

        Matches the signature and return types of DINOv2EmbeddingEngine.extract_features:
            global_embedding: 1D np.ndarray float32 of shape (384,)
            patch_embeddings: 2D np.ndarray float32 of shape (256, 384)
        """
        tensor_nchw = self.preprocess_image(image_input)

        # Run inference session
        outputs = self.session.run(None, {self.input_name: tensor_nchw})

        raw_cls = outputs[0]  # shape: (1, 384)
        raw_patches = outputs[1]  # shape: (1, 256, 384)

        # Ensure global CLS is strictly L2-normalized unit vector
        cls_1d = raw_cls.squeeze(0).astype(np.float32)
        norm = np.linalg.norm(cls_1d)
        if norm > 1e-9:
            cls_1d = cls_1d / norm

        patches_2d = raw_patches.squeeze(0).astype(np.float32)

        return cls_1d, patches_2d

    @property
    def model_size_mb(self) -> float:
        """Returns the on-disk file size of the active ONNX model in Megabytes."""
        return self.model_path.stat().st_size / (1024 * 1024)

