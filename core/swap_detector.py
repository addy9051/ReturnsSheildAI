"""Identity and Swap Matcher Engine for ReturnShield AI.

Implements Stage 2 visual fraud detection:
- Extracts DINOv2 global feature vectors and dense patch tokens.
- Computes cosine similarity against catalog reference embeddings.
- Computes patch-level distance matrix and spatial anomaly heatmap.
- Categorizes returned unit as VERIFIED_OEM, SUSPECTED_SWAP, or CONFIRMED_SWAP.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image

from core.catalog_manager import CatalogManager, DINOv2EmbeddingEngine
from core.schemas import IdentityMatchStatus, SwapDetectionResult

logger = logging.getLogger("returnshield.swap_detector")


class SwapDetector:
    """Detects model, colorway, and generation swaps using DINOv2 visual representations."""

    def __init__(
        self,
        catalog_manager: Optional[CatalogManager] = None,
        embedding_engine: Optional[Any] = None,
        high_threshold: float = 0.88,
        low_threshold: float = 0.70,
        patch_anomaly_threshold: float = 0.45,
        heatmaps_dir: Optional[Union[str, Path]] = None,
    ):
        """Initializes the SwapDetector with thresholds and embedding engine.

        Args:
            catalog_manager: CatalogManager instance containing cached SKU embeddings.
            embedding_engine: Optional DINOv2EmbeddingEngine or EdgeEmbeddingEngine instance.
            high_threshold: Minimum cosine similarity required for instant OEM verification (>= 0.88).
            low_threshold: Maximum cosine similarity below which a return is confirmed fraud (< 0.70).
            patch_anomaly_threshold: Peak patch distance score triggering localized tamper flag.
            heatmaps_dir: Directory where dispute-evidence heatmap overlays are saved.
        """
        self.catalog_manager = catalog_manager or CatalogManager()
        self.embedding_engine = embedding_engine or self.catalog_manager.embedding_engine
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.patch_anomaly_threshold = patch_anomaly_threshold

        base_storage = self.catalog_manager.storage_dir
        self.heatmaps_dir = Path(heatmaps_dir or base_storage / "heatmaps")
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_rgb_bgr(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[Image.Image, np.ndarray]:
        """Converts input into both PIL RGB (for DINOv2) and BGR numpy (for OpenCV overlays)."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Chassis image not found: {path}")
            pil_rgb = Image.open(path).convert("RGB")
            cv_bgr = cv2.imread(str(path))
        elif isinstance(image_input, Image.Image):
            pil_rgb = image_input.convert("RGB")
            cv_bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            if image_input.ndim == 2:
                cv_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                pil_rgb = Image.fromarray(image_input, mode="L").convert("RGB")
            elif image_input.shape[2] == 4:
                cv_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
                pil_rgb = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB))
            else:
                cv_bgr = image_input.copy()
                pil_rgb = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        return pil_rgb, cv_bgr

    def compute_patch_anomaly_map(
        self,
        query_patches: np.ndarray,
        reference_patches: np.ndarray,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, float]:
        """Computes dense patch-level anomaly grid and smooths it to original image dimensions.

        Args:
            query_patches: (N, D) float32 patch embeddings from query image.
            reference_patches: (M, D) float32 patch embeddings from catalog reference.
            target_size: (width, height) of destination image for overlay.

        Returns:
            Tuple of (upsampled_heatmap_float32 [0.0, 1.0], peak_anomaly_score).
        """
        # Normalize patches along embedding dimension
        q_norm = query_patches / (np.linalg.norm(query_patches, axis=-1, keepdims=True) + 1e-8)
        r_norm = reference_patches / (np.linalg.norm(reference_patches, axis=-1, keepdims=True) + 1e-8)

        # Pairwise cosine similarity matrix: (N_query, M_ref)
        sim_matrix = np.matmul(q_norm, r_norm.T)

        # Nearest-neighbor distance per patch: 1.0 - max(similarity)
        max_similarity = np.max(sim_matrix, axis=1)
        anomaly_vector = np.clip(1.0 - max_similarity, 0.0, 1.0)

        # Reshape to 2D spatial grid (e.g. 224/14 = 16x16)
        grid_dim = int(np.sqrt(anomaly_vector.shape[0]))
        if grid_dim * grid_dim != anomaly_vector.shape[0]:
            grid_dim = 16
            anomaly_vector = anomaly_vector[:256]
        grid = anomaly_vector.reshape(grid_dim, grid_dim).astype(np.float32)

        # Spatial Gaussian filter to remove single-patch high-frequency noise
        smoothed_grid = cv2.GaussianBlur(grid, (3, 3), sigmaX=1.0)

        # Resize to original image dimensions using bicubic interpolation
        w, h = target_size
        heatmap_resized = cv2.resize(smoothed_grid, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_clipped = np.clip(heatmap_resized, 0.0, 1.0)

        peak_score = float(np.max(smoothed_grid))
        return heatmap_clipped, peak_score

    def render_and_save_heatmap(
        self,
        original_bgr: np.ndarray,
        anomaly_map: np.ndarray,
        destination_path: Path,
    ) -> str:
        """Renders an OpenCV JET colormap overlay on the returned chassis and persists to disk."""
        h, w = original_bgr.shape[:2]
        if anomaly_map.shape[:2] != (h, w):
            anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_CUBIC)

        # Scale float [0, 1] to uint8 [0, 255]
        heatmap_uint8 = np.uint8(np.clip(anomaly_map, 0.0, 1.0) * 255.0)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend original chassis image (65%) with heatmap overlay (35%)
        blended = cv2.addWeighted(original_bgr, 0.65, heatmap_color, 0.35, 0)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination_path), blended)
        return str(destination_path)

    def detect_swap(
        self,
        query_image: Union[str, Path, Image.Image, np.ndarray],
        sku_id: Optional[str] = None,
        reference_image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        save_heatmap: bool = True,
        custom_heatmap_path: Optional[Union[str, Path]] = None,
    ) -> SwapDetectionResult:
        """Evaluates a return chassis crop against catalog reference to detect bait-and-switch swaps.

        Args:
            query_image: Return chassis image (path, PIL, or ndarray).
            sku_id: Target SKU identifier to look up in catalog cache.
            reference_image: Optional explicit reference photo if not using SKU catalog.
            save_heatmap: If True, writes the visual anomaly overlay to disk for dispute packets.
            custom_heatmap_path: Optional specific path to write the heatmap artifact.

        Returns:
            Strongly typed SwapDetectionResult with identity status, similarity, and heatmap path.
        """
        pil_query, cv_bgr_query = self._load_image_rgb_bgr(query_image)
        orig_w, orig_h = pil_query.size

        # 1. Retrieve reference representations
        ref_global: Optional[np.ndarray] = None
        ref_patches: Optional[np.ndarray] = None
        effective_threshold = self.high_threshold

        if sku_id:
            profile = self.catalog_manager.get_sku(sku_id)
            if profile:
                effective_threshold = profile.similarity_threshold
            ref_global = self.catalog_manager.get_sku_embedding(sku_id)
            ref_patches = self.catalog_manager.get_sku_patch_embeddings(sku_id)

        if ref_global is None and reference_image is not None:
            pil_ref, _ = self._load_image_rgb_bgr(reference_image)
            ref_global, ref_patches = self.embedding_engine.extract_features(pil_ref)

        if ref_global is None:
            raise KeyError(
                f"No catalog reference or cached DINOv2 embedding found for SKU '{sku_id}'. "
                "Register SKU reference images first."
            )

        # 2. Extract query features
        query_global, query_patches = self.embedding_engine.extract_features(pil_query)

        # 3. Global CLS cosine similarity (unit vectors dot product)
        cosine_sim = float(np.dot(ref_global, query_global))
        cosine_sim = max(-1.0, min(1.0, cosine_sim))

        # 4. Dense patch anomaly map computation
        spatial_diff_score = 0.0
        heatmap_path_str: Optional[str] = None

        if ref_patches is not None and len(ref_patches) > 0:
            anomaly_map, spatial_diff_score = self.compute_patch_anomaly_map(
                query_patches=query_patches,
                reference_patches=ref_patches,
                target_size=(orig_w, orig_h),
            )

            if save_heatmap:
                if custom_heatmap_path:
                    dest_file = Path(custom_heatmap_path)
                else:
                    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    dest_file = self.heatmaps_dir / f"diff_{sku_id or 'sku'}_{stamp}_{uuid.uuid4().hex[:6]}.png"
                heatmap_path_str = self.render_and_save_heatmap(cv_bgr_query, anomaly_map, dest_file)

        # 5. Multi-tiered triage decision logic
        if cosine_sim >= effective_threshold and spatial_diff_score <= self.patch_anomaly_threshold:
            status = IdentityMatchStatus.VERIFIED_OEM
            is_swap = False
            explanation = (
                f"Authentic OEM match confirmed (Cosine: {cosine_sim:.3f} >= {effective_threshold:.2f}, "
                f"Peak patch anomaly: {spatial_diff_score:.3f})."
            )
        elif cosine_sim < self.low_threshold:
            status = IdentityMatchStatus.CONFIRMED_SWAP
            is_swap = True
            explanation = (
                f"Bait-and-Switch Fraud Detected: Visual identity severely diverges from catalog reference "
                f"(Cosine: {cosine_sim:.3f} < {self.low_threshold:.2f}). Probable model, generation, or colorway swap."
            )
        else:
            status = IdentityMatchStatus.SUSPECTED_SWAP
            is_swap = True
            explanation = (
                f"Suspected Swap or Localized Tamper: Visual similarity in borderline zone "
                f"(Cosine: {cosine_sim:.3f}, Peak patch anomaly: {spatial_diff_score:.3f}). "
                "Secondary review required with spatial heatmap inspection."
            )

        logger.info(
            "Swap detection for SKU '%s': %s (Cosine=%.3f, PatchPeak=%.3f)",
            sku_id, status.value, cosine_sim, spatial_diff_score
        )

        return SwapDetectionResult(
            identity_status=status,
            cosine_similarity=round(cosine_sim, 4),
            threshold=effective_threshold,
            is_swap_detected=is_swap,
            explanation=explanation,
            spatial_diff_score=round(spatial_diff_score, 4),
            heatmap_path=heatmap_path_str,
        )

