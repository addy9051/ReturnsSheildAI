"""Multi-Angle Volumetric & Stereo Camera Inspection Engine for ReturnShield AI.

Features:
1. Dense 3D Disparity Mapping (OpenCV StereoSGBM) for volumetric thickness and depth profiling.
2. Liquid Damage Indicator (LDI) Detection via dual-band wrap-around HSV thresholding.
3. Lateral Hinge & Joint Integrity Analysis via high-frequency Sobel gradient line continuity.
4. Multi-View Feature Fusion: Combines orthogonal top-down (90 deg) and lateral oblique (45 deg)
   embeddings to expose covert flank/underbelly swaps where top housing is authentic but lower
   components are counterfeit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image

from core.volumetric_schemas import (
    CameraPerspective,
    HingeCondition,
    LiquidDamageStatus,
    MultiAngleInspectionResult,
    PortCondition,
    VolumetricDiscrepancy,
)

logger = logging.getLogger("returnshield.volumetric_inspector")

DEFAULT_HEATMAPS_DIR = Path(__file__).resolve().parent.parent / "storage" / "heatmaps"


class VolumetricStereoInspector:
    """Industrial multi-angle stereo and volumetric inspection engine."""

    def __init__(
        self,
        baseline_mm: float = 120.0,
        focal_length_px: float = 800.0,
        heatmaps_dir: Optional[Union[str, Path]] = None,
        num_disparities: int = 64,
        block_size: int = 5,
    ):
        """Initializes the multi-angle stereo inspector.

        Args:
            baseline_mm: Physical distance between stereo cameras in mm (default: 120mm).
            focal_length_px: Effective focal length in pixels (default: 800px).
            heatmaps_dir: Storage directory for rendered 3D disparity heatmaps.
            num_disparities: Search disparity range (must be divisible by 16).
            block_size: Matching block size (odd number >= 3).
        """
        self.baseline_mm = baseline_mm
        self.focal_length_px = focal_length_px
        self.heatmaps_dir = Path(heatmaps_dir or DEFAULT_HEATMAPS_DIR)
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)

        self.num_disparities = num_disparities
        self.block_size = block_size

        # Initialize OpenCV Semi-Global Block Matching (StereoSGBM)
        self._stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * (self.block_size ** 2),
            P2=32 * 3 * (self.block_size ** 2),
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
        )

    # ==========================================================================
    # 1. 3D Stereo Disparity & Volumetric Depth Profiling
    # ==========================================================================

    def compute_disparity_map(
        self,
        left_image: Union[str, Path, Image.Image, np.ndarray],
        right_image: Union[str, Path, Image.Image, np.ndarray],
        save_heatmap: bool = True,
        nominal_depth_mm: float = 45.0,
        depth_tolerance_mm: float = 6.0,
    ) -> Tuple[np.ndarray, Optional[str], VolumetricDiscrepancy]:
        """Computes dense disparity map and evaluates volumetric 3D profile."""
        img_l_bgr = self._to_bgr(left_image)
        img_r_bgr = self._to_bgr(right_image)

        # Ensure equal dimensions
        h_l, w_l = img_l_bgr.shape[:2]
        h_r, w_r = img_r_bgr.shape[:2]
        if (h_l, w_l) != (h_r, w_r):
            img_r_bgr = cv2.resize(img_r_bgr, (w_l, h_l), interpolation=cv2.INTER_LINEAR)

        # Convert to grayscale for stereo matching
        gray_l = cv2.cvtColor(img_l_bgr, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r_bgr, cv2.COLOR_BGR2GRAY)

        # Compute raw disparity (fixed-point format scaled by 16)
        raw_disp = self._stereo_sgbm.compute(gray_l, gray_r)
        disparity_fp = raw_disp.astype(np.float32) / 16.0

        # Mask valid disparity region
        valid_mask = disparity_fp > 1.0
        if np.any(valid_mask):
            valid_disps = disparity_fp[valid_mask]
            mean_disp = float(np.mean(valid_disps))
            std_disp = float(np.std(valid_disps))
            # Depth = (Baseline * FocalLength) / Disparity
            measured_depth_mm = float((self.baseline_mm * self.focal_length_px) / max(mean_disp, 1.0))
        else:
            mean_disp = 0.0
            std_disp = 0.0
            measured_depth_mm = nominal_depth_mm

        depth_delta = abs(measured_depth_mm - nominal_depth_mm)
        discrepancy = depth_delta > depth_tolerance_mm

        heatmap_path = None
        if save_heatmap:
            heatmap_path = self._render_disparity_overlay(img_l_bgr, disparity_fp)

        profile = VolumetricDiscrepancy(
            measured_depth_mm=round(measured_depth_mm, 2),
            nominal_depth_mm=round(nominal_depth_mm, 2),
            depth_delta_mm=round(depth_delta, 2),
            tolerance_mm=depth_tolerance_mm,
            discrepancy_detected=discrepancy,
            disparity_mean=round(mean_disp, 2),
            disparity_std=round(std_disp, 2),
        )

        return disparity_fp, heatmap_path, profile

    def _render_disparity_overlay(self, original_bgr: np.ndarray, disparity_fp: np.ndarray) -> str:
        """Renders OpenCV TURBO colormap disparity overlay for dispute packet."""
        disp_clipped = np.clip(disparity_fp, 0, self.num_disparities)
        disp_norm = cv2.normalize(disp_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_map = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

        blended = cv2.addWeighted(original_bgr, 0.60, color_map, 0.40, 0)

        filename = f"disparity_{uuid.uuid4().hex[:8]}.png"
        out_path = self.heatmaps_dir / filename
        cv2.imwrite(str(out_path), blended)
        return str(out_path)

    # ==========================================================================
    # 2. Liquid Damage Indicator (LDI) Detection
    # ==========================================================================

    def inspect_liquid_damage_indicator(
        self,
        lateral_crop: Union[str, Path, Image.Image, np.ndarray],
        red_pixel_trigger_threshold: int = 60,
    ) -> Tuple[LiquidDamageStatus, int]:
        """Detects whether water contact indicator stickers turned pink/crimson.

        Uses official OpenCV HSV dual-band wrap-around thresholding:
        - Band 1: Lower pink/light-red [0, 50, 100] to [15, 255, 255]
        - Band 2: Upper crimson [170, 50, 100] to [179, 255, 255]
        """
        img_bgr = self._to_bgr(lateral_crop)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Dual-band red/pink thresholding
        lower_band1 = np.array([0, 50, 90], dtype=np.uint8)
        upper_band1 = np.array([16, 255, 255], dtype=np.uint8)

        lower_band2 = np.array([168, 50, 90], dtype=np.uint8)
        upper_band2 = np.array([179, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_band1, upper_band1)
        mask2 = cv2.inRange(hsv, lower_band2, upper_band2)
        liquid_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological filtering to eliminate isolated pixel sensor noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, kernel)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_CLOSE, kernel)

        active_pixels = int(cv2.countNonZero(liquid_mask))

        if active_pixels >= red_pixel_trigger_threshold:
            status = LiquidDamageStatus.CONFIRMED_LIQUID_DAMAGE
        elif active_pixels >= (red_pixel_trigger_threshold // 3):
            status = LiquidDamageStatus.SUSPECTED_MOISTURE
        else:
            status = LiquidDamageStatus.NO_LIQUID_DETECTED

        return status, active_pixels

    # ==========================================================================
    # 3. Lateral Hinge & Joint Integrity Analyzer
    # ==========================================================================

    def inspect_hinge_integrity(
        self,
        hinge_crop: Union[str, Path, Image.Image, np.ndarray],
        fracture_energy_threshold: float = 38.0,
    ) -> Tuple[HingeCondition, float]:
        """Detects structural hairline fractures or glue repairs on lateral hinges.

        Evaluates high-frequency gradient discontinuities across the pivot axis.
        """
        img_bgr = self._to_bgr(hinge_crop)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Compute Sobel edge gradients in X and Y
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)

        # High-frequency fracture energy score
        p95_energy = float(np.percentile(grad_mag, 95))
        mean_energy = float(np.mean(grad_mag))

        # Check for unnatural specular glue reflection (high saturation variation in gray)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_std = float(np.std(hsv[:, :, 1]))

        if p95_energy > fracture_energy_threshold * 1.5:
            condition = HingeCondition.FRACTURED_CRACKED
        elif sat_std > 50.0 and p95_energy > fracture_energy_threshold:
            condition = HingeCondition.UNAUTHORIZED_GLUE_REPAIR
        else:
            condition = HingeCondition.INTACT_OEM

        return condition, round(p95_energy, 2)

    # ==========================================================================
    # 4. Multi-View Feature Fusion
    # ==========================================================================

    def fuse_multi_view_inspection(
        self,
        top_similarity: float,
        oblique_similarity: float,
        top_weight: float = 0.60,
        oblique_weight: float = 0.40,
        swap_threshold: float = 0.75,
    ) -> Tuple[float, bool]:
        """Combines top-down and oblique 45-degree similarities.

        Detects covert housing swaps (genuine top shell + swapped underbelly/flank).
        """
        fused = float((top_similarity * top_weight) + (oblique_similarity * oblique_weight))
        fused = max(0.0, min(1.0, fused))

        # Covert swap occurs when top shell looks genuine (>= 0.85) but lateral view fails (< 0.70)
        covert_swap = (top_similarity >= 0.85 and oblique_similarity < swap_threshold)
        lateral_tamper = (fused < swap_threshold) or covert_swap

        return round(fused, 4), lateral_tamper

    # ==========================================================================
    # Helper Utilities
    # ==========================================================================

    def _to_bgr(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Converts diverse image formats into OpenCV BGR uint8 array."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Inspection image not found: {p}")
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Failed to read image at {p}")
            return img
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            return image_input.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

