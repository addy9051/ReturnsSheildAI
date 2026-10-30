"""Bill of Materials (BOM) Kitting Auditor Engine for ReturnShield AI.

Implements Stage 3 accessory tray inspection:
- Evaluates accessory tray ROI against the registered SKU Bill of Materials.
- Detects filled component slots vs empty recessed foam cavities using
  edge gradient density and luminance entropy.
- Calculates itemized missing accessory deductions.
- Produces strongly typed KittingAuditResult.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from core.catalog_manager import CatalogManager
from core.schemas import (
    AccessoryStatus,
    BOMItem,
    KittedItemAudit,
    KittingAuditResult,
    SKUProfile,
)

logger = logging.getLogger("returnshield.kitting_auditor")


class KittingAuditor:
    """Verifies presence, absence, and authenticity of required packaging accessories."""

    def __init__(
        self,
        catalog_manager: Optional[CatalogManager] = None,
        edge_energy_threshold: float = 12.0,
        entropy_threshold: float = 3.5,
    ):
        """Initializes the KittingAuditor.

        Args:
            catalog_manager: CatalogManager instance containing registered SKU BOMs.
            edge_energy_threshold: Minimum Sobel gradient density to classify a slot as populated.
            entropy_threshold: Minimum Shannon entropy in slot to classify as populated.
        """
        self.catalog_manager = catalog_manager or CatalogManager()
        self.edge_energy_threshold = edge_energy_threshold
        self.entropy_threshold = entropy_threshold

    def _load_tray_image(
        self,
        tray_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Loads tray capture into BGR numpy format."""
        if isinstance(tray_input, (str, Path)):
            path = Path(tray_input)
            if not path.exists():
                raise FileNotFoundError(f"Tray image not found: {path}")
            bgr = cv2.imread(str(path))
        elif isinstance(tray_input, Image.Image):
            bgr = cv2.cvtColor(np.array(tray_input), cv2.COLOR_RGB2BGR)
        elif isinstance(tray_input, np.ndarray):
            bgr = tray_input.copy()
        else:
            raise TypeError(f"Unsupported tray image input: {type(tray_input)}")
        return bgr

    @staticmethod
    def compute_slot_metrics(slot_bgr: np.ndarray) -> Tuple[float, float]:
        """Computes edge gradient energy and Shannon luminance entropy for a tray slot.

        Why: An empty molded foam cavity exhibits near-uniform surface color and minimal
        internal high-frequency gradients. A slot populated with a coiled cable, adapter,
        or printed manual exhibits high gradient energy and elevated luminance entropy.
        """
        gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Edge gradient energy via Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_energy = float(np.mean(magnitude))

        # 2. Shannon luminance entropy
        hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        return edge_energy, entropy

    @staticmethod
    def compute_color_variance(slot_bgr: np.ndarray) -> float:
        """Computes color standard deviation across RGB channels to distinguish foam from accessories."""
        return float(np.mean([np.std(slot_bgr[:, :, c]) for c in range(3)]))

    def extract_slot_crops(
        self,
        tray_bgr: np.ndarray,
        bom: List[BOMItem],
        sku_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        """Extracts slot crops either via SKU-configured ROI bounding boxes or spatial partitioning."""
        h, w = tray_bgr.shape[:2]
        slots = []

        # Check for explicit SKU-specific slot ROIs in metadata
        configured_rois = (sku_metadata or {}).get("slot_rois", {})
        if configured_rois and isinstance(configured_rois, dict):
            for item in bom:
                if item.item_id in configured_rois:
                    roi = configured_rois[item.item_id]  # [x1, y1, x2, y2] normalized or pixels
                    if len(roi) == 4:
                        x1, y1, x2, y2 = roi
                        if all(0.0 <= val <= 1.0 for val in roi):
                            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                        crop = tray_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            slots.append(crop)
                            continue
                # Fallback to partition
                slots.append(self.partition_tray_slots(tray_bgr, len(bom))[min(len(slots), len(bom)-1)])
            return slots

        return self.partition_tray_slots(tray_bgr, len(bom))

    def partition_tray_slots(
        self,
        tray_bgr: np.ndarray,
        num_items: int,
    ) -> List[np.ndarray]:
        """Partitions the tray image into heuristic spatial sub-regions if explicit ROIs are not defined."""
        h, w = tray_bgr.shape[:2]
        slots = []

        if num_items <= 2:
            # Split vertically into top and bottom halves
            mid_y = h // 2
            slots.append(tray_bgr[0:mid_y, :])
            slots.append(tray_bgr[mid_y:h, :])
        elif num_items == 3:
            # 2 top slots (left/right) + 1 bottom large cavity (standard headphone/drill packaging)
            mid_y = int(h * 0.45)
            mid_x = w // 2
            slots.append(tray_bgr[0:mid_y, 0:mid_x])      # Top Left (cable/accessories)
            slots.append(tray_bgr[0:mid_y, mid_x:w])      # Top Right (audio cable/manual)
            slots.append(tray_bgr[mid_y:h, 0:w])          # Bottom Large Cavity (case/device)
        else:
            # Grid partitioning 2x2
            mid_y = h // 2
            mid_x = w // 2
            slots.append(tray_bgr[0:mid_y, 0:mid_x])
            slots.append(tray_bgr[0:mid_y, mid_x:w])
            slots.append(tray_bgr[mid_y:h, 0:mid_x])
            slots.append(tray_bgr[mid_y:h, mid_x:w])

        return slots[:num_items]

    def audit_kitting(
        self,
        tray_input: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        sku_id: Optional[str] = None,
        bom_override: Optional[List[BOMItem]] = None,
        detected_item_ids: Optional[List[str]] = None,
    ) -> KittingAuditResult:
        """Audits the returned packaging tray against the SKU's required Bill of Materials.

        Args:
            tray_input: Accessory tray crop image (optional if detected_item_ids passed directly).
            sku_id: SKU identifier to retrieve required BOM list.
            bom_override: Optional explicit list of BOMItems.
            detected_item_ids: Optional pre-detected accessory IDs (e.g. from vision detector).

        Returns:
            KittingAuditResult with per-item presence, missing counts, and total deductions.
        """
        bom: List[BOMItem] = []
        if bom_override is not None:
            bom = bom_override
        elif sku_id:
            profile = self.catalog_manager.get_sku(sku_id)
            if profile:
                bom = profile.bom
            else:
                raise KeyError(f"SKU '{sku_id}' not found in catalog.")
        else:
            raise ValueError("Either sku_id or bom_override must be provided.")

        if not bom:
            # Product has no accessories defined in BOM
            return KittingAuditResult(
                expected_items_count=0,
                detected_items_count=0,
                missing_items_count=0,
                all_items_present=True,
                item_audits=[],
                total_accessory_deduction=0.0,
            )

        audits: List[KittedItemAudit] = []
        total_deduction = 0.0

        # Mode A: Direct detected IDs provided
        if detected_item_ids is not None:
            detected_set = set(detected_item_ids)
            for item in bom:
                if item.item_id in detected_set:
                    audits.append(KittedItemAudit(
                        bom_item_id=item.item_id,
                        name=item.name,
                        status=AccessoryStatus.PRESENT_OEM,
                        confidence=0.98,
                        deduction_fee=0.0,
                    ))
                else:
                    audits.append(KittedItemAudit(
                        bom_item_id=item.item_id,
                        name=item.name,
                        status=AccessoryStatus.MISSING,
                        confidence=0.95,
                        deduction_fee=item.deduction_fee,
                    ))
                    total_deduction += item.deduction_fee

        # Mode B: Image-based visual tray inspection
        elif tray_input is not None:
            tray_bgr = self._load_tray_image(tray_input)
            sku_metadata = profile.metadata if profile else {}
            slot_crops = self.extract_slot_crops(tray_bgr, bom, sku_metadata=sku_metadata)

            for idx, item in enumerate(bom):
                if idx < len(slot_crops):
                    slot_bgr = slot_crops[idx]
                    edge_energy, entropy = self.compute_slot_metrics(slot_bgr)
                    color_std = self.compute_color_variance(slot_bgr)

                    # Populated slots exhibit significant internal edge complexity and entropy
                    is_populated = (
                        (edge_energy >= self.edge_energy_threshold or entropy >= self.entropy_threshold)
                        and color_std >= 6.0
                    )
                    conf = min(1.0, max(0.60, edge_energy / (self.edge_energy_threshold * 1.5)))

                    if is_populated:
                        audits.append(KittedItemAudit(
                            bom_item_id=item.item_id,
                            name=item.name,
                            status=AccessoryStatus.PRESENT_OEM,
                            confidence=round(conf, 3),
                            deduction_fee=0.0,
                        ))
                    else:
                        audits.append(KittedItemAudit(
                            bom_item_id=item.item_id,
                            name=item.name,
                            status=AccessoryStatus.MISSING,
                            confidence=round(1.0 - conf * 0.3, 3),
                            deduction_fee=item.deduction_fee,
                        ))
                        total_deduction += item.deduction_fee
                else:
                    # Slot not visible in partition
                    audits.append(KittedItemAudit(
                        bom_item_id=item.item_id,
                        name=item.name,
                        status=AccessoryStatus.MISSING,
                        confidence=0.50,
                        deduction_fee=item.deduction_fee,
                    ))
                    total_deduction += item.deduction_fee
        else:
            # Neither image nor detected list provided -> assume all missing
            for item in bom:
                audits.append(KittedItemAudit(
                    bom_item_id=item.item_id,
                    name=item.name,
                    status=AccessoryStatus.MISSING,
                    confidence=0.50,
                    deduction_fee=item.deduction_fee,
                ))
                total_deduction += item.deduction_fee

        detected_count = sum(1 for a in audits if a.status == AccessoryStatus.PRESENT_OEM)
        missing_count = len(bom) - detected_count
        all_present = (missing_count == 0)

        logger.info(
            "Kitting audit for SKU '%s': Expected=%d, Detected=%d, Missing=%d, Deduction=$%.2f",
            sku_id, len(bom), detected_count, missing_count, total_deduction
        )

        return KittingAuditResult(
            expected_items_count=len(bom),
            detected_items_count=detected_count,
            missing_items_count=missing_count,
            all_items_present=all_present,
            item_audits=audits,
            total_accessory_deduction=round(total_deduction, 2),
        )
