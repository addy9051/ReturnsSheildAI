"""Master Return Triage Pipeline for ReturnShield AI.

Coordinates the end-to-end execution of all inspection modules:
- Visual Identity & Swap Detection (DINOv2)
- Dual-Serial OCR Cross-Verification (OpenCV + RapidOCR)
- Bill of Materials (BOM) Kitting Audit
- Algorithmic Financial Disposition & Cryptographic Evidence Sealing
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional, Union

from core.catalog_manager import CatalogManager
from core.disposition_engine import DispositionEngine
from core.kitting_auditor import KittingAuditor
from core.schemas import (
    ReturnInspectionRequest,
    ReturnInspectionResult,
)
from core.serial_verifier import SerialVerifier
from core.swap_detector import SwapDetector
from core.volumetric_inspector import VolumetricStereoInspector
from core.volumetric_schemas import (
    HingeCondition,
    LiquidDamageStatus,
    MultiAngleInspectionResult,
)

logger = logging.getLogger("returnshield.pipeline")


class ReturnTriagePipeline:
    """End-to-end orchestrator executing warehouse triage within sub-3-second SLA."""

    def __init__(
        self,
        catalog_manager: Optional[CatalogManager] = None,
        swap_detector: Optional[SwapDetector] = None,
        serial_verifier: Optional[SerialVerifier] = None,
        kitting_auditor: Optional[KittingAuditor] = None,
        disposition_engine: Optional[DispositionEngine] = None,
        volumetric_inspector: Optional[VolumetricStereoInspector] = None,
    ):
        """Initializes all sub-inspection engines."""
        self.catalog_manager = catalog_manager or CatalogManager()
        self.swap_detector = swap_detector or SwapDetector(catalog_manager=self.catalog_manager)
        self.serial_verifier = serial_verifier or SerialVerifier()
        self.kitting_auditor = kitting_auditor or KittingAuditor(catalog_manager=self.catalog_manager)
        self.disposition_engine = disposition_engine or DispositionEngine()
        self.volumetric_inspector = volumetric_inspector or VolumetricStereoInspector()

    def inspect(self, request: ReturnInspectionRequest) -> ReturnInspectionResult:
        """Executes full automated triage scan on a return package.

        Args:
            request: ReturnInspectionRequest with RMA, SKU ID, and image crop paths.

        Returns:
            Master ReturnInspectionResult with sub-inspection findings, disposition,
            and cryptographic audit evidence certificate.
        """
        start_time = time.time()
        sku_id = request.sku_id
        profile = self.catalog_manager.get_sku(sku_id)

        if not profile:
            raise KeyError(f"SKU '{sku_id}' is not registered in catalog database.")

        evidence_snapshots = {}

        # 1. Visual Identity & Swap Detection
        swap_result = None
        if request.chassis_crop_path or request.overhead_image_path:
            chassis_input = request.chassis_crop_path or request.overhead_image_path
            evidence_snapshots["chassis_crop"] = str(chassis_input)
            try:
                swap_result = self.swap_detector.detect_swap(
                    query_image=chassis_input,
                    sku_id=sku_id,
                    save_heatmap=True,
                )
                if swap_result.heatmap_path:
                    evidence_snapshots["swap_heatmap"] = swap_result.heatmap_path
            except Exception as err:
                logger.error("Swap detector failed for RMA '%s': %s", request.rma_number, err)

        # 2. Dual-Serial Number Verification
        serial_result = None
        box_input = request.box_barcode_raw or request.overhead_image_path
        chassis_serial_input = request.serial_plate_crop_path or request.overhead_image_path

        if request.serial_plate_crop_path:
            evidence_snapshots["serial_plate"] = str(request.serial_plate_crop_path)

        if box_input or chassis_serial_input:
            try:
                serial_result = self.serial_verifier.verify_serials(
                    box_barcode_input=box_input,
                    chassis_serial_input=chassis_serial_input,
                    expected_regex=profile.expected_serial_regex,
                )
            except Exception as err:
                logger.error("Serial verifier failed for RMA '%s': %s", request.rma_number, err)

        # 3. Kitting Tray & BOM Checklist Audit
        kitting_result = None
        tray_input = request.tray_crop_path or request.overhead_image_path
        if request.tray_crop_path:
            evidence_snapshots["tray_crop"] = str(request.tray_crop_path)

        try:
            kitting_result = self.kitting_auditor.audit_kitting(
                tray_input=tray_input,
                sku_id=sku_id,
            )
        except Exception as err:
            logger.error("Kitting auditor failed for RMA '%s': %s", request.rma_number, err)

        # 4. Multi-Angle Oblique & Volumetric Stereo Inspection
        volumetric_result = None
        if request.oblique_image_path:
            evidence_snapshots["oblique_image"] = str(request.oblique_image_path)
            try:
                top_input = request.chassis_crop_path or request.overhead_image_path
                findings = []
                disparity_path = None
                volumetric_profile = None

                if top_input:
                    _, disparity_path, volumetric_profile = self.volumetric_inspector.compute_disparity_map(
                        left_image=top_input,
                        right_image=request.oblique_image_path,
                        save_heatmap=True,
                    )
                    if disparity_path:
                        evidence_snapshots["disparity_map"] = disparity_path

                # Liquid Damage Indicator check
                ldi_status, ldi_pixels = self.volumetric_inspector.inspect_liquid_damage_indicator(
                    request.oblique_image_path
                )
                if ldi_status == LiquidDamageStatus.CONFIRMED_LIQUID_DAMAGE:
                    findings.append(f"Confirmed liquid damage: {ldi_pixels} pink/crimson LDI pixels detected.")

                # Lateral hinge integrity check
                hinge_status, hinge_energy = self.volumetric_inspector.inspect_hinge_integrity(
                    request.oblique_image_path
                )
                if hinge_status != HingeCondition.INTACT_OEM:
                    findings.append(f"Lateral joint anomaly: {hinge_status.value} (fracture energy {hinge_energy})")

                # Multi-view similarity evaluation
                top_sim = swap_result.cosine_similarity if swap_result else 1.0
                oblique_sim = top_sim
                if hasattr(self.swap_detector, "embedding_engine"):
                    try:
                        pil_oblique, _ = self.swap_detector._load_image_rgb_bgr(request.oblique_image_path)
                        oblique_cls, _ = self.swap_detector.embedding_engine.extract_features(pil_oblique)
                        ref_cls = self.catalog_manager.get_sku_embedding(sku_id)
                        if ref_cls is not None:
                            import numpy as np
                            oblique_sim = float(np.dot(ref_cls, oblique_cls))
                    except Exception as e:
                        logger.debug("Oblique embedding extraction error: %s", e)

                fused_sim, lateral_tamper = self.volumetric_inspector.fuse_multi_view_inspection(
                    top_similarity=top_sim,
                    oblique_similarity=oblique_sim,
                )
                if lateral_tamper:
                    findings.append(f"Multi-view divergence: Top {top_sim:.3f} vs Oblique {oblique_sim:.3f} indicates covert flank/underbelly swap.")

                volumetric_result = MultiAngleInspectionResult(
                    liquid_damage_status=ldi_status,
                    liquid_indicator_pixels=ldi_pixels,
                    hinge_condition=hinge_status,
                    volumetric_profile=volumetric_profile,
                    disparity_map_path=disparity_path,
                    multi_view_fused_similarity=fused_sim,
                    top_similarity=top_sim,
                    oblique_similarity=oblique_sim,
                    lateral_tamper_detected=lateral_tamper,
                    lateral_findings=findings,
                )
            except Exception as err:
                logger.error("Volumetric inspector failed for RMA '%s': %s", request.rma_number, err)

        # 5. Surface Wear Analysis
        surface_wear_score = 0.0
        # If spatial diff is available, use it as a proxy for physical abrasion
        if swap_result and swap_result.spatial_diff_score:
            surface_wear_score = min(1.0, swap_result.spatial_diff_score * 0.8)

        # 6. Financial Disposition & Audit Certificate Generation
        disposition, certificate = self.disposition_engine.calculate_disposition(
            sku_profile=profile,
            rma_number=request.rma_number,
            swap_result=swap_result,
            serial_result=serial_result,
            kitting_result=kitting_result,
            surface_wear_score=surface_wear_score,
            evidence_snapshot_paths=evidence_snapshots,
            volumetric_result=volumetric_result,
        )

        processing_time_ms = round((time.time() - start_time) * 1000.0, 2)

        return ReturnInspectionResult(
            rma_number=request.rma_number,
            sku_id=sku_id,
            fraud_flags=certificate.fraud_flags,
            is_fraud=(disposition.action.value == "REJECT_FRAUD_HOLD"),
            swap_result=swap_result,
            serial_result=serial_result,
            kitting_result=kitting_result,
            volumetric_result=volumetric_result,
            disposition=disposition,
            certificate=certificate,
            processing_time_ms=processing_time_ms,
            operator_id=request.operator_id,
            bench_id=request.bench_id,
        )
