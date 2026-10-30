"""Financial Disposition and Cryptographic Dispute Evidence Engine for ReturnShield AI.

Implements Stage 4 financial triage:
- Consolidates findings from SwapDetector, SerialVerifier, and KittingAuditor.
- Calculates exact restocking fee rates, accessory deductions, and wear penalties.
- Assigns actionable disposition: FULL_REFUND, PARTIAL_REFUND, or REJECT_FRAUD_HOLD.
- Generates cryptographically sealed (SHA-256) Audit Evidence Certificates
  for Stripe, Shopify, and Amazon SAFE-T chargeback defense.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from core.schemas import (
    AccessoryStatus,
    AuditEvidenceCertificate,
    ConditionGrade,
    DispositionAction,
    DispositionResult,
    FraudFlag,
    IdentityMatchStatus,
    ItemizedDeduction,
    KittingAuditResult,
    SerialMatchStatus,
    SerialVerificationResult,
    SKUProfile,
    SwapDetectionResult,
)
from core.volumetric_schemas import (
    HingeCondition,
    LiquidDamageStatus,
    MultiAngleInspectionResult,
)

logger = logging.getLogger("returnshield.disposition_engine")


class DispositionEngine:
    """Calculates algorithmic refund deductions and generates cryptographic dispute certificates."""

    # Default wear deduction rates relative to MSRP
    WEAR_PENALTIES = {
        ConditionGrade.GRADE_A: 0.0,    # Like New / Mint (0% deduction)
        ConditionGrade.GRADE_B: 0.10,   # Light Scuffs / Minor blemishes (10% deduction)
        ConditionGrade.GRADE_C: 0.25,   # Heavy Scratches / Dents (25% deduction)
        ConditionGrade.FRAUD_HOLD: 1.0, # 100% Forfeit / Rejection
    }

    def determine_condition_grade(
        self,
        surface_wear_score: float,
        is_fraud: bool,
    ) -> ConditionGrade:
        """Assigns cosmetic condition grade based on surface abrasion severity."""
        if is_fraud:
            return ConditionGrade.FRAUD_HOLD
        if surface_wear_score <= 0.10:
            return ConditionGrade.GRADE_A
        elif surface_wear_score <= 0.30:
            return ConditionGrade.GRADE_B
        else:
            return ConditionGrade.GRADE_C

    def evaluate_fraud_flags(
        self,
        swap_result: Optional[SwapDetectionResult],
        serial_result: Optional[SerialVerificationResult],
        kitting_result: Optional[KittingAuditResult],
        condition_grade: ConditionGrade,
        volumetric_result: Optional[MultiAngleInspectionResult] = None,
    ) -> List[FraudFlag]:
        """Synthesizes active fraud or anomaly flags across all inspection sub-modules."""
        flags: List[FraudFlag] = []

        # 1. Identity swap check
        if swap_result and swap_result.identity_status == IdentityMatchStatus.CONFIRMED_SWAP:
            flags.append(FraudFlag.BAIT_AND_SWITCH)

        # 2. Serial verification check
        if serial_result:
            if not serial_result.is_match:
                flags.append(FraudFlag.SERIAL_MISMATCH)
            if serial_result.match_status == SerialMatchStatus.OCR_UNREADABLE:
                flags.append(FraudFlag.TAMPERED_SERIAL_LABEL)

        # 3. Missing accessories check
        if kitting_result and not kitting_result.all_items_present:
            flags.append(FraudFlag.MISSING_ACCESSORY)

        # 4. Physical damage & volumetric check
        if condition_grade == ConditionGrade.GRADE_C:
            flags.append(FraudFlag.PHYSICAL_DAMAGE)

        if volumetric_result:
            if volumetric_result.lateral_tamper_detected and FraudFlag.BAIT_AND_SWITCH not in flags:
                flags.append(FraudFlag.BAIT_AND_SWITCH)
            if volumetric_result.liquid_damage_status == LiquidDamageStatus.CONFIRMED_LIQUID_DAMAGE and FraudFlag.PHYSICAL_DAMAGE not in flags:
                flags.append(FraudFlag.PHYSICAL_DAMAGE)
            if volumetric_result.hinge_condition == HingeCondition.FRACTURED_CRACKED and FraudFlag.PHYSICAL_DAMAGE not in flags:
                flags.append(FraudFlag.PHYSICAL_DAMAGE)

        # If critical evidence checks are completely missing, flag incomplete evidence
        if swap_result is None and serial_result is None:
            flags.append(FraudFlag.INCOMPLETE_EVIDENCE)
        elif not flags:
            flags.append(FraudFlag.CLEAN)

        return flags

    def calculate_disposition(
        self,
        sku_profile: SKUProfile,
        rma_number: str,
        swap_result: Optional[SwapDetectionResult] = None,
        serial_result: Optional[SerialVerificationResult] = None,
        kitting_result: Optional[KittingAuditResult] = None,
        surface_wear_score: float = 0.0,
        evidence_snapshot_paths: Optional[Dict[str, str]] = None,
        volumetric_result: Optional[MultiAngleInspectionResult] = None,
        order_paid_amount: Optional[float] = None,
        currency: str = "USD",
        policy_version: str = "2026.1",
        tenant_id: str = "tenant-default",
    ) -> Tuple[DispositionResult, AuditEvidenceCertificate]:
        """Calculates exact financial disposition and seals cryptographic audit certificate.

        Args:
            sku_profile: Registered catalog SKU with MSRP and BOM fee manifests.
            rma_number: Return authorization identifier.
            swap_result: Output from SwapDetector.
            serial_result: Output from SerialVerifier.
            kitting_result: Output from KittingAuditor.
            surface_wear_score: Normalized cosmetic abrasion severity [0.0, 1.0].
            evidence_snapshot_paths: Dictionary mapping ROI names to disk image paths.
            volumetric_result: Optional output from VolumetricStereoInspector.
            order_paid_amount: Authoritative customer order amount (source of truth).
            currency: Currency ISO code (defaults to 'USD').
            policy_version: Versioned returns policy identifier.
            tenant_id: Multi-tenant partition identifier.

        Returns:
            Tuple of (DispositionResult, AuditEvidenceCertificate).
        """
        msrp = sku_profile.msrp
        basis_amount = order_paid_amount if order_paid_amount is not None else msrp
        basis_type = "ORDER_PAID" if order_paid_amount is not None else "CATALOG_MSRP"
        deductions: List[ItemizedDeduction] = []

        # 1. Evaluate fraud severity
        has_swap_fraud = swap_result is not None and swap_result.is_swap_detected and swap_result.identity_status == IdentityMatchStatus.CONFIRMED_SWAP
        has_serial_fraud = serial_result is not None and (not serial_result.is_match) and serial_result.match_status == SerialMatchStatus.MISMATCH
        has_lateral_fraud = volumetric_result is not None and volumetric_result.lateral_tamper_detected
        is_hard_fraud = has_swap_fraud or has_serial_fraud or has_lateral_fraud

        # Check for missing critical evidence
        has_incomplete_evidence = (swap_result is None or serial_result is None)

        # 2. Determine cosmetic condition grade
        condition_grade = self.determine_condition_grade(surface_wear_score, is_hard_fraud)

        # 3. Collect fraud flags
        fraud_flags = self.evaluate_fraud_flags(
            swap_result, serial_result, kitting_result, condition_grade, volumetric_result=volumetric_result
        )
        if has_incomplete_evidence and FraudFlag.INCOMPLETE_EVIDENCE not in fraud_flags:
            fraud_flags.append(FraudFlag.INCOMPLETE_EVIDENCE)
            if FraudFlag.CLEAN in fraud_flags:
                fraud_flags.remove(FraudFlag.CLEAN)

        # 4. Financial Calculation Engine
        if is_hard_fraud:
            # Full rejection: 0% refund, 100% deduction
            action = DispositionAction.REJECT_FRAUD_HOLD
            deductions.append(ItemizedDeduction(
                reason="Return Rejected: Fraud Flagged (Bait-and-Switch or Serial Mismatch)",
                category="fraud_forfeit",
                amount=basis_amount,
            ))
            total_deductions = basis_amount
            net_refund = 0.0
            summary = (
                f"REJECTED - FRAUD DETECTED: Return {rma_number} rejected due to critical anomalies "
                f"({', '.join(f.value for f in fraud_flags)}). Refund withheld in full ({currency} 0.00)."
            )

        else:
            # Open-box return: apply itemized deductions against basis amount
            # A. Open-box restocking fee
            if sku_profile.restocking_fee_rate > 0.0:
                restock_amt = round(basis_amount * sku_profile.restocking_fee_rate, 2)
                deductions.append(ItemizedDeduction(
                    reason=f"Standard Open-Box Restocking Fee ({int(sku_profile.restocking_fee_rate * 100)}%)",
                    category="restocking_fee",
                    amount=restock_amt,
                ))

            # B. Missing BOM accessory deductions
            if kitting_result and not kitting_result.all_items_present:
                for item_audit in kitting_result.item_audits:
                    if item_audit.status != AccessoryStatus.PRESENT_OEM and item_audit.deduction_fee > 0.0:
                        deductions.append(ItemizedDeduction(
                            reason=f"Missing OEM Accessory: {item_audit.name}",
                            category="accessory",
                            amount=item_audit.deduction_fee,
                        ))

            # C. Cosmetic wear penalty
            wear_rate = self.WEAR_PENALTIES.get(condition_grade, 0.0)
            if wear_rate > 0.0:
                wear_amt = round(basis_amount * wear_rate, 2)
                deductions.append(ItemizedDeduction(
                    reason=f"Cosmetic Surface Wear Penalty: {condition_grade.value}",
                    category="cosmetic_wear",
                    amount=wear_amt,
                ))

            # D. Volumetric, liquid, and lateral joint deductions
            if volumetric_result:
                if volumetric_result.liquid_damage_status == LiquidDamageStatus.CONFIRMED_LIQUID_DAMAGE:
                    liquid_fee = round(min(150.0, basis_amount * 0.40), 2)
                    deductions.append(ItemizedDeduction(
                        reason="Liquid Ingress Damage (Triggered LDI)",
                        category="liquid_damage",
                        amount=liquid_fee,
                    ))
                if volumetric_result.hinge_condition == HingeCondition.FRACTURED_CRACKED:
                    hinge_fee = round(min(80.0, basis_amount * 0.20), 2)
                    deductions.append(ItemizedDeduction(
                        reason="Fractured Structural Hinge / Pivot Joint",
                        category="hinge_damage",
                        amount=hinge_fee,
                    ))
                if volumetric_result.volumetric_profile and volumetric_result.volumetric_profile.discrepancy_detected:
                    vol_fee = round(min(50.0, basis_amount * 0.15), 2)
                    deductions.append(ItemizedDeduction(
                        reason=f"3D Volumetric Thickness Discrepancy ({volumetric_result.volumetric_profile.depth_delta_mm:.1f}mm delta)",
                        category="volumetric_deviation",
                        amount=vol_fee,
                    ))

            total_deductions = min(basis_amount, round(sum(d.amount for d in deductions), 2))
            net_refund = max(0.0, round(basis_amount - total_deductions, 2))

            # Action routing: Fail-closed on missing evidence
            if has_incomplete_evidence:
                action = DispositionAction.MANUAL_SECONDARY_REVIEW
                net_refund = 0.0
                summary = (
                    f"MANUAL REVIEW REQUIRED: Return {rma_number} missing critical evidence (visual chassis or serial unverified). "
                    f"Automated refund withheld ({currency} 0.00) pending bench inspection."
                )
            elif swap_result and swap_result.identity_status == IdentityMatchStatus.SUSPECTED_SWAP:
                action = DispositionAction.MANUAL_SECONDARY_REVIEW
                summary = (
                    f"MANUAL REVIEW REQUIRED: Return {rma_number} exhibits visual ambiguity in swap detection. "
                    f"Calculated tentative partial refund: {currency} {net_refund:.2f}."
                )
            elif total_deductions == 0.0:
                action = DispositionAction.FULL_REFUND
                summary = f"FULL REFUND APPROVED: Return {rma_number} is pristine Grade A with complete accessories ({currency} {net_refund:.2f})."
            else:
                action = DispositionAction.PARTIAL_REFUND
                summary = (
                    f"PARTIAL REFUND APPROVED: Return {rma_number} net refund {currency} {net_refund:.2f} "
                    f"after {currency} {total_deductions:.2f} in itemized deductions."
                )

        threshold_exceeded = (total_deductions > 50.0) or (basis_amount > 0 and (total_deductions / basis_amount) >= 0.20)
        requires_approval = bool(is_hard_fraud or has_incomplete_evidence or threshold_exceeded or action == DispositionAction.MANUAL_SECONDARY_REVIEW)
        approval_reason = None
        if requires_approval:
            if is_hard_fraud:
                approval_reason = "Hard fraud flagged; supervisor approval required for refund forfeiture."
            elif has_incomplete_evidence:
                approval_reason = "Critical inspection evidence unavailable; manual bench inspection required."
            elif threshold_exceeded:
                approval_reason = f"Itemized deductions ({currency} {total_deductions:.2f}) exceed automatic authorization threshold."
            else:
                approval_reason = "Manual secondary review required before disbursement."

        disposition_obj = DispositionResult(
            action=action,
            condition_grade=condition_grade,
            original_msrp=msrp,
            basis_amount=basis_amount,
            basis_type=basis_type,
            currency=currency,
            policy_version=policy_version,
            total_deductions=total_deductions,
            net_refund_amount=net_refund,
            requires_human_approval=requires_approval,
            approval_reason=approval_reason,
            deduction_breakdown=deductions,
            recommendation_summary=summary,
        )

        # 5. Raw Evidence Artifact Hashing
        artifact_hashes: Dict[str, str] = {}
        if evidence_snapshot_paths:
            for key, p_str in evidence_snapshot_paths.items():
                if p_str and Path(p_str).is_file():
                    digest = AuditEvidenceCertificate.compute_file_sha256(p_str)
                    if digest:
                        artifact_hashes[key] = digest

        # 6. Cryptographic Dispute Certificate Sealing & Signing
        evidence_payload = {
            "rma_number": rma_number,
            "sku_id": sku_profile.sku_id,
            "tenant_id": tenant_id,
            "action": action.value,
            "condition_grade": condition_grade.value,
            "original_msrp": msrp,
            "basis_amount": basis_amount,
            "basis_type": basis_type,
            "currency": currency,
            "net_refund": net_refund,
            "total_deductions": total_deductions,
            "fraud_flags": [f.value for f in fraud_flags],
            "swap_cosine": swap_result.cosine_similarity if swap_result else None,
            "serial_matched": serial_result.is_match if serial_result else None,
            "missing_accessories_count": kitting_result.missing_items_count if kitting_result else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        evidence_hash = AuditEvidenceCertificate.compute_evidence_hash(evidence_payload)

        signing_secret = os.getenv("RETURNSHIELD_SIGNING_SECRET", "rs_hmac_signing_secret_dev_2026")
        service_signature = AuditEvidenceCertificate.sign_manifest(evidence_payload, artifact_hashes, signing_secret)

        # Generate human-readable dispute narrative
        narrative = (
            f"ReturnShield AI Inspection Audit for RMA {rma_number} (SKU: {sku_profile.sku_id}, Basis: {currency} {basis_amount:.2f}).\n"
            f"Decision: {action.value} | Condition Grade: {condition_grade.value}.\n"
            f"Serial Verification: {'MATCH' if (serial_result and serial_result.is_match) else ('UNAVAILABLE' if not serial_result else 'MISMATCH')}.\n"
            f"Visual Identity (DINOv2): {'OEM VERIFIED' if (swap_result and not swap_result.is_swap_detected) else ('UNAVAILABLE' if not swap_result else 'SWAP DETECTED')}.\n"
            f"Missing Accessories: {kitting_result.missing_items_count if kitting_result else 0} item(s).\n"
            f"Itemized Deductions Applied: {currency} {total_deductions:.2f}. Net Disbursed: {currency} {net_refund:.2f}.\n"
            f"Consistency Checksum (SHA-256): {evidence_hash}\n"
            f"Service Signature (HMAC-SHA256): {service_signature[:16]}..."
        )

        certificate = AuditEvidenceCertificate(
            rma_number=rma_number,
            sku_id=sku_profile.sku_id,
            tenant_id=tenant_id,
            disposition_action=action,
            condition_grade=condition_grade,
            fraud_flags=fraud_flags,
            evidence_hash=evidence_hash,
            artifact_hashes=artifact_hashes,
            service_signature=service_signature,
            signing_algorithm="HMAC-SHA256",
            evidence_snapshot_paths=evidence_snapshot_paths or {},
            audit_narrative=narrative,
            dispute_ready_payload=evidence_payload,
        )

        logger.info(
            "Disposition evaluated for RMA '%s': %s (Refund=%s %.2f, Deductions=%s %.2f, Signature=%s...)",
            rma_number, action.value, currency, net_refund, currency, total_deductions, service_signature[:12]
        )

        return disposition_obj, certificate


        logger.info(
            "Disposition evaluated for RMA '%s': %s (Refund=$%.2f, Deductions=$%.2f, Hash=%s...)",
            rma_number, action.value, net_refund, total_deductions, evidence_hash[:12]
        )

        return disposition_obj, certificate
