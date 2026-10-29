"""ERP Platform Payload Adapters for ReturnShield AI.

Transforms canonical ReturnShield triage results into native, official payload schemas:
1. Shopify Admin REST Refund API (POST /admin/api/latest/orders/{order_id}/refunds.json)
2. Narvar Post-Purchase Return Status API
3. Amazon SP-API / Seller Central SAFE-T Reimbursement Claim Format
4. Flipkart Seller Protection Fund (SPF) Claim Incident Format
5. Canonical Generic Webhook Format
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from core.erp_schemas import (
    CanonicalWebhookPayload,
    ERPPlatform,
    WebhookEvent,
)
from core.schemas import (
    ConditionGrade,
    DispositionAction,
    FraudFlag,
    ReturnInspectionResult,
)


class BaseERPAdapter:
    """Base class for all ERP platform payload transformations."""

    @staticmethod
    def build_payload(canonical: CanonicalWebhookPayload) -> Dict[str, Any]:
        raise NotImplementedError


class ShopifyAdapter(BaseERPAdapter):
    """Adapts ReturnShield triage into official Shopify Admin REST Refund API payload."""

    @staticmethod
    def build_payload(canonical: CanonicalWebhookPayload) -> Dict[str, Any]:
        """Constructs official Shopify /admin/api/latest/orders/{order_id}/refunds.json payload."""
        is_approved = canonical.disposition_action in [
            DispositionAction.FULL_REFUND,
            DispositionAction.PARTIAL_REFUND,
        ]

        # Inventory restock decision: restock only if authentic; quarantine if fraud
        restock_type = "return" if is_approved else "no_restock"

        # Itemized notes
        if canonical.disposition_action == DispositionAction.FULL_REFUND:
            note = f"ReturnShield AI: Verified authentic {canonical.sku_id}. Full refund released."
            order_tag = "returnshield:verified_oem"
        elif canonical.disposition_action == DispositionAction.PARTIAL_REFUND:
            reasons = "; ".join(f"{d.name} (-${d.deduction_amount:.2f})" for d in canonical.deduction_breakdown)
            note = f"ReturnShield AI: Partial refund approved. Deductions applied: {reasons}."
            order_tag = "returnshield:partial_deduction"
        elif canonical.disposition_action == DispositionAction.REJECT_FRAUD_HOLD:
            flags = ", ".join(f.value for f in canonical.fraud_flags)
            note = f"ReturnShield AI: FRAUD BLOCKED. Violations: {flags}. Refund withholding enforced."
            order_tag = "returnshield:fraud_blocked"
        else:
            note = "ReturnShield AI: Secondary manual review required. Refund held."
            order_tag = "returnshield:manual_review"

        shopify_payload: Dict[str, Any] = {
            "refund": {
                "currency": canonical.currency,
                "notify": True,
                "note": note,
                "shipping": {"full_refund": canonical.disposition_action == DispositionAction.FULL_REFUND},
                "refund_line_items": [
                    {
                        "sku": canonical.sku_id,
                        "quantity": 1,
                        "restock_type": restock_type,
                    }
                ],
            },
            "metafields": [
                {
                    "namespace": "returnshield",
                    "key": "evidence_hash",
                    "value": canonical.evidence_hash or "",
                    "type": "single_line_text_field",
                },
                {
                    "namespace": "returnshield",
                    "key": "certificate_id",
                    "value": canonical.certificate_id or "",
                    "type": "single_line_text_field",
                },
            ],
            "tags_to_add": [order_tag],
        }

        # Add transaction if refunding money
        if is_approved and canonical.net_refund_amount > 0:
            shopify_payload["refund"]["transactions"] = [
                {
                    "amount": round(canonical.net_refund_amount, 2),
                    "kind": "refund",
                    "gateway": "shopify_payments",
                }
            ]

        # Add order adjustment line if deductions exist
        if canonical.total_deductions > 0:
            shopify_payload["refund"]["order_adjustments"] = [
                {
                    "kind": "refund_discrepancy",
                    "reason": f"Restocking and condition deductions (${canonical.total_deductions:.2f})",
                    "amount": f"-{canonical.total_deductions:.2f}",
                }
            ]

        return shopify_payload


class NarvarAdapter(BaseERPAdapter):
    """Adapts ReturnShield triage into Narvar Post-Purchase Return status payload."""

    @staticmethod
    def build_payload(canonical: CanonicalWebhookPayload) -> Dict[str, Any]:
        status_map = {
            DispositionAction.FULL_REFUND: "approved",
            DispositionAction.PARTIAL_REFUND: "approved",
            DispositionAction.REJECT_FRAUD_HOLD: "rejected",
            DispositionAction.MANUAL_SECONDARY_REVIEW: "investigate",
        }

        return {
            "retailer_moniker": "returnshield_partner",
            "return_info": {
                "rma_number": canonical.rma_number,
                "order_number": canonical.order_id or "UNKNOWN",
                "return_status": status_map.get(canonical.disposition_action, "investigate"),
                "condition_grade": canonical.condition_grade.value,
                "inspection_summary": {
                    "original_msrp": canonical.original_msrp,
                    "deductions": canonical.total_deductions,
                    "net_refund": canonical.net_refund_amount,
                    "is_fraud": canonical.is_fraud,
                    "fraud_flags": [f.value for f in canonical.fraud_flags],
                },
                "dispute_evidence": {
                    "certificate_id": canonical.certificate_id,
                    "sha256_hash": canonical.evidence_hash,
                    "dossier_pdf_url": canonical.arbitration_pdf_url,
                },
            },
        }


class AmazonSAFE_T_Adapter(BaseERPAdapter):
    """Formats dispute dossier payload for Amazon SAFE-T Reimbursement Claims."""

    @staticmethod
    def build_payload(canonical: CanonicalWebhookPayload) -> Dict[str, Any]:
        # Resolve SAFE-T reason code
        reason_code = "OTHER"
        if FraudFlag.BAIT_AND_SWITCH in canonical.fraud_flags:
            reason_code = "DIFFERENT_ITEM_RETURNED"
        elif FraudFlag.SERIAL_MISMATCH in canonical.fraud_flags:
            reason_code = "DIFFERENT_ITEM_RETURNED_SERIAL_MISMATCH"
        elif FraudFlag.MISSING_ACCESSORY in canonical.fraud_flags:
            reason_code = "CUSTOMER_RETURNED_ITEMS_MISSING_PARTS"
        elif FraudFlag.PHYSICAL_DAMAGE in canonical.fraud_flags:
            reason_code = "CUSTOMER_DAMAGED_ITEM"

        claimed_amount = canonical.total_deductions if canonical.total_deductions > 0 else canonical.original_msrp

        return {
            "safe_t_claim": {
                "order_id": canonical.order_id or canonical.rma_number,
                "sku": canonical.sku_id,
                "reason_code": reason_code,
                "claimed_amount": round(claimed_amount, 2),
                "currency": canonical.currency,
                "claim_description": canonical.audit_narrative or "Return inspection audit failure",
                "evidence_attachments": {
                    "pdf_dossier_url": canonical.arbitration_pdf_url,
                    "tamper_proof_sha256": canonical.evidence_hash,
                    "certificate_id": canonical.certificate_id,
                },
                "serial_forensics": {
                    "flags": [f.value for f in canonical.fraud_flags],
                    "condition": canonical.condition_grade.value,
                },
            }
        }


class FlipkartSPFAdapter(BaseERPAdapter):
    """Formats dispute dossier for Flipkart Seller Protection Fund (SPF)."""

    @staticmethod
    def build_payload(canonical: CanonicalWebhookPayload) -> Dict[str, Any]:
        incident_type = "WRONG_PRODUCT_RECEIVED" if canonical.is_fraud else "MISSING_OR_DAMAGED_ITEMS"
        return {
            "flipkart_spf_incident": {
                "return_id": canonical.rma_number,
                "order_item_id": canonical.order_id or "UNKNOWN",
                "incident_type": incident_type,
                "claim_amount": canonical.total_deductions or canonical.original_msrp,
                "currency": canonical.currency,
                "inspection_findings": canonical.audit_narrative,
                "sha256_evidence_seal": canonical.evidence_hash,
                "arbitration_dossier_url": canonical.arbitration_pdf_url,
            }
        }


def adapt_canonical_payload(canonical: CanonicalWebhookPayload, platform: ERPPlatform) -> Dict[str, Any]:
    """Translates canonical payload into platform-native structure."""
    if platform == ERPPlatform.SHOPIFY:
        return ShopifyAdapter.build_payload(canonical)
    elif platform == ERPPlatform.NARVAR:
        return NarvarAdapter.build_payload(canonical)
    elif platform == ERPPlatform.AMAZON_SAFE_T:
        return AmazonSAFE_T_Adapter.build_payload(canonical)
    elif platform == ERPPlatform.FLIPKART_SPF:
        return FlipkartSPFAdapter.build_payload(canonical)
    return canonical.model_dump(mode="json")


def create_canonical_payload_from_inspection(
    result: ReturnInspectionResult,
    order_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    arbitration_pdf_url: Optional[str] = None,
) -> CanonicalWebhookPayload:
    """Constructs CanonicalWebhookPayload directly from ReturnInspectionResult."""
    disp = result.disposition
    cert = result.certificate

    action = disp.action if disp else (DispositionAction.REJECT_FRAUD_HOLD if result.is_fraud else DispositionAction.FULL_REFUND)
    grade = disp.condition_grade if disp else (ConditionGrade.FRAUD_HOLD if result.is_fraud else ConditionGrade.GRADE_A)
    msrp = disp.original_msrp if disp else 0.0
    deductions = disp.total_deductions if disp else 0.0
    net_refund = disp.net_refund_amount if disp else 0.0
    breakdown = disp.deduction_breakdown if disp else []

    # Map disposition action to WebhookEvent
    if action == DispositionAction.FULL_REFUND:
        event = WebhookEvent.REFUND_APPROVED
    elif action == DispositionAction.PARTIAL_REFUND:
        event = WebhookEvent.REFUND_PARTIAL
    elif action == DispositionAction.REJECT_FRAUD_HOLD:
        event = WebhookEvent.FRAUD_ALERT if result.is_fraud else WebhookEvent.REFUND_REJECTED
    else:
        event = WebhookEvent.RETURN_INSPECTED

    return CanonicalWebhookPayload(
        event=event,
        rma_number=result.rma_number,
        sku_id=result.sku_id,
        order_id=order_id,
        customer_id=customer_id,
        disposition_action=action,
        condition_grade=grade,
        is_fraud=result.is_fraud,
        fraud_flags=result.fraud_flags,
        original_msrp=msrp,
        total_deductions=deductions,
        net_refund_amount=net_refund,
        currency="USD",
        deduction_breakdown=breakdown,
        certificate_id=cert.certificate_id if cert else None,
        evidence_hash=cert.evidence_hash if cert else None,
        arbitration_pdf_url=arbitration_pdf_url,
        audit_narrative=cert.audit_narrative if cert else None,
    )

