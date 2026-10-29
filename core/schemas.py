"""Pydantic data contracts for ReturnShield AI.

Defines schemas for SKU catalog onboarding, visual identity extraction,
dual-serial OCR verification, kitting audit, condition grading,
refund deductions, and cryptographic dispute audit certificates.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# Domain Enums
# ==============================================================================

class FraudFlag(str, Enum):
    """Categorization of fraud or anomaly detected during inspection."""
    CLEAN = "CLEAN"
    BAIT_AND_SWITCH = "BAIT_AND_SWITCH"
    SERIAL_MISMATCH = "SERIAL_MISMATCH"
    MISSING_ACCESSORY = "MISSING_ACCESSORY"
    UNAUTHORIZED_ACCESSORY_SUBSTITUTION = "UNAUTHORIZED_ACCESSORY_SUBSTITUTION"
    PHYSICAL_DAMAGE = "PHYSICAL_DAMAGE"
    TAMPERED_SERIAL_LABEL = "TAMPERED_SERIAL_LABEL"
    COUNTERFEIT = "COUNTERFEIT"


class ConditionGrade(str, Enum):
    """Cosmetic and functional condition grade assigned to returned merchandise."""
    GRADE_A = "GRADE_A"          # Like New / Mint (0% wear deduction)
    GRADE_B = "GRADE_B"          # Light Wear / Minor Blemishes (10-15% deduction)
    GRADE_C = "GRADE_C"          # Heavy Wear / Scratches / Dents (30-50% deduction)
    FRAUD_HOLD = "FRAUD_HOLD"    # Rejected / Counterfeit / Bait-and-Switch (0% refund)


class DispositionAction(str, Enum):
    """Actionable financial triage decision."""
    FULL_REFUND = "FULL_REFUND"
    PARTIAL_REFUND = "PARTIAL_REFUND"
    REJECT_FRAUD_HOLD = "REJECT_FRAUD_HOLD"
    MANUAL_SECONDARY_REVIEW = "MANUAL_SECONDARY_REVIEW"


class SerialMatchStatus(str, Enum):
    """Status of cross-matching box label barcode against chassis serial."""
    EXACT_MATCH = "EXACT_MATCH"
    NORMALIZED_MATCH = "NORMALIZED_MATCH"   # OCR noise corrected (e.g. 0/O, 1/I)
    MISMATCH = "MISMATCH"
    MISSING_LABEL = "MISSING_LABEL"
    OCR_UNREADABLE = "OCR_UNREADABLE"


class IdentityMatchStatus(str, Enum):
    """Visual identity confirmation using DINOv2 representation."""
    VERIFIED_OEM = "VERIFIED_OEM"
    SUSPECTED_SWAP = "SUSPECTED_SWAP"
    CONFIRMED_SWAP = "CONFIRMED_SWAP"


class AccessoryStatus(str, Enum):
    """Inspection status of each required Bill of Materials (BOM) item."""
    PRESENT_OEM = "PRESENT_OEM"
    MISSING = "MISSING"
    SUBSTITUTE_NON_OEM = "SUBSTITUTE_NON_OEM"
    DAMAGED = "DAMAGED"


# ==============================================================================
# BOM & SKU Catalog Schemas
# ==============================================================================

class BOMItem(BaseModel):
    """Item required in product box bill of materials."""
    model_config = ConfigDict(extra="ignore")

    item_id: str = Field(..., description="Unique accessory ID, e.g. 'cable-usbc'")
    name: str = Field(..., description="Accessory display name, e.g. 'USB-C Braided Cable'")
    category: str = Field(default="accessory", description="Category: cable, charger, manual, case, etc.")
    required_qty: int = Field(default=1, ge=1, description="Expected quantity")
    deduction_fee: float = Field(default=15.0, ge=0.0, description="Deduction penalty if missing ($)")
    expected_tray_slot: Optional[str] = Field(default=None, description="Tray slot ROI identifier")
    is_critical: bool = Field(default=False, description="Missing this marks item incomplete/dispute")
    reference_embedding: Optional[List[float]] = Field(default=None, description="Optional accessory visual embedding")


class SKUProfile(BaseModel):
    """Complete registered SKU profile in catalog."""
    model_config = ConfigDict(extra="ignore")

    sku_id: str = Field(..., description="Unique SKU identifier, e.g. 'SKU-SONY-WH1000XM5-BLK'")
    title: str = Field(..., description="Product commercial title")
    category: str = Field(..., description="E-commerce taxonomy category")
    msrp: float = Field(..., gt=0.0, description="Manufacturer Suggested Retail Price in USD")
    restocking_fee_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Baseline restocking fee rate (0.0 to 1.0)")
    expected_serial_regex: Optional[str] = Field(default=r"^[A-Z0-9]{8,18}$", description="Regex pattern for serial validation")
    reference_image_paths: List[str] = Field(default_factory=list, description="Paths to golden reference photos")
    bom: List[BOMItem] = Field(default_factory=list, description="Bill of materials accessory checklist")
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Cosine similarity threshold for DINOv2 match")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SKUOnboardRequest(BaseModel):
    """Payload to register or update an SKU in the catalog."""
    model_config = ConfigDict(extra="ignore")

    sku_id: str = Field(..., description="Unique SKU identifier")
    title: str = Field(..., description="Product title")
    category: str = Field(..., description="Category name")
    msrp: float = Field(..., gt=0.0, description="MSRP value ($)")
    restocking_fee_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_serial_regex: Optional[str] = Field(default=r"^[A-Z0-9]{8,18}$")
    bom: List[BOMItem] = Field(default_factory=list)
    reference_image_paths: List[str] = Field(default_factory=list)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SKUEmbeddingRecord(BaseModel):
    """Serialized vector representation cached for an SKU."""
    model_config = ConfigDict(extra="ignore")

    sku_id: str
    global_embedding: List[float] = Field(..., description="L2-normalized DINOv2 CLS / global embedding vector")
    embedding_dim: int = Field(default=384, description="Embedding dimension")
    model_name: str = Field(default="dinov2_vits14", description="Backbone vision transformer model name")
    patch_token_count: Optional[int] = Field(default=None, description="Number of patch tokens stored")
    image_path: str = Field(..., description="Source reference image used to extract embeddings")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ==============================================================================
# Triage Station Inspection Requests & Results
# ==============================================================================

class ReturnInspectionRequest(BaseModel):
    """Warehouse triage scan request submitted from bench or camera overhead."""
    model_config = ConfigDict(extra="ignore")

    rma_number: str = Field(..., description="Customer return authorization / order ID")
    sku_id: str = Field(..., description="Catalog SKU identifier")
    expected_customer_serial: Optional[str] = Field(default=None, description="Serial number from original customer invoice")
    box_barcode_raw: Optional[str] = Field(default=None, description="Raw scanned barcode from shipping/outer box")
    overhead_image_path: Optional[str] = Field(default=None, description="Full bench overhead camera capture")
    chassis_crop_path: Optional[str] = Field(default=None, description="Isolated product chassis ROI")
    serial_plate_crop_path: Optional[str] = Field(default=None, description="Cropped serial number engraving or label")
    tray_crop_path: Optional[str] = Field(default=None, description="Cropped accessory tray ROI")
    operator_id: Optional[str] = Field(default="OPERATOR-01", description="3PL warehouse inspector ID")
    bench_id: Optional[str] = Field(default="BENCH-01", description="Workstation station ID")


class SwapDetectionResult(BaseModel):
    """Output of DINOv2 visual identity and swap matching."""
    model_config = ConfigDict(extra="ignore")

    identity_status: IdentityMatchStatus
    cosine_similarity: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score vs reference embedding")
    threshold: float = Field(default=0.85, description="Decision threshold applied")
    is_swap_detected: bool = Field(..., description="True if item deviates from catalog reference")
    explanation: str = Field(..., description="Human-readable explanation of visual comparison")
    spatial_diff_score: Optional[float] = Field(default=None, description="Patch-level spatial anomaly score")
    heatmap_path: Optional[str] = Field(default=None, description="Saved visual diff heatmap path")


class SerialVerificationResult(BaseModel):
    """Output of Dual-Serial OCR cross-verification."""
    model_config = ConfigDict(extra="ignore")

    box_barcode: Optional[str] = None
    chassis_serial_raw: Optional[str] = None
    chassis_serial_normalized: Optional[str] = None
    match_status: SerialMatchStatus
    is_match: bool = Field(..., description="True if serials match exactly or within normalized OCR variance")
    similarity_ratio: float = Field(default=1.0, ge=0.0, le=1.0, description="String distance ratio between barcodes")
    ocr_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence of chassis serial OCR read")
    notes: str = Field(default="", description="Detailed serial matching notes")


class KittedItemAudit(BaseModel):
    """Verification record for a specific BOM accessory."""
    model_config = ConfigDict(extra="ignore")

    bom_item_id: str
    name: str
    status: AccessoryStatus
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    deduction_fee: float = Field(default=0.0, ge=0.0)


class KittingAuditResult(BaseModel):
    """Output of accessory tray and BOM slot verification."""
    model_config = ConfigDict(extra="ignore")

    expected_items_count: int
    detected_items_count: int
    missing_items_count: int
    all_items_present: bool
    item_audits: List[KittedItemAudit] = Field(default_factory=list)
    total_accessory_deduction: float = Field(default=0.0, ge=0.0)


class ItemizedDeduction(BaseModel):
    """Specific dollar deduction applied to customer refund."""
    model_config = ConfigDict(extra="ignore")

    reason: str = Field(..., description="Explanation, e.g. 'Missing USB-C Braided Cable'")
    category: str = Field(..., description="Category: 'accessory', 'cosmetic_wear', 'restocking_fee', 'fraud_forfeit'")
    amount: float = Field(..., ge=0.0, description="Dollar deduction amount")


class DispositionResult(BaseModel):
    """Final condition grade and financial disposition."""
    model_config = ConfigDict(extra="ignore")

    action: DispositionAction
    condition_grade: ConditionGrade
    original_msrp: float = Field(..., ge=0.0)
    total_deductions: float = Field(..., ge=0.0)
    net_refund_amount: float = Field(..., ge=0.0)
    deduction_breakdown: List[ItemizedDeduction] = Field(default_factory=list)
    recommendation_summary: str = Field(...)


class AuditEvidenceCertificate(BaseModel):
    """Cryptographically sealed dispute evidence bundle for Stripe/Amazon SAFE-T/Shopify."""
    model_config = ConfigDict(extra="ignore")

    certificate_id: str = Field(default_factory=lambda: f"CERT-{uuid.uuid4().hex[:12].upper()}")
    rma_number: str
    sku_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    disposition_action: DispositionAction
    condition_grade: ConditionGrade
    fraud_flags: List[FraudFlag] = Field(default_factory=list)
    evidence_hash: str = Field(..., description="SHA-256 hash digest of all inspection evidence")
    evidence_snapshot_paths: Dict[str, str] = Field(default_factory=dict)
    audit_narrative: str
    dispute_ready_payload: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def compute_evidence_hash(payload: Dict[str, Any]) -> str:
        """Computes deterministic SHA-256 hash digest of evidence dictionary."""
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


class ReturnInspectionResult(BaseModel):
    """Master end-to-end inspection record for a returned merchandise unit."""
    model_config = ConfigDict(extra="ignore")

    inspection_id: str = Field(default_factory=lambda: f"INSP-{uuid.uuid4().hex[:10].upper()}")
    rma_number: str
    sku_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    fraud_flags: List[FraudFlag] = Field(default_factory=list)
    is_fraud: bool = Field(default=False)
    swap_result: Optional[SwapDetectionResult] = None
    serial_result: Optional[SerialVerificationResult] = None
    kitting_result: Optional[KittingAuditResult] = None
    disposition: Optional[DispositionResult] = None
    certificate: Optional[AuditEvidenceCertificate] = None
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    operator_id: Optional[str] = None
    bench_id: Optional[str] = None

    @model_validator(mode="after")
    def sync_fraud_status(self) -> ReturnInspectionResult:
        """Ensures is_fraud is automatically True if any fraud flags exist."""
        has_fraud_flags = any(
            f in [FraudFlag.BAIT_AND_SWITCH, FraudFlag.SERIAL_MISMATCH, FraudFlag.COUNTERFEIT, FraudFlag.TAMPERED_SERIAL_LABEL]
            for f in self.fraud_flags
        )
        if has_fraud_flags:
            self.is_fraud = True
        return self

