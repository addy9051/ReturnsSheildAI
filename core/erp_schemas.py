"""ERP and Webhook Domain Models for ReturnShield AI.

Defines schemas for:
- ERP platform identifiers (Shopify, Narvar, Amazon SAFE-T, Flipkart SPF).
- Webhook lifecycle events and subscriptions.
- Canonical normalized webhook payloads.
- Outbound delivery attempts, retry policies, and audit logs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from core.schemas import ConditionGrade, DispositionAction, FraudFlag, ItemizedDeduction


class ERPPlatform(str, Enum):
    """Supported e-commerce ERP platforms and reverse logistics channels."""
    SHOPIFY = "shopify"
    NARVAR = "narvar"
    RETURNLY = "returnly"
    AMAZON_SAFE_T = "amazon_safe_t"
    FLIPKART_SPF = "flipkart_spf"
    CUSTOM = "custom"


class WebhookEvent(str, Enum):
    """Normalized webhook event topics emitted by ReturnShield AI."""
    RETURN_INSPECTED = "return.inspected"
    REFUND_APPROVED = "refund.approved"
    REFUND_PARTIAL = "refund.partial"
    REFUND_REJECTED = "refund.rejected"
    FRAUD_ALERT = "fraud.alert"


class WebhookSubscription(BaseModel):
    """Registered webhook listener for ERP notification dispatch."""
    model_config = ConfigDict(extra="ignore")

    subscription_id: str = Field(default_factory=lambda: f"SUB-{uuid.uuid4().hex[:10].upper()}")
    platform: ERPPlatform = Field(default=ERPPlatform.CUSTOM)
    target_url: str = Field(..., description="Target HTTPS callback endpoint")
    secret_token: str = Field(..., description="Shared secret used for HMAC-SHA256 signature sealing")
    subscribed_events: List[WebhookEvent] = Field(
        default_factory=lambda: [
            WebhookEvent.RETURN_INSPECTED,
            WebhookEvent.REFUND_APPROVED,
            WebhookEvent.REFUND_PARTIAL,
            WebhookEvent.REFUND_REJECTED,
            WebhookEvent.FRAUD_ALERT,
        ]
    )
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = Field(default=None, description="Human readable label or shop domain")

    def to_public_dict(self) -> Dict[str, Any]:
        """Returns safe public dictionary with secret_token masked."""
        d = self.model_dump(mode="json")
        d["secret_token"] = "••••••••"
        return d


class WebhookSubscriptionResponse(BaseModel):
    """Sanitized public response for webhook subscriptions concealing secret tokens."""
    model_config = ConfigDict(extra="ignore")

    subscription_id: str
    platform: ERPPlatform = ERPPlatform.CUSTOM
    target_url: str
    secret_token: str = Field(default="••••••••", description="Masked secret token")
    subscribed_events: List[WebhookEvent] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None



class CanonicalWebhookPayload(BaseModel):
    """Canonical, normalized outbound event payload dispatched to ERP listeners."""
    model_config = ConfigDict(extra="ignore")

    event: WebhookEvent
    event_id: str = Field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:12].upper()}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rma_number: str
    sku_id: str
    order_id: Optional[str] = None
    customer_id: Optional[str] = None
    disposition_action: DispositionAction
    condition_grade: ConditionGrade
    is_fraud: bool
    fraud_flags: List[FraudFlag] = Field(default_factory=list)
    original_msrp: float = Field(..., ge=0.0)
    total_deductions: float = Field(..., ge=0.0)
    net_refund_amount: float = Field(..., ge=0.0)
    currency: str = Field(default="USD")
    deduction_breakdown: List[ItemizedDeduction] = Field(default_factory=list)
    certificate_id: Optional[str] = None
    evidence_hash: Optional[str] = None
    arbitration_pdf_url: Optional[str] = None
    audit_narrative: Optional[str] = None
    platform_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-formatted, platform-specific payload for direct REST consumption",
    )


class WebhookDeliveryAttempt(BaseModel):
    """Audit log entry capturing outbound webhook transmission status."""
    model_config = ConfigDict(extra="ignore")

    delivery_id: str = Field(default_factory=lambda: f"DELIV-{uuid.uuid4().hex[:10].upper()}")
    subscription_id: str
    target_url: str
    event: WebhookEvent
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool
    status_code: Optional[int] = None
    duration_ms: float = 0.0
    attempts: int = 1
    request_headers: Dict[str, str] = Field(default_factory=dict)
    response_body: Optional[str] = None
    error_message: Optional[str] = None

