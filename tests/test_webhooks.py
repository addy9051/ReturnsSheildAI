"""Comprehensive Test Suite for Real-Time E-Commerce ERP & Platform Webhooks.

Validates:
1. Cryptographic HMAC-SHA256 signing and verification (Shopify Base64 and Standard Hex).
2. Anti-replay timestamp tolerance defense (300-second window).
3. Platform payload transformations: Shopify Admin REST, Narvar, Amazon SAFE-T, Flipkart SPF.
4. Asynchronous WebhookDispatcher with exponential backoff and delivery audit logging.
5. FastAPI webhook subscription CRUD, inbound webhook authentication, and end-to-end dispatch.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient
import httpx

from core.erp_adapters import (
    AmazonSAFE_T_Adapter,
    FlipkartSPFAdapter,
    NarvarAdapter,
    ShopifyAdapter,
    create_canonical_payload_from_inspection,
)
from core.erp_schemas import (
    CanonicalWebhookPayload,
    ERPPlatform,
    WebhookEvent,
    WebhookSubscription,
)
from core.schemas import (
    ConditionGrade,
    DispositionAction,
    FraudFlag,
    ItemizedDeduction,
    ReturnInspectionResult,
)
from core.webhook_dispatcher import WebhookDispatcher
from core.webhook_security import (
    build_outbound_webhook_headers,
    sign_payload_base64,
    sign_payload_hex,
    verify_signature_base64,
    verify_signature_hex,
    verify_timestamp_freshness,
)
from app.api import app


# ==============================================================================
# 1. Cryptographic Security & Anti-Replay Defense Tests
# ==============================================================================

def test_shopify_base64_hmac_sign_and_verify():
    """Validates official Shopify Base64 HMAC-SHA256 signature verification."""
    secret = "shpss_test_secret_key_12345"
    raw_payload = b'{"order_id": 998811, "status": "return_received"}'

    # Compute signature
    sig = sign_payload_base64(raw_payload, secret)
    assert len(sig) > 20
    assert not sig.startswith("sha256=")

    # Verify matching signature
    assert verify_signature_base64(raw_payload, secret, sig) is True

    # Reject tampered payload
    tampered = b'{"order_id": 998811, "status": "refund_fraud_tampered"}'
    assert verify_signature_base64(tampered, secret, sig) is False

    # Reject wrong secret
    assert verify_signature_base64(raw_payload, "wrong_secret", sig) is False


def test_standard_hex_hmac_sign_and_verify():
    """Validates standard hex-encoded HMAC-SHA256 (Stripe/GitHub style) with sha256= prefix."""
    secret = "whsec_standard_secret_998877"
    raw_payload = b'{"event": "refund.approved", "amount": 399.99}'

    sig_hex = sign_payload_hex(raw_payload, secret)
    assert len(sig_hex) == 64

    # Verify with and without prefix
    assert verify_signature_hex(raw_payload, secret, sig_hex) is True
    assert verify_signature_hex(raw_payload, secret, f"sha256={sig_hex}") is True

    # Tampered verification failure
    assert verify_signature_hex(b"tampered", secret, sig_hex) is False


def test_timestamp_freshness_tolerance_replay_defense():
    """Validates 300-second anti-replay attack protection."""
    now_epoch = time.time()

    # Fresh timestamps
    assert verify_timestamp_freshness(now_epoch, tolerance_seconds=300) is True
    assert verify_timestamp_freshness(now_epoch - 100, tolerance_seconds=300) is True
    assert verify_timestamp_freshness(now_epoch + 10, tolerance_seconds=300) is True

    # Replay attack: expired timestamps (> 300s old)
    assert verify_timestamp_freshness(now_epoch - 301, tolerance_seconds=300) is False
    assert verify_timestamp_freshness(now_epoch - 3600, tolerance_seconds=300) is False

    # ISO-8601 string parsing
    iso_fresh = datetime.now(timezone.utc).isoformat()
    assert verify_timestamp_freshness(iso_fresh, tolerance_seconds=300) is True

    # Invalid timestamp
    assert verify_timestamp_freshness("invalid-timestamp-string") is False


# ==============================================================================
# 2. Platform Adapter Payload Transformation Tests
# ==============================================================================

@pytest.fixture
def clean_canonical_payload() -> CanonicalWebhookPayload:
    return CanonicalWebhookPayload(
        event=WebhookEvent.REFUND_APPROVED,
        rma_number="RMA-SONY-001",
        sku_id="SKU-SONY-WH1000XM5-BLK",
        order_id="ORD-1001",
        customer_id="CUST-501",
        disposition_action=DispositionAction.FULL_REFUND,
        condition_grade=ConditionGrade.GRADE_A,
        is_fraud=False,
        original_msrp=399.99,
        total_deductions=0.0,
        net_refund_amount=399.99,
        currency="USD",
        certificate_id="CERT-TEST-01",
        evidence_hash="abcdef0123456789abcdef0123456789",
        arbitration_pdf_url="/audit/certificates/CERT-TEST-01/pdf",
        audit_narrative="Pristine authentic unit. Verified OEM.",
    )


@pytest.fixture
def fraud_canonical_payload() -> CanonicalWebhookPayload:
    return CanonicalWebhookPayload(
        event=WebhookEvent.FRAUD_ALERT,
        rma_number="RMA-SONY-002",
        sku_id="SKU-SONY-WH1000XM5-BLK",
        order_id="ORD-1002",
        customer_id="CUST-502",
        disposition_action=DispositionAction.REJECT_FRAUD_HOLD,
        condition_grade=ConditionGrade.FRAUD_HOLD,
        is_fraud=True,
        fraud_flags=[FraudFlag.BAIT_AND_SWITCH, FraudFlag.SERIAL_MISMATCH],
        original_msrp=399.99,
        total_deductions=399.99,
        net_refund_amount=0.0,
        currency="USD",
        certificate_id="CERT-TEST-02",
        evidence_hash="feedface12345678feedface12345678",
        arbitration_pdf_url="/audit/certificates/CERT-TEST-02/pdf",
        audit_narrative="Fraudulent unit swap: older WH-1000XM4 returned with counterfeit serial plate.",
    )


def test_shopify_adapter_full_refund(clean_canonical_payload):
    """Validates Shopify Admin REST Refund API transformation for full authentic refund."""
    adapted = ShopifyAdapter.build_payload(clean_canonical_payload)
    refund = adapted["refund"]

    assert refund["currency"] == "USD"
    assert refund["notify"] is True
    assert "Verified authentic" in refund["note"]
    assert refund["refund_line_items"][0]["restock_type"] == "return"
    assert len(refund["transactions"]) == 1
    assert refund["transactions"][0]["amount"] == 399.99
    assert refund["transactions"][0]["kind"] == "refund"
    assert "returnshield:verified_oem" in adapted["tags_to_add"]


def test_shopify_adapter_fraud_blocked(fraud_canonical_payload):
    """Validates Shopify Admin REST Refund API transformation for fraud rejection."""
    adapted = ShopifyAdapter.build_payload(fraud_canonical_payload)
    refund = adapted["refund"]

    assert "FRAUD BLOCKED" in refund["note"]
    assert refund["refund_line_items"][0]["restock_type"] == "no_restock"
    assert "transactions" not in refund
    assert "returnshield:fraud_blocked" in adapted["tags_to_add"]


def test_narvar_adapter(clean_canonical_payload, fraud_canonical_payload):
    """Validates Narvar post-purchase status schema transformation."""
    narvar_clean = NarvarAdapter.build_payload(clean_canonical_payload)
    assert narvar_clean["return_info"]["return_status"] == "approved"
    assert narvar_clean["return_info"]["condition_grade"] == "GRADE_A"
    assert narvar_clean["return_info"]["dispute_evidence"]["certificate_id"] == "CERT-TEST-01"

    narvar_fraud = NarvarAdapter.build_payload(fraud_canonical_payload)
    assert narvar_fraud["return_info"]["return_status"] == "rejected"
    assert narvar_fraud["return_info"]["inspection_summary"]["is_fraud"] is True


def test_amazon_safe_t_adapter(fraud_canonical_payload):
    """Validates Amazon SAFE-T dispute packet transformation."""
    safe_t = AmazonSAFE_T_Adapter.build_payload(fraud_canonical_payload)
    claim = safe_t["safe_t_claim"]

    assert claim["reason_code"] == "DIFFERENT_ITEM_RETURNED"
    assert claim["claimed_amount"] == 399.99
    assert claim["evidence_attachments"]["pdf_dossier_url"] == "/audit/certificates/CERT-TEST-02/pdf"
    assert "BAIT_AND_SWITCH" in claim["serial_forensics"]["flags"]


def test_flipkart_spf_adapter(fraud_canonical_payload):
    """Validates Flipkart SPF incident dossier transformation."""
    spf = FlipkartSPFAdapter.build_payload(fraud_canonical_payload)
    incident = spf["flipkart_spf_incident"]

    assert incident["incident_type"] == "WRONG_PRODUCT_RECEIVED"
    assert incident["claim_amount"] == 399.99
    assert incident["sha256_evidence_seal"] == "feedface12345678feedface12345678"


# ==============================================================================
# 3. Webhook Dispatcher & Async Delivery Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_webhook_dispatcher_lifecycle(tmp_path: Path, clean_canonical_payload):
    """Tests subscription registration, delivery with mock transport, and audit logging."""
    dispatcher = WebhookDispatcher(storage_dir=tmp_path / "webhooks")

    # 1. Register subscription
    sub = dispatcher.register_subscription(
        target_url="https://mock.shopify-store.com/webhooks/returns",
        secret_token="whsec_test_secret_token",
        platform=ERPPlatform.SHOPIFY,
        description="Production Shopify Store Webhook",
    )
    assert sub.platform == ERPPlatform.SHOPIFY
    assert len(dispatcher.list_subscriptions()) == 1

    # 2. Mock HTTP Transport simulating successful 200 response
    def mock_handler(request: httpx.Request):
        assert "X-ReturnShield-Signature" in request.headers
        assert "X-Shopify-Hmac-SHA256" in request.headers
        body = json.loads(request.content.decode("utf-8"))
        assert body["event"] == "refund.approved"
        return httpx.Response(200, json={"received": True, "status": "processed"})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        delivery = await dispatcher.deliver_to_subscription(
            subscription=sub,
            canonical_payload=clean_canonical_payload,
            client=mock_client,
        )

    assert delivery.success is True
    assert delivery.status_code == 200
    assert delivery.attempts == 1
    assert delivery.duration_ms > 0

    # 3. Verify audit deliveries log
    deliveries = dispatcher.list_deliveries()
    assert len(deliveries) == 1
    assert deliveries[0].delivery_id == delivery.delivery_id

    # 4. Delete subscription
    deleted = dispatcher.delete_subscription(sub.subscription_id)
    assert deleted is True
    assert len(dispatcher.list_subscriptions()) == 0


@pytest.mark.asyncio
async def test_webhook_dispatcher_retry_on_server_error(tmp_path: Path, clean_canonical_payload):
    """Tests exponential backoff retries when receiver endpoints return 500 errors."""
    dispatcher = WebhookDispatcher(
        storage_dir=tmp_path / "webhooks",
        max_retries=3,
        initial_backoff_sec=0.01,
    )

    sub = dispatcher.register_subscription(
        target_url="https://flaky.erp-endpoint.com/webhook",
        secret_token="whsec_flaky_endpoint",
        platform=ERPPlatform.CUSTOM,
    )

    attempt_count = 0

    def flaky_handler(request: httpx.Request):
        nonlocal attempt_count
        attempt_count += 1
        return httpx.Response(503, text="Service Unavailable")

    transport = httpx.MockTransport(flaky_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        delivery = await dispatcher.deliver_to_subscription(
            subscription=sub,
            canonical_payload=clean_canonical_payload,
            client=mock_client,
        )

    assert delivery.success is False
    assert delivery.status_code == 503
    assert delivery.attempts == 3
    assert attempt_count == 3


# ==============================================================================
# 4. FastAPI REST Endpoints & Inbound Webhook Authentication Tests
# ==============================================================================

@pytest.fixture
def api_client():
    """Provides a TestClient instance with admin credentials for FastAPI testing."""
    from fastapi.testclient import TestClient
    with TestClient(app, headers={"X-API-Key": "rs_admin_key_sec_2026", "X-Tenant-ID": "tenant_test"}) as client:
        yield client


def test_api_webhook_endpoints_and_inbound_auth(api_client: TestClient):
    """Tests Webhook subscriptions CRUD and inbound webhook signature authentication in FastAPI."""
    # 1. Create Subscription
    create_res = api_client.post(
        "/webhooks/subscriptions",
        json={
            "target_url": "https://partner-erp.example.com/webhooks/returnshield",
            "secret_token": "partner_secret_12345",
            "platform": "shopify",
            "description": "Partner ERP Triage Listener",
        },
    )
    assert create_res.status_code == 201
    sub_data = create_res.json()
    sub_id = sub_data["subscription_id"]
    assert sub_data["platform"] == "shopify"

    # 2. List Subscriptions
    list_res = api_client.get("/webhooks/subscriptions")
    assert list_res.status_code == 200
    subs = list_res.json()
    assert any(s["subscription_id"] == sub_id for s in subs)

    # 3. Test Inbound Shopify Webhook Authentication (Valid Signature)
    inbound_secret = "returnshield_shared_secret"
    raw_inbound_payload = b'{"id": 883322, "topic": "returns/create"}'
    valid_shopify_sig = sign_payload_base64(raw_inbound_payload, inbound_secret)

    inbound_res = api_client.post(
        "/webhooks/inbound/shopify",
        content=raw_inbound_payload,
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Hmac-SHA256": valid_shopify_sig,
            "X-Shopify-Topic": "returns/create",
        },
    )
    assert inbound_res.status_code == 200
    assert inbound_res.json()["verified"] is True

    # 4. Inbound Shopify Webhook (Invalid Signature -> 401)
    inbound_bad = api_client.post(
        "/webhooks/inbound/shopify",
        content=raw_inbound_payload,
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Hmac-SHA256": "invalid_base64_sig==",
            "X-Shopify-Topic": "returns/create",
        },
    )
    assert inbound_bad.status_code == 401

    # 5. Inbound Narvar Webhook Authentication
    now_ts = str(int(time.time()))
    narvar_res = api_client.post(
        "/webhooks/inbound/narvar",
        content=b'{"rma": "RMA-999", "status": "initiated"}',
        headers={
            "Content-Type": "application/json",
            "X-Narvar-Topic": "initiated",
            "X-Narvar-Timestamp": now_ts,
        },
    )
    assert narvar_res.status_code == 200
    assert narvar_res.json()["platform"] == "narvar"

    # 6. Delete Subscription
    del_res = api_client.delete(f"/webhooks/subscriptions/{sub_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "DELETED"
