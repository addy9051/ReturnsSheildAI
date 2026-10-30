"""Integration tests for ReturnShield AI FastAPI service (Phase 4).

Validates:
1. System health check and GPU diagnostics endpoint.
2. SKU catalog retrieval and onboarding endpoints.
3. Automated return inspection endpoint with JSON payload.
4. Cryptographic dispute certificate lookup and tamper verification endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from app.api import app, catalog_manager, pipeline
from core.catalog_manager import CatalogManager
from core.pipeline import ReturnTriagePipeline
from core.schemas import BOMItem, SKUProfile


@pytest.fixture(scope="module")
def api_client():
    """Provides TestClient with pre-warmed lifespan and admin credentials."""
    with TestClient(app, headers={"X-API-Key": "rs_admin_key_sec_2026", "X-Tenant-ID": "tenant_test"}) as client:
        yield client


def test_api_health_endpoint(api_client: TestClient):
    """Validates /health reports operational state and registered SKUs."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ReturnShield AI Triage Engine"
    assert "device" in data
    assert "registered_skus_count" in data
    assert data["registered_skus_count"] >= 5


def test_api_list_and_get_skus(api_client: TestClient):
    """Validates /skus and /skus/{sku_id} retrieval."""
    # List skus
    res_list = api_client.get("/skus")
    assert res_list.status_code == 200
    skus = res_list.json()
    assert len(skus) >= 5
    first_sku = skus[0]
    sku_id = first_sku["sku_id"]

    # Get specific sku
    res_get = api_client.get(f"/skus/{sku_id}")
    assert res_get.status_code == 200
    retrieved = res_get.json()
    assert retrieved["sku_id"] == sku_id
    assert "msrp" in retrieved
    assert "bom" in retrieved

    # Get non-existent sku -> 404
    res_404 = api_client.get("/skus/NON-EXISTENT-SKU-999")
    assert res_404.status_code == 404


def test_api_onboard_sku(api_client: TestClient):
    """Validates /skus/onboard endpoint."""
    payload = {
        "sku_id": "SKU-API-TEST-DEVICE",
        "title": "API Test ANC Earbuds",
        "category": "Audio",
        "msrp": 199.99,
        "restocking_fee_rate": 0.15,
        "bom": [
            {
                "item_id": "case-charging",
                "name": "Wireless Charging Case",
                "category": "case",
                "deduction_fee": 50.0,
            }
        ],
        "expected_serial_regex": r"^SN-TEST-[0-9]{6}$",
    }
    response = api_client.post("/skus/onboard", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU-API-TEST-DEVICE"
    assert data["msrp"] == 199.99


def test_api_inspect_return_and_certificate_lifecycle(api_client: TestClient):
    """Validates /returns/inspect, certificate storage, and /audit/verify."""
    # 1. Submit return inspection request
    req = {
        "rma_number": "RMA-API-TEST-001",
        "sku_id": "SKU-SONY-WH1000XM5-BLK",
        "box_barcode_raw": "SN-SNY-8842109",
        "operator_id": "OP-BENCH-42",
        "bench_id": "BENCH-DOCK-EAST",
    }
    response = api_client.post("/returns/inspect", json=req)
    assert response.status_code == 200
    result = response.json()

    assert result["rma_number"] == "RMA-API-TEST-001"
    assert "disposition" in result
    assert "certificate" in result
    assert result["certificate"] is not None

    cert_id = result["certificate"]["certificate_id"]
    evidence_hash = result["certificate"]["evidence_hash"]
    dispute_payload = result["certificate"]["dispute_ready_payload"]

    # 2. Retrieve certificate via GET /audit/certificates/{cert_id}
    res_cert = api_client.get(f"/audit/certificates/{cert_id}")
    assert res_cert.status_code == 200
    cert_data = res_cert.json()
    assert cert_data["certificate_id"] == cert_id
    assert cert_data["evidence_hash"] == evidence_hash

    # 3. Cryptographically verify authentic certificate
    verify_req = {
        "dispute_ready_payload": dispute_payload,
        "evidence_hash": evidence_hash,
    }
    res_verify = api_client.post("/audit/verify", json=verify_req)
    assert res_verify.status_code == 200
    verify_data = res_verify.json()
    assert verify_data["is_valid"] is True
    assert verify_data["tamper_detected"] is False
    assert verify_data["status"] == "AUTHENTIC_VERIFIED"

    # 4. Cryptographically verify tampered certificate
    tampered_payload = dict(dispute_payload)
    tampered_payload["net_refund"] = 8888.88
    res_tampered = api_client.post("/audit/verify", json={
        "dispute_ready_payload": tampered_payload,
        "evidence_hash": evidence_hash,
    })
    assert res_tampered.status_code == 200
    tampered_data = res_tampered.json()
    assert tampered_data["is_valid"] is False
    assert tampered_data["tamper_detected"] is True
    assert tampered_data["status"] == "TAMPER_DETECTED_INVALID"

    # 5. Download dispute PDF dossier via GET /audit/certificates/{cert_id}/pdf
    res_pdf = api_client.get(f"/audit/certificates/{cert_id}/pdf?platform=AMAZON_SAFE_T")
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert len(res_pdf.content) > 5000
    assert res_pdf.content[:4] == b"%PDF"


def test_mobile_doorstep_pwa_endpoints(api_client: TestClient):
    """Validates Mobile Doorstep PWA routes, manifest, and service worker."""
    # 1. PWA root page
    res_ui = api_client.get("/doorstep/")
    assert res_ui.status_code == 200
    assert "text/html" in res_ui.headers["content-type"]
    assert "ReturnShield QC" in res_ui.text
    assert "cameraVideo" in res_ui.text

    # 2. PWA Web App Manifest
    res_manifest = api_client.get("/doorstep/manifest.json")
    assert res_manifest.status_code == 200
    manifest = res_manifest.json()
    assert manifest["name"] == "ReturnShield AI Doorstep QC"
    assert manifest["display"] == "standalone"

    # 3. PWA Service Worker
    res_sw = api_client.get("/doorstep/sw.js")
    assert res_sw.status_code == 200
    assert "CACHE_NAME" in res_sw.text

    # 4. Mobile shortcut redirect
    res_redir = api_client.get("/mobile", follow_redirects=False)
    assert res_redir.status_code in [307, 302]
    assert res_redir.headers["location"] == "/doorstep/"


# ==============================================================================
# Security & Operational Invariant Regression Tests (Audit Remediation)
# ==============================================================================

def test_unauthenticated_requests_rejected():
    """Loophole 1: Rejects requests missing authentication credentials."""
    with TestClient(app) as unauth_client:
        res = unauth_client.get("/skus")
        assert res.status_code == 401

        res = unauth_client.post("/returns/inspect", json={"rma_number": "RMA-001", "sku_id": "SKU-TEST"})
        assert res.status_code == 401

        res = unauth_client.post("/webhooks/subscriptions", json={
            "target_url": "https://example.com/webhook",
            "secret_token": "secret123",
            "platform": "shopify",
        })
        assert res.status_code == 401


def test_rbac_role_separation():
    """Loophole 1: Ensures Inspector key cannot execute Admin catalog or webhook mutations."""
    with TestClient(app, headers={"X-API-Key": "rs_inspector_key_sec_2026"}) as inspector_client:
        # Inspector attempting to onboard SKU -> 403 Forbidden
        res = inspector_client.post("/skus/onboard", json={
            "sku_id": "SKU-HACK",
            "title": "Hacked SKU",
            "category": "Audio",
            "msrp": 100.0,
        })
        assert res.status_code == 403
        assert "lacks required 'ADMIN' permission" in res.json()["detail"]


def test_empty_inspection_fails_closed(api_client: TestClient):
    """Loophole 2: Incomplete inspection request must fail-closed to manual review, holding refund."""
    req = {
        "rma_number": "RMA-EMPTY-EVIDENCE-01",
        "sku_id": "SKU-SONY-WH1000XM5-BLK",
        "order_paid_amount": 399.99,
        "currency": "USD",
        "returns_policy_version": "v2.1",
    }
    response = api_client.post("/returns/inspect", json=req)
    assert response.status_code == 200
    data = response.json()

    assert data["disposition"]["action"] == "MANUAL_SECONDARY_REVIEW"
    assert "INCOMPLETE_EVIDENCE" in data["fraud_flags"]
    assert data["disposition"]["net_refund_amount"] == 0.0
    assert data["stage_checks"]["visual_identity"] == "UNAVAILABLE"


def test_path_traversal_blocked(api_client: TestClient):
    """Loophole 4: Blocks path traversal in SKU IDs and image paths."""
    # 1. Invalid SKU ID with directory traversal
    res_traversal = api_client.post("/skus/onboard", json={
        "sku_id": "../etc/passwd",
        "title": "Malicious SKU",
        "category": "Audio",
        "msrp": 100.0,
    })
    assert res_traversal.status_code in [400, 422]

    # 2. Path traversal in inspection image path
    res_path = api_client.post("/returns/inspect", json={
        "rma_number": "RMA-TRAVERSAL-01",
        "sku_id": "SKU-SONY-WH1000XM5-BLK",
        "overhead_image_path": "../../../windows/win.ini",
    })
    assert res_path.status_code in [400, 422]


def test_server_authorized_pickup_flow(api_client: TestClient):
    """Loophole 3: Validates server-authorized OTP generation and fraud pickup blocking."""
    from app.api import CERTIFICATES_DIR
    tenant_dir = CERTIFICATES_DIR / "tenant_test"
    tenant_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean return record allows OTP issuance
    clean_record = {
        "certificate": {"certificate_id": "CERT-TEST-CLEAN-01"},
        "inspection_result": {
            "rma_number": "RMA-CLEAN-AUTH-01",
            "is_fraud": False,
            "disposition": {"action": "FULL_REFUND"},
        },
    }
    with open(tenant_dir / "CERT-TEST-CLEAN-01.json", "w", encoding="utf-8") as f:
        json.dump(clean_record, f)

    clean_auth = api_client.post("/returns/authorize-pickup", json={"rma_number": "RMA-CLEAN-AUTH-01"})
    assert clean_auth.status_code == 200
    auth_data = clean_auth.json()
    assert auth_data["authorized"] is True
    assert auth_data["otp"] is not None
    assert auth_data["pickup_token"].startswith("PKUP-")

    # 2. Fraud hold return is blocked from pickup authorization and OTP release
    fraud_record = {
        "certificate": {"certificate_id": "CERT-TEST-FRAUD-01"},
        "inspection_result": {
            "rma_number": "RMA-FRAUD-AUTH-01",
            "is_fraud": True,
            "disposition": {"action": "REJECT_FRAUD_HOLD"},
        },
    }
    with open(tenant_dir / "CERT-TEST-FRAUD-01.json", "w", encoding="utf-8") as f:
        json.dump(fraud_record, f)

    fraud_auth = api_client.post("/returns/authorize-pickup", json={"rma_number": "RMA-FRAUD-AUTH-01"})
    assert fraud_auth.status_code == 200
    f_data = fraud_auth.json()
    assert f_data["authorized"] is False
    assert f_data["otp"] is None
    assert "OTP release prohibited" in f_data["reason"]

    # 3. Non-existent inspection record blocks pickup
    unknown_auth = api_client.post("/returns/authorize-pickup", json={"rma_number": "RMA-NON-EXISTENT-999"})
    assert unknown_auth.status_code == 200
    u_data = unknown_auth.json()
    assert u_data["authorized"] is False
    assert "No valid inspection record found" in u_data["reason"]


def test_webhook_ssrf_protection(api_client: TestClient):
    """Loophole 5: Blocks registration of private/loopback/cloud metadata webhook endpoints."""
    # Loopback IP
    res_loopback = api_client.post("/webhooks/subscriptions", json={
        "target_url": "http://127.0.0.1:8080/callback",
        "secret_token": "secret_token_123",
        "platform": "shopify",
    })
    assert res_loopback.status_code == 400
    assert "SSRF violation" in res_loopback.json()["detail"]

    # Cloud metadata IP
    res_metadata = api_client.post("/webhooks/subscriptions", json={
        "target_url": "http://169.254.169.254/latest/meta-data/",
        "secret_token": "secret_token_123",
        "platform": "shopify",
    })
    assert res_metadata.status_code == 400
    assert "SSRF violation" in res_metadata.json()["detail"]

