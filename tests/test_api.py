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
    """Provides TestClient with pre-warmed lifespan."""
    with TestClient(app) as client:
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
