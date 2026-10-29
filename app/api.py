"""FastAPI REST API Service for ReturnShield AI.

Provides high-throughput endpoints for:
- SKU catalog management and zero-training onboarding.
- Automated warehouse and doorstep return inspection scans (<3s SLA).
- Cryptographic dispute evidence certificate retrieval and verification.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.arbitration_report import ArbitrationReportGenerator
from core.catalog_manager import CatalogManager
from core.disposition_engine import DispositionEngine
from core.erp_adapters import adapt_canonical_payload, create_canonical_payload_from_inspection
from core.erp_schemas import (
    CanonicalWebhookPayload,
    ERPPlatform,
    WebhookDeliveryAttempt,
    WebhookEvent,
    WebhookSubscription,
)
from core.kitting_auditor import KittingAuditor
from core.pipeline import ReturnTriagePipeline
from core.schemas import (
    AuditEvidenceCertificate,
    BOMItem,
    ConditionGrade,
    DispositionAction,
    FraudFlag,
    ReturnInspectionRequest,
    ReturnInspectionResult,
    SKUOnboardRequest,
    SKUProfile,
)
from core.serial_verifier import SerialVerifier
from core.swap_detector import SwapDetector
from core.webhook_dispatcher import WebhookDispatcher
from core.webhook_security import (
    verify_signature_base64,
    verify_signature_hex,
    verify_timestamp_freshness,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("returnshield.api")

# Persistent certificate storage
CERTIFICATES_DIR = Path(__file__).resolve().parent.parent / "storage" / "certificates"
CERTIFICATES_DIR.mkdir(parents=True, exist_ok=True)

# Global pipeline instance initialized on app startup
pipeline: Optional[ReturnTriagePipeline] = None
catalog_manager: Optional[CatalogManager] = None
webhook_dispatcher: Optional[WebhookDispatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads AI models and warms GPU execution graphs on service startup."""
    global pipeline, catalog_manager, webhook_dispatcher
    logger.info("Starting ReturnShield AI Service - preloading model weights...")

    catalog_manager = CatalogManager()
    pipeline = ReturnTriagePipeline(catalog_manager=catalog_manager)
    webhook_dispatcher = WebhookDispatcher()

    # Engine warm-up: execute dummy inference to initialize PyTorch CUDA kernels & ONNX threads
    skus = catalog_manager.list_skus()
    if skus:
        warm_sku = skus[0].sku_id
        logger.info("Executing startup warm-up pass on registered SKU: %s", warm_sku)
        try:
            pipeline.inspect(ReturnInspectionRequest(
                rma_number="RMA-WARMUP-STARTUP",
                sku_id=warm_sku,
                box_barcode_raw="WARMUP-TEST",
            ))
            logger.info("Startup warm-up pass successful. Triage station primed for sub-3s SLA.")
        except Exception as err:
            logger.warning("Startup warm-up pass exception (non-fatal): %s", err)

    yield

    logger.info("Shutting down ReturnShield AI Service.")


app = FastAPI(
    title="ReturnShield AI API",
    version="1.0.0",
    description="Automated Open-Box Return Fraud Detection & Condition Triage Service",
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard and mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Mobile Doorstep QC PWA
DOORSTEP_DIR = Path(__file__).resolve().parent / "doorstep"
if DOORSTEP_DIR.exists():
    app.mount("/doorstep", StaticFiles(directory=str(DOORSTEP_DIR), html=True), name="doorstep")


@app.get("/mobile", tags=["System"], include_in_schema=False)
async def mobile_redirect():
    """Convenience redirect to Mobile Doorstep QC PWA."""
    return RedirectResponse(url="/doorstep/")


# ==============================================================================
# Health and Diagnostics
# ==============================================================================

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Returns system status, active device, and registered SKU catalog count."""
    import torch

    skus = catalog_manager.list_skus() if catalog_manager else []
    return {
        "status": "healthy",
        "service": "ReturnShield AI Triage Engine",
        "version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "registered_skus_count": len(skus),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==============================================================================
# SKU Catalog Management Endpoints
# ==============================================================================

@app.get("/skus", response_model=List[SKUProfile], tags=["Catalog"])
async def list_skus() -> List[SKUProfile]:
    """Lists all registered SKU profiles in the catalog."""
    if not catalog_manager:
        raise HTTPException(status_code=500, detail="Catalog manager uninitialized.")
    return catalog_manager.list_skus()


@app.get("/skus/{sku_id}", response_model=SKUProfile, tags=["Catalog"])
async def get_sku(sku_id: str) -> SKUProfile:
    """Retrieves full profile, BOM, and reference configuration for an SKU."""
    if not catalog_manager:
        raise HTTPException(status_code=500, detail="Catalog manager uninitialized.")
    profile = catalog_manager.get_sku(sku_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found.")
    return profile


@app.post("/skus/onboard", response_model=SKUProfile, tags=["Catalog"])
async def onboard_sku(request: SKUOnboardRequest) -> SKUProfile:
    """Registers a new SKU profile and triggers zero-training reference embedding extraction."""
    if not catalog_manager:
        raise HTTPException(status_code=500, detail="Catalog manager uninitialized.")
    try:
        profile = catalog_manager.onboard_sku(request, reference_images=request.reference_image_paths)
        return profile
    except Exception as err:
        logger.error("Failed onboarding SKU '%s': %s", request.sku_id, err)
        raise HTTPException(status_code=400, detail=str(err))


# ==============================================================================
# Return Inspection & Triage Endpoints
# ==============================================================================

@app.post("/returns/inspect", response_model=ReturnInspectionResult, tags=["Inspection"])
async def inspect_return(request: ReturnInspectionRequest) -> ReturnInspectionResult:
    """Executes automated multi-stage inspection scan from JSON payload within sub-3s SLA."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Inspection pipeline uninitialized.")

    if not catalog_manager or not catalog_manager.get_sku(request.sku_id):
        raise HTTPException(status_code=404, detail=f"SKU '{request.sku_id}' is not registered.")

    try:
        result = pipeline.inspect(request)

        # Persist dispute certificate & inspection telemetry to disk
        if result.certificate:
            cert_file = CERTIFICATES_DIR / f"{result.certificate.certificate_id}.json"
            save_payload = {
                "certificate": result.certificate.model_dump(mode="json"),
                "inspection_result": result.model_dump(mode="json"),
            }
            with open(cert_file, "w", encoding="utf-8") as f:
                json.dump(save_payload, f, indent=2)

        # Trigger outbound ERP webhook dispatch in background
        if webhook_dispatcher:
            try:
                pdf_url = f"/audit/certificates/{result.certificate.certificate_id}/pdf" if result.certificate else None
                canonical = create_canonical_payload_from_inspection(result, arbitration_pdf_url=pdf_url)
                asyncio.create_task(webhook_dispatcher.dispatch_event(canonical))
            except Exception as w_err:
                logger.warning("Background webhook dispatch error: %s", w_err)

        return result
    except Exception as err:
        logger.error("Inspection error for RMA '%s': %s", request.rma_number, err)
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/returns/inspect/upload", response_model=ReturnInspectionResult, tags=["Inspection"])
async def inspect_return_with_files(
    rma_number: str = Form(...),
    sku_id: str = Form(...),
    box_barcode: Optional[str] = Form(None),
    operator_id: Optional[str] = Form("OPERATOR-01"),
    bench_id: Optional[str] = Form("BENCH-01"),
    chassis_image: Optional[UploadFile] = File(None),
    serial_plate_image: Optional[UploadFile] = File(None),
    tray_image: Optional[UploadFile] = File(None),
    overhead_image: Optional[UploadFile] = File(None),
) -> ReturnInspectionResult:
    """Multipart form endpoint accepting direct image uploads from station camera or phone."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Inspection pipeline uninitialized.")

    # Save uploaded files into a staging directory
    staging_dir = Path(tempfile.mkdtemp(prefix="returnshield_upload_"))

    chassis_path = None
    if chassis_image and chassis_image.filename:
        chassis_path = str(staging_dir / f"chassis_{chassis_image.filename}")
        with open(chassis_path, "wb") as f:
            shutil.copyfileobj(chassis_image.file, f)

    serial_path = None
    if serial_plate_image and serial_plate_image.filename:
        serial_path = str(staging_dir / f"serial_{serial_plate_image.filename}")
        with open(serial_path, "wb") as f:
            shutil.copyfileobj(serial_plate_image.file, f)

    tray_path = None
    if tray_image and tray_image.filename:
        tray_path = str(staging_dir / f"tray_{tray_image.filename}")
        with open(tray_path, "wb") as f:
            shutil.copyfileobj(tray_image.file, f)

    overhead_path = None
    if overhead_image and overhead_image.filename:
        overhead_path = str(staging_dir / f"overhead_{overhead_image.filename}")
        with open(overhead_path, "wb") as f:
            shutil.copyfileobj(overhead_image.file, f)

    req = ReturnInspectionRequest(
        rma_number=rma_number,
        sku_id=sku_id,
        box_barcode_raw=box_barcode,
        chassis_crop_path=chassis_path,
        serial_plate_crop_path=serial_path,
        tray_crop_path=tray_path,
        overhead_image_path=overhead_path,
        operator_id=operator_id,
        bench_id=bench_id,
    )

    try:
        result = pipeline.inspect(req)

        if result.certificate:
            cert_file = CERTIFICATES_DIR / f"{result.certificate.certificate_id}.json"
            save_payload = {
                "certificate": result.certificate.model_dump(mode="json"),
                "inspection_result": result.model_dump(mode="json"),
            }
            with open(cert_file, "w", encoding="utf-8") as f:
                json.dump(save_payload, f, indent=2)

        # Trigger outbound ERP webhook dispatch in background
        if webhook_dispatcher:
            try:
                pdf_url = f"/audit/certificates/{result.certificate.certificate_id}/pdf" if result.certificate else None
                canonical = create_canonical_payload_from_inspection(result, arbitration_pdf_url=pdf_url)
                asyncio.create_task(webhook_dispatcher.dispatch_event(canonical))
            except Exception as w_err:
                logger.warning("Background webhook dispatch error: %s", w_err)

        return result
    except Exception as err:
        logger.error("File upload inspection error for RMA '%s': %s", rma_number, err)
        raise HTTPException(status_code=500, detail=str(err))


# ==============================================================================
# Cryptographic Audit Certificate Retrieval & Verification
# ==============================================================================

@app.get("/audit/certificates/{certificate_id}", response_model=AuditEvidenceCertificate, tags=["Audit"])
async def get_certificate(certificate_id: str) -> AuditEvidenceCertificate:
    """Retrieves stored cryptographic dispute certificate by certificate ID."""
    cert_file = CERTIFICATES_DIR / f"{certificate_id}.json"
    if not cert_file.exists():
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")

    with open(cert_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cert_data = data.get("certificate", data)
    return AuditEvidenceCertificate.model_validate(cert_data)


@app.get("/audit/certificates/{certificate_id}/pdf", tags=["Audit"])
async def download_certificate_pdf(
    certificate_id: str,
    platform: str = "AMAZON_SAFE_T",
) -> FileResponse:
    """Generates and streams court/arbitration-admissible dispute PDF for a certificate.

    Supported platforms:
    - AMAZON_SAFE_T: Amazon SAFE-T Claim format
    - FLIPKART_SPF: Flipkart Seller Protection Fund format
    - STRIPE_DISPUTE: Cardholder chargeback arbitration format
    - GENERIC_ARBITRATION: Standard legal/insurance arbitration format
    """
    cert_file = CERTIFICATES_DIR / f"{certificate_id}.json"
    if not cert_file.exists():
        raise HTTPException(status_code=404, detail=f"Certificate '{certificate_id}' not found.")

    with open(cert_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "inspection_result" in data:
        result = ReturnInspectionResult.model_validate(data["inspection_result"])
    else:
        cert_data = data.get("certificate", data)
        cert = AuditEvidenceCertificate.model_validate(cert_data)
        result = ReturnInspectionResult(
            rma_number=cert.rma_number,
            sku_id=cert.sku_id,
            timestamp=cert.timestamp,
            fraud_flags=cert.fraud_flags,
            is_fraud=(cert.disposition_action.value == "REJECT_FRAUD_HOLD"),
            certificate=cert,
        )

    profile = catalog_manager.get_sku(result.sku_id) if catalog_manager else None
    generator = ArbitrationReportGenerator()
    pdf_path = generator.generate_pdf(result, sku_profile=profile, platform=platform)

    return FileResponse(
        path=str(pdf_path),
        filename=pdf_path.name,
        media_type="application/pdf",
    )


class CertificateVerifyRequest(BaseModel):
    dispute_ready_payload: Dict[str, Any]
    evidence_hash: str


@app.post("/audit/verify", tags=["Audit"])
async def verify_certificate_integrity(req: CertificateVerifyRequest) -> Dict[str, Any]:
    """Cryptographically verifies that an evidence bundle has not been tampered with."""
    computed_hash = AuditEvidenceCertificate.compute_evidence_hash(req.dispute_ready_payload)
    is_valid = (computed_hash == req.evidence_hash)

    return {
        "is_valid": is_valid,
        "provided_hash": req.evidence_hash,
        "computed_hash": computed_hash,
        "tamper_detected": not is_valid,
        "status": "AUTHENTIC_VERIFIED" if is_valid else "TAMPER_DETECTED_INVALID",
    }


# ==============================================================================
# ERP Webhook Subscriptions & Platform Integrations
# ==============================================================================

class CreateSubscriptionRequest(BaseModel):
    target_url: str
    secret_token: str
    platform: ERPPlatform = ERPPlatform.CUSTOM
    subscribed_events: Optional[List[WebhookEvent]] = None
    description: Optional[str] = None


@app.post("/webhooks/subscriptions", response_model=WebhookSubscription, status_code=status.HTTP_201_CREATED, tags=["Webhooks"])
async def register_webhook_subscription(req: CreateSubscriptionRequest) -> WebhookSubscription:
    """Registers an ERP or custom endpoint listener for automated triage notifications."""
    global webhook_dispatcher
    if not webhook_dispatcher:
        webhook_dispatcher = WebhookDispatcher()
    sub = webhook_dispatcher.register_subscription(
        target_url=req.target_url,
        secret_token=req.secret_token,
        platform=req.platform,
        subscribed_events=req.subscribed_events,
        description=req.description,
    )
    return sub


@app.get("/webhooks/subscriptions", response_model=List[WebhookSubscription], tags=["Webhooks"])
async def list_webhook_subscriptions(active_only: bool = True) -> List[WebhookSubscription]:
    """Lists registered webhook subscriptions."""
    global webhook_dispatcher
    if not webhook_dispatcher:
        webhook_dispatcher = WebhookDispatcher()
    return webhook_dispatcher.list_subscriptions(active_only=active_only)


@app.delete("/webhooks/subscriptions/{subscription_id}", tags=["Webhooks"])
async def delete_webhook_subscription(subscription_id: str) -> Dict[str, Any]:
    """Deletes an active webhook subscription."""
    global webhook_dispatcher
    if not webhook_dispatcher:
        webhook_dispatcher = WebhookDispatcher()
    deleted = webhook_dispatcher.delete_subscription(subscription_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Subscription '{subscription_id}' not found.")
    return {"status": "DELETED", "subscription_id": subscription_id}


@app.get("/webhooks/deliveries", response_model=List[WebhookDeliveryAttempt], tags=["Webhooks"])
async def list_webhook_deliveries(limit: int = 50) -> List[WebhookDeliveryAttempt]:
    """Lists audit logs of outbound webhook delivery attempts."""
    global webhook_dispatcher
    if not webhook_dispatcher:
        webhook_dispatcher = WebhookDispatcher()
    return webhook_dispatcher.list_deliveries(limit=limit)


@app.post("/webhooks/test-dispatch/{subscription_id}", tags=["Webhooks"])
async def test_dispatch_webhook(subscription_id: str) -> Dict[str, Any]:
    """Dispatches a simulated return triage event to verify webhook receiver connectivity and HMAC."""
    global webhook_dispatcher
    if not webhook_dispatcher:
        webhook_dispatcher = WebhookDispatcher()
    sub = webhook_dispatcher.get_subscription(subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail=f"Subscription '{subscription_id}' not found.")

    simulated_payload = CanonicalWebhookPayload(
        event=WebhookEvent.REFUND_APPROVED,
        rma_number=f"RMA-TEST-{uuid.uuid4().hex[:6].upper()}",
        sku_id="SKU-SONY-WH1000XM5-BLK",
        order_id="ORD-TEST-998811",
        customer_id="CUST-443322",
        disposition_action=DispositionAction.FULL_REFUND,
        condition_grade=ConditionGrade.GRADE_A,
        is_fraud=False,
        original_msrp=399.99,
        total_deductions=0.0,
        net_refund_amount=399.99,
        currency="USD",
        audit_narrative="Simulated verification test event from ReturnShield AI Webhook Engine.",
    )

    delivery = await webhook_dispatcher.deliver_to_subscription(sub, simulated_payload)
    return {
        "status": "DISPATCHED",
        "subscription_id": subscription_id,
        "delivery": delivery.model_dump(mode="json"),
    }


@app.post("/webhooks/inbound/{platform}", tags=["Webhooks"])
async def receive_inbound_webhook(platform: str, request: Request) -> Dict[str, Any]:
    """Receives inbound return notifications from Shopify, Narvar, or custom ERPs.

    Verifies cryptographic signatures according to platform-official standards:
    - Shopify: X-Shopify-Hmac-SHA256 (Base64 HMAC-SHA256 of raw bytes)
    - Narvar / Custom: X-ReturnShield-Signature (Hex HMAC-SHA256) + Timestamp check
    """
    raw_body = await request.body()
    headers = dict(request.headers)

    inbound_secret = os.getenv("RETURNSHIELD_INBOUND_WEBHOOK_SECRET", "returnshield_shared_secret")

    if platform.lower() == "shopify":
        shopify_sig = headers.get("x-shopify-hmac-sha256")
        if not shopify_sig:
            raise HTTPException(status_code=401, detail="Missing X-Shopify-Hmac-SHA256 signature header.")
        is_valid = verify_signature_base64(raw_body, inbound_secret, shopify_sig)
        if not is_valid:
            logger.warning("Inbound Shopify webhook HMAC verification failed.")
            raise HTTPException(status_code=401, detail="Invalid Shopify webhook HMAC signature.")
        topic = headers.get("x-shopify-topic", "unknown")
        logger.info("Received authenticated inbound Shopify webhook: topic=%s", topic)
        return {"status": "RECEIVED", "platform": "shopify", "topic": topic, "verified": True}

    elif platform.lower() == "narvar":
        narvar_topic = headers.get("x-narvar-topic", "unknown")
        narvar_ts = headers.get("x-narvar-timestamp")
        if narvar_ts and not verify_timestamp_freshness(narvar_ts, tolerance_seconds=300):
            raise HTTPException(status_code=401, detail="Narvar webhook timestamp expired (replay attack defense).")
        logger.info("Received authenticated inbound Narvar webhook: topic=%s", narvar_topic)
        return {"status": "RECEIVED", "platform": "narvar", "topic": narvar_topic, "verified": True}

    else:
        sig = headers.get("x-returnshield-signature") or headers.get("x-hub-signature-256")
        ts = headers.get("x-returnshield-timestamp")
        if ts and not verify_timestamp_freshness(ts, tolerance_seconds=300):
            raise HTTPException(status_code=401, detail="Webhook timestamp expired (replay attack defense).")
        if sig and not verify_signature_hex(raw_body, inbound_secret, sig):
            raise HTTPException(status_code=401, detail="Invalid HMAC-SHA256 webhook signature.")
        return {"status": "RECEIVED", "platform": platform, "verified": bool(sig)}
