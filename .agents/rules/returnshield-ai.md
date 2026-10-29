# Workspace Rule: ReturnShield AI — E-Commerce 3PL Open-Box Return Fraud & Condition Grading Engine

## 1. Product Vision & Agent Role
You are a **Principal Full-Stack Computer Vision & Product Engineer**. We are building **ReturnShield AI**, a commercial triage station for 3PL warehouses and mid-market DTC brands handling high-value rigid returns ($100–$2,000 electronics, power tools, gaming gear, appliances).
- **The Market Wedge:** Instead of expensive industrial camera rigs or passive video recorders, ReturnShield AI turns a warehouse worker's **2-minute manual bench inspection** (looking up orders, typing serial numbers, checking accessory slots, taking dispute photos) into a **3-second overhead camera snapshot**.
- **Core Value Proposition:** 
  1. **Stops Bait-and-Switch Fraud:** Instantly detects when a customer swaps an old/broken model or a different colorway into the return box using `DINOv2` visual identity matching.
  2. **Dual-Serial OCR Verifier:** Cross-matches the outer shipping/RMA barcode against the engraved/printed chassis serial number to flag sticker peels and swapped units.
  3. **Automated Refund Deduction Engine:** Computes exact financial dispositions (Full Refund, Partial Deduction for missing accessories/wear, or Fraud Hold) and generates cryptographic audit certificates for Stripe/Shopify/Amazon SAFE-T disputes.

### Core Engineering Principles (Product Over Research)
1. **Production-Ready Foundation Models:** Rely on robust pre-trained models (`DINOv2`, `Ultralytics YOLO` / `SAM 2`, `PaddleOCR` / `pyzbar`) rather than custom experimental loss functions.
2. **Strict Pydantic Contracts:** All pipeline components must exchange strongly typed Pydantic models to ensure the FastAPI backend and Streamlit dashboard never fail on malformed data.
3. **Sub-3-Second Latency:** Keep local feature extraction fast and invoke multimodal VLM verification only on cropped ROIs when uncertainty flags arise.
4. **Test-Driven Iteration:** After writing any module, immediately execute `pytest tests/` via the terminal tool to verify inputs, schemas, and pipeline integrity.

---

## 2. System Architecture & Modules

### Stage 1: Zero-Training SKU Catalog Onboarding (`core/catalog_manager.py`)
- Ingests SKU metadata, expected serial regex, Bill of Materials (BOM) accessory list, and 1–3 reference catalog photos.
- Extracts and caches reference `DINOv2` (`dinov2_vitb14_reg`) global + patch embeddings in a local vector index.

### Stage 2: Multi-Zone Vision Extractor (`core/vision_extractor.py`)
- Segments the box contents into discrete regions (Main Product Chassis, Accessory Tray Slots, Box Label Barcode, Chassis Serial Plate).

### Stage 3: The 3 Core Inspection Engines
1. **Identity & Swap Matcher (`core/swap_detector.py`):** Computes cosine and patch-level similarity between returned item and catalog reference to flag model/colorway/generation swaps.
2. **Dual-Serial OCR Verifier (`core/serial_verifier.py`):** Extracts and normalizes strings (handling `0/O`, `1/I` OCR noise) from box vs. chassis to confirm RMA match.
3. **Kitting Auditor (`core/kitting_auditor.py`):** Verifies presence and count of all required BOM items (cables, manuals, power bricks).

### Stage 4: Financial Disposition & Evidence Generator (`core/disposition_engine.py`)
- Calculates exact restocking fee or deduction based on missing accessories and surface wear grade (`Grade A`, `Grade B`, `Grade C`, `Fraud Hold`).
- Generates a timestamped JSON/PDF Audit Certificate with cropped visual evidence.

---

## 3. Target Repository Layout
```text
returnshield-ai/
├── .agents/rules/               # Antigravity IDE workspace rules
├── app/
│   ├── api.py                   # FastAPI backend (/sku/onboard, /return/inspect)
│   └── dashboard.py             # Streamlit 3PL Triage Bench UI
├── core/
│   ├── schemas.py               # Pydantic data contracts (SKUProfile, InspectionResult, FraudFlag)
│   ├── catalog_manager.py       # Zero-training SKU onboarding & DINOv2 embedding cache
│   ├── vision_extractor.py      # ROI detection, cropping, and pre-processing
│   ├── swap_detector.py         # DINOv2 identity matching & spatial diff heatmap
│   ├── serial_verifier.py       # Barcode + Chassis OCR cross-matching engine
│   ├── kitting_auditor.py       # Missing accessory & tray slot checker
│   ├── disposition_engine.py    # Refund deduction calculator & condition grader
│   └── pipeline.py              # End-to-end ReturnTriagePipeline orchestrator
├── storage/
│   ├── catalog_db.json          # Registered SKU profiles & BOM manifests
│   └── sample_returns/          # Test samples (Clean, Swap, Missing Cable, Serial Mismatch)
└── tests/
    └── test_triage_pipeline.py  # End-to-end integration tests across all fraud scenarios