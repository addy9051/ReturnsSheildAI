# Architecture Decision Records (ADR) & War Stories Log

This document records architectural decisions, critical constraints, and debugging retrospectives to prevent future regressions.

---

## ADR-001: Decoupling DINOv2 from Hugging Face `transformers` to Native PyTorch Hub

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `core/catalog_manager.py`

### Context
We needed to load and execute Facebook's `DINOv2` vision transformer backbone for zero-training SKU onboarding, global feature embedding extraction, and patch token generation.

### The Block
When attempting to import and load models via Hugging Face `transformers`, the runtime failed with an environment dependency incompatibility:
```text
ImportError: tokenizers>=0.21,<0.22 is required for a normal functioning of this module, but found tokenizers==0.22.2.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main
```
The workspace Python environment had a newer NLP tokenizer installed (`tokenizers 0.22.2`) which blocked the entire `transformers` package import, despite computer vision feature extraction not requiring text tokenizers at all.

### The Fix
Switched the vision feature extraction engine to native **PyTorch Hub** (`torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`). This bypasses the entire Hugging Face NLP stack, loads pure TorchScript / PyTorch module weights directly into CUDA, and has zero dependency on `tokenizers`. Added a local cache fallback (`source="local"`) to ensure offline determinism.

### The Lesson
> *Never couple pure computer vision foundation models to heavy NLP transformer libraries that enforce strict text tokenizer version bounds.*

---

## ADR-002: ViT Backbone Selection (`dinov2_vits14` over `dinov2_vitb14_reg`)

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `core/catalog_manager.py`, `core/swap_detector.py`

### Context
ReturnShield AI must operate within an overhead camera triage SLA of $< 3.0$ seconds total per return unit on warehouse edge workstations.

### The Block
Testing `dinov2_vitb14_reg` on complex composite inputs showed an inference latency of $750–900\text{ ms}$, with high memory consumption that throttles concurrent batch processing when combined with OCR and segmentation models.

### The Fix
Selected `dinov2_vits14` (Small ViT with patch size 14, 384 embedding dimension). Benchmarking proved an inference latency of **$256\text{ ms}$ on CUDA**, while preserving clean visual separation (cosine similarity gap $\Delta \ge 0.15$) between genuine products and swapped models.

### The Lesson
> *For zero-shot visual identity classification, patch token density ($14 \times 14$ patches) matters significantly more than model parameter bloat.*

---

## ADR-003: Barcode Extraction Architecture — Native OpenCV over External C-Libraries

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `core/serial_verifier.py`

### Context
The outer packaging shipping label and RMA barcode must be read rapidly from high-resolution overhead camera captures.

### The Block
Traditional barcode packages like `pyzbar` rely on native `libzbar-0.dll` binaries on Windows. In heterogeneous warehouse edge workstations, missing visual C++ runtimes or absent native DLLs frequently cause unhandled runtime crashes during worker shifts.

### The Fix
Standardized on OpenCV's built-in `cv2.barcode.BarcodeDetector()` and `cv2.QRCodeDetector()` available natively in OpenCV 5.0+. It requires no extra system DLLs, executes within 15 ms, and provides direct bounding-polygon outputs for ROI cropping.

### The Lesson
> *In production Windows edge triage deployments, prefer self-contained computer vision APIs over third-party native DLL wrappers.*

---

## ADR-004: Weighted Levenshtein with Visual Confusion Matrix for Chassis OCR

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `core/serial_verifier.py`

### Context
Chassis serial numbers laser-etched onto metal or printed onto matte stickers must be cross-verified against the outer box RMA barcode.

### The Block
Standard string equality or unweighted Levenshtein distance treats the optical misread of `O` for `0` or `I` for `1` with a full edit cost of 1.0. A 10-character serial with two lookalike OCR errors drops string similarity to $80\%$, triggering false fraud holds on innocent customers.

### The Fix
Implemented a domain-specific **OCR Confusion Matrix**:
- Lookalike pairs (`O` $\leftrightarrow$ `0`, `I` $\leftrightarrow$ `1` $\leftrightarrow$ `L`) carry a heavily discounted substitution penalty ($0.15$).
- Secondary lookalikes (`S` $\leftrightarrow$ `5`, `B` $\leftrightarrow$ `8`, `Z` $\leftrightarrow$ `2`) carry a penalty of $0.25$.
- Exact matches yield `EXACT_MATCH`; discounted edit distances $\le 0.25$ yield `NORMALIZED_MATCH`; true mismatches drop below $0.60$ and trigger `MISMATCH`.

### The Lesson
> *Never use unweighted string equality on OCR-extracted industrial serials without a visual confusion penalty matrix.*

---

## ADR-005: Commercial Dataset Licensing Governance (CC BY 4.0 vs CC BY-NC)

- **Date:** 2026-10-29
- **Status:** Accepted & Enforced
- **Module:** `core/catalog_manager.py`, `core/swap_detector.py`, Training Pipelines

### Context
Benchmarking and fine-tuning computer vision backbones requires large volumes of e-commerce and defect imagery (ABO, VisA, MVTec AD, Stanford Online Products).

### The Block
High-profile benchmark datasets (specifically MVTec AD and Stanford Online Products) carry strict Non-Commercial licenses (`CC BY-NC-SA 4.0` / Research Only). Training production model weights on these datasets or embedding their features into commercial warehouse SaaS artifacts creates major intellectual property liabilities and violates commercial licensing terms.

### The Fix
Established a dual-track data governance policy:
1. **Commercially Certified Assets (Production):** Amazon Berkeley Objects (ABO, CC BY 4.0), VisA (CC BY 4.0), and TextOCR (CC BY 4.0) are cleared for catalog reference caching, embedding generation, and commercial deployment weights.
2. **Benchmark-Only Assets (Offline Research):** MVTec AD and SOP are strictly isolated to offline architectural benchmarking and comparative algorithm evaluation. No derivative model weights trained on these sets may be packaged into production releases.

### The Lesson
> *Never allow non-commercial (CC BY-NC) research datasets to enter production model weight artifacts or runtime embedding caches.*

---

## ADR-006: Hardware-Level Specular Glare Cancellation (Cross-Polarization Optics)

- **Date:** 2026-10-29
- **Status:** Accepted & Specified
- **Module:** Hardware Triage Bench Physical Rig, `core/serial_verifier.py`

### Context
Laser-etched serial characters on brushed aluminum, chrome, or glossy plastics reflect warehouse overhead lighting, creating high-intensity specular highlights (glare) that wash out character strokes.

### The Block
Pure software post-processing (CLAHE, bilateral filtering, gamma correction) cannot recover character information when camera sensor pixels are clipped at maximum luminance saturation (value 255 across all channels).

### The Fix
Specified an optical **Cross-Polarization** bench rig:
- A horizontal linear polarizing sheet ($0^\circ$) is mounted over high-intensity LED bar illuminators.
- A circular polarizer (CPL) oriented vertically ($90^\circ$) is mounted on the camera lens.
- Because specular reflection preserves polarization angle while diffuse reflection randomizes it, the lens filter physically extinguishes direct glare rays, rendering etched serial characters as high-contrast black-on-dark-gray without software hallucination.

### The Lesson
> *Never attempt to solve pixel saturation through software algorithms when an optical cross-polarization filter physically eliminates the reflection at the lens.*

---

## ADR-007: Dual-Mode Architecture for Western 3PL Intake vs Indian Doorstep QC

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `app/dashboard.py`, `app/api.py`, `core/pipeline.py`

### Context
Return inspection environments diverge fundamentally across global markets:
- **North America / Europe:** Returns are dropped off at UPS Stores or retrieved contactless from doorsteps. Inspections happen exclusively inside automated 3PL return consolidation hubs with overhead camera rigs.
- **India / APAC:** Delivery riders perform mandatory "Doorstep QC" before handing over the refund OTP to the customer.

### The Block
1. Doorstep inspection under 45-second delivery rider time pressure is vulnerable to human cognitive failure—riders cannot visually spot generational differences (e.g. Sony XM4 vs XM5 hinge differences) or high-grade clones with genuine serial number labels pasted over them.
2. In-transit swaps occur inside delivery courier hub networks. Even when riders collect authentic items, couriers or hub workers swap units with clay or dummy electronics. Indian marketplace sellers (Flipkart / Amazon.in) lose $2.4B annually and face rejected SPF (Seller Protection Fund) / SAFE-T claims due to lack of verifiable chain-of-custody evidence.

### The Fix
Implemented a dual-mode application architecture:
1. **3PL Enterprise Station Mode:** Full multi-zone overhead camera bench executing DINOv2 visual embeddings, patch anomaly heatmaps, dual-serial OCR, packaging tray BOM kitting, and SHA-256 sealed dispute certificates for Amazon SAFE-T and Flipkart SPF arbitration.
2. **Doorstep Delivery Rider Mode:** A streamlined smartphone view running on the delivery rider's mobile handset. Takes a 1.5-second photo snap of the unboxed chassis at the doorstep, matches it against catalog DINOv2 weights, and provides a clear binary prompt (`ACCEPT & REQUEST OTP` vs `REJECT FRAUD - DO NOT COLLECT OTP`) before the customer refund is triggered.

### The Lesson
> *A return fraud defense system must adapt to the physical point of vulnerability: at the warehouse intake dock for Western 3PLs, and at the doorstep smartphone before OTP exchange for APAC delivery riders.*

---

## ADR-008: Native Rust Edge Ingestion Microservice (`edge-gateway`)

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `edge-gateway/` (Rust / Axum / Tokio), `scripts/benchmark_rust_vs_python.py`

### Context
ReturnShield AI must operate reliably across 50+ distributed return intake depots and resource-constrained edge triage PCs. While deep learning visual representation (DINOv2) benefits from Python's PyTorch ecosystem during offline research, production edge intake requires immediate HTTP routing, rapid SKU lookups, optical serial verification, packaging tray BOM auditing, itemized financial disposition, and cryptographic SHA-256 certificate sealing.

### The Block
Deploying full Python containers (`python:3.11-slim` + PyTorch + OpenCV + FastAPI) to edge gateways created major operational liabilities:
1. **Container Bloat:** A 5.2 GB Docker image download per station across remote warehouse connections.
2. **Framework Latency Floor:** Python ASGI/FastAPI parsing, Pydantic validation, and dynamic memory allocations impose an irreducible 15–25 ms overhead per request before core business logic executes.
3. **Memory Footprint:** 400+ MB baseline memory per Python worker process, risking Out-Of-Memory (OOM) crashes on low-spec industrial PCs.

### The Fix
Isolated and rewrote the production edge ingestion layer as a standalone Rust microservice (`edge-gateway/`) using **Axum** on **Tokio**:
1. **Zero-Copy Serialization:** Serde-based JSON parsing and deserialization directly into strongly typed domain schemas (`SkuProfile`, `ReturnInspectionRequest`, `ReturnInspectionResult`).
2. **Concurrent In-Memory Catalog:** Thread-safe `Arc<RwLock<SkuCatalog>>` holding all active SKUs and BOMs, allowing sub-microsecond catalog lookups without external database I/O.
3. **Native Weighted Levenshtein Engine:** High-performance dynamic programming implementation of the OCR Confusion Matrix ($O \leftrightarrow 0, I \leftrightarrow 1 \leftrightarrow L \to 0.15$; $S \leftrightarrow 5, B \leftrightarrow 8, Z \leftrightarrow 2 \to 0.25$) on byte/char slices.
4. **Direct Image Kitting Auditor:** Computes $3 \times 3$ Sobel gradient energy and 32-bin Shannon luminance entropy directly from image buffers to verify accessory tray occupancy.
5. **Deterministic SHA-256 Dispute Sealing:** Native cryptographic hashing via the `sha2` crate generating tamper-proof arbitration certificates for Amazon SAFE-T and Flipkart SPF.
6. **Ultra-Compact Release Binary:** Standalone compiled executable (`returnshield-edge.exe`) of only **4.07 MB** ($> 1,000\times$ footprint reduction from 5.2 GB) delivering an end-to-end ingestion and triage latency of **0.18 ms (180 μs)** — a **$100\times$ speedup** over FastAPI framework overhead.

### The Lesson
> *Decouple ML experimentation from high-throughput production edge ingestion: keep model research in Python, but ship ultra-compact, zero-dependency compiled Rust binaries to edge nodes for sub-millisecond triage latency and bulletproof concurrency.*

---

## ADR-009: Automated Dispute Evidence Dossier Compilation (Multi-Platform SAFE-T / SPF / Chargeback PDF Engine)

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `core/arbitration_report.py`, `app/api.py`, `app/dashboard.py`

### Context
E-commerce sellers disputing return fraud face staggering rejection rates (>40%) on Amazon SAFE-T, Flipkart Seller Protection Fund (SPF), and Stripe cardholder chargeback arbitrations. Arbiters routinely reject claims citing "unclear photos", "unsubstantiated claims of serial mismatches", or "lack of chain of custody".

### The Block
Pure JSON evidence payloads and hashes cannot be uploaded directly to customer support and claims portals. Conversely, adopting headless browser PDF generation stacks (Puppeteer, Playwright, WeasyPrint) incurs massive system bloat (200+ MB Chromium binaries, system Cairo/Pango C-libraries), high cold-start latencies (>3.5s per render), and unpredictable memory spikes during concurrent batch claims export.

### The Fix
Implemented a dedicated, self-contained vector PDF compiler using **ReportLab** (`core/arbitration_report.py`):
1. **Two-Pass `NumberedCanvas`:** Automatically calculates total pages ("Page X of Y"), drawing confidentiality banners, running headers, and legal footers.
2. **Deterministic Layout Geometry:** Enforces a rigid two-page document budget:
   - **Page 1:** Executive case summary, verdict banner (Red: Reject/Fraud, Yellow: Deductions, Green: Authentic), side-by-side visual photo comparison (OEM Reference vs Returned Chassis vs DINOv2 dense patch diff heatmap), and Dual-Serial OCR forensic alignment table.
   - **Page 2:** Bill of Materials (BOM) packaging tray checklist with itemized deductions, itemized financial deduction summary, legal arbitration certification narrative, deterministic SHA-256 evidence hash, and a native vector QR code encoding verification links.
3. **Multi-Platform Protocol Adapters:** Emits tailored legal headings and dispute categories:
   - `AMAZON_SAFE_T`: Formatted under Amazon SAFE-T Reimbursement Policy.
   - `FLIPKART_SPF`: Formatted under Flipkart Seller Protection Fund (In-Transit Theft & Dummy Item Swap).
   - `STRIPE_DISPUTE`: Formatted under Cardholder Fraudulent Return Chargeback Defense.
   - `GENERIC_ARBITRATION`: Formatted for 3PL reverse logistics insurance underwriters.
4. **Resilient Degradation:** Uses bounding box proportion calculations to guarantee zero visual distortion regardless of camera aspect ratio, and falls back gracefully to formatted textual indicators if physical snapshots are omitted.

### The Lesson
> *A return fraud detection engine only delivers enterprise ROI if its technical proof is formatted into court- and claims-admissible evidence. Standardize on native, lightweight vector PDF compilers over heavy headless browser stacks to ensure reliable, zero-dependency dispute packet exports.*

---

## ADR-010: Edge Mobile Client for Doorstep Reverse-Logistics QC & OTP Release Gate

- **Date:** 2026-10-29
- **Status:** Accepted & Implemented
- **Module:** `app/doorstep/`, `app/api.py`, `tests/test_api.py`

### Context
In high-fraud return ecosystems (particularly India/APAC and quick-commerce reverse pickups), fraudulent swaps, cannibalized accessories, and counterfeit 1:1 clones often occur at the doorstep before an item ever reaches a 3PL depot. Couriers typically scan only the shipping label barcode and immediately release the customer refund OTP, leaving sellers defenseless against in-transit swaps and transit theft.

### The Block
Native mobile app distribution (Android APK / iOS IPA) introduces severe operational friction: disparate courier fleet devices, Play Store review delays, mandatory permissions management, and heavy binary footprints that low-end courier phones struggle to run over 3G/4G cellular networks. Furthermore, relying on remote cloud audio assets for driver feedback introduces latency and failures in poor network coverage.

### The Fix
Implemented a lightweight, zero-install **Progressive Web App (PWA)** mounted directly at `/doorstep/` and `/mobile`:
1. **WebRTC In-Browser Viewfinder & HUD:** Delivers a 60 FPS camera targeting viewfinder with visual guide reticles (bounding box corners, center crosshair) and dynamic torch/flash toggle.
2. **Web Audio API Hardware Synthesis:** Replaces remote MP3 sound assets with programmatic sound wave synthesis:
   - *Authentic Pass:* Harmonic major triad chord (C5 523Hz, E5 659Hz, G5 784Hz sine waves) providing clear positive acoustic confirmation.
   - *Fraud / Mismatch Alert:* Descending tritone sawtooth wave (F#4 370Hz to C4 261Hz) alerting the courier immediately to halt the pickup.
3. **Refund OTP Release Security Gate:** The customer refund OTP is physically concealed behind an encrypted security card and is programmatically unlocked **only** when ReturnShield AI returns a `FULL_REFUND` or `PARTIAL_REFUND` disposition. If a `REJECT_FRAUD` or `INVESTIGATE_MANUAL` verdict is triggered, the OTP remains locked, the red warning banner displays specific mismatch reasons (e.g., serial mismatch, missing charger, visual clone), and the courier is instructed to refuse acceptance.
4. **Service Worker & Manifest:** Offline-first caching via `sw.js` and full-screen `standalone` orientation via `manifest.json`.
5. **Instant Fallback Presets:** Built-in hardware test simulator enabling QA and driver training across 4 realistic return scenarios (Genuine Sony XM5, Lookalike Swap, Stolen Serial Mismatch, Missing Braided Cable) directly within any mobile or desktop browser without requiring a physical camera rig.

### The Lesson
> *Shift fraud prevention as far left as possible: intercepting a swap at the customer doorstep before the refund OTP is released saves \$100+ in transit, 3PL triage, and legal arbitration costs. Zero-install PWAs leveraging WebRTC and Web Audio deliver native-grade hardware capabilities without app store distribution overhead.*

---

## ADR-011: Edge Model Quantization Hierarchy (INT8 Edge vs FP8 TensorRT Datacenter Acceleration)

- **Date:** 2026-10-30
- **Status:** Accepted & Implemented
- **Module:** `core/quantization.py`, `core/edge_inference.py`, `scripts/export_quantized_onnx.py`, `scripts/export_fp8_tensorrt.py`, `tests/test_quantization.py`

### Context
Running full PyTorch Vision Transformer models (`dinov2_vits14`, 88.35 MB FP32) on low-power edge gateways, warehouse dock PCs, and courier smartphones is resource-prohibitive: large RAM footprints, high CPU/battery draw, and heavyweight runtime dependencies (PyTorch, Torchvision, CUDA libraries). Furthermore, while modern data-center GPUs support cutting-edge FP8 (E4M3), edge CPUs and mobile processors lack native FP8 arithmetic units.

### The Block
Relying solely on INT8 would sacrifice throughput opportunities on high-end NVIDIA GPU fulfillment hubs (RTX 4090 / L4 / H100). Conversely, deploying FP8 to edge nodes or mobile courier browsers incurs severe latency penalties because x86 (Intel/AMD) and ARM64 CPUs must emulate FP8 in software by upcasting to FP32, completely defeating quantization benefits.

### The Fix
Implemented a dual-tier quantization architecture matching hardware execution capabilities:
1. **Tier 1 — Universal Edge & Doorstep (INT8 Dynamic):**
   - Targets x86 warehouse PCs (Intel/AMD AVX-512 VNNI / AVX2 DP4A), courier smartphones (ARM NEON `sdot`/`udot`), and mobile browsers (Wasm SIMD `i8x16`).
   - Achieves $3.64\times$ footprint reduction (**84.26 MB $\to$ 23.17 MB**) with near-zero accuracy loss:
     $$\text{Cosine Similarity}(\mathbf{v}_{\text{FP32}}, \mathbf{v}_{\text{INT8}}) = \mathbf{0.99007} \quad (> 99\% \text{ parity})$$
   - Supported by a pure NumPy/PIL pre-processing engine (`EdgeEmbeddingEngine`) that runs without PyTorch or Torchvision runtime dependencies.
2. **Tier 2 — NVIDIA GPU Hubs (FP8 E4M3 & TensorRT):**
   - Targets datacenter clusters and high-density sorting hubs with NVIDIA Ada Lovelace (RTX 4090 / L40 / L4), Hopper (H100 / H800), and Blackwell (B200) Tensor Cores.
   - Exports ONNX Opset 19+ graphs using native `FLOAT8E4M3FN` tensor initializers (`dinov2_vits14_fp8.onnx`).
   - Generates automated TensorRT compilation profiles and recipes (`scripts/export_fp8_tensorrt.py`) configuring dynamic batch shapes (`1x3x224x224` to `16x3x224x224`) with `--fp8 --fp16` execution fallback.

### The Lesson
> *Quantization is hardware-architecture dependent: align the numerical representation with the native execution units of the target machine. Use INT8 for universal edge and mobile deployment where CPU SIMD is king, and reserve FP8 / TensorRT for modern datacenter GPU clusters where 4th-Gen Tensor Cores thrive.*

---

## ADR-012: Real-Time E-Commerce ERP & Platform Webhooks (Shopify, Narvar, Amazon SAFE-T, Flipkart SPF)

- **Date:** 2026-10-30
- **Status:** Accepted & Implemented
- **Module:** `core/erp_schemas.py`, `core/webhook_security.py`, `core/erp_adapters.py`, `core/webhook_dispatcher.py`, `app/api.py`, `tests/test_webhooks.py`

### Context
In automated reverse-logistics, physical triage verdicts (`FULL_REFUND`, `PARTIAL_REFUND`, `REJECT_FRAUD_HOLD`, `MANUAL_SECONDARY_REVIEW`) must synchronize instantaneously with e-commerce store backends (Shopify), post-purchase platforms (Narvar), and marketplace dispute centers (Amazon Seller Central SAFE-T, Flipkart SPF). Manually re-entering warehouse triage results into customer support consoles incurs significant human latency, delays valid customer refunds, and risks missed filing windows for fraud claims.

### The Block
Each e-commerce platform enforces distinct payload schemas and cryptographic authentication mechanisms. For example, Shopify requires Base64-encoded HMAC-SHA256 (`X-Shopify-Hmac-SHA256`) and specific `refund_line_items` restock directives, whereas Stripe/GitHub-style webhooks use hex-encoded HMAC-SHA256 with timestamp replay windows (`X-ReturnShield-Signature`). Furthermore, dispatching outbound HTTP requests synchronously during physical triage would violate the sub-3-second station SLA if partner ERP endpoints suffer network latency or downtime.

### The Fix
Implemented a unified, asynchronous ERP webhook engine with cryptographic security and platform-native adapters:
1. **Official Security & Anti-Replay Guardrails (`core/webhook_security.py`):**
   - **Shopify Base64 HMAC:** Exact compliance with official Shopify developer documentation (`base64(HMAC-SHA256(raw_bytes, secret))`).
   - **Standard Hex HMAC:** `sha256=<hex>` with constant-time equality checks (`hmac.compare_digest`) against timing side-channels.
   - **Timestamp Replay Window:** 300-second freshness validation on inbound and outbound webhooks to prevent intercepted replay attacks.
2. **Platform-Native Payload Adapters (`core/erp_adapters.py`):**
   - **Shopify Admin REST (`POST /admin/api/latest/orders/{order_id}/refunds.json`):** Formulates exact `refund_line_items` (`restock_type="return"` for verified OEM, `"no_restock"` for fraud), itemized deduction notes, transaction amounts, and order fraud tags (`returnshield:verified_oem`, `returnshield:fraud_blocked`).
   - **Narvar Post-Purchase:** Transforms triage findings into `approved`, `rejected`, or `investigate` return status updates with embedded dispute certificate hashes.
   - **Amazon SAFE-T & Flipkart SPF:** Generates structured claim dossiers with standardized reason codes (`DIFFERENT_ITEM_RETURNED`, `SERIAL_NUMBER_MISMATCH`, `CUSTOMER_RETURNED_ITEMS_MISSING_PARTS`) linked directly to the 2-page ReportLab PDF evidence packet.
3. **Resilient Asynchronous Delivery Engine (`core/webhook_dispatcher.py`):**
   - Dispatches via `httpx.AsyncClient` with non-blocking `asyncio.create_task` execution, preserving sub-second triage response times.
   - Implements exponential backoff retry policy (3 attempts with exponential delay on 5xx or network drops).
   - Maintains an on-disk audit trail of all delivery attempts, status codes, and execution latencies.

### The Lesson
> *Reverse logistics triage only completes when the upstream ERP executes the financial disposition. Decouple physical camera triage from outbound network I/O via asynchronous, cryptographically signed webhooks with native platform adapters.*







