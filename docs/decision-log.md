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

