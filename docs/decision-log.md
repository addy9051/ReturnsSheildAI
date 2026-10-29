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
