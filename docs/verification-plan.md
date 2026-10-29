# ReturnShield AI: Mathematical Verification Plan & Runbook

This runbook defines the repeatable, automated, and human-auditable steps required to mathematically prove system integrity across each phase.

---

## 1. Phase 1 Verification (Core Schemas & Catalog Manager)

### Step 1.1: Pydantic Schema Validation & Automatic Invariants
Run the isolated schema unit tests:
```powershell
pytest tests/test_triage_pipeline.py -k "test_sku_profile_validation or test_return_inspection_result_fraud_sync or test_cryptographic_evidence_certificate" -v
```
**Expected Mathematical Proof:**
- `test_sku_profile_validation`: Fails if MSRP $\le 0.0$ or Restocking Fee $> 1.0$.
- `test_return_inspection_result_fraud_sync`: Proves `is_fraud == True` if and only if fraud flags contain `BAIT_AND_SWITCH` or `SERIAL_MISMATCH`.
- `test_cryptographic_evidence_certificate`: Proves SHA-256 hash output is 64 characters long and changes deterministically upon 1-bit payload modification.

### Step 1.2: DINOv2 Feature Extractor & Catalog Persistence
Run the catalog manager lifecycle tests:
```powershell
pytest tests/test_triage_pipeline.py -k "test_catalog_manager_onboarding_and_caching or test_dinov2_similarity_scoring or test_catalog_search" -v
```
**Expected Mathematical Proof:**
- Global feature vector L2-norm: $\|\mathbf{v}\|_2 = 1.0 \pm 10^{-4}$.
- Self-similarity: $\text{sim}(\text{ref}, \text{ref}) \ge 0.99$.
- Clean separation gap: $\text{sim}(\text{genuine}) - \text{sim}(\text{swapped}) > 0.15$.
- Retrieval accuracy: Catalog search returns target SKU at Rank 1.

### Step 1.3: Synthetic Dataset Scenarios
Run end-to-end evaluation over all 5 mock return scenarios:
```powershell
pytest tests/test_triage_pipeline.py -k "test_scenario" -v
```
**Expected Proof Gates:**
- `CASE-01-CLEAN`: Identity `VERIFIED_OEM`, Serial `EXACT_MATCH`, BOM missing $= 0$, Disposition `FULL_REFUND`.
- `CASE-02-SWAP`: Identity `CONFIRMED_SWAP`, Serial `MISMATCH`, Disposition `REJECT_FRAUD_HOLD`.
- `CASE-03-SERIAL-MISMATCH`: Identity `VERIFIED_OEM`, Serial `MISMATCH`, Disposition `REJECT_FRAUD_HOLD`.
- `CASE-04-MISSING-ACCESSORIES`: Identity `VERIFIED_OEM`, Serial `EXACT_MATCH`, Deductions $= \$50.00$, Disposition `PARTIAL_REFUND`.
- `CASE-05-COSMETIC-WEAR`: Condition `GRADE_C`, Disposition `PARTIAL_REFUND`.

---

## 2. Phase 2 Verification (Swap Detector & Dual-Serial Verifier)

*To be executed upon implementation of `core/swap_detector.py` and `core/serial_verifier.py`:*

### Step 2.1: Spatial Diff Heatmap Generation Proof
```powershell
pytest tests/test_swap_detector.py -v
```
**Verification Assertions:**
- Output heatmap file must be generated at `<storage>/sample_returns/<case>/heatmap.png`.
- Output image channels $= 3$ (RGB/BGR), dimensions match original input ROI exactly.
- Peak anomaly score on genuine return $\le 0.35$; peak anomaly score on swapped model $\ge 0.60$.

### Step 2.2: Dual-Serial OCR Ambiguity Resolution Proof
```powershell
pytest tests/test_serial_verifier.py -v
```
**Verification Assertions:**
- Optical noise (`0` vs `O`, `1` vs `I`) resolves to `NORMALIZED_MATCH` with similarity score $\ge 0.95$.
- Genuine chassis serial mismatch resolves to `MISMATCH` with similarity score $< 0.60$.
- Barcode decoding latency $< 50\text{ ms}$; OCR latency $< 400\text{ ms}$.

---

## 3. Phase 3 Verification (Kitting Auditor, Financial Disposition & Master Pipeline)

### Step 3.1: BOM Slot Anomaly Metrics Proof
```powershell
pytest tests/test_kitting_auditor.py -v
```
**Verification Assertions:**
- Populated slot Sobel edge gradient energy $> 2.5\times$ empty molded foam cavity.
- Populated slot Shannon luminance entropy significantly exceeds uniform empty cavity.
- Missing BOM items correctly deduct itemized penalty fee from net refund without triggering fraudulent swap false positive.

### Step 3.2: Financial Disposition & Dispute Certificate Proof
```powershell
pytest tests/test_disposition_engine.py -v
```
**Verification Assertions:**
- Mint/Like-new return $\to$ `ConditionGrade.GRADE_A`, `DispositionAction.FULL_REFUND`.
- Confirmed bait-and-switch $\to$ `ConditionGrade.FRAUD_HOLD`, `DispositionAction.REJECT_FRAUD_HOLD`, $\$0.00$ refund.
- 1-bit payload modification changes deterministic SHA-256 seal (tamper detection).

### Step 3.3: Master Pipeline End-to-End & Station SLA Gate
```powershell
pytest tests/test_pipeline_e2e.py -v
```
**Verification Assertions:**
- End-to-end multi-module inspection executes warm within sub-3.0 second SLA (Measured: $\approx 1.1\text{ s}$).
- All 5 standard return scenarios (`CASE-01-CLEAN` through `CASE-05-COSMETIC-WEAR`) route to their mathematically expected disposition.

---

## 4. Phase 4 Verification (FastAPI REST Service & Streamlit Station)

### Step 4.1: API Endpoint Integration Proof
```powershell
pytest tests/test_api.py -v
```
**Verification Assertions:**
- `GET /health` reports active GPU acceleration and catalog SKU count.
- `GET /skus` and `POST /skus/onboard` correctly manage SKU metadata and BOM.
- `POST /returns/inspect` executes automated triage and persists cryptographic dispute certificates.
- `POST /audit/verify` confirms authentic dispute certificates and detects payload tampering.

### Step 4.2: Streamlit Dashboard Live Launch
To launch the interactive dual-mode operator station:
```powershell
streamlit run app/dashboard.py
```
- **Port:** `8501`
- **Features Tested:** 3PL Intake Station, Doorstep Delivery Rider QC, Zero-Training SKU Catalog.

