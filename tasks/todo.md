# Government-ID ML Foundation Tasks

## Task 1: Streaming dataset loader and contract validation

**Description:** Load file-backed and TAR-backed records from the canonical
JSONL schema without extracting archives, validate annotations at the boundary,
and expose identity-safe samples to training code.

**Acceptance criteria:**
- [ ] File and TAR records load with image, document polygon, fields, labels, and split.
- [ ] Missing/corrupt members and invalid annotations produce structured errors without exposing field values.
- [ ] The loader refuses unknown schema versions and unapproved dataset IDs.

**Verification:**
- [ ] Focused unit tests cover valid, missing, corrupt, and unsafe records.
- [ ] Full manifest smoke test loads all 4,890 records.
- [ ] Existing normalization tests remain green.

**Dependencies:** None

**Files likely touched:** `src/data/`, `tests/data/`, training configuration

**Estimated scope:** Medium

## Task 2: Evaluation harness and locked-test guard

**Description:** Implement reusable metrics for detection, OCR, classification,
field extraction, calibration, and hard-negative false positives before model
training begins.

**Acceptance criteria:**
- [ ] Metrics are broken down by dataset, country, ID type, script, modality, and quality slice.
- [ ] Calibration and locked-test datasets cannot be silently used for fitting.
- [ ] A deterministic dummy predictor produces a versioned baseline report.

**Verification:**
- [ ] Metric unit tests use hand-calculated examples.
- [ ] End-to-end evaluation runs on a small fixture.
- [ ] Repeated runs produce identical reports.

**Dependencies:** Task 1

**Files likely touched:** `src/evaluation/`, `tests/evaluation/`, `configs/evaluation/`

**Estimated scope:** Medium

## Checkpoint: Foundation

- [ ] Tasks 1-2 tests, linting, formatting, and runtime smoke test pass.
- [ ] Human reviews the metric definitions before model training.

## Task 3: Document detection and rectification baseline

**Description:** Train a lightweight detector for ID-document quadrilaterals and
produce perspective-corrected document crops for downstream models.

**Acceptance criteria:**
- [ ] Training uses train identities only and selects checkpoints on validation identities.
- [ ] Evaluation reports recall/IoU for full, rotated, perspective, and partially cropped documents.
- [ ] Rectification preserves source coordinates so findings can map back to page/image locations.

**Verification:**
- [ ] Overfit a tiny training subset as a pipeline sanity check.
- [ ] Run full validation evaluation and save reproducible configuration/checkpoint hashes.
- [ ] Visually inspect a fixed, non-PII sample grid.

**Dependencies:** Tasks 1-2

**Files likely touched:** `src/models/detection/`, `src/training/`, `configs/detection/`, `tests/models/`

**Estimated scope:** Medium

## Task 4: Quality model and degradation suite

**Description:** Generate controlled degradation labels and train quality heads
for blur, glare, shadow, low resolution, crop, skew, and compression.

**Acceptance criteria:**
- [ ] Augmentations are applied only after identity splitting.
- [ ] Each degradation has severity metadata and an unmodified control.
- [ ] Quality output selects recovery processing without changing field values.

**Verification:**
- [ ] Deterministic augmentation tests pass.
- [ ] Quality metrics are reported per degradation and severity.
- [ ] Recovery improves or preserves downstream OCR on validation data.

**Dependencies:** Tasks 1-3

**Files likely touched:** `src/augmentation/`, `src/models/quality/`, `configs/quality/`, `tests/augmentation/`

**Estimated scope:** Medium

## Task 5: Multilingual OCR baseline

**Description:** Train or fine-tune OCR on rectified template/capture pairs for
the active Indian scripts and Nepal, while retaining a general Latin/Cyrillic
baseline from MIDV-2020.

**Acceptance criteria:**
- [ ] OCR never invents or silently repairs characters in exported evidence.
- [ ] CER, WER, and exact-field accuracy are reported per script and field type.
- [ ] Multiple preprocessing/OCR passes retain individual hypotheses and confidence.

**Verification:**
- [ ] Tiny-subset overfit and full validation evaluation pass.
- [ ] Numeric-ID confusion tests cover O/0, I/1, B/8, separators, and native digits.
- [ ] Captured-view predictions map back through the template/rectification link.

**Dependencies:** Tasks 1-4

**Files likely touched:** `src/models/ocr/`, `src/training/`, `configs/ocr/`, `tests/models/`

**Estimated scope:** Medium

## Checkpoint: Perception

- [ ] Tasks 3-5 meet documented baseline gates without touching locked test data.
- [ ] Error analysis identifies the dominant data gaps by script and capture condition.

## Task 6: Lawful priority-India dataset expansion

**Description:** Add documented mock/synthetic or authorized data for PAN,
Indian passport, driving licence, EPIC/voter ID, and vehicle RC, preserving
issuer variants, old/new layouts, front/back, photocopies, and mobile captures.

**Acceptance criteria:**
- [ ] Every source has provenance, licence/authorization, generator version, and approval status.
- [ ] Each ID class has independent identities across all four splits and realistic hard negatives.
- [ ] No scraped, leaked, or unconsented identity document enters training.

**Verification:**
- [ ] Registry/integrity audit passes.
- [ ] Identity and template leakage audit reports zero crossings.
- [ ] Human reviews representative synthetic/mock samples for layout fidelity.

**Dependencies:** Tasks 1-2; priority order decision

**Files likely touched:** `datasets/manifests/`, `scripts/`, `datasets/schemas/`, `tests/data/`

**Estimated scope:** Medium per ID type

## Task 7: Document-type classifier

**Description:** Classify supported ID types using rectified visual features,
OCR keywords, script, layout, and negative evidence.

**Acceptance criteria:**
- [ ] Unsupported and non-ID documents have explicit negative/unknown labels.
- [ ] Per-class precision/recall and confusion matrices are reported.
- [ ] A class cannot be enabled in confirmed output without approved train/calibration/test coverage.

**Verification:**
- [ ] Hard-negative suite includes invoices, forms, hospital records, cards, and number-like text.
- [ ] Ablation verifies that no single logo/keyword artifact dominates classification.
- [ ] Validation metrics and calibration curves are reproducible.

**Dependencies:** Tasks 3, 5, 6

**Files likely touched:** `src/models/classification/`, `configs/classification/`, `tests/models/`

**Estimated scope:** Medium

## Task 8: Field localization and semantic extraction

**Description:** Detect and label ID number, name, DOB, address, photo, QR/MRZ,
issuer, and other supported fields using layout and OCR evidence.

**Acceptance criteria:**
- [ ] Output includes page/image coordinates and source OCR spans for every field.
- [ ] Duplicate/multilingual printed fields are linked without collapsing their evidence.
- [ ] Field-level exact match and localization metrics are reported per ID type.

**Verification:**
- [ ] Unit tests cover coordinate transforms and front/back documents.
- [ ] Validation evaluation reports each field separately.
- [ ] Masked sample outputs are manually inspected.

**Dependencies:** Tasks 3, 5, 7

**Files likely touched:** `src/models/fields/`, `src/geometry/`, `configs/fields/`, `tests/models/`

**Estimated scope:** Medium

## Task 9: ID validators and calibrated evidence fusion

**Description:** Combine independent visual, OCR, layout, checksum, QR/MRZ, and
issuer evidence into per-claim decisions without a universal fixed threshold.

**Acceptance criteria:**
- [ ] Each ID type has explicit applicable validators and evidence provenance.
- [ ] Calibration is fitted only on the calibration split and measured per ID/field.
- [ ] Confirmed output requires the configured evidence policy; conflicts are never silently resolved.

**Verification:**
- [ ] Validator unit tests cover valid, invalid, corrupted, masked, and conflicting inputs.
- [ ] Fusion ablations show the contribution of each evidence family.
- [ ] False confirmations are measured on the hard-negative suite.

**Dependencies:** Tasks 5, 7, 8

**Files likely touched:** `src/validation/`, `src/fusion/`, `configs/policy/`, `tests/validation/`

**Estimated scope:** Medium

## Checkpoint: Structured recognition

- [ ] Tasks 6-9 pass per-ID data, metric, and evidence-policy gates.
- [ ] Unsupported IDs remain non-confirmable.

## Task 10: Adversarial challenge evaluation

**Description:** Evaluate frozen candidates against blur, shadows, glare,
occlusion, partial crop, rotation, compression, screenshots, photocopies,
multi-page PDFs, multiple IDs per page, and non-ID number-heavy documents.

**Acceptance criteria:**
- [ ] Challenge set is identity-disjoint and contains both positive and negative cases.
- [ ] Results include false positives per 10,000 negatives and failure slices.
- [ ] Locked test is run only after the candidate and decision policy are frozen.

**Verification:**
- [ ] Challenge generation is deterministic and audited for leakage.
- [ ] Repeated inference produces identical structured findings.
- [ ] Every critical failure is reproduced by a retained non-sensitive test case.

**Dependencies:** Tasks 3-9

**Files likely touched:** `src/evaluation/`, `datasets/challenges/`, `configs/evaluation/`, `tests/evaluation/`

**Estimated scope:** Medium

## Task 11: Model export and scanner integration contract

**Description:** Export optimized local models and define the software-facing
input/output contract for file/page/frame findings, evidence, errors, and masked
reporting.

**Acceptance criteria:**
- [ ] Exported models match reference inference within defined tolerances.
- [ ] Contract includes ID type, fields, page/frame coordinates, evidence, status, and model versions.
- [ ] No raw PII appears in operational logs or default reports.

**Verification:**
- [ ] Cross-platform smoke tests run on representative Windows/macOS hardware.
- [ ] Large PDF/ZIP orchestration tests are bounded and resumable.
- [ ] Versioned integration fixture passes end to end.

**Dependencies:** Task 10

**Files likely touched:** `src/export/`, `src/contracts/`, `configs/runtime/`, `tests/integration/`

**Estimated scope:** Medium

## Checkpoint: Model handoff

- [ ] Full test, lint, format, security, reproducibility, and documentation gates pass.
- [ ] Frozen locked-test report and model/checkpoint hashes are recorded.
- [ ] Human approves the model package before endpoint-software integration.
