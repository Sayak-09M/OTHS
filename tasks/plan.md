# Implementation Plan: Government-ID ML Foundation

## Overview

Build an India-first, multi-stage identity-document recognition system from the
verified dataset foundation. The first trustworthy release will detect and
rectify documents, classify supported ID types, extract fields, validate
ID-specific evidence, calibrate its decisions, and produce reproducible metrics.
Nepal follows India. Bangladesh and Sri Lanka remain later coverage expansions.

The current data can support an Aadhaar-focused research baseline and generic
foreign-ID pretraining. It cannot support production claims for PAN, Indian
passport, driving licence, EPIC/voter ID, vehicle RC, or other Indian IDs until
lawful mock/synthetic or authorized data for those types is added.

Tasks are tracked in `tasks/todo.md`.

## Architecture Decisions

- Use a staged pipeline rather than one monolithic model: quality assessment,
  document detection/rectification, OCR, type classification, field extraction,
  ID-specific validation, and calibrated evidence fusion.
- Establish metrics and locked-test protections before training models.
- Split by source identity; no view, scan, rotation, or augmentation of one
  identity may cross a split boundary.
- Use the calibration split only for confidence/decision policy. Never tune on
  the locked test set.
- Treat ID type and field value as separate claims. A document may be classified
  correctly while a number remains unconfirmed.
- Never generate missing digits or silently repair OCR. Confirmed findings must
  be supported by visual, textual, structural, and validator evidence.
- Keep source documents local. Reports and training logs must mask government-ID
  numbers and avoid copying raw PII.

## Dependency Graph

```text
Loader + contract validation
            |
            v
Evaluation harness ---- Lawful India-ID data expansion
            |                         |
            v                         v
Detection/rectification ------> Type classifier
            |                         |
            v                         v
Quality + OCR ---------------> Field extraction
            \                         /
             v                       v
                Validators + fusion
                         |
                         v
             Adversarial locked evaluation
                         |
                         v
               Export/integration contract
```

## Task List

### Phase 1: Measurable training foundation

- [ ] Task 1: Implement streaming dataset loader and contract validation
- [ ] Task 2: Implement evaluation harness and locked-test guard

### Checkpoint: Foundation

- [ ] Every image and annotation can be loaded reproducibly
- [ ] Metrics run on a deterministic dummy predictor
- [ ] Locked-test access is explicit and logged

### Phase 2: First Aadhaar-capable perception path

- [ ] Task 3: Train document detection and rectification baseline
- [ ] Task 4: Build quality/degradation model and augmentation suite
- [ ] Task 5: Train multilingual OCR baseline on rectified documents

### Checkpoint: Perception

- [ ] Detection, rectification, quality, and OCR metrics meet the initial gates
- [ ] Failure slices are reported by script, modality, blur, crop, and rotation

### Phase 3: India ID breadth and structured findings

- [ ] Task 6: Expand lawful datasets for priority Indian ID classes
- [ ] Task 7: Train document-type classifier
- [ ] Task 8: Train field-localization and semantic extraction models
- [ ] Task 9: Implement ID-specific validators and evidence fusion

### Checkpoint: Structured recognition

- [ ] Every supported ID has independent data and validation coverage
- [ ] Findings include type, field, location, evidence, and masked value
- [ ] Unsupported IDs cannot be emitted as confirmed findings

### Phase 4: Accuracy proof and integration

- [ ] Task 10: Run adversarial challenge evaluation and error analysis
- [ ] Task 11: Export models and define the scanner integration contract

### Checkpoint: Model handoff

- [ ] Locked-test results are generated once from a frozen candidate
- [ ] False-positive and false-negative gates are met per ID type
- [ ] Models reproduce locally from pinned configuration and checkpoints

## Initial Quality Gates

These are entry gates, not claims that the finished system is perfect:

- Document detection: recall reported at strict and relaxed IoU, broken down by
  capture condition and script.
- OCR: character error rate, word error rate, and exact field match by script.
- Type classification: macro F1 and per-class precision/recall; no aggregate-only
  score may hide a weak ID class.
- PII findings: per-field precision/recall plus false positives per 10,000
  non-ID pages/images.
- Confidence: reliability diagram, expected calibration error, and selective
  accuracy at the confirmed-output policy.
- Robustness: separate results for blur, shadow, glare, low resolution, rotation,
  perspective, partial crop, compression, photocopy, screenshot, and long PDF.

Numerical release thresholds must be set only after the baseline and challenge
set reveal what is attainable. The locked test remains untouched until then.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Only 85 unique Indian Aadhaar identities and 5 Nepal identities are currently available | High | Use them for baseline work only; expand lawful identity-level diversity before production claims |
| No approved PAN/DL/EPIC/passport/RC training sets yet | High | Build authorized mock/synthetic sets with documented generators and provenance before enabling these labels |
| Near-duplicate views inflate scores | High | Preserve identity-group splitting and add automated leakage checks to every training run |
| OCR appears plausible but changes digits | High | Preserve multi-pass OCR evidence, checksum/QR/MRZ validation, and exact-value abstention |
| Thresholds overfit one dataset | High | Tune only on calibration data and validate against hard negatives and institution-specific pilots |
| Synthetic layouts fail on real hospital/hotel scans | High | Add authorized, de-identified pilot evaluation without automatically retaining customer documents |

## Open Questions

- The priority order after Aadhaar must be finalized before Task 6. A reasonable
  default is PAN, Indian passport, driving licence, EPIC/voter ID, and vehicle RC.
- Production acceptance thresholds require an operational decision about the
  relative cost of a missed ID versus a false confirmed finding.

