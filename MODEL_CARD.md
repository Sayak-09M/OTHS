# Model card: Foundation v2 identity-document candidate

## Summary

Foundation v2 is a local-first candidate model for identity-document presence,
type classification, and document-corner localization. It is used with native
text extraction, OCR, format validators, and evidence fusion; the visual model
does not independently confirm an identity number.

- Architecture: `mobilenet-v3-small-corner-heatmap-v1`
- Checkpoint: `artifacts/document/foundation-v2-best.pt`
- SHA-256: `1f2f64831a0df868806d14b71f13b07a1a0ce24b42020f5550c2a2f9dd2b988d`
- Priority region: India, followed by Nepal, Bangladesh, and Sri Lanka
- Status: candidate-ready, not production-qualified

## Intended use

The model is intended to supply evidence to an authorized, local PII scanner.
It supports document/page inputs produced by the scanner and is not a drive
crawler, an OS-permission bypass, an issuing-authority verification service, or
a final compliance decision maker.

## Supported identity labels

The runtime policy enables Aadhaar, PAN, EPIC, ABHA, Indian passport and driving
licence candidates; Nepal National ID; Bangladesh NID; and Sri Lanka NIC. Some
additional MIDV document labels are retained in the visual checkpoint for
pretraining compatibility.

Only Aadhaar has a locally implemented checksum in this release. Other
identifiers are supported by format, context, layout, and OCR evidence, which
does not prove that an issuing authority assigned the value.

## Validation evidence

The aggregate visual validation report is stored at
`artifacts/reports/foundation-v2-validation.json`. On 1,986 validation examples
it reports 0.9935 document-type accuracy, 0.9329 mean polygon IoU, 0.9823 recall
at IoU 0.50, and 0.9635 recall at IoU 0.75. These figures measure the visual
checkpoint on the prepared validation split; they are not proof of production
PII extraction accuracy or zero error on hospital/hotel data.

Before production, evaluate exact identifier extraction per entity and quality
slice on identity-disjoint, authorized hospital/hotel pilots. Also complete the
cross-platform parity, calibration, signed-manifest, SBOM, licence-BOM, and
runtime assurance gates emitted by `scripts/check_dgx_readiness.py`.

## Data and privacy

Training uses approved mock/publicly licensed MIDV sources and locally generated
fictional, visibly marked synthetic records. Raw and processed datasets are not
included in this repository. Legacy datasets with unclear provenance or
apparent personal data are quarantined and excluded from version control and
training. See `datasets/manifests/registry.yaml` for provenance and approval
status.

Generated scan reports, audit logs, local paths, and raw PII are intentionally
excluded. The default report masks identifier values.

## Known limitations

- No model can guarantee perfect accuracy on all unseen formats or degraded
  images. Unsupported, corrupt, encrypted, inaccessible, or resource-limited
  inputs must remain explicit gaps.
- Nepal coverage is weaker than the aggregate result and requires expanded
  held-out evidence.
- Spreadsheet embedded images and legacy binary `.xls` need an upstream
  extractor; current XLSX support covers XML text and numeric cell content.
- The release is not production-qualified until every readiness gate has
  independent evidence.

## Reproduction

Use Python 3.12 and the pinned `requirements.lock`, then follow
`docs/DETECTIONGODX_MODEL_SOP.md`. Run the readiness check, complete test suite,
and aggregate checkpoint evaluation before integration.
