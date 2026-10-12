# Mixed Identity-PII Scanner

Local-first identity-number discovery for mixed directories. The runtime scans
every supported page independently; the visual ID model supplies evidence but
does not decide whether OCR or native-text extraction runs.

## Enabled identity detectors

- India: Aadhaar, PAN, EPIC/voter ID, ABHA, passport-number candidates, and
  driving-licence-number candidates.
- Nepal: 10-digit National Identity Number with Nepal-specific context.
- Bangladesh: 10-, 13-, or 17-digit NID with Bangladesh-specific context.
- Sri Lanka: old and new NIC formats with Sri Lanka-specific context.

Only Aadhaar has a locally implemented checksum in this release. Other types
are reported as format-and-context validated, not as proof that the issuing
authority assigned or authenticated the number.

## Inputs

- Raster images: BMP, GIF, JPEG, PNG, TIFF, and WebP.
- PDF, including native text plus rendered-page OCR.
- Plain text and structured text: TXT, CSV, TSV, JSON, XML, HTML, Markdown,
  YAML, and logs.
- MATLAB source plus v5/v6/v7 (non-HDF5) MAT and FIG containers. Text and
  identity-labelled small numeric arrays are extracted without executing code;
  large scientific matrices are not converted into candidate-number noise.
- Office Open XML: DOCX, XLSX, and PPTX XML text. XLSX shared strings,
  inline text, numeric cell values, and cached formula results are inspected.
  Spreadsheet cell addresses and embedded Office images are not yet emitted or
  OCR-scanned; legacy binary `.xls` is unsupported and is logged explicitly.
- EML message text.
- ZIP archives, including supported nested members within configured limits.
- Textual files with unknown or missing extensions when conservative content
  inspection identifies them as text.

Unsupported files, unreadable paths, corrupt files, encrypted content,
resource-limit failures, and skipped symbolic links are written to the report;
they are not silently treated as clean.

MATLAB v7.3/HDF5 remains an explicit `matlab_v73_backend_unavailable` gap in
this standalone worker. Proprietary binary packages and database files also
need their DetectionGodX extractor before A3 model inference; their bytes are
not guessed as text.

## Duplicate handling

The scanner does not skip content because another file has identical bytes or
because the same identifier was already found elsewhere. Every filesystem path,
PDF page, archive entry, and repeated identifier occurrence is scanned and
reported with its own location or polygon. Multiple OCR preprocessing passes for
one physical occurrence are fused as supporting evidence rather than emitted as
extra duplicates. ZIP entries that reuse the same member name receive an
`archive-entry` occurrence suffix so their report locations remain distinct.

## Run

```bash
git clone https://github.com/Sayak-09M/OTHS.git
cd OTHS
source .venv/bin/activate
python scripts/scan_identity.py "/path/to/scan" \
  --report artifacts/reports/mixed-identity-scan.json
```

The v2 foundation checkpoint is now the default. For images and rendered PDF
pages, full-page OCR always runs. When the visual model finds usable document
corners, a second perspective-rectified OCR route runs and maps every candidate
back to original-page coordinates. The visual model never suppresses OCR.

The reviewed checkpoint is versioned at
`artifacts/document/foundation-v2-best.pt`; generated scan reports, training
logs, raw datasets, and non-release checkpoints remain local and are ignored by
Git.

The JSON report has separate `findings` and `review_candidates`. Values are
masked by default. Review candidates preserve recall and rejection reasons
without presenting an unverified value as confirmed.

## Measure end-to-end extraction accuracy

The visual-checkpoint report does not measure whether OCR extracted the exact
identity number. Use the validation evaluator to compare in-memory OCR values
with labelled `id_number` fields while keeping complete identifiers out of the
saved report:

```bash
python scripts/evaluate_identity_e2e.py \
  --checkpoint artifacts/document/foundation-v2-best.pt \
  --policy configs/inference/mixed-identities.yaml \
  --split validation \
  --entity IN_AADHAAR \
  --entity IN_PAN \
  --device mps \
  --output artifacts/reports/identity-e2e-aadhaar-pan-validation.json
```

Candidate-level metrics show whether OCR and the detectors discovered the exact
value. Confirmed-level metrics additionally apply the active inference policy.
Images without an annotated number are reported separately as weak-positive
coverage and cannot inflate exact precision or recall. The locked test requires
`--final-evaluation` and is reserved for a frozen checkpoint and policy after
validation-driven improvements are complete. Labelled values that fail the
active format or checksum requirements remain in candidate-level OCR metrics,
but are listed under `ground_truth_issues` instead of being counted as required
confirmed findings.

The scanner reads only locations that the operating system permits. Full-disk
access must be granted to the eventual endpoint application through the normal
Windows or macOS security controls; this project does not bypass them.

## Build and train the v2 visual foundation

```bash
python scripts/build_foundation_v2.py
python scripts/train_document.py \
  --config configs/detection/foundation-v2.yaml
```

Foundation v2 adds privacy-safe synthetic coverage for the priority South Asian
ID classes and `NON_ID` hospital/hotel documents. `NON_ID` trains the presence
head and is excluded from document-type and corner losses. Model selection now
also measures presence accuracy and penalizes false positives.

## DetectionGodX integration

This project is the server-side A3 candidate producer, not the endpoint drive
crawler and not the A4 publisher. The in-process adapter in
`pii_id.integration.dgx` emits raw-free `dgx.observation.v1` objects with page
regions, model/detector digests, masked values, client-scoped HMAC fingerprints,
and protected evidence references. It never emits `dgx.finding.v1` records.

The integration contract is pinned in
`configs/integration/detectiongodx-v2.yaml`. Run the local integrity/readiness
check before attaching a bundle:

```bash
python scripts/check_dgx_readiness.py
```

This reports `candidate_ready`, not `production_qualified`. Production requires
the held-out institutional slices, cross-platform runtime evidence, signed
runtime manifest, SBOM/license BOM, and A4 calibration listed in the report.

See `docs/DETECTIONGODX_MODEL_SOP.md` for the complete terminal SOP.
