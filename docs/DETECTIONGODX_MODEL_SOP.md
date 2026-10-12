# DetectionGodX ML model SOP

This SOP validates and runs the local ML candidate. It does not commit, push,
upload, or send documents over a network.

## 1. Activate the pinned environment

```bash
cd "/path/to/OTHS"
source .venv/bin/activate
```

Use Python 3.12. Do not install packages at runtime on a hospital or hotel
server; the release bundle must be built and signed offline.

## 2. Check the bundle and DetectionGodX contract pin

```bash
python scripts/check_dgx_readiness.py
```

Expected status: `candidate_ready`. Also verify:

- `checkpoint_sha256` matches the validation report;
- `missing_visual_labels` is empty;
- `ocr_available` is true;
- `producer_role` is `A3`;
- `output_contract` is `dgx.observation.v1`;
- `production_qualified` remains false until every listed production gate has
  independent evidence.

## 3. Run the fast correctness gates

```bash
python -m pytest -q
python -m ruff check .
```

These tests include duplicate occurrences, corrupt inputs, ZIP members,
bounded MATLAB v5 extraction, unknown-extension text detection, multi-page
media behavior, full-page plus rectified OCR, raw-value leakage, and the A3
observation contract.

## 4. Re-evaluate the frozen checkpoint

This is the longer GPU/MPS/CPU validation command. Run it only when you want to
recreate the metrics:

```bash
python scripts/evaluate_document.py \
  --checkpoint artifacts/document/foundation-v2-best.pt \
  --output artifacts/reports/foundation-v2-validation.json
```

Do not retrain merely because integration code changed. Retrain only after an
approved dataset/config change, and never tune on the locked test split.

## 5. Run an authorized pilot scan

```bash
python scripts/scan_identity.py "/path/to/authorized/pilot" \
  --checkpoint artifacts/document/foundation-v2-best.pt \
  --policy configs/inference/mixed-identities.yaml \
  --max-pdf-pages 2000 \
  --report artifacts/reports/authorized-pilot.json
```

The scanner processes every supported PDF page up to the explicit limit and
every allowed ZIP member within the archive limits. Identical files and repeated
ID occurrences are not deduplicated away. Corrupt, encrypted, unsupported,
unreadable, and resource-limited inputs are errors/gaps, never “clean”.

MATLAB `.m`, v5 `.mat`, and v5 `.fig` files are handled locally without code
execution. The extractor reads textual values, structural field names, and
small numeric arrays only when their labels identify an identity field. MATLAB
v7.3/HDF5 and other proprietary binary formats remain explicit gaps until their
isolated DetectionGodX extractors are bundled and qualified.

For a full computer deployment, DetectionGodX A1 enumerates authorized Windows
roots and A2/A3 process the resulting artifacts. The ML worker itself must not
bypass OS permissions or independently crawl drives.

## 6. Review accuracy by slice

Do not accept only the aggregate 99% number. Before a production release,
measure each enabled ID type and each quality slice separately:

- precision, recall, and exact field match;
- false positives per 10,000 non-ID pages;
- blur, shadow, glare, crop, rotation, perspective, compression, photocopy,
  screenshot, and camera capture;
- native text, scan-only PDF, hybrid PDF, image, Office extraction, and ZIP;
- English, Hindi/Devanagari, and every enabled regional script;
- India first, followed by Nepal, Bangladesh, and Sri Lanka;
- documents from authorized, identity-disjoint hospital/hotel pilots.

The current Nepal slice is materially weaker than the aggregate result, so it
must stay conditional until expanded data and held-out results meet the release
gate.

## 7. Attach to DetectionGodX

The software supplies one already-authorized rendered page, immutable IDs and
digests, a protected evidence reference, and an ephemeral client-scoped
fingerprint key. Call `MixedIdentityScanner.scan_image_candidates`, then
`build_page_observations`. Put those observations in the A3
`dgx.worker-result.v1` batch with empty `findings` and empty
`assurance_decisions` arrays. A4 performs calibrated assurance and is the only
role allowed to publish findings.

The server bundle must additionally pin/sign the exact checkpoint, policy,
preprocessor, Python/Torch/Tesseract runtime, OCR language packs, contract
schemas, SBOM, license BOM, execution provider, and known-answer readiness test.
