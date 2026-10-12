# Dataset workspace

Only datasets marked `approved` in `manifests/registry.yaml` may enter a
training, validation, calibration, or test pipeline.

## Policy

- Prefer issuer, author, institutional, or canonical research-host sources.
- Preserve the original licence, README, citation, URL, checksums, and archive.
- Do not train on leaked, scraped, or otherwise unconsented identity documents.
- Do not treat a public download link as proof that personal data is lawful to
  reuse for model training.
- Keep synthetic/mock identity datasets separate from consented real-world
  evaluation data.
- Split documents by source identity/template before augmentation to prevent
  train/test leakage.
- Never copy production customer scans into this workspace automatically.

## Layout

```text
datasets/
  generated/   local, visibly watermarked fictional training images
  raw/         immutable source archives and source metadata
  manifests/   provenance, licence, integrity, and approval records
  schemas/     canonical machine-readable annotation contracts
  processed/   generated manifests; raw images remain in raw archives
```

The legacy top-level `archive/` and `adhaar/` folders contain apparent real
identity-document images. They are intentionally excluded pending provenance,
consent, privacy, and legal review. The top-level custom-generated Aadhaar set
is also excluded until its licence and generation provenance are established.

## Build the normalized foundation

```bash
python3 scripts/normalize_and_split.py
```

The command reads only the two approved MIDV registry entries and writes
`datasets/processed/id-foundation-v1/`. It does not extract or modify source
archives. Each underlying document identity is assigned to exactly one of
`train`, `validation`, `calibration`, or `locked_test`; every captured, scanned,
and rotated view follows that assignment.

The normalized records conform to
`schemas/canonical-id-record.schema.json`. Field values are artificial or
modified research data, but the generated manifests should still be handled as
restricted training material and should not be published automatically.

## Build foundation v2

```bash
python scripts/build_foundation_v2.py
```

The v2 builder preserves every v1 identity assignment and adds India-priority
synthetic Aadhaar, PAN, EPIC, ABHA, passport-like, and driving-licence-like
samples; Nepal, Bangladesh, and Sri Lanka ID samples; and hospital/hotel hard
negatives. Generated cards are visibly marked `SYNTHETIC / NOT VALID`, use
geometric avatars, and contain no imported personal records.

The build refuses unapproved sources, keeps all variants of one fictional
identity in one split, verifies generated Aadhaar checksums, checks identifier
uniqueness per class, and fully decodes every referenced image. Number formats
without a locally implemented official checksum are explicitly described as
format examples rather than verified issued identities.
