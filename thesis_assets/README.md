# IdentityGate Thesis Assets

This directory contains submission-facing artifacts derived from the frozen
IdentityGate experimental record.

## Structure

- `tables/`
  Final report-oriented CSV tables derived from EXP053 and EXP054.

- `plots/`
  Final SVG quantitative figures generated only from frozen result CSV files.
  These figures do not rerun SAM 3 and do not alter any experimental outcome.

- `qualitative_candidates/`
  Copies of pre-existing visual audit material from frozen historical
  experiments. These are candidates for thesis qualitative panels and must be
  accompanied by an explicit outcome-independent selection rule if a subset is
  used in the thesis.

- `drafts/`
  Evidence-backed text drafts for Chapters 5 and 6, front matter, and figure /
  table captions.

- `ASSET_MANIFEST.csv`
  SHA256 manifest for the submission-facing assets.

## Critical interpretation boundary

The frozen HARD_TEST80 primary contrast is POR@30(B3-S)-POR@30(B2). Its
frozen-threshold estimate is +0.007905 with BCa 95% CI
[-0.002037, +0.020882], but its pooled write-rate difference is 0.032703,
which exceeds the frozen 0.02 tolerance.

Therefore the final primary status is:

`RATE_MISMATCH_NO_MATCHED_RATE_PRIMARY_INTERPRETATION`

No TEST threshold retuning or TEST rerun is permitted.

## Reproducibility provenance

Primary source outputs remain under `experiments/` and protocol/freeze records
remain under the repository root and `docs/`. Files in this directory are
derived reporting artifacts and do not replace the canonical experimental
record.
