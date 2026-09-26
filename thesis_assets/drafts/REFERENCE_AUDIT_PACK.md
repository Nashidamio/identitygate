# Final Reference Audit Pack

## Status

This file is a submission-facing citation-control artifact.

The project already audited the following identifiers and they MUST NOT be
re-audited unless new contradictory evidence appears:

- SENTRY — arXiv:2606.24449
- Rethinking Memory Design in SAM-Based Visual Object Tracking — arXiv:2512.22624
- SAM3-DMS — arXiv:2601.09699
- DAM4SAM — arXiv:2509.13864
- QDMN — arXiv:2207.07922
- OAMVOS — arXiv:2604.22837
- SurgSLOT — arXiv:2511.16618v2
- CMR — arXiv:2608.22064
- VOS-Agent — arXiv:2608.12721
- MOSEv2 — arXiv:2508.05630

## Mandatory final-thesis citation roles

### Dataset

MOSEv2 is the primary dataset citation and is mandatory.

MOSE may be cited only as predecessor/background if the thesis text discusses
the dataset lineage.

### Frozen substrate

The thesis core substrate is SAM 3 VOS/PVS, not SAM 3.1 Multiplex.

Any citation sentence must preserve that distinction.

### Memory-selection related work

Core related-work comparison should cover at least:

- SENTRY
- Rethinking Memory Design
- SAM3-DMS
- DAM4SAM
- QDMN

Supporting related-work citations may include:

- CMR
- VOS-Agent
- OAMVOS
- SurgSLOT
- SAMURAI
- ReMeDI-SAM3

Do not create a novelty claim merely from absence of another paper.

## Claim-safe positioning

Do NOT write:

- nobody filters SAM memory;
- this is the first memory gate;
- identity information generally does not work;
- SAM 3 has no memory mitigation at all.

Preferred framing:

The thesis performs a controlled comparison of quality, temporal, and tested
native-pointer identity information for write admission under a frozen SAM 3
VOS/PVS substrate and a pre-specified write-budget protocol.

## Reference inclusion rule

A reference enters the final bibliography only if:

1. it is cited in the actual thesis text;
2. its exact title/authors/year or venue are available from an audited source;
3. its citation supports the sentence where it is used.

Do not pad the bibliography with DAVIS, LVOS, SA-V, SAMITE, HiM2SAM, or other
stretch references unless the final manuscript actually discusses them.

## Remaining metadata work

`REFERENCE_METADATA_AUDIT.csv` intentionally leaves title/authors/venue fields
blank where exact bibliographic metadata is not preserved in the current
submission-facing assets.

These blanks must NOT be filled from memory.

The next bibliography step is metadata transfer from already-audited project
sources or the users actual reference manager/manuscript bibliography.
