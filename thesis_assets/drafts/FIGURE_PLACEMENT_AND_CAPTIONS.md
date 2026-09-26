# Final Figure Placement and Caption Pack

## Figure 1.1

Title: Controlled Memory-Write Admission in Frozen SAM 3

Recommended file:
`fig01_controlled_memory_write_admission.png`

Placement:
Chapter 1, after the research question / experimental-principle introduction.

Caption:
**Figure 1.1: Controlled memory-write admission in frozen SAM 3.**
A frozen SAM 3 VOS/PVS tracker produces per-object candidate predictions.
Quality, temporal, and, when available, native pointer-identity information
are used by nested memory-write policies to decide whether a candidate write
is admitted to or blocked from the existing SAM 3 memory state. SAM 3
parameters remain fixed; the intervention affects only memory-write
admission. Operating points are selected on development data and remain fixed
during TEST evaluation. The visual examples are schematic and are not
empirical TEST evidence.

## Figure 2.1

Title: Memory Management and Identity Reasoning in Video Object Segmentation

Recommended file:
`fig02_memory_management_related_work_landscape.png`

Placement:
Chapter 2, after Section 2.1.

Caption:
**Figure 2.1: Positioning of the thesis within memory management and identity
reasoning in video object segmentation.**
Prior work spans memory representation and retrieval, selective and
reliability-oriented memory management, and identity-aware or
competitor-relative reasoning. The thesis converts these themes into a
controlled question at the memory-write interface of a frozen SAM 3 tracker:
whether learned quality-plus-temporal information, self identity, or
tracked-competitor identity provides useful evidence for write admission.
The nested study uses a pre-specified write-budget protocol, explicit TEST
rate-mismatch handling, and video-clustered inference.

## Figure 3.1

Title: Experimental Pipeline for Controlled Memory-Write Evaluation

Recommended file:
`fig03_controlled_memory_write_evaluation_pipeline.png`

Placement:
Chapter 3, after Section 3.1.

Caption:
**Figure 3.1: Experimental pipeline for controlled memory-write evaluation.**
A frozen SAM 3 VOS/PVS tracker produces candidate object predictions from
which the final quality, temporal, and, when valid, native pointer-identity
features are extracted. The nested B1/B2/B3-S/B3-R policies intervene only
at memory-write admission. Operating points are selected using Fresh DEV and
frozen before a single final TEST campaign. Realized TEST write rates are
checked against the prospectively specified absolute tolerance of 0.02;
comparisons exceeding this tolerance are reported as `RATE_MISMATCH` and are
not interpreted as matched-rate effects. POR@30 is the sole primary endpoint,
ITR@30 is secondary, and uncertainty is quantified using paired
video-clustered BCa 95% confidence intervals.

## Figure 3.2

Title: Leakage-Controlled Construction of Development and Test Cohorts

Recommended file:
`fig05_leakage_controlled_cohort_construction.png`

Placement:
Chapter 3, after the development-exposure / cohort-construction sections.

Caption:
**Figure 3.2: Leakage-controlled construction of development and test cohorts
from MOSEv2.**
Recovery events are enumerated from the densely annotated 3,666-video
training partition. Final whole-scene-labelled events are excluded from
primary-event eligibility but retained as a tagged control stratum, while
videos previously exposed during development are excluded at the video level.
The resulting primary eligible population contains 2,701 events from 1,170
videos. Frozen outcome-independent difficulty and visual-diversity rules
define a 120-video hard pool, partitioned into Fresh DEV (40 videos, 233
primary events) and Hard TEST (80 videos, 506 events). Representative TEST
contains 40 videos and 76 primary events sampled independently after excluding
the entire hard pool.

## Figure 4.1

Title: Signal Composition and Memory-Write Decision Architecture

Recommended file:
`fig04_signal_composition_and_write_decision_architecture.png`

Placement:
Chapter 4, after the dual-risk architecture section.

Caption:
**Figure 4.1: Signal composition and memory-write decision architecture.**
B2 uses five quality, geometry, and temporal features; B3-S adds the native
self-anchor pointer cosine, and B3-R further adds the maximum
tracked-competitor anchor cosine when the required identity pointers are
valid. Independent drift- and theft-risk heads are combined conservatively
through the minimum safe probability. Pointer validity is used only for
availability and routing, not as a predictive feature. Per-object safety
scores are aggregated to a frame-level write decision. SAM 3 remains frozen.

## Chapter 5 quantitative figures

### Figure 5.1
File: `chapter5_hard_test_write_rates.svg`
Title: Realized Hard TEST Write Rates Across Final Methods

### Figure 5.2
File: `chapter5_hard_test_bca_effects.svg`
Title: Paired Video-Clustered BCa Effects on Hard TEST POR@30

### Figure 5.3
File: `chapter5_por30_hard_vs_representative.svg`
Title: POR@30 on Hard and Representative TEST Cohorts

### Figure 5.4
File: `chapter5_por30_vs_write_rate.svg`
Title: POR@30 Across Development-Supported Hard TEST Write Rates

The write-rate curve must retain its disclosed zoomed vertical range and must
not be interpreted as evidence at unsupported interpolated operating points.
