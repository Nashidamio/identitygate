# Amendment A12 - Final TEST Execution Freeze

Date: 2026-09-24

Status: FROZEN WHEN THE FINAL EXP053 IMPLEMENTATION COMMIT IS COMMITTED TO main

## 1. Scientific choices unchanged

A12 changes no model, feature, threshold, r_star, split, event definition,
endpoint, statistical cluster, practical threshold, or inference rule.

Fresh DEV outcomes from EXP051 and EXP052 have already been observed.
This amendment only freezes the operational scope of the one-touch final TEST
campaign before any TEST prediction is generated.

## 2. Final TEST cohorts

HARD_TEST80:
- 80 videos.
- 506 frozen primary events.
- membership SHA256: 6bf5a059c05d07f82e43fec9bd6b4c4b723bc551a21b72c988912e57b28ce582

REPRESENTATIVE_TEST40:
- 40 videos.
- 76 frozen primary events.
- membership SHA256: ba23def8c8d0de7a83af64c6f952544d5f3e44ad6ca018f9e4d2b6cd82ebfb66

HARD_TEST80 is the primary hard-stratum evaluation cohort.

REPRESENTATIVE_TEST40 is the development-untouched external-validity anchor.
It receives every final headline variant once and the corresponding exact-K
matched neutral controls.

## 3. Headline operating point

r_star = 0.30

Frozen headline thresholds:
- B1 tau=0.1
- B2 tau=0.1
- B3_S tau=0.2
- B3_R tau=0.2
- B5 tau=0.7

B0 remains at native write rate.

Each gated headline method receives an outcome-independent exact per-video
neutral control using the same physical admission count.

## 4. A9 supported curve points

Exactly 19 successful DEV mappings are frozen as SUPPORTED.
Exactly 26 requested mappings remain UNSUPPORTED.

SUPPORTED mappings:
- target=0.1 variant=B1 tau=0.4 DEV_rate=0.107692307692
- target=0.1 variant=B2 tau=0.9 DEV_rate=0.084990619137
- target=0.1 variant=B3_S tau=0.9 DEV_rate=0.098686679174
- target=0.1 variant=B3_R tau=0.95 DEV_rate=0.093245778612
- target=0.1 variant=B5 tau=0.925 DEV_rate=0.103377110694
- target=0.2 variant=B1 tau=0.2 DEV_rate=0.201688555347
- target=0.2 variant=B2 tau=0.5 DEV_rate=0.205440900563
- target=0.2 variant=B3_S tau=0.6 DEV_rate=0.188180112570
- target=0.2 variant=B3_R tau=0.7 DEV_rate=0.201313320826
- target=0.2 variant=B5 tau=0.8500000000000001 DEV_rate=0.182739212008
- target=0.3 variant=B1 tau=0.1 DEV_rate=0.283677298311
- target=0.3 variant=B2 tau=0.1 DEV_rate=0.309193245779
- target=0.3 variant=B3_S tau=0.2 DEV_rate=0.303939962477
- target=0.3 variant=B3_R tau=0.2 DEV_rate=0.290431519700
- target=0.3 variant=B5 tau=0.7 DEV_rate=0.298123827392
- target=0.4 variant=B2 tau=0.0125 DEV_rate=0.382363977486
- target=0.4 variant=B3_S tau=0.0125 DEV_rate=0.395121951220
- target=0.4 variant=B3_R tau=0.00625 DEV_rate=0.399624765478
- target=0.4 variant=B5 tau=0.3 DEV_rate=0.402063789869

No unsupported TEST threshold is generated.

The five target=0.30 supported mappings are identical to the headline
trajectories and are reused rather than rerun.

HARD_TEST80 executes all 19 supported curve points and exact-K neutral
controls. REPRESENTATIVE_TEST40 executes headline variants and headline
neutral controls only because its frozen role is external-validity anchoring,
not threshold-curve construction.

## 5. TEST rate handling

TEST thresholds are never searched, interpolated, extrapolated, refined, or
retuned.

For any direct gated pairwise comparison, an absolute pooled TEST write-rate
difference greater than 0.02 is labelled RATE_MISMATCH and is not described
as a matched-rate result.

Every method-versus-neutral comparison remains exactly budget matched per
video by construction.

## 6. Endpoints and inference

POR@30 remains the sole primary endpoint.

Primary contrast:
    POR@30(B3-S) - POR@30(B2)

ITR@30 remains secondary.

The video remains the statistical cluster.

Final inferential intervals use the A11 paired video-clustered BCa 95 percent
procedure.

The minimum practically important hard-set POR benefit remains +0.08.

## 7. One-touch campaign semantics

The initial TEST command is mode run.

Immediately before the first TEST trajectory is allowed to access video
frames or annotations, EXP053 creates a persistent TEST campaign marker
binding the campaign to the frozen repository commit, config SHA256, and
cohort memberships.

If engineering interruption occurs after that marker exists, only mode resume
under the exact same frozen commit/config may continue unfinished cached
trajectories.

Resume is continuation of the same frozen campaign. It cannot change code,
configuration, thresholds, cohorts, endpoints, comparisons, or statistics.

No partial POR or ITR result may be used to change the campaign while it is
in progress.

## 8. Outcome firewall

No TEST outcome may alter:
- B1/B2/B3-S/B3-R/B5 models;
- thresholds;
- r_star;
- neutral construction;
- supported/unsupported curve status;
- cohorts;
- event definitions;
- endpoint hierarchy;
- RATE_MISMATCH tolerance;
- bootstrap implementation;
- +0.08 practical threshold.

Negative, null, harmful, positive, and inconclusive TEST outcomes are all
admissible.
