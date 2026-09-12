# Amendment A7 — Final Gate Feature, Composition, and Missingness Freeze

Status: FROZEN PENDING COMMIT

## 1. Purpose

This amendment closes the remaining pre-DEV gate-definition ambiguity.

It freezes the final B2 feature set, the inherited B3-S/B3-R feature sets,
the dual-head admission-score composition, and base-feature missingness
handling before any fresh final DEV outcome is inspected.

No TEST data are touched.

## 2. Supersession

The historical expectation that final B2 must contain all originally
enumerated quality/temporal Features 1-8 is superseded.

Final B2 is the causal, definition-independent five-feature subset already
implemented and trained in EXP029:

1. mask_conf_iou_head
2. occ_score_logit
3. area_norm
5. area_ratio_anchor
6. temporal_iou_prev

This remains a quality-plus-temporal gate.

## 3. Excluded deferred features

Feature 4 — area ratio vs rolling clean reference — is excluded from the
final gate.

Reason:
Its definition requires an IdentityGate-maintained clean-reference state
and a tau_clean promotion policy. No final non-oracle clean-state rule was
pre-specified. Introducing one now would add another recurrent gate state
and threshold after development work has already occurred.

Feature 7 — normalized centroid displacement — is excluded from the final
gate.

Reason:
The primitives were cached, but the exact reference and normalization rule
were never frozen. Selecting a definition after the existing development
experiments would introduce avoidable researcher degrees of freedom.

Feature 8 — frames since last clean write — is excluded from the final
gate.

Reason:
It depends on the same unresolved clean-write state and tau_clean policy as
Feature 4.

No GT/oracle clean-state substitute may be introduced.

## 4. Final ladder feature sets

B1:
Frozen manual A6 rule.

B2:
- mask_conf_iou_head
- occ_score_logit
- area_norm
- area_ratio_anchor
- temporal_iou_prev

B3-S:
B2 plus:
- ptr_sim_anchor_fp32

B3-R:
B3-S plus:
- max_comp_anchor_cos_fp32

pointer_valid is an availability/routing mask only and is never a
predictive input.

## 5. Frozen model weights

Final B2 weights are the existing TRAIN-only EXP029 model:

experiments/EXP029_b2core_train/model.json

SHA256:
6160a6be9da2058808c16182abe03443014806273df46fe59a86411ffc869ecf

Although historical artifacts call this model B2-core, A7 promotes this
exact frozen five-feature model to final B2. The historical filenames and
records are not rewritten.

Final B3-S and B3-R weights are the existing TRAIN-only EXP031 model:

experiments/EXP031_b3_train/model.json

SHA256:
235076b86aa0975b3ec624aae703575fd4f030381b6af4008c3808f2db71847a

No retraining is performed by this amendment.

## 6. Dual-head composition

Each learned variant has independent drift and theft unsafe-probability
heads.

For an object:

p_safe_drift = 1 - p_unsafe_drift
p_safe_theft = 1 - p_unsafe_theft

Final object admission score:

p_safe = min(p_safe_drift, p_safe_theft)

Thus either predicted failure mode can veto a write.

No additional learned calibration layer, weighting coefficient, or
post-hoc combination is introduced.

## 7. Base-feature missingness

For B2, B3-S, and B3-R:

If any required B2 base feature is non-finite, the object fails closed:

object admission score = 0

The object remains in A3 frame-level aggregation and therefore may block
the whole-frame write.

Missing base features are not imputed and the affected object is not
silently dropped.

## 8. Identity missingness

A5 remains unchanged.

B3-S:
- self identity available -> B3-S
- self identity unavailable -> B2

B3-R:
- relational identity available -> B3-R
- self identity only -> B3-S
- self identity unavailable -> B2

For a single-object frame, B3-R routes to B3-S when self identity is
available.

pointer_valid remains an availability/routing mask only.

Unexpected non-finite identity when identity is expected to be computable
is an engineering defect and must STOP execution rather than silently
route around the defect.

## 9. Physical intervention

A3 remains unchanged.

Object-level scores are aggregated with:

frame_score = minimum object score over all tracked objects

At a frozen operating threshold tau:

ADMIT iff frame_score >= tau

Otherwise the current non-conditioning frame is physically evicted from
the SAM3 memory banks according to the verified A3 intervention.

Conditioning frames are retained.

## 10. Operating thresholds and matched write rate

A7 does not select an operating threshold.

A4 remains controlling:

- DEV target write rate is selected using write rate only.
- Threshold matching uses the frozen A4 deterministic procedure.
- Full rate curves are mandatory.
- TEST does not retune thresholds.
- Direct TEST comparisons with pooled write-rate mismatch > 0.02 are
  marked RATE_MISMATCH.

## 11. Statistical path

Primary comparative endpoint:
POR@30.

Primary comparative inference:
paired video-clustered BCa bootstrap 95% confidence interval for POR@30
differences.

Video is the statistical cluster.

The minimum practically important hard-set POR effect remains +0.08.

Interpretation remains:
- CI entirely above 0: positive evidence.
- CI entirely below 0: evidence of harm.
- CI upper bound below +0.08: rules out a +8 percentage-point benefit.
- CI spanning both 0 and +0.08: inconclusive.

## 12. Compute feasibility

EXP033 executed B1, B2, B3-S, and B3-R sequentially inside frozen SAM3 on
the RTX 4080 SUPER 16 GB machine.

Observed gated-variant peak VRAM was approximately 6.06-6.18 GB and B0
peak VRAM was approximately 6.27 GB.

Therefore the frozen gate architecture is feasible on the available
16 GB GPU under the verified mechanism-sanity workload.

Full DEV evaluation must still monitor peak VRAM and is not assumed to
have executed until recorded.

## 13. Claim boundary

A7 is an architecture/protocol freeze, not evidence of tracking benefit.

The existing TRAIN-only descriptive results do not establish whether B1,
B2, B3-S, or B3-R improves tracking.

No novelty claim is made by this amendment.

Fresh final DEV remains untouched at this freeze.

TEST remains untouched.

## 14. Consequence

After A7 is committed, the final learned gate architecture and score
composition are frozen.

The next scientific step is to establish the current final DEV population
without using the superseded EXP017 final split, then execute the A4
matched-write-rate DEV evaluation.

No gate feature, model weight, composition rule, or missingness policy may
be changed in response to DEV tracking outcomes without a new prospective
amendment that explicitly records the resulting loss of confirmatory
status.
