# Amendment A6 - B1 Manual Rule Baseline

Date: 2026-09-12

Status: PROPOSED FOR FREEZE

## 1. Purpose

This amendment freezes the architecture of B1, the non-learned rule-gate
baseline.

B1 answers whether learning is necessary beyond a defensible manually designed
quality-and-temporal admission rule.

B1 is a comparator, not an IdentityGate novelty claim.

## 2. Literature basis

The canonical thesis rules require B1 to be modeled on published MOSEv2
Memory Quality Filtering (MQF) and OAMVOS-style reliability heuristics rather
than an invented weak baseline.

Verified references for this architecture decision:
- MOSEv2: A More Challenging Dataset for Video Object Segmentation in Complex
  Scenes, arXiv:2508.05630.
- OAMVOS:2nd Report for 5th PVUW MOSE Track, arXiv:2604.22837.

MOSEv2 MQF forms a multiplicative mask-decoder quality score and applies a
quality threshold. OAMVOS uses predicted-IoU and geometry-consistency cues and
treats failure of reliability cues as evidence that the current state should
not be trusted.

B1 adapts those general published reliability principles to the already
verified causal signals available in the frozen SAM3 IdentityGate substrate.
It is not claimed to reproduce either external method exactly.

No published numeric threshold is imported because their scores and host
trackers differ from this experiment.

## 3. Required causal inputs

For object o at eligible non-conditioning frame t, B1 uses only:

- mask_conf_iou_head;
- occ_score_logit;
- area_ratio_anchor;
- temporal_iou_prev.

Definitions follow the currently frozen signal semantics.

area_ratio_anchor is current predicted binary-mask area divided by the
permanent frame-0 GT prompt area.

temporal_iou_prev compares the current binary tracker prediction with the
previous tracker binary prediction; for frame 1 the previous reference is the
frame-0 GT prompt mask.

No future GT, outcome label, POR, ITR, J&F, unsafe-admission outcome, DEV
tracking outcome, or TEST result enters B1 scoring.

## 4. Object-level B1 score

Define:

    q_iou = clip(mask_conf_iou_head, 0, 1)

    q_obj = sigmoid(occ_score_logit)

    q_conf = q_iou * q_obj

For finite area_ratio_anchor r >= 0:

    q_area = 0                         if r == 0
    q_area = min(r, 1 / r)             if r > 0

Define:

    q_temp = clip(temporal_iou_prev, 0, 1)

The manual object admission score is:

    B1_score = min(q_conf, q_area, q_temp)

Thus all components lie in [0,1].

The multiplicative q_conf follows the MQF-style principle that multiple
quality cues jointly determine reliability.

The minimum over confidence, geometry consistency, and temporal consistency
implements a conjunctive reliability rule: a weak required cue can veto a
write.

There are no learned parameters.

## 5. Missing or invalid required signals

If any required raw B1 input is non-finite, the object decision is
FAIL_CLOSED and the object is not admitted at any tau_B1.

This is an engineering missingness rule for B1 only.

It does not resolve the separately OPEN final missingness policy for the
learned B2-core feature family.

Finite r == 0 is not missing; it yields q_area == 0.

## 6. Frame aggregation and physical intervention

A3 remains controlling.

For each frame:

    frame_score_B1(t) = min_o B1_score(o,t)

The eligible non-conditioning frame is admitted only when every tracked object
passes the threshold:

    ADMIT iff frame_score_B1(t) >= tau_B1

Conditioning and prompt frames are always retained.

Physical blocking uses the already frozen A3 whole-frame memory-eviction
mechanism.

## 7. Threshold selection

A6 freezes the scoring rule, not a favorable operating threshold.

A4 remains controlling.

The scalar B1 admission-control parameter is tau_B1.

Initial threshold grid:

    0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00

Only DEV frame-write rate may be inspected for matched-rate threshold
selection.

If A4 permits midpoint refinement for matching, the same deterministic A4
procedure applies.

POR, ITR, J&F, UAR, contamination, failure labels, or any other tracking
outcome may not select tau_B1.

TEST never retunes tau_B1.

## 8. Statistical and feasibility boundary

B0 remains the ungated practical reference.

B1 participates in the same A4 common matched-rate comparison and deterministic
neutral matched-budget control as the learned variants.

Primary POR@30 comparative inference remains paired video-clustered BCa
bootstrap with 95 percent confidence intervals.

The locked minimum practically important hard-set POR effect remains +0.08.

B1 adds no SAM parameters and negligible scalar arithmetic. It introduces no
new 16 GB VRAM feasibility concern relative to the already verified frozen-SAM3
closed-loop path.

## 9. Claim boundary

B1 is a defensible manual quality-and-temporal baseline.

It is not described as MOSEv2 MQF, OAMVOS, or a reproduction of either.

It makes no novelty claim.

No tracking benefit is implied by freezing this rule. Performance is
established only by the later matched-rate closed-loop evaluation.
