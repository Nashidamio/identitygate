# Amendment A5 - Hierarchical Missing-Identity Routing

Date: 2026-09-12

Status: PROPOSED FOR FREEZE

## 1. Scope

This amendment resolves only the missing-identity routing rule for B3-S and
B3-R.

It does not change:
- the locked research question;
- frozen SAM3;
- the A3 frame-level physical intervention;
- the A4 matched-write-rate protocol;
- the dual drift/theft labels;
- POR@30 or video-clustered inference;
- the one-touch TEST rule.

It does not freeze final B2/B3 feature completion, dual-head composition,
missingness of the B2-core signals, calibration thresholds, the ITR
denominator, or the F1 endpoint/co-primary question.

## 2. Canonical identity validity

pointer_valid is an availability mask, not a predictive feature.

For a propagated object at frame t:

    pointer_valid = object_score_logit > 0

When pointer_valid is false, SAM uses the learned no-object pointer sentinel.
That pointer is not treated as an identity embedding.

No object is dropped from A3 ALL-SAFE aggregation because identity is
unavailable.

## 3. B3-S routing

B3-S requires a valid self-identity value:

    self_identity_available =
        pointer_valid
        AND finite(ptr_sim_anchor_fp32)

Routing is:

    if self_identity_available:
        use the frozen B3-S development model
    else:
        use the frozen B2-core development model

The fallback therefore removes the unavailable identity increment rather than
turning pointer_valid into a learned or hand-coded risk signal.

The object remains present in A3 ALL-SAFE frame aggregation.

## 4. B3-R routing

B3-R requires both self identity and a tracked-competitor identity value:

    relational_identity_available =
        self_identity_available
        AND at least one distinct tracked competitor anchor exists
        AND finite(max_comp_anchor_cos_fp32)

Routing is hierarchical:

    if relational_identity_available:
        use the frozen B3-R development model
    elif self_identity_available:
        use the frozen B3-S development model
    else:
        use the frozen B2-core development model

For a single-object video there is no distinct tracked competitor, so B3-R
routes to B3-S while self identity is available.

The fallback uses the richest available member of the predeclared nested
ladder and never deletes the object from the frame decision.

## 5. Unexpected numeric failures

The expected scientific missingness case is identity unavailability described
above.

If pointer_valid is true but a required identity value that should be
computable is unexpectedly non-finite because of an implementation or numeric
failure, the run must STOP and record an engineering defect.

Such a failure must not be silently converted into a scientific missingness
case after outcomes are observed.

Missingness of B2-core quality or temporal inputs is not resolved by this
amendment and remains separately OPEN.

## 6. Causality and leakage

Routing may use only:
- current pointer_valid;
- current identity values computed from the current generated pointer;
- permanent frame-0 prompted identity anchors;
- the set of tracked frame-0 objects.

No future GT, future prediction, future label, POR, ITR, J&F, unsafe-admission
outcome, or TEST result may affect routing.

pointer_valid is never supplied as a numeric input to B2, B3-S, or B3-R.

## 7. Calibration and matched-rate evaluation

A5 freezes routing, not final score calibration.

The routed object score remains subject to the separately frozen final
dual-head composition and threshold/calibration protocol.

A4 remains controlling for matched-rate evaluation:
- thresholds are selected from DEV write rate only;
- headline gated methods use the frozen common target-rate procedure;
- each signal method receives an exact per-video neutral matched-budget
  control;
- TEST thresholds are never retuned;
- direct TEST contrasts outside the A4 rate tolerance are labeled
  RATE_MISMATCH.

The video remains the statistical cluster. Primary POR@30 comparative
inference remains paired video-clustered BCa bootstrap, and the locked minimum
practically important hard-set POR effect remains +0.08.

## 8. Evidence boundary

EXP024 established that identity additions did not demonstrate incremental
utility over B2-core in the TRAIN-only held-out utility probe, with theft
comparisons favoring B2-core under the specified clustered bootstrap.

EXP031 established only that actual B3-S and B3-R development weights can be
trained on the valid-identity TRAIN population.

A5 does not claim that identity improves tracking, improves POR@30, or
generalizes. It defines how the nested models behave when identity information
is unavailable so that closed-loop evaluation can proceed without dropping
objects or using pointer validity as a predictive feature.

Negative closed-loop identity results will be retained.

## 9. Feasibility and claim boundary

The routing operation adds no SAM parameters and only selects among the small
already-trained gate models. It does not change the frozen SAM3 memory path.

Based on prior closed-loop measurements in this project, this routing does not
introduce a new 16 GB VRAM feasibility concern. Actual VRAM remains measured
and reported for the closed-loop runs.

No novelty claim is made by this amendment, and no new literature claim is
introduced.
