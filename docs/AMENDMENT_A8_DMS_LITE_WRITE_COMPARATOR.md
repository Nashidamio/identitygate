# AMENDMENT A8 - DMS-LITE WRITE-SIDE COMPARATOR

Date: 2026-09-19
Status: FROZEN

## 1. Purpose

This amendment freezes the B5 comparator definition required by the locked
B0/B1/B2/B3-S/B3-R/B5 ladder.

B5 is named DMS-lite write-side comparator.

B5 is NOT an exact reproduction of SAM3-DMS. It is a write-side comparator
derived prospectively from the reliability signal used by SAM3-DMS, adapted
to the already-frozen A3 physical write intervention and A4 matched-write-rate
protocol.

## 2. Upstream provenance

The inspected upstream source is:

- repository: FudanCVL/SAM3-DMS
- inspected commit: 88475d9fc2b6267f8542710375dec92b40590353

At that revision, the upstream tracker defines a memory reliability score from
object existence confidence and the SAM IoU score. The upstream memory
selection mechanism filters previously stored memories for later use rather
than physically blocking the current memory write.

The upstream decoupled mode creates separate tracker state for individual
objects, changing group-level memory assessment to per-object
self-assessment.

Because the locked IdentityGate research question concerns memory-write
admission and A4 matches physical frame write rate, the upstream read-side
selection mechanism is not used directly as B5.

No claim is made that B5 reproduces official SAM3-DMS results.

## 3. B5 object-level reliability score

For tracked object i at eligible non-conditioning frame t, use the existing
frozen SAM3 outputs already exposed by the closed-loop evaluator:

    mask_conf_iou_head_i = current_out["iou_score"][i]
    occ_score_logit_i = current_out["object_score_logits"][i]

No detector signal, identity signal, learned gate output, or additional model
is used by B5.

If either required scalar is nonfinite:

    b5_object_score_i = 0
    missing_policy = FAIL_CLOSED

Otherwise define:

    presence_i = 0                                      if occ_score_logit_i <= 0
    presence_i = 2 * sigmoid(occ_score_logit_i) - 1   otherwise

and:

    b5_object_score_i = presence_i * mask_conf_iou_head_i

No clipping or additional calibration is introduced.

This formula is frozen before observing any B5 tracking result.

## 4. Frame-level aggregation and physical action

A3 remains unchanged.

For every eligible non-conditioning frame:

    frame_score = minimum b5_object_score_i across tracked objects

The physical write decision is:

    ADMIT if frame_score >= tau
    BLOCK otherwise

Conditioning and prompt frames are never blocked.

BLOCK uses the already-verified A3 whole-frame non-conditioning memory
eviction mechanism.

B5 therefore differs from upstream SAM3-DMS in intervention semantics:
upstream selection is read-side memory selection, whereas B5 applies the
upstream-derived reliability signal prospectively to the frozen write-side
intervention.

## 5. Threshold and matched-rate protocol

The only B5 operating parameter is tau.

For engineering sanity and final A4 matching, B5 uses the deterministic
scalar threshold search:

    coarse tau = 0.00, 0.10, ..., 1.00

If a target rate is bracketed by adjacent tested thresholds but no coarse
threshold matches the A4 tolerance, use at most four deterministic midpoint
refinements.

All remaining target-rate order, tolerance, tie-breaking, stopping,
write-rate accounting, neutral-control, full-curve, and TEST mismatch rules
are inherited unchanged from Amendment A4.

Threshold selection uses realized write rate only. POR, ITR, J&F, unsafe
admission, contamination, or any tracking outcome must not influence B5
threshold selection.

## 6. Training and parameter count

B5 has no learned parameters and requires no training.

It reuses frozen SAM3 outputs and the frozen A3 intervention. The external
SAM3-DMS repository is provenance for the comparator signal definition and is
not an additional runtime model dependency for B5.

## 7. First implementation experiment

The first B5 implementation experiment will be a new experiment after EXP038.

Historical EXP033, EXP037, and EXP038 experiment files and recorded hashes
must not be rewritten to add B5.

The first B5 sanity must use development-exposed data only and must not touch
fresh final DEV or TEST.

It must verify at minimum:

- exact score calculation;
- nonfinite FAIL_CLOSED behavior;
- A3 frame-min aggregation;
- physical write-block integrity;
- deterministic A4 threshold matching;
- write counts and realized write rate;
- peak allocated VRAM.

## 8. Resource feasibility

B5 reuses the same frozen SAM3 execution and existing per-object
iou_score/object_score_logits tensors. Its additional computation is scalar
arithmetic plus the existing A3 write intervention.

EXP038 observed maximum peak allocated VRAM of approximately 6.27 GB on the
current sanity scope. B5 is therefore expected to remain below the 16 GB lab
GPU limit, but this is not assumed as a result and must be measured in the B5
sanity execution.

## 9. Final evaluation path

B5 remains mandatory for final common-rate selection on fresh DEV.

After final data exclusions and the fresh DEV/TEST split are frozen:

- B1, B2, B3-S, B3-R, and B5 must all participate in A4 common-rate
  selection;
- B5 must satisfy the same write-rate tolerance as the other gated methods;
- POR@30 remains the primary endpoint;
- full write-rate sweep curves remain mandatory;
- paired video-clustered BCa 95 percent confidence intervals remain the
  primary comparative inference;
- whole-scene negative control and J&F protection remain applicable;
- TEST remains one-touch after all final choices are frozen.

## 10. Supersession boundary

This amendment does not supersede A3, A4, A5, A6, or A7.

It only resolves the previously open operational definition of B5.

The previously audited SAM3-DMS literature position is retained. B5 is an
existing-method-inspired comparator adaptation, not a novelty claim.
