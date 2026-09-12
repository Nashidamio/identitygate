# Amendment A4 - Matched Frame-Write-Rate Evaluation Protocol

Date: 2026-09-12

Status: PROPOSED FOR FREEZE

## 1. Scope

This amendment resolves the matched-write-rate protocol left OPEN by
THESIS_RULES item 6 and Amendment A3.

It does not change:
- the locked research question;
- frozen SAM3;
- the A3 frame-level physical intervention;
- POR@30;
- video-clustered BCa bootstrap inference;
- the +8 percentage-point hard-set POR minimum effect of interest;
- the whole-scene negative control;
- the one-touch TEST rule.

It also does not resolve the still-open ITR denominator, missing-identity
fallback, deferred feature definitions, F1 endpoint-status contradiction,
or final gate/calibration choices.

## 2. Superseded B0-rate instruction

The old v5 instruction to match gated methods directly to B0 physical write
rate is superseded.

Under the verified A3 intervention, B0 retains every eligible
non-conditioning frame. Matching that physical rate would therefore force a
no-block operating point and would not test admission intelligence.

B0 remains the ungated practical reference and is reported at its native
physical frame-write rate. It is not the matched-budget reference.

A raw B0-versus-gated outcome difference may be reported, but it must not be
described as evidence that one selector is better at the same memory budget.

## 3. Physical write-rate denominator

For video v:

    N_v = number of eligible non-conditioning frame-write opportunities
    K_v = number of those opportunities physically admitted

and:

    frame_write_rate_v = K_v / N_v

Conditioning and prompt frames are excluded from both numerator and
denominator.

One eligible frame contributes one opportunity regardless of object count.

For threshold selection, the DEV pooled frame write rate is:

    pooled_write_rate = sum_v K_v / sum_v N_v

The macro mean and distribution of per-video write rates are also reported,
but are not the threshold-matching statistic.

## 4. Common matched operating rate

Headline B1, B2, B3-S, B3-R, and B5 comparisons use one common target
frame-write rate selected on fresh DEV without using POR, ITR, J&F, UAR,
contamination, or any other tracking outcome.

Candidate target rates are tested in this fixed order:

    0.50,
    0.40,
    0.60,
    0.30,
    0.70,
    0.20,
    0.80,
    0.10,
    0.90

The first target for which every required gated method can be matched on DEV
within the tolerance in Section 5 becomes r_star.

This ordering is frozen before matched-rate DEV outcomes are inspected. It
prioritizes the non-degenerate midpoint and then moves symmetrically outward.

B5 enters the common-rate headline comparison only through its frozen
thresholdable implementation. If the approved B5 path cannot expose a
reproducible compatible admission-control parameter within its existing
timebox, use the already-approved DMS-lite fallback. Do not invent a post-hoc
budget wrapper after outcomes are seen.

## 5. Matching tolerance and threshold selection

The headline matching tolerance is:

    abs(realized_DEV_pooled_write_rate - r_star) <= 0.02

This is an absolute two-percentage-point write-rate tolerance.

For learned sigmoid gates B2, B3-S, and B3-R, initial admission-threshold
candidates are:

    tau = 0.00, 0.10, 0.20, ..., 0.90, 1.00

Threshold selection may use only realized DEV write rate.

If no coarse threshold is within tolerance but two adjacent tested thresholds
bracket the target write rate, at most four deterministic midpoint refinements
of that threshold interval are allowed.

At every refinement:
- only write-rate values may be inspected;
- POR, ITR, J&F, UAR, contamination, and other outcome metrics remain hidden
  from the threshold-selection procedure.

If several tested thresholds are within tolerance, choose:
1. smallest absolute write-rate error;
2. then lower admission threshold as the deterministic tie-break.

B1 and B5 use the same write-rate-only rule over the frozen scalar
admission-control parameter defined in their own configs.

If a required variant cannot satisfy the tolerance at any candidate target,
STOP. That variant is not silently called matched. Any protocol change then
requires a new written amendment before outcome inspection.

## 6. Deterministic neutral matched-budget control

Every gated method at the headline operating point receives an
outcome-independent neutral selector control.

For method m and video v, first run the frozen method and record its physical
admission count K_m,v over N_v eligible opportunities.

The neutral control admits exactly K_m,v opportunities from the same video.

For 0 < K_m,v < N_v, order eligible non-conditioning opportunities
chronologically with zero-based index 0,...,N_v-1 and select:

    q_j = floor((j + 0.5) * N_v / K_m,v)

for:

    j = 0,...,K_m,v-1

For K_m,v = 0, select none.
For K_m,v = N_v, select all.

Therefore the neutral control has exactly the same physical write count as
the corresponding signal method in every video:

    K_neutral,v = K_method,v

Per-video neutral budget mismatch tolerance is exactly zero frame
opportunities.

The neutral selector uses no GT, gate score, prediction quality, POR, ITR,
future outcome, or object identity. K_m,v is an admission-budget count from
the frozen signal run, not a tracking outcome.

The neutral selector is an evaluation control, not a deployable online gate.

## 7. Full write-rate curves

Full closed-loop curves remain mandatory.

The target write-rate grid is:

    R_curve = {
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    }

For each gated method, DEV maps its frozen admission-control parameter to
these target rates using the same write-rate-only selection rule in Section 5.

The selected parameter values are frozen before TEST.

Report:
- POR@30 versus realized frame write rate;
- ITR versus realized frame write rate once the ITR denominator is separately
  frozen;
- UAR versus realized frame write rate;
- contamination descriptively versus realized frame write rate;
- B0 as the ungated practical-reference point.

No conclusion may depend only on one favorable threshold.

## 8. TEST handling

All target rates, selected thresholds/control parameters, matching tolerance,
and selector definitions are frozen before TEST.

TEST thresholds are never retuned.

TEST reports the realized physical frame write rate.

For a direct pairwise matched-rate contrast, if the two frozen methods differ
by more than 0.02 absolute pooled write rate on TEST, label that direct
contrast RATE_MISMATCH and do not describe it as a matched-rate result.

Each method-versus-neutral comparison remains exactly budget matched per
video because the neutral control copies that method's physical admission
count.

No TEST outcome may be used to replace r_star, thresholds, tolerance, grid,
or matching rules.

## 9. Statistical inference and effect-size interpretation

The video remains the statistical cluster.

Primary comparative inference remains a paired video-clustered BCa bootstrap
95% confidence interval for POR@30 differences.

Frame-level p-values are not primary evidence.

The locked hard-set minimum practically important POR effect is +0.08.

For a POR difference Delta:
- if the 95% CI is entirely above 0, report statistically supported positive
  evidence;
- if the 95% CI is entirely below 0, report statistically supported harm;
- if the upper 95% CI is below +0.08, the locked +8 pp practically important
  benefit is ruled out under the tested protocol;
- if the CI contains both 0 and +0.08, the result is inconclusive with respect
  to both existence and practical importance of the benefit.

A statistically positive effect smaller than +0.08 is reported as positive
but below the locked practical-effect target.

This amendment does not resolve the separate F1 question of which contrast
is formally co-primary.

## 10. Scientific interpretation

Matched-rate comparisons answer whether the information used by a selector
chooses better memory writes, rather than merely whether it writes less.

The neutral matched-budget control isolates signal-based selection from the
effect of write quantity.

B0 answers the separate practical question of how the gated system compares
with ungated frozen SAM3.

Negative results are retained. No threshold, target rate, selector, or
comparison rule may be changed because POR, ITR, or another outcome is
unfavorable.
