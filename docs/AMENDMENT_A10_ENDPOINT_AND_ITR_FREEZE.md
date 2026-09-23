# Amendment A10 - Final Endpoint and ITR Freeze

Date: 2026-09-23

Status: FROZEN WHEN THE COMMIT INTRODUCING THIS FILE IS COMMITTED TO main

## 1. Scope and timing

A10 resolves the remaining endpoint-status and Identity-Theft Rate
denominator questions before any fresh-DEV tracking-performance outcome is
computed or inspected.

At A10 freeze:
- EXP050 used fresh DEV only for outcome-independent physical write-rate
  selection;
- fresh-DEV POR, ITR, J&F, UAR, and contamination outcomes remain unseen;
- TEST remains untouched.

A10 does not alter the frozen models, features, routing, r_star, thresholds,
matched-rate tolerance, split membership, SAM3 substrate, physical
write intervention, or TEST rules.

## 2. Sole primary endpoint

POR@30 is the sole primary endpoint metric.

A qualifying reappearance event and POR@30 use the already-frozen event
construction:
- GT visibility is defined by non-empty target GT mask;
- a qualifying event follows a visibility gap of at least 5 frames;
- recovery means target IoU > 0.5 within the first 30 evaluable
  GT-visible frames after reappearance;
- the existing overlapping-event truncation rule remains unchanged.

POR@15 and POR@60 remain sensitivity endpoints.

## 3. Primary research-question contrast

The single primary research-question contrast is:

    POR@30(B3-S) - POR@30(B2)

at the frozen headline matched-write-rate operating point r_star=0.30.

This contrast isolates the incremental contribution of self-identity
information because B3-S adds self-identity to the frozen B2
quality-plus-temporal feature set.

Primary inference is the paired video-clustered BCa bootstrap 95% confidence
interval for this POR@30 difference.

The video is the statistical cluster.

The frozen +0.08 hard-set POR minimum practically important benefit and the
A4 confidence-interval interpretation rules remain unchanged.

No outcome may be used to replace this primary contrast.

## 4. Secondary scientific contrasts

The following are secondary, pre-specified contrasts:

- B3-R minus B3-S:
  incremental contribution of tracked-competitor relational identity.
- B3-R minus B2:
  total incremental effect of the B3-R identity family over
  quality-plus-temporal information.
- B2 minus B1:
  learned quality-plus-temporal gate versus the frozen manual-rule gate.
- B5 versus the relevant gated methods:
  external DMS-lite write-side comparator.
- gated methods versus B0:
  practical comparison with ungated frozen SAM3, not a same-budget
  selector-quality contrast.

All matched gated contrasts use frozen r_star=0.30 operating points and the
A4 TEST RATE_MISMATCH rule.

B0 remains at its native physical write rate.

These secondary contrasts cannot replace the primary B3-S-versus-B2
contrast after outcomes are observed.

## 5. Identity-Theft Rate definition

The historical theft-frame definition remains unchanged.

A frame is a theft frame when:

    target_iou < 0.3
    AND
    max_other_iou > 0.5

A theft episode requires at least 5 consecutive chronological video frames
that satisfy the theft-frame definition.

The primary ITR reporting unit is the qualifying reappearance event.

For each qualifying reappearance event, use the same chronological
post-reappearance evaluation interval underlying POR@30: from the
reappearance frame through the 30th evaluable GT-visible frame, subject to
the already-frozen overlapping-event truncation rule.

Define:

    theft_event_indicator = 1

if at least one qualifying >=5-frame theft episode occurs within that event
interval, and 0 otherwise.

Then:

    ITR@30 =
        number of qualifying reappearance events with theft_event_indicator=1
        /
        number of qualifying reappearance events evaluated

The denominator includes all qualifying events in the evaluated cohort,
including events with no observed competitor overlap. This measures the
empirical risk that a qualifying reappearance event develops sustained
identity theft under the tested multi-object dataset protocol.

Multiple qualifying reappearance cycles for the same object are separate
events, consistent with POR.

## 6. ITR secondary reporting

ITR@30 is a secondary endpoint, not a co-primary endpoint.

Report:
- event-level ITR@30;
- raw theft-event count;
- number of videos containing at least one theft event.

Historical per-track or per-object-frame normalizations may be reported only
as clearly labeled descriptive diagnostics. They are not substitutes for
the frozen event-level ITR@30 endpoint.

ITR comparative uncertainty must preserve video clustering.

Sparse theft evidence must be reported as sparse; zero or near-zero counts
must not be converted into a stronger conclusion by changing the
denominator.

## 7. Resolution of the F1 contradiction

The historical endpoint contradiction is resolved as follows:

- POR@30 is the sole primary endpoint metric.
- B3-S versus B2 at matched r_star=0.30 is the single primary
  research-question contrast.
- ITR@30 is secondary.
- B0 comparisons are practical-reference comparisons.
- B3-R, B1, and B5 contrasts are secondary.

Therefore the historical proposal to treat POR and ITR as co-primary
endpoints is superseded.

No endpoint or contrast may be promoted or demoted after DEV or TEST
tracking-performance outcomes are observed.

## 8. Outcome firewall

At A10 freeze, no fresh-DEV tracking-performance outcome has been used to
choose:
- the primary endpoint;
- the primary contrast;
- the ITR denominator;
- r_star;
- any gate threshold;
- any model;
- any split.

TEST remains untouched.

Negative, null, harmful, positive, and inconclusive outcomes remain
admissible.
