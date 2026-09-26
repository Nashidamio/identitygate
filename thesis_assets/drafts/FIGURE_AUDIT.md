# Thesis Figure Audit

Status: PRE-INTEGRATION SCIENTIFIC AUDIT

## Figure A - Simple SIGMA pipeline

Decision: DO NOT INCLUDE AS-IS.

Reasons:
- labels the core tracker as SAM 3.1; final core substrate is SAM 3;
- shows a 128-dimensional identity representation, whereas the final native
  SAM 3 object pointer is 256-dimensional;
- substantially overlaps the more informative end-to-end pipeline figure.

Preferred action:
Drop this figure unless it is redrawn from the final implementation.

## Figure B - Related-work landscape and signal-family comparison

Decision: KEEP AFTER CORRECTION.

Intended placement:
Chapter 2, near the end of Related Work.

Required checks:
- exact paper names, years, and citations must match the audited bibliography;
- correct any abbreviated-name spelling errors;
- do not state that all compared methods use the same memory budget;
- replace "Same memory budget (matched write rate)" with wording equivalent to
  "Pre-specified write-budget protocol with explicit rate-mismatch handling";
- retain SIGMA as the controlled comparison framework rather than a claimed
  state-of-the-art method.

## Figure C - End-to-end SIGMA and evaluation pipeline

Decision: KEEP AFTER MINOR CORRECTION.

Intended placement:
Opening overview of Chapter 3 or transition from Chapter 1 to methodology.

Required changes:
- "SIGMA (proposed)" -> "SIGMA evaluation framework";
- "Matched-rate setup" -> "Pre-specified write-budget protocol";
- indicate that the frozen TEST interpretation includes the prospectively
  defined rate-mismatch rule;
- ensure feature-family labels match the final B2/B3-S/B3-R definitions.

If the cat/video imagery is generated rather than sampled from MOSEv2, the
caption must state that the imagery is schematic and is not experimental
dataset evidence.

## Figure D - Gate architecture

Decision: HOLD UNTIL CODE-LEVEL VERIFICATION.

Intended placement:
Chapter 4 implementation section.

Items requiring verification against the final EXP029/EXP031 implementation:
- whether the model uses a genuinely shared backbone;
- exact hidden dimensions and nonlinearities;
- exact relationship between drift and theft heads;
- exact p_safe composition;
- exact object-level to frame-level aggregation;
- B2/B3-S/B3-R feature counts and routing.

No architecture element should be included because it was planned; it must
match the executed implementation.

## Figure E - Dataset and cohort construction

Decision: KEEP IF ALL COUNTS AND RULES VERIFY AGAINST FINAL ARTIFACTS.

Intended placement:
Chapter 3 dataset/evaluation section.

Must verify:
- whole-scene event/video counts;
- post-exclusion eligible-pool counts;
- hard-cohort video/event counts;
- Fresh DEV event count;
- DI-v1 component definitions;
- DINOv2 diversity construction;
- event-level versus video-level exclusion wording;
- final Hard TEST and Representative TEST counts.

The final 80-video / 506-event Hard TEST and 40-video / 76-event
Representative TEST labels must remain unchanged.

## General figure rule

Schematic figures explain design; they are not empirical evidence.
Experimental claims must continue to come from frozen tables, plots, hashes,
and TEST inference artifacts.

Institution-specific rules for generated imagery remain to be checked before
final PDF submission.
