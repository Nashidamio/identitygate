# IdentityGate Final Submission Integration Pack

Status: DERIVED THESIS-WRITING ARTIFACT

Scientific source of truth remains the frozen repository experimental record.

Critical final interpretation:
- POR@30 is the sole primary endpoint.
- Primary contrast: B3-S minus B2.
- HARD_TEST80 primary realized-rate difference exceeds the frozen 0.02
  tolerance.
- Therefore:
  RATE_MISMATCH_NO_MATCHED_RATE_PRIMARY_INTERPRETATION.
- No TEST rerun or TEST threshold retuning is permitted.

---

# Final Thesis Front-Matter Pack

## Recommended title

When Should SAM 3 Write to Memory? A Controlled Study of Signal-Informed Gating for Video Object Segmentation

## Abstract

Memory-based video object segmentation depends not only on what a tracker
remembers, but also on which predictions are allowed to become future memory.
This thesis asks a controlled design question under a frozen SAM 3 VOS/PVS
substrate: "What information should a memory-write gate use - quality signals,
temporal signals, or identity signals?"

To isolate memory-write selection from model adaptation, SAM 3 was kept frozen
and a physically verified closed-loop intervention was implemented to admit or
block non-conditioning memory writes. The evaluation used a nested ladder:
an ungated reference (B0), a manual quality-and-temporal rule (B1), a learned
quality-plus-temporal gate (B2), a self-identity extension using the native
SAM 3 object pointer (B3-S), a tracked-competitor relational identity extension
(B3-R), and a prospectively frozen DMS-lite write-side comparator (B5).
Operating points were selected from development data using realized write rate,
with target r*=0.30, supported rate sweeps, exact per-video neutral budget
controls, and no TEST retuning.

The sole primary endpoint was post-occlusion recovery within 30 evaluable
frames (POR@30). Final evaluation used HARD_TEST80 (80 videos, 506 primary
events) and REPRESENTATIVE_TEST40 (40 videos, 76 primary events). Inference
used paired video-clustered BCa 95% confidence intervals with 50,000 bootstrap
replicates and seed 52.

On HARD_TEST80, the clearest positive matched-rate evidence favored learned
quality-plus-temporal gating: B2 exceeded the manual B1 rule by 2.96 percentage
points in POR@30, with BCa 95% CI [+0.61, +6.43]. The frozen primary B3-S-minus-
B2 comparison had a +0.79 percentage-point POR@30 difference, BCa 95% CI
[-0.20, +2.09], but its 3.27 percentage-point realized write-rate difference
exceeded the prospectively frozen 2-point tolerance. It therefore cannot be
interpreted as a matched-rate primary identity effect. At matched TEST write
rate, B3-R produced no POR@30 increment over B3-S: 0.00 percentage points,
BCa 95% CI [-1.13, +1.43]. On REPRESENTATIVE_TEST40, B2, B3-S, and B3-R each
obtained POR@30=0.7237.

The results therefore give a bounded answer to the research question. Within
the tested frozen SAM 3 formulation, learned quality-plus-temporal admission
was supported relative to the manual rule, whereas the tested native
self-identity and tracked-competitor pointer signals did not establish
incremental post-occlusion recovery benefit. The study also demonstrates why
memory-write policies require controlled write-budget interpretation: a
numerically favorable result can cease to support the intended comparison when
its realized write rate violates the pre-specified tolerance. The thesis
contributes a reproducible closed-loop intervention, a signal-family
comparison, a controlled write-budget protocol, dual drift/theft outcomes, and
video-clustered inference for studying memory admission without post-TEST
retuning.

## Final contribution list

1. A controlled experimental formulation of memory-write admission in frozen
   SAM 3 that asks which information families contribute to useful write
   decisions rather than treating gating only as an end-to-end leaderboard
   method.

2. A physically verified closed-loop memory-write intervention for SAM 3
   VOS/PVS, enabling admit/block decisions without fine-tuning, LoRA, detector
   modification, or changes to the SAM architecture.

3. A nested signal-family comparison separating a manual quality/temporal
   rule, learned quality-plus-temporal gating, native self-identity, and
   tracked-competitor relational identity information, with explicit
   missing-identity routing.

4. A controlled write-budget evaluation protocol with development-only
   operating-point selection, supported rate sweeps, exact per-video neutral
   controls, explicit TEST rate-mismatch handling, and no TEST retuning.

5. A leakage-controlled outcome and inference framework using POR@30 as the
   sole primary endpoint, ITR@30 as a secondary endpoint, dual drift/theft
   definitions, and paired video-clustered BCa inference.

6. An empirical answer with an explicit interpretation boundary: learned
   quality-plus-temporal gating improved over the manual rule in the matched
   Hard TEST comparison, while the tested native self and relational identity
   additions did not establish incremental POR@30 benefit; the frozen
   matched-rate rule prevented a numerically favorable but rate-mismatched
   identity result from being overinterpreted.

---

# Chapter 5 - Results

## 5.1 Performance Evaluation

Final evaluation used two pre-frozen cohorts: HARD_TEST80 with 80 videos and
506 primary reappearance events, and REPRESENTATIVE_TEST40 with 40 videos and
76 primary events. All methods were executed under the frozen SAM 3 VOS/PVS
substrate and the final TEST thresholds were not retuned.

On HARD_TEST80, POR@30 was 0.7411 for B0, 0.7273 for B1, 0.7569 for B2,
0.7648 for B3-S, 0.7648 for B3-R, and 0.7530 for B5. The corresponding
ITR@30 values were 0.0593, 0.0514, 0.0316, 0.0395, 0.0336, and 0.0336.
The realized physical write rates were 1.0000 for B0, 0.3199 for B1,
0.3233 for B2, 0.2906 for B3-S, 0.2923 for B3-R, and 0.3080 for B5.

On REPRESENTATIVE_TEST40, POR@30 was 0.5789 for B0, 0.6711 for B1,
0.7237 for B2, 0.7237 for B3-S, 0.7237 for B3-R, and 0.6974 for B5.
These representative-cohort results are reported descriptively as the
development-untouched external-validity anchor.

## 5.2 Analysis of Design Solutions

The learned quality-plus-temporal gate B2 improved over the manual B1 rule on
the Hard TEST at closely matched realized write rates. The paired difference
B2 minus B1 was +0.02964 POR@30, with a video-clustered BCa 95% confidence
interval from +0.00612 to +0.06430. The realized write-rate difference was
0.00335, within the frozen 0.02 tolerance.

Adding the tested self-identity pointer signal in B3-S increased the frozen
Hard TEST point estimate over B2 by only +0.00791 POR@30. Its BCa 95%
confidence interval was -0.00204 to +0.02088. However, B3-S and B2 differed
in realized Hard TEST write rate by 0.03270, exceeding the frozen 0.02
tolerance. The primary contrast is therefore RATE_MISMATCH and cannot be
interpreted as a matched-rate primary effect.

Adding tracked-competitor identity in B3-R produced no POR@30 change relative
to B3-S: delta = 0.00000, BCa 95% CI [-0.01129, +0.01431]. This comparison
remained within the frozen TEST write-rate tolerance. Thus, under the tested
frozen formulation, relational pointer identity did not establish incremental
POR@30 benefit over self-identity.

B5, the frozen DMS-lite write-side comparator, differed from B2 by -0.00395
POR@30 with BCa 95% CI [-0.01961, +0.01250] on Hard TEST, while satisfying
the TEST write-rate tolerance.

## 5.3 Final Design Adjustments

All final design changes used in TEST were frozen prospectively. The final B2
feature set contained mask IoU-head confidence, object-presence logit,
normalized mask area, frame-0 anchor area ratio, and temporal IoU to the
previous prediction. B3-S added FP32 self-anchor pointer cosine, while B3-R
added maximum tracked-competitor anchor cosine. Pointer validity remained a
routing mask rather than a predictive input. Missing identity was handled
hierarchically by routing B3-R to B3-S or B2, and B3-S to B2, according to
which identity signals were valid.

The final common DEV target rate was r*=0.30. Frozen headline thresholds were
0.10 for B1, 0.10 for B2, 0.20 for B3-S, 0.20 for B3-R, and 0.70 for B5.
TEST thresholds were never retuned.

## 5.4 Statistical Analysis

The video was the statistical cluster. Primary uncertainty used a paired
video-clustered BCa 95% confidence interval with 50,000 bootstrap replicates
and seed 52. The sole primary endpoint was POR@30 and the primary
research-question contrast was B3-S minus B2.

The primary frozen-threshold point estimate was +0.7905 percentage points,
with BCa 95% CI [-0.2037, +2.0882] percentage points. The upper confidence
bound was below the pre-registered +8 percentage-point practically important
effect threshold. Nevertheless, because the two methods exceeded the frozen
TEST write-rate mismatch tolerance, this interval is not interpreted as a
matched-rate primary-effect estimate.

The secondary B2-minus-B1 contrast showed +2.964 percentage points,
BCa 95% CI [+0.612, +6.430] percentage points, at matched TEST write rate.
The B3-R-minus-B3-S contrast was 0.000 percentage points,
BCa 95% CI [-1.129, +1.431] percentage points, also at matched TEST write
rate.

For the secondary ITR@30 endpoint, B2 minus B1 was -1.976 percentage points,
BCa 95% CI [-4.848, -0.680] percentage points, at matched TEST write rate.
B3-S minus B2 was +0.791 percentage points, BCa 95% CI
[+0.187, +2.407] percentage points, but this pair was RATE_MISMATCH.
B3-R minus B3-S was -0.593 percentage points with BCa 95% CI
[-2.283, +0.412] percentage points.

## 5.5 Comparisons and Relationships

Outcome-independent exact-K neutral controls matched each signal method
exactly in physical write count for every video. None of the reported
signal-versus-neutral POR@30 confidence intervals established a clear
positive signal-selection advantage. This indicates that part of the observed
performance is attributable to write quantity and emphasizes the importance
of the matched-budget control.

The supported write-rate sweep also showed that method ordering was not
constant across operating rates. Conclusions are therefore based on the
pre-frozen headline operating point together with the supported curve rather
than on a selectively chosen threshold.

## 5.6 Discussion

The clearest positive comparative evidence on the Hard TEST concerned learned
quality-plus-temporal selection: B2 exceeded the manual B1 rule at matched
write rate. In contrast, the native SAM 3 pointer-based self-identity addition
did not establish an incremental POR@30 advantage, and the primary B3-S
versus B2 comparison additionally lost its matched-rate interpretation because
the frozen thresholds realized write rates more than two percentage points
apart on TEST.

The relational identity extension B3-R also did not improve POR@30 over B3-S
at a matched TEST write rate. Representative TEST reinforced the absence of an
observable POR@30 separation among B2, B3-S, and B3-R, all of which obtained
0.7237. These findings should not be generalized to identity information in
general. They apply to the tested frozen SAM 3 pointer-based identity
formulation, its hierarchical missing-identity routing, and the evaluated
MOSEv2 protocol.

---

# Chapter 6 - Conclusions and Future Work

## 6.1 Answer to the Research Question

This thesis asked: "What information should a memory-write gate use - quality
signals, temporal signals, or identity signals?"

The final evidence supports a bounded answer rather than a universal ranking of
signal types. Under the frozen SAM 3 VOS/PVS formulation evaluated here, the
strongest positive matched-rate evidence favored the learned
quality-plus-temporal gate. On HARD_TEST80, B2 improved over the manual B1 rule
by 2.964 percentage points in POR@30, with a paired video-clustered BCa 95%
confidence interval of [0.612, 6.430] percentage points while satisfying the
frozen matched-write-rate tolerance.

The tested native pointer-based identity additions did not establish
incremental POR@30 benefit. At the frozen operating points, B3-S exceeded B2
by 0.791 percentage points, with BCa 95% CI [-0.204, 2.088] percentage points.
However, the pair realized a 3.270 percentage-point write-rate difference,
which exceeded the frozen 2-percentage-point tolerance. The prospectively
specified protocol therefore forbids interpreting this numerically favorable
difference as a matched-rate primary identity effect.

The relational extension provided a cleaner incremental identity comparison.
B3-R produced a 0.000 percentage-point POR@30 difference relative to B3-S at
matched TEST write rate, with BCa 95% CI [-1.129, 1.431]. On
REPRESENTATIVE_TEST40, B2, B3-S, and B3-R each obtained POR@30=0.7237.

Taken together, these results support learned quality-plus-temporal admission
relative to the manual rule, but do not establish additional post-occlusion
recovery benefit from the tested native self-identity or tracked-competitor
pointer signals. This is a result about the tested signal definitions and the
frozen SAM 3 memory interface; it is not evidence that identity information in
general is useless for memory management.

## 6.2 What the Study Establishes

The thesis makes three broader empirical points.

First, memory-write admission is an experimentally meaningful intervention.
The gate was connected to the physical closed-loop write path rather than
evaluated only as an offline classifier, so admit/block decisions were capable
of changing subsequent tracker state.

Second, learned selection can improve on a manual reliability rule without
changing the underlying segmentation model. The B2-versus-B1 result shows
that, in the tested Hard TEST setting, the learned quality-plus-temporal
formulation captured useful admission information beyond the manually
specified rule.

Third, raw performance ordering is not sufficient evidence for a signal
family. B3-S achieved numerically higher POR@30 than B2, but the frozen
matched-rate tolerance was violated. The protocol therefore changed the
scientific interpretation of the observed result. This is an important
outcome of the study: controlling how often a method writes is necessary
before attributing downstream differences to which frames it selected.

## 6.3 Contributions

The main contributions are:

1. a controlled formulation of memory-write admission in frozen SAM 3 around
   the question of what information a write gate should use;

2. a physically verified closed-loop admit/block intervention in SAM 3
   VOS/PVS without fine-tuning or architectural modification;

3. a nested signal-family ladder spanning a manual quality/temporal rule,
   learned quality-plus-temporal gating, self-identity, and
   tracked-competitor relational identity;

4. a leakage-controlled missing-identity routing protocol that treats pointer
   validity as signal availability rather than as a predictive feature;

5. a controlled write-budget protocol with development-only operating-point
   selection, supported write-rate curves, exact per-video neutral controls,
   explicit TEST rate-mismatch handling, and no TEST retuning;

6. a dual drift/theft outcome framework with POR@30 as the sole primary
   endpoint, ITR@30 as a secondary endpoint, and paired video-clustered BCa
   inference with a frozen 50,000-replicate implementation; and

7. a bounded empirical finding: learned quality-plus-temporal gating improved
   over the manual rule in the matched Hard TEST comparison, whereas the
   tested native self-identity and relational identity additions did not
   establish incremental POR@30 benefit.

## 6.4 Limitations

The primary B3-S versus B2 TEST contrast missed the frozen write-rate
tolerance. Consequently, the study cannot make a matched-rate primary-effect
claim for self-identity over B2 even though the B3-S point estimate was
numerically higher.

The representative cohort contained 40 videos and 76 primary events and was
used as an external-validity anchor rather than as a replacement primary
cohort. Its equal POR@30 values for B2, B3-S, and B3-R are descriptive evidence
within that cohort, not proof of equivalence.

B5 was a prospectively frozen DMS-lite write-side comparator inspired by the
SAM3-DMS reliability signal; it was not an exact reproduction of the external
SAM3-DMS system.

Identity evidence was restricted to the frozen SAM 3 native object-pointer
space and the pre-specified reference construction. Failure to establish an
incremental benefit for these signals does not imply that learned identity
embeddings, alternative temporal references, different memory architectures,
or jointly optimized read/write systems would behave similarly.

The study intentionally performed one frozen TEST campaign. No subgroup
search, TEST threshold retuning, or additional TEST experiment was performed
after observing the final outcomes. This protects the interpretation of the
registered comparisons but also means that potentially interesting
heterogeneity remains prospective rather than demonstrated.

## 6.5 Interpretation and Future Work

A useful hypothesis emerging from the results is that an informative local
frame is not necessarily a useful future memory write. A prediction may appear
high-quality, temporally consistent, or identity-consistent while still being
redundant with existing memory or unhelpful for future recovery. The present
study did not directly measure causal per-write utility, so this distinction
is an interpretation and future-work hypothesis rather than a demonstrated
mechanism.

A prospective follow-up could measure write utility directly by comparing
downstream trajectories under controlled inclusion or exclusion of individual
candidate writes. Such a design could test whether quality, temporal, and
identity scores rank actual future memory utility, rather than merely
correlating with current-frame reliability.

Future work should also evaluate stronger identity representations explicitly
optimized for discrimination, learned temporal identity references,
alternative trusted-memory architectures, and adaptive rate control that
preserves a predeclared memory budget.

Replication on additional densely annotated multi-object VOS datasets would
test whether the observed quality-versus-identity pattern generalizes beyond
MOSEv2. Cross-model replication could determine whether the findings are
specific to the SAM 3 VOS memory interface or reflect a broader property of
foundation-model video trackers.

Any such analyses should be treated as new prospective experiments rather
than post-hoc modifications of the present frozen TEST result.

---

# Final Tables and Figures — Suggested Captions

## Tables

### Main final TEST table

Final headline performance on the frozen HARD_TEST80 and
REPRESENTATIVE_TEST40 cohorts. POR@30 is the sole primary endpoint; ITR@30 is
secondary. Realized write rate is the physical fraction of eligible
non-conditioning frame-write opportunities admitted. B0 retains its native
write rate and is not a matched-budget reference.

### Statistical comparison table

Paired video-clustered BCa 95% confidence intervals for pre-specified Hard
TEST contrasts. Bootstrap inference used 50,000 replicates with deterministic
seed 52. Direct gated comparisons with an absolute pooled TEST write-rate
difference greater than 0.02 are labelled RATE_MISMATCH and are not interpreted
as matched-rate effects. Signal-versus-neutral controls are exactly budget
matched per video.

## Figures

### Hard versus Representative POR@30

POR@30 for the six final headline methods on the frozen HARD_TEST80 and
REPRESENTATIVE_TEST40 cohorts. Representative TEST is reported as a
development-untouched external-validity anchor rather than as a replacement
primary cohort.

### Hard TEST realized memory write rates

Realized physical frame-write rates for the final Hard TEST headline methods.
The dashed line denotes the development-selected target r*=0.30. B0 is the
ungated native-write reference and therefore operates at write rate 1.0.
Frozen TEST thresholds were not retuned to restore rate matching.

### Hard TEST BCa effects

Pre-specified Hard TEST POR@30 differences with paired video-clustered BCa 95%
confidence intervals. The vertical zero line denotes no difference and the
dashed +8 percentage-point line denotes the prospectively frozen minimum
practically important benefit. Rate-match status is shown separately because
the primary B3-S-minus-B2 comparison exceeded the frozen TEST write-rate
tolerance.

### POR@30 versus realized write rate

POR@30 across all development-supported Hard TEST operating points for B1,
B2, B3-S, B3-R, and B5. Only thresholds prospectively supported by the
development write-rate mapping were evaluated; unsupported targets were not
interpolated or retuned on TEST. The y-axis is restricted to 0.70–0.78 to make
the observed variation readable.

---

# Appendix Integration Checklist

## Appendix A — Frozen experimental protocol

Include or summarize:
- frozen SAM 3 substrate and pinned commit;
- no fine-tuning / no LoRA / no detector modification;
- gate parameter-count constraint;
- B0, B1, B2, B3-S, B3-R, B5 definitions;
- A5 missing-identity routing;
- A6 B1 manual rule;
- A7 final feature definitions;
- A8 B5 DMS-lite definition;
- A9 supported development rate mapping;
- A10 endpoint hierarchy;
- A11 paired video-clustered BCa procedure;
- A12 TEST freeze;
- A13 final TEST inference lock.

## Appendix B — Frozen operating points

Final headline thresholds:
- B1: 0.10
- B2: 0.10
- B3-S: 0.20
- B3-R: 0.20
- B5: 0.70
- B0: native write rate

Development target:
- r*=0.30

TEST matched-rate tolerance:
- absolute pooled write-rate difference <= 0.02

## Appendix C — Statistical reporting

Report:
- HARD_TEST80: 80 videos, 506 primary events;
- REPRESENTATIVE_TEST40: 40 videos, 76 primary events;
- paired video-clustered BCa 95% confidence intervals;
- 50,000 bootstrap replicates;
- seed 52;
- video as statistical cluster;
- +0.08 minimum practically important POR benefit.

## Appendix D — Reproducibility package

Reference:
- RESEARCH_HISTORY.md
- THESIS_RULES.md
- SUPERVISOR_CONDITIONS.md
- EXPERIMENT_REGISTRY.md
- RESEARCH_STATE.md
- requirements.lock.txt
- SIGNAL_SCHEMA-v1.md
- thesis_assets/ASSET_MANIFEST.csv

## Appendix E — Qualitative illustration boundary

The EXP025 GIF/contact sheet is a TRAIN-exposed closed-loop mechanism
illustration only. It is not inferential TEST evidence and must not be
described as proof of final B1/B2/B3 performance.

---

# Whole-Thesis Mandatory Consistency Replacements

Before PDF freeze, verify that no chapter states:

- SAM 3.1 as the final substrate;
- MOSEv2 valid split as usable final validation;
- EXP017 as the final TEST split;
- memory encoding as proof of valid identity pointer;
- negative identity margin as identity theft;
- a single failure label instead of drift/theft;
- B3 as a single undifferentiated final condition;
- ITR@30 as co-primary;
- B3-S versus B2 as a valid matched-rate primary TEST effect;
- any TEST threshold retuning;
- any claim that native pointer identity generally does not work.

Required terminology:

- SAM 3
- B3-S
- B3-R
- POR@30 sole primary endpoint
- ITR@30 secondary endpoint
- paired video-clustered BCa
- RATE_MISMATCH_NO_MATCHED_RATE_PRIMARY_INTERPRETATION
- frozen TEST
- no TEST retuning
