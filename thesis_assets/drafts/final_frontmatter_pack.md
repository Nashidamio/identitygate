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
