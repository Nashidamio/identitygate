# Final Thesis Front-Matter Pack

## Recommended title

SIGMA: Signal-Informed Memory-Write Gating in Frozen SAM 3 Video Object Segmentation

### Suggested subtitle

Evaluating Quality, Temporal, and Identity Signals under Controlled Write Budgets

## Abstract

Memory-based video object segmentation systems can accumulate unreliable
predictions when current-frame information is written back into persistent
memory. This thesis studies a narrower design question under a frozen SAM 3
VOS/PVS substrate: what information should a memory-write gate use — quality
signals, temporal signals, or identity signals?

A physically verified closed-loop intervention was developed to admit or block
non-conditioning memory writes without modifying or fine-tuning SAM 3. The
evaluation used a nested ladder comprising an ungated reference (B0), a manual
quality-and-temporal rule (B1), a learned quality-plus-temporal gate (B2), a
self-identity extension using the native SAM 3 object pointer (B3-S), a
tracked-competitor relational identity extension (B3-R), and a prospectively
frozen DMS-lite write-side comparator (B5). Thresholds were selected from
development data using realized write rate only, with a common target
r*=0.30, exact per-video neutral budget controls, and no TEST retuning.

The sole primary endpoint was post-occlusion recovery within 30 evaluable
frames (POR@30). Final evaluation used HARD_TEST80 (80 videos, 506 primary
events) together with REPRESENTATIVE_TEST40 (40 videos, 76 primary events).
Uncertainty was quantified using paired video-clustered BCa 95% confidence
intervals with 50,000 bootstrap replicates and seed 52.

On HARD_TEST80, B2 achieved POR@30=0.7569 and B3-S achieved 0.7648. The frozen
primary B3-S-minus-B2 difference was +0.79 percentage points, with BCa 95% CI
[-0.20, +2.09] percentage points. However, their realized write rates differed
by 3.27 percentage points, exceeding the prospectively frozen 2-point
tolerance; consequently, this result does not support a matched-rate primary
effect interpretation. In the matched secondary comparison, B2 exceeded the
manual B1 rule by +2.96 percentage points in POR@30, BCa 95% CI
[+0.61, +6.43]. B3-R produced no POR@30 increment over B3-S at matched write
rate: 0.00 percentage points, BCa 95% CI [-1.13, +1.43]. On
REPRESENTATIVE_TEST40, B2, B3-S, and B3-R each achieved POR@30=0.7237.

The evidence therefore supports a bounded conclusion. Under the tested frozen
SAM 3 formulation, learned quality-plus-temporal admission improved on the
manual rule, whereas the tested native self-identity and relational
pointer-identity additions did not establish incremental post-occlusion
recovery benefit. The thesis contributes a reproducible closed-loop
memory-write intervention, matched-write-rate evaluation protocol,
signal-family comparison, dual drift/theft outcome framework, and
video-clustered inferential procedure for studying memory admission without
post-TEST retuning.

## Final contribution list

1. A physically verified closed-loop memory-write blocking mechanism for
   frozen SAM 3 VOS/PVS, enabling controlled intervention on memory admission
   without SAM fine-tuning or architectural modification.

2. A nested, prospectively frozen signal-family comparison separating manual
   reliability rules, learned quality-plus-temporal information, self-identity
   information, and tracked-competitor relational identity information.

3. A matched-write-rate evaluation protocol with development-only threshold
   selection, exact per-video neutral budget controls, supported rate-sweep
   operating points, explicit TEST rate-mismatch handling, and no TEST
   retuning.

4. A leakage-controlled evaluation framework combining POR@30 as the sole
   primary endpoint, event-level ITR@30 as a secondary endpoint, dual
   drift/theft failure definitions, and paired video-clustered BCa inference.

5. A reproducible negative-result boundary showing that, under the tested
   frozen SAM 3 native pointer formulation, adding self or relational identity
   information did not establish incremental POR@30 benefit over the
   quality-plus-temporal gate.
