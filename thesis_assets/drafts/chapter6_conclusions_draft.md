# Chapter 6 - Conclusions and Future Work

## 6.1 Findings

The thesis asked: "What information should a memory-write gate use - quality
signals, temporal signals, or identity signals?"

Under the frozen SAM 3 VOS/PVS protocol tested here, the strongest positive
comparative evidence was obtained from the learned quality-plus-temporal gate.
B2 improved over the manual B1 rule on Hard TEST by 2.964 percentage points
in POR@30, with a video-clustered BCa 95% confidence interval of
[0.612, 6.430] percentage points, while satisfying the frozen matched-write-
rate tolerance.

The tested native pointer-based identity additions did not establish
incremental POR@30 benefit. B3-S exceeded B2 by only 0.791 percentage points
at the frozen thresholds, with BCa 95% CI [-0.204, 2.088] percentage points.
Moreover, this primary pair realized a 3.270 percentage-point write-rate
difference, exceeding the frozen 2-percentage-point tolerance, so it cannot be
reported as a matched-rate primary effect.

B3-R produced no POR@30 improvement over B3-S at matched TEST write rate:
0.000 percentage-point difference with BCa 95% CI [-1.129, 1.431].
On the Representative TEST cohort, B2, B3-S, and B3-R all obtained the same
POR@30 value of 0.7237.

The evidence therefore supports a bounded conclusion: under this frozen SAM 3
pointer-based formulation, learned quality-plus-temporal information was
useful relative to the manual gate, whereas the tested self and relational
identity-pointer additions did not establish incremental post-occlusion
recovery benefit.

## 6.2 Contributions

This thesis contributes a reproducible experimental protocol for studying
memory-write admission in a frozen SAM 3 video tracker rather than relying
only on end-to-end leaderboard changes.

The main contributions are:

1. a physically verified closed-loop memory-write intervention in frozen
   SAM 3 VOS/PVS;
2. a nested signal-family comparison spanning manual reliability rules,
   learned quality-plus-temporal gating, self-identity, and tracked-competitor
   identity;
3. a leakage-controlled missing-identity routing protocol that treats pointer
   validity as availability rather than as a predictive feature;
4. a matched-write-rate evaluation protocol with exact per-video neutral
   budget controls and supported write-rate curves;
5. a dual drift/theft outcome framework with POR@30 as the sole primary
   endpoint and event-level ITR@30 as a secondary endpoint;
6. video-clustered BCa inference with a frozen 50,000-replicate implementation;
7. a pre-registered negative-result path showing that the tested native SAM 3
   pointer identity signals did not establish the hypothesized incremental
   POR benefit.

## 6.3 Limitations

The primary B3-S versus B2 TEST contrast missed the frozen write-rate
tolerance, preventing a matched-rate primary-effect interpretation.

The representative cohort contained 40 videos and 76 primary events and was
used as an external-validity anchor rather than as a replacement primary
cohort.

B5 was a prospectively frozen DMS-lite write-side comparator inspired by the
SAM3-DMS reliability signal; it was not an exact reproduction of the external
SAM3-DMS system.

Identity evidence used the frozen SAM 3 native object pointer space. A null
incremental result for these signals does not imply that stronger learned
identity representations, alternative reference construction, or other
memory architectures would also fail.

The study intentionally performed a single frozen TEST campaign. No subgroup
search, threshold retuning, or additional TEST experiment was performed after
observing final outcomes.

## 6.4 Future Work

Future work should evaluate stronger identity representations that are
explicitly optimized for identity discrimination rather than relying on the
native pointer geometry alone. Promising directions include learned temporal
identity references, alternative trusted-memory architectures, and adaptive
rate control that preserves a predeclared memory budget.

Replication on additional densely annotated multi-object VOS datasets would
test whether the observed quality-versus-identity pattern generalizes beyond
MOSEv2. Cross-model replication could determine whether the findings are
specific to the SAM 3 VOS memory interface or reflect a broader property of
foundation-model video trackers.

Future studies may also investigate whether identity information becomes more
useful when memory readout and write admission are optimized jointly. Such
extensions should be treated as new prospective experiments rather than as
post-hoc modifications of the present frozen TEST result.
