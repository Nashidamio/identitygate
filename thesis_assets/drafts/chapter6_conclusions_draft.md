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
