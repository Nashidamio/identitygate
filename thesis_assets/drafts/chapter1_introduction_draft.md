# Chapter 1 - Introduction

## 1.1 Background

Video object segmentation (VOS) requires a tracker to preserve object identity
and segmentation quality across long sequences containing occlusion,
appearance change, clutter, and visually similar objects. Modern
memory-based trackers address this problem by retaining information from
earlier frames and reusing it during subsequent prediction.

Memory is useful only when the information written into it is useful. A
prediction that is inaccurate, stale, or associated with the wrong object can
become part of the state used for future tracking. This makes memory management
a sequential decision problem: the tracker must not only predict the current
object mask, but also decide whether the current prediction should influence
future frames.

This thesis studies that decision in a frozen SAM 3 VOS/PVS tracker. The focus
is deliberately narrow. SAM 3 itself is not fine-tuned, modified with LoRA, or
replaced by a new segmentation architecture. Instead, the work isolates the
memory-write interface and asks what information should govern admission of a
candidate prediction into memory.

[[CITE:SAM3]]

## 1.2 Motivation

The motivating failure mode was observed before the learned gate was built.
Instrumentation of the SAM 3 VOS path showed that the memory-encoding path is
executed throughout propagation, including frames in which undesirable
predictions can occur. Earlier experiments also demonstrated predictions for
objects that were absent in the ground truth, establishing that imperfect
tracker state can reach the memory-processing path.

This observation does not imply that SAM 3 contains no memory-management
mechanisms. Confidence handling, tracking logic, retention policy, and other
mitigations may operate elsewhere in the system. The narrower issue studied
here is whether an explicit admission decision at the write interface can use
observable signals to decide which candidate writes should be allowed to
affect future tracking.

Recent work on VOS and SAM-family tracking has explored quality-aware memory,
motion and temporal consistency, distractor-aware memory management,
trajectory validation, and object-wise memory selection. These systems
motivate memory management as an important interface, but they do not by
themselves answer the controlled explanatory question studied here:
when the segmentation model is fixed, which information family contributes
useful evidence for deciding whether to write?

[[CITE:QDMN;DAM4SAM;SENTRY;SAM3-DMS;RethinkingMemory]]

## 1.3 Research Question

The research question is locked as:

> What information should a memory-write gate use — quality signals, temporal
> signals, or identity signals?

The thesis therefore treats memory admission primarily as an explanatory
comparison rather than as a claim that a new gate must outperform all existing
methods.

The experimental framework is referred to as SIGMA: a signal-informed
memory-write gating framework used to implement and compare the candidate
information families under a frozen tracker.

## 1.4 Experimental Principle

A central difficulty in evaluating memory-write policies is that a method can
change performance simply by changing how often it writes. A policy that
writes less frequently is not automatically better at selecting frames, and a
policy that writes more frequently may receive additional opportunities to
adapt.

For this reason, the study uses a pre-specified write-budget protocol.
Operating points are selected on development data, supported write-rate curves
are retained, and final TEST comparisons are interpreted using a frozen
absolute write-rate tolerance. A comparison that violates this tolerance is
reported as a rate mismatch rather than being retrospectively treated as a
matched-rate effect.

This rule became consequential in the final experiment. B3-S obtained a
numerically higher POR@30 value than B2 on HARD_TEST80, but their realized
write-rate difference exceeded the frozen tolerance. The study therefore does
not interpret that numerical ordering as evidence of a matched-rate
self-identity effect.

## 1.5 Signal-Family Ladder

The final experimental ladder is:

- B0: native frozen SAM 3 reference;
- B1: manually specified quality-and-temporal admission rule;
- B2: learned quality-plus-temporal gate;
- B3-S: B2 augmented with native self-identity information;
- B3-R: relational extension adding tracked-competitor identity information;
- B5: prospectively frozen DMS-lite write-side comparator.

The ladder is intentionally nested. The central scientific comparisons ask
whether learning improves on the manual rule and whether native identity
information adds useful evidence beyond the learned quality-plus-temporal
formulation.

Identity availability is handled through explicit routing. Pointer validity is
an availability condition rather than a predictive input. Missing identity
therefore does not silently become an additional signal.

## 1.6 Evaluation Design

The study uses MOSEv2 because its densely annotated training partition permits
frame-level event construction and post-occlusion analysis. The provided
validation partition is unsuitable for the thesis endpoints because it does
not provide the required dense per-frame masks. Consequently, leakage-
controlled development and held-out cohorts are constructed from the densely
annotated training partition at video level.

[[CITE:MOSEv2]]

The sole primary endpoint is POR@30: successful post-occlusion recovery,
defined as reaching target IoU greater than 0.5 within the first 30 evaluable
ground-truth-visible frames after a qualifying reappearance. ITR@30 is a
secondary endpoint concerned with identity-theft behaviour.

Final uncertainty is quantified with paired video-clustered BCa 95% confidence
intervals. Video, rather than frame or event, is the statistical cluster.
The frozen final implementation uses 50,000 bootstrap replicates and seed 52.

The final evaluation comprises HARD_TEST80 with 80 videos and 506 primary
events and REPRESENTATIVE_TEST40 with 40 videos and 76 primary events.
TEST is touched once after the final freeze, with no post-TEST threshold
retuning.

## 1.7 Summary of Findings

The final evidence gives a bounded answer to the research question.

The clearest positive matched-rate result was obtained for learned
quality-plus-temporal admission. On HARD_TEST80, B2 exceeded the manual B1
rule by 2.964 percentage points in POR@30, with paired video-clustered BCa
95% CI [0.612, 6.430] percentage points.

The tested native pointer-based identity additions did not establish
incremental POR@30 benefit. B3-S exceeded B2 numerically by 0.791 percentage
points, but the comparison violated the frozen write-rate tolerance and
therefore cannot support the intended matched-rate primary interpretation.
B3-R produced a 0.000 percentage-point POR@30 difference relative to B3-S at
matched TEST write rate, with BCa 95% CI [-1.129, 1.431]. On the
Representative TEST cohort, B2, B3-S, and B3-R each obtained POR@30=0.7237.

The appropriate conclusion is not that identity information is generally
useless. Rather, under the tested frozen SAM 3 pointer representation,
reference construction, routing policy, and evaluation protocol, native self
and tracked-competitor identity signals did not establish additional
post-occlusion recovery benefit beyond the learned quality-plus-temporal gate.

## 1.8 Contributions

This thesis contributes:

1. a controlled formulation of memory-write admission in frozen SAM 3 around
   the question of what information a write gate should use;

2. a physically verified closed-loop admit/block intervention in SAM 3
   VOS/PVS without fine-tuning or architectural modification;

3. a nested comparison of manual quality/temporal rules, learned
   quality-plus-temporal gating, self-identity, and tracked-competitor
   relational identity;

4. leakage-controlled missing-identity routing that treats pointer validity
   as availability rather than as a predictive feature;

5. a controlled write-budget protocol with development-only operating-point
   selection, supported write-rate curves, exact per-video neutral controls,
   explicit TEST rate-mismatch handling, and no TEST retuning;

6. a dual drift/theft outcome framework with POR@30 as the sole primary
   endpoint, ITR@30 as a secondary endpoint, and paired video-clustered BCa
   inference; and

7. an empirical result showing positive matched-rate evidence for learned
   quality-plus-temporal gating over the manual rule, while the tested native
   self and relational identity additions did not establish incremental
   POR@30 benefit.

## 1.9 Scope and Claim Boundaries

The thesis does not claim that SIGMA is a universally superior memory policy.
It does not claim that identity information cannot improve VOS memory, and it
does not claim that B5 is an exact implementation of SAM3-DMS.

The empirical claims are restricted to the frozen SAM 3 VOS/PVS substrate,
the pre-specified features and routing rules, the MOSEv2-derived evaluation
cohorts, and the frozen operating-point and statistical protocol.

The title therefore reflects the explanatory nature of the work:

**When Should SAM 3 Write to Memory? A Controlled Study of Signal-Informed
Gating for Video Object Segmentation**

## 1.10 Thesis Organization

Chapter 2 positions the study within memory representation, selective memory
management, and identity-aware video tracking. Chapter 3 defines the dataset,
cohort construction, signal families, labels, endpoints, statistical protocol,
and TEST firewall. Chapter 4 describes the physical SAM 3 intervention and the
B1/B2/B3-S/B3-R/B5 implementations. Chapter 5 reports the frozen TEST results
and uncertainty. Chapter 6 interprets the findings, states their limitations,
and identifies prospective follow-up questions without modifying the frozen
TEST conclusions.
