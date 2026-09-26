<!-- GENERATED THESIS NARRATIVE ASSEMBLY. -->
<!-- Source chapter files remain authoritative for editing. -->

<!-- SOURCE: thesis_assets/drafts/final_frontmatter_pack.md -->

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

<!-- SOURCE: thesis_assets/drafts/chapter1_introduction_draft.md -->

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

---

<!-- SOURCE: thesis_assets/drafts/chapter2_related_work_draft.md -->

# Chapter 2 - Related Work and Thesis Positioning

## 2.1 Scope of This Review

This chapter positions the thesis within three connected lines of work in
video object segmentation (VOS): memory representation and retrieval,
selective or reliability-oriented memory management, and identity-aware
tracking. These lines are related but answer different questions.

Memory-representation methods ask how historical information should be stored
and retrieved. Memory-management methods ask which observations should be
retained, trusted, or reused. Identity-aware methods ask how object-specific
information should be preserved when multiple similar or competing objects are
present.

The thesis lies at the intersection of the latter two questions. Its focus is
not a new segmentation backbone, a new detector, or an end-to-end retrained
tracker. Instead, it studies a controlled memory-write decision in a frozen
SAM 3 VOS/PVS system and asks which information family should govern write
admission.

The literature is therefore reviewed according to the information and
decision role that each line contributes, rather than as a chronological list
of leaderboard methods.

[FIGURE 2.X HERE: Memory Management and Identity Reasoning in Video Object
Segmentation]

## 2.2 Memory Representation and Retrieval in VOS

Memory-based VOS predates foundation-model tracking. Earlier systems such as
STM, STCN, MiVOS, XMem, and later object-aware approaches established the
general principle that information from previous frames can be stored and
retrieved to support future segmentation.

[[CITE:STM]]
[[CITE:STCN]]
[[CITE:MiVOS]]
[[CITE:XMem]]
[[CITE:Cutie]]

Across this lineage, the central design problem evolved from simply retaining
past frame-mask information toward more selective retrieval, longer-term
memory organization, and representations that better preserve object-specific
information. This progression is important for the present thesis because it
establishes memory as an active component of the tracker state rather than a
passive log of previous predictions.

However, improvements in memory representation do not by themselves determine
which current predictions should be allowed to become future memory. A tracker
may possess a strong memory encoder and retrieval mechanism while still facing
the separate decision of whether a particular candidate observation is useful
or safe to write.

This distinction motivates the thesis focus on memory-write admission.

## 2.3 Selective and Reliability-Oriented Memory Management

A second line of work addresses the risk that unreliable observations can
enter or influence memory.

### 2.3.1 Learned quality-aware admission before foundation-model tracking

QDMN is an important predecessor because it demonstrates learned
quality-oriented memory control in pre-foundation VOS.

[[CITE:QDMN]]

Its relevance to this thesis is conceptual: memory updates need not be governed
only by fixed temporal schedules. Learned signals can instead estimate whether
a candidate observation is suitable for memory.

The present study differs in scope. It evaluates a frozen SAM 3 tracker,
separates quality-plus-temporal information from native pointer-identity
information, and uses a controlled write-budget protocol to interpret their
incremental effects.

### 2.3.2 Motion and temporal heuristics in SAM-family tracking

SAMURAI represents a training-free approach in which motion information is
used to improve memory selection and tracking stability.

[[CITE:SAMURAI]]

This establishes motion and temporal consistency as plausible evidence for
memory management. It also motivates the thesis B1/B2 family, where temporal
consistency is evaluated together with quality-related signals.

The thesis does not attempt to reproduce SAMURAI. Its question is narrower:
whether learned quality-plus-temporal information improves on a manually
specified reliability rule and whether tested native identity signals add
information beyond that learned formulation.

### 2.3.3 Distractor-aware and structured memory

DAM4SAM provides a close conceptual precedent for treating memory reliability
and distractor handling as explicit design problems rather than relying only
on the base tracker state.

[[CITE:DAM4SAM]]

This work is important because it weakens any broad claim that SAM-family
trackers lack memory-management mechanisms. The defensible distinction for the
present thesis is therefore not that memory governance is absent from prior
work, but that the thesis performs a controlled signal-family comparison at
the write-admission interface while keeping the underlying SAM 3 tracker
fixed.

### 2.3.4 Trajectory and cycle-based validation

SENTRY provides training-free trajectory/cycle-style validation before
accepting information into memory.

[[CITE:SENTRY]]

Its importance to the thesis is twofold. First, it supports the premise that
a current prediction should not automatically be considered a trustworthy
future memory. Second, it provides a strong contemporary example of
write-oriented validation without requiring the present thesis to claim that
memory filtering itself is novel.

The methodological distinction is that SENTRY uses hand-crafted
trajectory/geometric validation, whereas this thesis compares learned
quality-plus-temporal admission with native-pointer identity extensions under
a frozen experimental protocol.

### 2.3.5 Reliability-based memory selection in SAM 3

SAM3-DMS is particularly relevant because it uses per-object confidence and
reliability information for memory selection in a SAM 3 setting.

[[CITE:SAM3-DMS]]

The thesis therefore includes B5, a prospectively frozen DMS-lite write-side
comparator derived from the reliability signal used by SAM3-DMS. B5 is not an
exact reproduction of the external SAM3-DMS system. The intervention semantics
differ: the thesis applies the reliability score to its already-frozen
physical write-admission mechanism so that it can be evaluated within the same
write-budget protocol as the other thesis conditions.

This distinction is necessary for a fair interpretation. B5 is a
thesis-internal comparator inspired by an external reliability signal, not a
claim to reproduce official SAM3-DMS results.

### 2.3.6 Memory-policy studies in SAM-generation trackers

Rethinking Memory Design in SAM-Based Visual Object Tracking studies memory
architecture and memory-policy choices across SAM-generation trackers and
provides evidence that explicit memory design remains consequential even when
the underlying segmentation model is strong.

[[CITE:RethinkingMemory]]

This line is close to the thesis because it treats memory policy as an
independent design axis. The present work uses that insight differently:
rather than comparing bundled memory architectures as end-to-end systems, it
fixes the SAM 3 substrate and evaluates the information available to a
write-admission decision.

OAMVOS and other recent reliability- or occlusion-aware systems further
reinforce the broader point that temporal validation, occlusion handling, and
selective memory are active research problems.

[[CITE:OAMVOS]]

## 2.4 Identity-Aware Tracking and Competitor Reasoning

Memory reliability is not only a question of mask quality. In multi-object
scenes, a plausible mask can correspond to the wrong object. This motivates a
separate identity-oriented line of work.

Earlier identity-aware VOS and tracking systems such as AOT, DeAOT, and AOST
demonstrate the value of object-specific identity representations and
identity-aware propagation.

[[CITE:AOT]]
[[CITE:DeAOT]]
[[CITE:AOST]]

More recent systems extend this idea through explicit re-identification,
semantic identity modelling, or competitor-aware reasoning.

[[CITE:ReMeDI-SAM3]]
[[CITE:SurgSLOT]]
[[CITE:CMR]]
[[CITE:VOS-Agent]]

These systems motivate the scientific plausibility of the thesis identity
hypothesis. If memory contamination can arise when a target becomes confused
with another tracked object, then an identity-sensitive write decision could
in principle reject a frame that appears locally plausible but is associated
with the wrong object.

The thesis tests this proposition using the native SAM 3 object-pointer space.
B3-S augments the learned quality-plus-temporal gate with self-anchor pointer
similarity. B3-R adds a relational signal based on the maximum similarity to
anchors of other currently tracked objects.

The term tracked competitor is intentional. B3-R does not implement a generic
external distractor detector. Its relational evidence is restricted to objects
that are already present in the tracked set.

## 2.5 Why Identity at Write Time Remains an Open Controlled Question

Identity-aware tracking and competitor-relative memory reasoning do not
automatically imply that identity is useful for write admission.

A representation may be useful during memory readout or correspondence yet add
little information to a write gate once quality and temporal evidence are
already available. Conversely, identity may be valuable only under particular
occlusion or multi-object conditions.

This distinction motivates the nested experimental ladder:

- B1: manual quality-plus-temporal rule;
- B2: learned quality-plus-temporal gate;
- B3-S: B2 plus native self-identity;
- B3-R: B3-S plus tracked-competitor relational identity.

The ladder is designed to ask incremental questions rather than compare
unrelated end-to-end systems.

B2 versus B1 asks whether learned admission adds value over a manual
reliability rule.

B3-S versus B2 asks whether native self-identity adds information beyond the
learned quality-plus-temporal formulation.

B3-R versus B3-S asks whether tracked-competitor relational identity adds
information beyond self identity.

These comparisons are meaningful only if differences in write frequency are
controlled or explicitly acknowledged. This requirement leads directly to the
pre-specified write-budget protocol described in Chapter 3.

## 2.6 Closest Methodological Neighbours

The closest related systems illuminate different parts of the thesis problem,
but none should be presented as an identical experimental design.

QDMN establishes learned quality-aware memory admission in earlier VOS, but
does not answer the frozen-SAM-3 quality-versus-identity question.

SAMURAI contributes motion-aware selection but uses a training-free heuristic
rather than the thesis supervised signal-family comparison.

DAM4SAM establishes distractor-aware and structured memory as an important
design direction, but the thesis does not reproduce its memory architecture.

SENTRY validates candidate memory using training-free trajectory/cycle
evidence rather than learned native-pointer identity features.

SAM3-DMS supplies a contemporary SAM 3 reliability-based memory-selection
reference; the thesis B5 comparator adopts only a prospectively frozen
reliability signal within the thesis write-side intervention.

Rethinking Memory Design demonstrates that memory policy itself remains an
important axis in SAM-generation tracking, while the thesis focuses on the
controlled information content of the write decision.

CMR motivates competitor-relative evidence but is primarily relevant as a
neighbour in relational identity reasoning rather than as the same
write-admission mechanism.

The literature therefore supports the importance of all three candidate
families—quality, temporal consistency, and identity—without deciding their
incremental value under the controlled conditions used in this thesis.

## 2.7 Thesis Positioning

The thesis is best positioned as a controlled explanatory study of
memory-write information, not as a claim that memory filtering is new or that
a named gate universally outperforms prior systems.

Its scientific contribution is the experimental isolation of the write
decision under a fixed SAM 3 substrate.

The study holds the segmentation model constant, uses a physically verified
closed-loop write intervention, defines nested signal families, selects
operating points without using TEST outcomes, and applies an explicit
rate-mismatch rule before interpreting direct TEST comparisons.

This framing is especially important given the final results. Learned
quality-plus-temporal admission produced positive matched-rate evidence over
the manual B1 rule, while the tested native self and relational identity
extensions did not establish incremental POR@30 benefit. The result therefore
does not invalidate the motivation for identity-aware tracking in general. It
answers a narrower question about the tested native SAM 3 pointer signals at
the write interface.

The contribution is consequently both empirical and methodological: it
provides evidence about which tested signals added value and a protocol that
prevents differences in memory-write frequency from being mistaken for
differences in signal quality.

## 2.8 Claim Boundaries

The following claims are intentionally avoided:

- that no prior work filters or manages SAM memory;
- that this thesis introduces the first memory gate;
- that identity information is generally ineffective;
- that B5 reproduces SAM3-DMS;
- that all external systems operate at the same memory-write interface;
- that a null incremental identity result invalidates identity-aware tracking.

The supported positioning is narrower:

> The thesis performs a controlled comparison of learned quality-plus-temporal
> information and tested native-pointer identity information for memory-write
> admission in a frozen SAM 3 VOS/PVS tracker, under a pre-specified
> write-budget and video-clustered evaluation protocol.

## 2.9 Chapter Summary

Prior VOS research establishes the importance of memory representation,
selective memory management, temporal validation, distractor handling, and
identity-aware reasoning. These developments motivate the thesis signal
families but do not determine their incremental value at a fixed memory-write
interface.

Chapter 3 therefore converts this literature gap into an explicit experimental
protocol: a frozen SAM 3 substrate, leakage-controlled MOSEv2 cohorts,
pre-specified signal families and endpoints, controlled operating-point
selection, and video-clustered statistical inference.

---

<!-- SOURCE: thesis_assets/drafts/chapter3_experimental_methodology_draft.md -->

# Chapter 3 - Experimental Methodology

## 3.1 Study Design

This thesis uses a controlled intervention design to study one specific
decision inside a frozen video object segmentation tracker: whether a
non-conditioning prediction should be allowed to become future memory.

The research question is:

> What information should a memory-write gate use — quality signals, temporal
> signals, or identity signals?

The design therefore separates three issues that are often conflated in
end-to-end tracker comparisons:

1. the quality of the underlying segmentation model;
2. how frequently memory is updated; and
3. which information is used to decide whether a candidate write is admitted.

SAM 3 is held fixed throughout the final study. The experimental intervention
acts only on memory-write admission. Signal families are introduced through a
nested ladder so that each comparison asks an incremental information question
rather than comparing unrelated tracker architectures.

[FIGURE 3.X HERE: Experimental Pipeline for Controlled Memory-Write Evaluation]

## 3.2 Frozen SAM 3 Substrate

The final substrate is SAM 3 VOS/PVS, instantiated using
`build_sam3_video_model()` and the frozen `facebook/sam3/sam3.pt` checkpoint.

The SAM 3 model is not fine-tuned. No gradient, LoRA module, detector
modification, or additional SAM architecture is introduced. The gate is a
small external decision module whose parameter count remains well below the
50,000-parameter thesis limit.

The frozen repository revision used for the SAM 3 source is:

`8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`

SAM 3.1 Multiplex is not the final thesis substrate. Earlier hardware
evaluation showed that the Multiplex path required approximately 21.77 GB of
VRAM and therefore did not fit the available 16 GB RTX 4080 SUPER. The VOS/PVS
path both fit the available hardware and exposed the per-object signals needed
for the locked research question.

The experimental interpretation is consequently limited to the frozen SAM 3
VOS/PVS memory interface.

## 3.3 Dataset Choice

The study uses MOSEv2.

The densely annotated training partition contains 3,666 videos with per-frame
object masks and is used as the source population for development and held-out
evaluation cohort construction.

The provided MOSEv2 validation partition is not used for the thesis endpoints
because it does not contain the dense per-frame masks required to determine
visibility gaps, recovery events, target IoU trajectories, or relational theft
outcomes.

This is not a conventional use of the official train/validation split.
Instead, the thesis constructs leakage-controlled video-level development and
TEST cohorts from the densely annotated training partition.

[[CITE:MOSEv2]]

## 3.4 Recovery-Event Definition

A recovery event is defined from ground-truth visibility.

For an object to generate a qualifying event:

1. the object is visible;
2. it becomes absent for at least five consecutive frames; and
3. it subsequently becomes visible again.

The reappearance frame is the first frame after the qualifying gap with a
non-empty ground-truth target mask.

Post-reappearance evaluation uses only frames in which the target is
ground-truth visible.

A full scan of the 3,666-video MOSEv2 training partition produced:

- 1,691 videos containing at least one qualifying recovery event;
- 3,237 event-bearing tracks; and
- 4,469 raw recovery events.

These quantities describe the pre-exclusion event census and are not
themselves the final primary evaluation population.

## 3.5 Whole-Scene Event Control

Early analysis identified groups of objects disappearing and reappearing at
similar times. Timing synchronization was initially suspected to indicate
whole-scene occlusion, but visual audit showed that synchronization alone was
not a valid semantic definition. Some cases instead reflected camera motion,
out-of-view motion, or other local phenomena.

Consequently, synchronized timing is not used as the final whole-scene label.

Final whole-scene status is based on the frozen whole-scene labelling
procedure recorded in the experimental history. Whole-scene-labelled events
are retained in the corpus as a tagged control/sensitivity stratum but are
excluded from the primary POR population.

This distinction is important:

- whole-scene handling occurs at the event level; and
- development exposure is handled separately at the video level.

Whole-scene events are therefore not silently deleted from the dataset.

## 3.6 Development-Exposure Firewall

Any video that had previously been used for model development, exploratory
tracking analysis, gate training, or development evaluation was excluded from
the final eligible TEST population.

This rule uses video identity, not individual events, as the exposure
boundary. If a video was development-exposed, no event from that video could
later enter the held-out final TEST cohorts.

After excluding final whole-scene primary events and development-exposed
videos, the final primary eligible population contained:

- 2,701 primary events;
- 1,170 videos.

This population is the source from which the final hard and representative
evaluation cohorts were constructed.

[FIGURE 5.X HERE: Leakage-Controlled Construction of Development and Test
Cohorts]

## 3.7 Outcome-Independent Difficulty Index

The primary held-out cohort was deliberately enriched for difficult recovery
conditions without using tracker recovery outcomes.

Difficulty was therefore defined using observable, outcome-independent
covariates only.

The frozen DI-v1 contains five components:

1. frame-0 competitor identity pressure;
2. occlusion-gap duration;
3. pre-gap target size;
4. scene crowding; and
5. reappearance displacement.

Each component is converted to a within-population percentile rank. The final
DI-v1 score is the equal-weight arithmetic mean of the five ranks.

For the target-size component, smaller pre-gap targets are ranked as harder.

For identity pressure, only frame-0 anchor descriptors are used. No future
tracking outcome, recovery result, POR value, or TEST prediction enters the
difficulty score.

The difficulty index is therefore a dataset-construction variable rather than
an outcome variable.

## 3.8 Visual-Diversity Constraint

Difficulty ranking alone can concentrate visually similar videos. To reduce
this risk, a separate outcome-independent visual-diversity constraint is
applied during hard-pool construction.

One deterministic middle frame is selected from each of the 1,170 eligible
videos. A frozen DINOv2 ViT-S/14 backbone produces one 384-dimensional visual
embedding per video.

The embeddings are used only for diversity control.

They are:

- not gate features;
- not tracking outcomes;
- not used for threshold selection;
- not trained on the thesis data; and
- not exposed to Fresh DEV or TEST outcomes.

The frozen clustering procedure uses k=20 visual clusters, and the hard-set
construction restricts any one visual cluster to at most 15% of the selected
hard pool.

The visual embedding therefore affects cohort diversity, not gate inference.

## 3.9 Final Cohort Construction

The frozen DI-v1 ranking and visual-cluster constraint define a 120-video hard
pool containing 739 primary events.

This hard pool is partitioned at video level into:

### Fresh DEV

- 40 videos;
- 233 primary events.

Fresh DEV is used for threshold and operating-point selection only.

### Hard TEST

- 80 videos;
- 506 primary events.

Hard TEST is the primary held-out evaluation cohort.

### Representative TEST

A second held-out cohort is sampled uniformly from the primary eligible
population after excluding the entire 120-video hard pool.

It contains:

- 40 videos;
- 76 primary events.

Representative TEST is an external-validity anchor. It is not used to replace
the primary Hard TEST cohort and is not used for model or threshold selection.

## 3.10 Experimental Ladder

The final ladder contains six conditions.

### B0 — Native reference

B0 is the frozen SAM 3 tracker at its native physical write behaviour.

B0 is not forced to the common write budget and therefore is not treated as a
matched-rate reference.

### B1 — Manual quality-plus-temporal rule

B1 uses the prospectively frozen manual reliability rule defined before final
evaluation.

Its inputs correspond to quality, area consistency, and temporal consistency
signals. It serves as a non-learned reference for the same broad information
family used by B2.

### B2 — Learned quality-plus-temporal gate

B2 is the final learned five-feature gate.

Its frozen predictive inputs are:

- `mask_conf_iou_head`;
- `occ_score_logit`;
- `area_norm`;
- `area_ratio_anchor`;
- `temporal_iou_prev`.

These inputs represent prediction quality, presence confidence, mask geometry,
and temporal consistency.

### B3-S — Self-identity extension

B3-S inherits all five B2 features and adds:

- `ptr_sim_anchor_fp32`.

This feature is the FP32 cosine similarity between the current native SAM 3
object pointer and the trusted self anchor.

B3-S therefore asks whether native self-identity information adds useful
evidence beyond the learned B2 formulation.

### B3-R — Relational identity extension

B3-R inherits B3-S and adds:

- `max_comp_anchor_cos_fp32`.

This is the maximum FP32 cosine similarity between the current target pointer
and trusted anchors of other currently tracked objects.

B3-R therefore asks whether tracked-competitor relational identity adds
information beyond self identity.

The signal refers specifically to tracked competitors. It is not a generic
external distractor detector.

### B5 — DMS-lite write-side comparator

B5 is a prospectively frozen write-side comparator inspired by the reliability
signal used by SAM3-DMS.

It is not an exact reproduction of SAM3-DMS.

For object i, the final B5 score uses only the frozen SAM 3 mask-confidence
and object-presence outputs.

Presence is:

`0`, when `occ_score_logit <= 0`;

otherwise:

`2 * sigmoid(occ_score_logit) - 1`.

The object score is:

`presence * mask_conf_iou_head`.

No identity signal or learned gate output is used by B5.

## 3.11 Gate Training Labels

The learned gates predict two distinct failure types.

### Drift

Drift is defined from target overlap only:

`target_iou < 0.3`.

This definition is intentionally identity-free.

### Theft

Theft is relational and is defined using overlap with other tracked
ground-truth objects:

`max_other_iou > 0.5`.

The two labels are kept separate because they represent different failure
semantics.

Gray-zone examples that do not satisfy the frozen positive/negative training
definitions are excluded from gate training rather than being assigned an
arbitrary label.

The final learned models therefore use dual failure-typed heads: one for drift
risk and one for theft risk.

## 3.12 Learned Gate Architecture

For B2, B3-S, and B3-R, each failure type is predicted by an independent MLP
head with:

- input dimension d determined by the variant;
- Linear(d,64);
- ReLU;
- Linear(64,32);
- ReLU;
- Linear(32,1).

The two learned heads output drift and theft unsafe logits independently.

A shared TRAIN-only feature normalizer is used where specified by the frozen
training artifacts; the neural heads themselves are independent rather than a
single shared neural backbone.

[FIGURE 4.X HERE: Signal Composition and Memory-Write Decision Architecture]

## 3.13 Dual-Risk Safety Composition

For each object, the two unsafe probabilities are converted to safe
probabilities:

`p_safe_drift = 1 - p_unsafe_drift`

`p_safe_theft = 1 - p_unsafe_theft`

The final object admission score is:

`p_safe = min(p_safe_drift, p_safe_theft)`.

This conservative composition allows either predicted failure mode to veto a
write.

No post-hoc weighting coefficient or learned calibration layer is added after
the frozen gate definition.

## 3.14 Missingness and Identity Routing

Base-feature and identity missingness are handled differently.

### Non-finite B2 base features

If any required B2 base feature is non-finite, the object fails closed:

`object admission score = 0`.

The object remains part of the frame-level aggregation and can therefore block
the frame write.

### Pointer availability

The availability mask is:

`pointer_valid = object_score_logit > 0`.

`pointer_valid` is never supplied as a predictive input.

It is used only to determine which frozen gate variant is valid for the current
object.

The final routing is:

### B3-S

- self identity available -> use B3-S;
- self identity unavailable -> fall back to B2.

### B3-R

- relational identity available -> use B3-R;
- self identity only -> fall back to B3-S;
- self identity unavailable -> fall back to B2.

For a single-object frame, B3-R therefore reduces to B3-S when self identity is
available.

Unexpected non-finite identity values in a case where identity is expected to
be computable are treated as engineering defects rather than silently routed
around.

## 3.15 Identity Computation

All identity cosine computation is performed in FP32.

For the locked identity path:

- autocast is disabled;
- TF32 is disabled; and
- float32 matrix multiplication precision is set to `highest`.

The native SAM 3 object pointer is used directly. No learned external identity
embedding is introduced in the final thesis experiment.

## 3.16 Frame-Level Admission and Physical Intervention

The gate produces an object-level score for every tracked object.

The final frame score is:

`frame_score = minimum object score over all tracked objects`.

At the frozen operating threshold tau:

`ADMIT` if `frame_score >= tau`;

otherwise:

`BLOCK`.

The intervention applies to eligible non-conditioning frames.

A blocked frame is physically removed from the SAM 3 memory-write path using
the verified closed-loop intervention. Conditioning and prompt frames are
retained.

The intervention therefore changes future tracker state rather than merely
relabeling an offline prediction.

This distinction is central to the thesis: the gate is evaluated as a
closed-loop memory-write policy, not as an offline classifier.

## 3.17 Development Operating-Point Selection

The gated methods require a frozen operating threshold.

Threshold selection is performed using Fresh DEV only.

The target development write rate is:

`r* = 0.30`.

Thresholds are selected using realized write rate, not POR, ITR, or any other
tracking-performance outcome.

The final headline thresholds are:

- B1: 0.10;
- B2: 0.10;
- B3-S: 0.20;
- B3-R: 0.20;
- B5: 0.70.

B0 remains at its native write rate.

Supported development rate-sweep operating points are retained for
write-rate/performance curves rather than interpolating unsupported TEST
operating points after seeing outcomes.

## 3.18 Neutral Matched-Budget Controls

For each signal-based gated condition, the evaluation also includes a neutral
control with the same per-video write budget.

The neutral control admits exactly the same number of eligible writes per
video but does not use the signal policy to choose which frames are retained.

This separates two possible mechanisms:

1. benefit from writing less often; and
2. benefit from selecting which frames to write.

Neutral controls are therefore diagnostic controls for selection quality, not
alternative primary methods.

## 3.19 Frozen TEST Rate-Mismatch Rule

Development threshold matching does not guarantee identical realized write
rates on TEST.

The prospectively frozen direct-comparison tolerance is:

`absolute pooled TEST write-rate difference <= 0.02`.

If a direct comparison exceeds this tolerance, it is marked:

`RATE_MISMATCH`.

A rate-mismatched contrast may still be reported descriptively, including its
POR point estimate and confidence interval, but it is not interpreted as a
matched-rate signal-family effect.

No TEST threshold search or threshold retuning is permitted to repair a
mismatch.

This rule protects the scientific contrast from being changed after observing
TEST performance.

## 3.20 Primary Endpoint: POR@30

POR@30 is the sole primary endpoint.

For each qualifying reappearance event, the tracker is evaluated over the
first 30 evaluable ground-truth-visible frames after reappearance, subject to
the frozen overlapping-event truncation rule.

An event is counted as recovered if:

`target_iou > 0.5`

at least once within that evaluation interval.

POR@30 is:

`number of recovered qualifying events / number of qualifying events`.

The endpoint directly targets the thesis failure mode: whether the tracker
recovers the correct target after a qualifying absence.

## 3.21 Secondary Endpoint: ITR@30

ITR@30 is a secondary endpoint.

For each qualifying reappearance event, the same chronological evaluation
interval used by POR@30 is examined for sustained identity theft.

A theft episode requires the frozen relational theft condition to persist for
at least five consecutive evaluable frames.

The event-level theft indicator is one if at least one such qualifying theft
episode occurs within the event interval.

ITR@30 is:

`number of qualifying events with a theft episode / number of qualifying
events evaluated`.

The denominator includes qualifying events even when no competitor overlap is
observed.

ITR@30 is therefore not promoted to co-primary status.

## 3.22 Statistical Unit and Point Estimator

The video is the statistical cluster.

Frames and recovery events from the same video are not treated as independent
experimental units.

For each method, the headline POR@30 point estimate is the pooled event-level
recovery proportion over the frozen cohort.

For a paired method comparison, the observed statistic is the difference in
pooled POR@30 between the two methods.

The clustering affects uncertainty estimation, not the definition of the
headline pooled event proportion.

## 3.23 Paired Video-Clustered BCa Inference

Primary comparative uncertainty is quantified using paired
video-clustered BCa 95% confidence intervals.

The frozen implementation uses:

- 50,000 bootstrap replicates;
- seed 52;
- paired resampling of videos;
- midrank bias-correction z0; and
- delete-one-video jackknife acceleration.

The same sampled video multiplicities are applied to both methods in a paired
contrast.

The minimum practically important Hard TEST POR benefit is prospectively set
to:

`+0.08`.

This practical threshold is interpreted separately from statistical
uncertainty.

## 3.24 Primary and Secondary Comparisons

The single primary research-question contrast is:

`B3-S - B2`

for POR@30 on HARD_TEST80.

This comparison asks whether native self-identity information provides
incremental value beyond the learned quality-plus-temporal gate.

Other comparisons are secondary, including:

- B2 - B1;
- B3-R - B3-S;
- B3-R - B2;
- B5 - B2; and
- practical-reference comparisons with B0.

Representative TEST is reported as an external-validity evaluation rather than
a replacement primary analysis.

## 3.25 One-Touch TEST Firewall

The final TEST campaign is performed once after the final methodological
freeze.

After TEST outcomes are observed:

- no gate is retrained;
- no feature is added or removed;
- no threshold is changed;
- no split is modified;
- no endpoint is promoted or demoted;
- no TEST operating point is interpolated;
- no subgroup is selected to rescue a result; and
- no TEST rerun is performed.

Negative, null, harmful, positive, and inconclusive outcomes are all accepted
under the same frozen protocol.

This firewall is essential to the confirmatory interpretation of the final
results.

## 3.26 Reproducibility Controls

The final experimental protocol records:

- frozen model and source revisions;
- exact signal definitions;
- training and evaluation configurations;
- model hashes;
- output hashes;
- cohort membership hashes;
- deterministic seeds;
- experiment registry entries;
- research-state updates; and
- prospective amendments documenting methodological supersessions.

Experimental artifacts are retained in the repository rather than replacing
historical records when a method is superseded.

The repository therefore records not only the final successful path but also
the methodological decisions required to reach it without silently rewriting
history.

## 3.27 Chapter Summary

This chapter defined the controlled experimental conditions used to answer the
research question.

The design fixes the SAM 3 tracker, constructs leakage-controlled development
and TEST cohorts, introduces nested quality/temporal and identity signal
families, physically intervenes on memory writes, controls write-rate
differences, and uses video-clustered inference under a one-touch TEST
firewall.

Chapter 4 describes how these frozen methodological rules were implemented in
the SAM 3 VOS/PVS code path and how the B1, B2, B3-S, B3-R, and B5 conditions
were executed reproducibly.

---

<!-- SOURCE: thesis_assets/drafts/chapter4_implementation_reproducibility_draft.md -->

# Chapter 4 - Implementation and Reproducibility

## 4.1 Implementation Objective

The implementation goal was not to modify SAM 3 into a new segmentation
architecture. It was to create a controlled mechanism that could observe the
signals available during tracking and physically decide whether an eligible
non-conditioning frame was allowed to remain in the memory state used by
future propagation.

This distinction governed the engineering design.

SAM 3 remained frozen. The trainable components were limited to small
memory-admission heads operating on scalar tracker signals and native object
pointers. The intervention was evaluated in closed loop, so a write decision
could alter subsequent tracker state rather than merely classify an already
completed trajectory.

The implementation was developed in stages. Each stage first established an
engineering fact required by the research question, then froze the resulting
interface before later evaluation.

## 4.2 Execution Environment

All final thesis experiments were executed on the laboratory workstation with:

- NVIDIA RTX 4080 SUPER, 16 GB VRAM;
- 64 GB system RAM;
- WSL2 Ubuntu 22.04.5;
- Python 3.12.13;
- PyTorch 2.10.0+cu128; and
- CUDA 12.8.

The final SAM 3 source revision was:

`8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`.

The final substrate was constructed using:

`build_sam3_video_model()`.

SAM 3 parameters remained frozen throughout the gate experiments.

The reproducibility environment also pins additional dependencies required by
the SAM 3 and thesis code, including `setuptools<82`, `einops`,
`pycocotools`, and `scipy`. The complete environment is recorded in
`requirements.lock.txt`.

## 4.3 Substrate Selection Under the 16 GB Constraint

An early implementation question was whether the thesis should use the SAM 3
Object Multiplex path or the VOS/PVS tracking path.

The decision was made empirically rather than from the implementation plan.

The Object Multiplex configuration required approximately 21.77 GB of peak
VRAM on the available hardware. Shortening the input clip did not materially
remove this cost, indicating that the configuration did not fit reliably
inside the 16 GB GPU budget.

The non-Multiplex VOS/PVS path used substantially less memory and provided the
tracking interface required by the thesis.

The final thesis therefore uses SAM 3 VOS/PVS.

This is an engineering constraint with scientific consequences: conclusions
apply to the tested VOS/PVS memory interface and should not be presented as
results for the unexecuted Multiplex architecture.

## 4.4 Verification of the Native Memory Path

Before implementing a gate, the thesis first verified that the tracking path
actually exposed a memory-write process that could be controlled.

Instrumentation of the SAM 3 VOS path showed that the memory encoder was
invoked on every propagated frame in the mechanism probes, including a
97-frame run with 97 observed memory-encoding calls.

This observation motivated a write-interface intervention, but it is not
interpreted as evidence that SAM 3 has no other memory-management machinery.
The thesis claim is narrower: the tested VOS path exposed a per-frame
memory-processing opportunity at which an additional explicit admission
decision could be applied.

The observed VOS memory representation included per-object memory features,
while propagation also exposed ordered object identifiers, masks, object
scores, and object pointers required by the final signal definitions.

## 4.5 Native Object-Pointer Verification

The identity part of the research question required a target-specific
representation exposed by the actual VOS tracker.

A development-exposed signal probe verified that SAM 3 VOS exposes a
per-object native object pointer:

`obj_ptr in R^256`.

The pointer was available alongside per-object prediction outputs during
propagation.

This verification was a go/no-go condition for the identity comparison. The
final experiment does not introduce a learned external identity encoder.
Instead, B3-S and B3-R use FP32 cosine relationships derived from the native
SAM 3 object-pointer space.

Identity cosine operations are performed with:

- autocast disabled;
- TF32 disabled; and
- float32 matrix multiplication precision set to `highest`.

This isolates the identity comparison from mixed-precision differences in the
cosine calculation.

## 4.6 Physical Closed-Loop Write Intervention

The final gate is not an offline filter applied after tracking.

For every eligible non-conditioning frame, the implementation computes
per-object admission scores and then a frame-level score. If the frozen
policy blocks the frame, the current non-conditioning frame is physically
evicted from the SAM 3 memory state according to the verified intervention.

Conditioning and prompt frames are retained.

The decision therefore changes the state available to future frames.

The physical action is:

`ADMIT` if `frame_score >= tau`;

otherwise:

`BLOCK`.

A block means no memory write for that eligible non-conditioning frame. It
does not mean that the object or current segmentation output is discarded
from evaluation.

This distinction is important when interpreting the thesis figures: the gate
controls future memory state, not whether the tracker is allowed to produce a
current prediction.

## 4.7 Per-Object to Per-Frame Aggregation

SAM 3 can track multiple objects simultaneously, while the implemented
intervention blocks or retains the current frame-level memory update.

Consequently, object scores are conservatively aggregated as:

`frame_score = minimum object admission score over all tracked objects`.

A low safety score for any tracked object can therefore veto the whole-frame
write.

This aggregation rule is frozen across the learned gates and the B5
write-side comparator.

## 4.8 B1 Manual Quality-and-Temporal Baseline

B1 provides a non-learned reference for the quality-and-temporal signal
family.

The frozen manual rule uses:

- `mask_conf_iou_head`;
- `occ_score_logit`;
- `area_ratio_anchor`; and
- `temporal_iou_prev`.

The rule constructs three reliability terms.

Confidence reliability is:

`q_conf = mask_conf_iou_head * sigmoid(occ_score_logit)`.

Area reliability is based on the ratio between the current object area and
the frame-0 anchor area:

`q_area = min(r, 1/r)`,

where `r` is the current-to-anchor area ratio.

Temporal reliability is:

`q_temp = temporal_iou_prev`.

The B1 object score is the minimum of the frozen reliability terms.

B1 introduces no learned parameters. Its purpose is to determine whether
learning provides value beyond a transparent hand-designed rule using the
same broad quality-and-temporal information family.

## 4.9 B2 Learned Quality-Plus-Temporal Gate

B2 is the final learned quality-plus-temporal condition.

The final B2 input vector contains exactly five predictive features:

1. `mask_conf_iou_head`;
2. `occ_score_logit`;
3. `area_norm`;
4. `area_ratio_anchor`; and
5. `temporal_iou_prev`.

Historical plans contained additional candidate features. They are not part of
the final B2 implementation.

The final five-feature set was frozen because it was already implemented,
causally defined, and independent of unresolved clean-reference state.

The frozen B2 weights are stored in:

`experiments/EXP029_b2core_train/model.json`.

Model SHA256:

`6160a6be9da2058808c16182abe03443014806273df46fe59a86411ffc869ecf`.

The historical artifact name contains `b2core`, but the final feature freeze
promotes this exact model to final B2 rather than retraining it under a renamed
artifact.

## 4.10 B2 Training Architecture

B2 uses two independent failure-typed MLP heads, one for drift and one for
theft.

Each head has:

- input dimension 5;
- `Linear(5,64)`;
- ReLU;
- `Linear(64,32)`;
- ReLU;
- `Linear(32,1)`.

Each head therefore produces one unsafe logit.

The expected parameter count is:

- 2,497 parameters per head;
- 4,994 learned parameters in total.

This is far below the thesis limit of 50,000 trainable gate parameters.

Training was deterministic full-batch PyTorch CPU float64 using independently
weighted binary cross-entropy losses for the two failure types.

The optimizer was AdamW with:

- learning rate `0.001`;
- weight decay `0.0001`;
- betas `(0.9, 0.999)`;
- epsilon `1e-8`;
- 1,000 epochs; and
- no early stopping or DEV tuning.

The B2 normalizer is a TRAIN-only z-score transform fitted on finite B2
training rows. A zero standard deviation is replaced by 1.0.

No DEV or TEST examples are used to train the B2 weights or normalizer.

## 4.11 B3-S Self-Identity Extension

B3-S inherits the complete B2 feature vector and adds:

`ptr_sim_anchor_fp32`.

The resulting input dimension is six.

`ptr_sim_anchor_fp32` is the FP32 cosine similarity between the current native
SAM 3 object pointer and the trusted target anchor.

The architecture of each B3-S failure head remains:

`6 -> 64 -> 32 -> 1`

with ReLU hidden activations.

Expected learned parameter counts are:

- 2,561 parameters per head;
- 5,122 parameters total.

The B3-S comparison therefore changes the information available to the gate
without changing the underlying SAM 3 tracker or introducing a large model.

## 4.12 B3-R Relational Identity Extension

B3-R extends B3-S with:

`max_comp_anchor_cos_fp32`.

Its final input vector therefore contains seven features.

The relational feature is the maximum FP32 cosine similarity between the
current target pointer and the trusted anchors of other currently tracked
objects.

The signal is competitor-relative but restricted to the tracked object set.
It is not a generic external distractor detector.

Each B3-R head uses:

`7 -> 64 -> 32 -> 1`.

Expected learned parameter counts are:

- 2,625 parameters per head;
- 5,250 parameters total.

The frozen B3-S and B3-R weights are stored in:

`experiments/EXP031_b3_train/model.json`.

Model SHA256:

`235076b86aa0975b3ec624aae703575fd4f030381b6af4008c3808f2db71847a`.

## 4.13 Independent Drift and Theft Heads

The neural architecture does not use a shared learned backbone between the
drift and theft predictors.

For each variant, the drift and theft predictors are independent MLP heads.
The training configuration may share the relevant TRAIN-only normalization
procedure, but the neural weights are separate.

The two heads produce:

`p_unsafe_drift`

and:

`p_unsafe_theft`.

Safe probabilities are defined as:

`p_safe_drift = 1 - p_unsafe_drift`

and:

`p_safe_theft = 1 - p_unsafe_theft`.

The final object score is:

`p_safe = min(p_safe_drift, p_safe_theft)`.

Thus either predicted failure type can veto the object-level write score.

No additional post-hoc weighting coefficient, learned fusion layer, or TEST
calibration stage is introduced.

[FIGURE 4.X HERE: Signal Composition and Memory-Write Decision Architecture]

## 4.14 Base-Feature Missingness

Missingness is not silently imputed.

For B2, B3-S, and B3-R, if any required B2 base feature is non-finite, the
object fails closed:

`object admission score = 0`.

The object remains in the frame-level minimum aggregation. It can therefore
block the whole-frame write.

This rule prevents a difficult object from disappearing from the decision
simply because one of its required inputs is missing.

## 4.15 Identity Availability and Hierarchical Routing

Identity availability is handled separately from base-feature missingness.

The availability mask is:

`pointer_valid = object_score_logit > 0`.

`pointer_valid` is not a predictive feature.

It determines which identity-augmented model can be evaluated.

For B3-S:

- valid self identity -> B3-S;
- unavailable self identity -> B2.

For B3-R:

- relational identity available -> B3-R;
- self identity available but relational identity unavailable -> B3-S;
- self identity unavailable -> B2.

For single-object frames, B3-R therefore routes to B3-S when self identity is
available.

This routing rule is prospective and outcome-independent. It does not use
future GT, recovery outcomes, or TEST performance to decide which model is
applied.

An unexpected non-finite identity value in a case where identity is expected
to be computable is treated as an engineering defect rather than silently
rerouted.

## 4.16 B5 DMS-Lite Write-Side Comparator

B5 provides an external-reliability-inspired comparator while preserving the
same physical write intervention used by the thesis gates.

It is explicitly not an exact reproduction of SAM3-DMS.

For tracked object i, B5 uses:

- `mask_conf_iou_head_i`; and
- `occ_score_logit_i`.

If either scalar is non-finite, the object fails closed with score zero.

Otherwise:

`presence_i = 0`

when:

`occ_score_logit_i <= 0`.

For positive object-presence logits:

`presence_i = 2 * sigmoid(occ_score_logit_i) - 1`.

The object reliability score is:

`b5_object_score_i = presence_i * mask_conf_iou_head_i`.

The frame score is the minimum B5 object score across tracked objects, and the
same frozen thresholded admit/block intervention is applied.

The implementation therefore transfers a prospectively frozen reliability
signal into the thesis write-side protocol. It does not claim to reproduce the
read-side memory-selection semantics or reported performance of the external
SAM3-DMS system.

## 4.17 Closed-Loop Evaluator

The final evaluator executes the frozen tracking conditions over the selected
cohorts and records both tracking outcomes and memory-write accounting.

For each gated operating point it records, among other quantities:

- the frozen threshold;
- eligible write opportunities;
- admit count;
- block count;
- realized write rate; and
- trajectory/event outcomes used by POR@30 and ITR@30.

Accounting assertions require:

`admit_count + block_count = eligible_count`

and verify that the stored write rate agrees with:

`admit_count / eligible_count`.

The evaluator does not perform TEST threshold search.

The final TEST artifacts explicitly record:

`threshold_search_performed = False`

and:

`threshold_retuning_performed = False`.

## 4.18 Neutral Exact-Budget Controls

The implementation creates a neutral counterpart for each signal-based
condition.

For each video, the neutral control receives the exact number of admitted
writes used by its paired signal condition.

The identity of admitted frames is selected independently of the signal gate,
while the count is held fixed.

This gives the neutral control the same write budget without reproducing the
signal-based frame selection.

The implementation verifies that signal and neutral admit counts match before
the paired diagnostic is accepted.

## 4.19 Mechanism Sanity Before Final Evaluation

Closed-loop operation was verified before final DEV and TEST evaluation.

A unified mechanism sanity experiment executed B1, B2, B3-S, and B3-R
sequentially on a development-exposed TRAIN video.

The purpose was engineering verification, not a performance claim.

The sanity run checked that:

- the frozen SAM 3 substrate executed;
- each policy produced write decisions;
- the physical block mechanism changed the memory-write state;
- multi-object aggregation remained valid; and
- the variants could execute within the available 16 GB GPU.

Fresh DEV and TEST were not touched by this mechanism sanity run.

Qualitative frames retained from the earlier closed-loop sanity work are used
only as mechanism illustrations in the thesis. They are not presented as TEST
performance evidence.

## 4.20 Compute Feasibility

The final study was designed for the available 16 GB GPU rather than assuming
unlimited accelerator memory.

The Multiplex path was rejected after observed peak memory exceeded the GPU
budget.

The frozen VOS/PVS path and gate variants fit the available hardware.

Because the gates contain only a few thousand trainable parameters, their
parameter memory is negligible relative to frozen SAM 3. The dominant compute
cost remains the underlying tracker and closed-loop video propagation.

Final TEST reporting records peak VRAM and runtime per condition rather than
assuming equal computational cost.

## 4.21 Separation of Training and Tracking Precision

The learned gate heads were trained deterministically on CPU in float64 from
cached scalar features.

At deployment, the frozen SAM 3 tracker retains its native execution
precision, while identity cosine calculations follow the separately frozen
FP32 path.

This separation ensures that the identity comparison is not affected by
autocast or TF32 differences while avoiding unnecessary modification of the
SAM 3 inference path.

## 4.22 Reproducibility Discipline

Every scientific experiment is tied to a frozen repository state.

The workflow follows several rules:

1. implementation and configuration are committed before a scientific run;
2. the working tree is clean before execution;
3. large output artifacts are hashed;
4. experiment status is recorded in `EXPERIMENT_REGISTRY.md`;
5. current project state is recorded in `RESEARCH_STATE.md`;
6. methodological changes are appended as new amendments rather than silently
   rewriting previous decisions; and
7. superseded artifacts remain available for provenance.

This is especially important because the project contains several legitimate
methodological corrections, including the transition from the historical
SAM 3.1 Multiplex expectation to SAM 3 VOS/PVS, the replacement of the early
EXP017 final-split interpretation, and the later correction of synchronized
timing as a whole-scene proxy.

Historical records are therefore retained but are not treated as current
methodology when a later frozen amendment supersedes them.

## 4.23 Frozen Model and Artifact Provenance

The final learned models are identified by both path and hash.

B2:

`experiments/EXP029_b2core_train/model.json`

SHA256:

`6160a6be9da2058808c16182abe03443014806273df46fe59a86411ffc869ecf`

B3-S / B3-R:

`experiments/EXP031_b3_train/model.json`

SHA256:

`235076b86aa0975b3ec624aae703575fd4f030381b6af4008c3808f2db71847a`

The final cohort membership, operating points, TEST outputs, statistical
tables, and thesis-facing assets are likewise retained as repository
artifacts.

This allows the final report to identify not merely an algorithm description
but the exact frozen implementation that generated the reported evidence.

## 4.24 Implemented, Executed, and Superseded Scope

Several ideas appeared in historical planning documents but are not part of
the final experiment.

The final thesis does not use:

- SAM 3.1 Multiplex as the core substrate;
- a learned external identity embedding;
- the deferred phi identity model;
- rolling-clean or EMA identity references as final B3 features;
- centroid displacement as a B2 feature;
- frames-since-clean-write as a B2 feature;
- an oracle clean-state substitute;
- a single undifferentiated B3 condition; or
- TEST-driven threshold repair.

The final implemented and executed core is:

- frozen SAM 3 VOS/PVS;
- B0;
- B1;
- B2;
- B3-S;
- B3-R;
- B5;
- hierarchical missing-identity routing;
- physical closed-loop write blocking;
- development-selected frozen thresholds;
- neutral exact-budget controls; and
- one-touch final TEST evaluation.

This distinction between historical plans and executed methodology is retained
throughout the thesis.

## 4.25 Chapter Summary

The implementation converts the thesis question into a physical intervention
inside frozen SAM 3 tracking.

Native tracker outputs provide the final quality, geometry, temporal, and
pointer-identity signals. Small independent drift/theft MLP heads convert
those signals into conservative object safety scores. Missing identity is
handled by prospective hierarchical routing, object scores are aggregated by
a frame-level minimum, and the resulting threshold decision physically admits
or blocks the current non-conditioning memory write.

The implementation remains small relative to SAM 3, reproducible through
frozen model and artifact hashes, and feasible on the available 16 GB
workstation.

Chapter 5 evaluates the resulting policies under the pre-specified TEST
protocol and reports both their tracking outcomes and the write-rate
constraints that determine which comparisons are scientifically
interpretable.

---

<!-- SOURCE: thesis_assets/drafts/chapter5_results_draft.md -->

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

<!-- SOURCE: thesis_assets/drafts/chapter6_conclusions_draft.md -->

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

<!-- SOURCE: thesis_assets/drafts/APPENDIX_REPRODUCIBILITY_PACK.md -->

# Appendix — Reproducibility and Frozen Protocol

Status: DERIVED REPORTING ARTIFACT. The repository experimental record remains canonical.

## A. Frozen substrate and environment

- Substrate: SAM 3 VOS/PVS.
- Builder: `build_sam3_video_model()`.
- SAM source commit: `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`.
- SAM frozen: no gradient, LoRA, fine-tuning, detector modification, or new SAM architecture.
- Gate parameter budget: <=50K parameters.
- Hardware: NVIDIA RTX 4080 SUPER, 16 GB VRAM.
- OS: WSL2 Ubuntu 22.04.5.
- Python: 3.12.13.
- PyTorch: 2.10.0+cu128.
- CUDA: 12.8.

### Dependency lock

`requirements.lock.txt` SHA256: `c943caf335003594dc93d9042267695e70c0e955652257caae8a3ff11a0efa6a`

## B. Dataset and evaluation cohorts

- MOSEv2 train videos available: 3,666.
- MOSEv2 provided valid split is unusable for this evaluation because it lacks the required dense per-frame annotation.
- HARD_TEST80: 80 videos, 506 primary events.
- REPRESENTATIVE_TEST40: 40 videos, 76 primary events.
- Video is the statistical cluster.

## C. Frozen research question

> What information should a memory-write gate use — quality signals, temporal signals, or identity signals?

## D. Method ladder

- B0 — native ungated write reference.
- B1 — manual quality/temporal rule.
- B2 — learned quality + temporal gate.
- B3-S — B2 plus native self-identity pointer signal.
- B3-R — relational extension using tracked-competitor anchor similarity.
- B5 — DMS-lite write-side comparator.

## E. Final headline operating thresholds

| Method | tau |
|---|---:|
| B1 | 0.10 |
| B2 | 0.10 |
| B3-S | 0.20 |
| B3-R | 0.20 |
| B5 | 0.70 |

- Development target write rate: `r*=0.30`.
- B0 remains at its native physical write rate and is not a matched-budget reference.
- Frozen TEST matched-rate tolerance: absolute pooled write-rate difference <=0.02.

## F. Identity computation and routing

- Identity cosine is computed in FP32 with autocast off, TF32 off, and highest float32 matmul precision.
- `pointer_valid = object_score_logit > 0`.
- `pointer_valid` is an availability/routing condition, not a predictive feature.
- B3-S: valid self pointer -> B3-S; otherwise -> B2.
- B3-R: relational identity available -> B3-R; self only -> B3-S; no valid self pointer -> B2.

## G. Endpoints

- Sole primary endpoint: POR@30 — recovery IoU >0.5 within the first 30 evaluable GT-visible frames after reappearance.
- Qualifying disappearance gap: >=5 frames.
- ITR@30: secondary endpoint.
- Drift label: `target_iou < 0.3`.
- Theft label: `max_other_iou > 0.5`.
- Gray-zone examples are excluded from gate training.

## H. Statistical inference

- Paired video-clustered BCa 95% confidence intervals.
- Bootstrap replicates: 50,000.
- Seed: 52.
- Point estimator: pooled event difference.
- BCa bias correction: midrank z0.
- Acceleration: delete-one-video jackknife.
- Minimum practically important POR benefit: +0.08.

## I. Final HARD_TEST80 headline results

| Method | POR@30 | ITR@30 | Write rate | Peak VRAM (GB) | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| B0 | 0.7411 | 0.0593 | 1.0000 | 15.81 | 2572.7 |
| B1 | 0.7273 | 0.0514 | 0.3199 | 13.11 | 1852.5 |
| B2 | 0.7569 | 0.0316 | 0.3233 | 13.06 | 1878.9 |
| B3-S | 0.7648 | 0.0395 | 0.2906 | 13.06 | 1878.1 |
| B3-R | 0.7648 | 0.0336 | 0.2923 | 13.06 | 1888.7 |
| B5 | 0.7530 | 0.0336 | 0.3080 | 13.03 | 1845.7 |

## J. Representative TEST descriptive results

| Method | POR@30 | ITR@30 | Write rate |
|---|---:|---:|---:|
| B0 | 0.5789 | 0.0000 | 1.0000 |
| B1 | 0.6711 | 0.0132 | 0.3375 |
| B2 | 0.7237 | 0.0132 | 0.3739 |
| B3-S | 0.7237 | 0.0132 | 0.3581 |
| B3-R | 0.7237 | 0.0000 | 0.3652 |
| B5 | 0.6974 | 0.0132 | 0.3977 |

## K. Frozen primary inference

- Primary contrast: B3-S minus B2 on HARD_TEST80.
- Observed POR@30 difference: `0.007905`.
- BCa 95% CI: `[-0.002037, 0.020882]`.
- B3-S write rate: `0.290565`.
- B2 write rate: `0.323269`.
- Absolute rate difference: `0.032703`.
- Frozen status: `RATE_MISMATCH_NO_MATCHED_RATE_PRIMARY_INTERPRETATION`.

## L. TEST firewall

- TEST prediction-derived evidence was touched exactly once after final freeze.
- No TEST threshold retuning is permitted.
- No TEST rerun is permitted.
- Unsupported development write-rate targets are not interpolated post hoc on TEST.

## M. Canonical reproducibility artifacts

- `RESEARCH_STATE.md` — SHA256 `4c32d14bd42ca2f933efa77868a14e3ff8f0aacde42e2e31f088c9f123ce2831`
- `EXPERIMENT_REGISTRY.md` — SHA256 `df7a908cac695c97fef2295c682f0bd9bd81de6283881b7c389375e63457c046`
- `requirements.lock.txt` — SHA256 `c943caf335003594dc93d9042267695e70c0e955652257caae8a3ff11a0efa6a`
- `PREREGISTRATION.md` — SHA256 `8032b6de18287496524021621db302f4637f29d3e98073cab27c57275f37154e`
- `thesis_assets/ASSET_MANIFEST.csv` — canonical thesis-asset hash manifest.

## N. Repository freeze state

- Repository HEAD at appendix generation: `499620aba2b7ae6d0326f82f15d44d02216d51e1`.
- Branch: `main`.
- Final thesis-facing assets are stored under `thesis_assets/`.

---

