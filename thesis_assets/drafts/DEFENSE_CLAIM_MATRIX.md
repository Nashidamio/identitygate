# Thesis Defense Claim Matrix

## 1. What is the thesis actually asking?

Answer:
The thesis asks what information a memory-write gate should use: quality,
temporal, or identity information.

Defense boundary:
It is an explanatory controlled comparison, not a claim that a named gate is
universally superior.

Evidence:
Chapters 1, 2, and 3.

## 2. Why keep SAM 3 frozen?

Answer:
Freezing SAM 3 isolates the memory-write intervention. If the backbone were
fine-tuned simultaneously, downstream changes could not be attributed cleanly
to write selection.

Evidence:
Chapters 1, 3, and 4.

## 3. Why not use SAM 3.1 Multiplex?

Answer:
The final substrate decision was evidence-based. Multiplex required about
21.77 GB peak VRAM and did not fit the available 16 GB GPU reliably. The
VOS/PVS path fit and exposed the required per-object signals.

Boundary:
The final conclusions apply to SAM 3 VOS/PVS, not the unexecuted Multiplex
path.

Evidence:
Chapters 3 and 4.

## 4. Why is MOSEv2 valid not used?

Answer:
The official valid partition does not contain the dense per-frame masks needed
for the frozen recovery and theft endpoints. Leakage-controlled development
and TEST cohorts were therefore constructed from the densely annotated
training partition at video level.

Evidence:
Chapter 3.

## 5. Was the hard cohort selected because the tracker failed on it?

Answer:
No. Final difficulty selection used frozen outcome-independent covariates and
a visual-diversity constraint. Individual videos were not selected from TEST
tracking failures.

Evidence:
Chapter 3 and Figure 3.2.

## 6. Was the gate actually closed loop?

Answer:
Yes. Blocking changes the physical memory state available to subsequent
frames. It is not merely an offline classifier over a completed trajectory.

Evidence:
Chapters 1 and 4; Figure 1.1.

## 7. Why control write rate?

Answer:
A gate can change performance simply by writing more or less frequently.
Therefore signal-family effects cannot be interpreted cleanly without
controlling or explicitly checking realized write-rate differences.

Evidence:
Chapters 1, 3, and 5.

## 8. Why is B3-S versus B2 not claimed as a positive identity result?

Answer:
B3-S had a numerically higher Hard TEST POR@30, but the realized write-rate
difference was 0.0327, above the frozen 0.02 tolerance. The pre-specified rule
therefore marks the primary contrast as `RATE_MISMATCH` and forbids a
matched-rate identity-effect interpretation.

Evidence:
Chapters 1, 3, 5, and 6.

## 9. Did anything improve?

Answer:
Yes. B2 exceeded the manual B1 rule by 2.964 percentage points in Hard TEST
POR@30 at matched TEST write rate, with paired video-clustered BCa 95% CI
[0.612, 6.430] percentage points.

Evidence:
Chapters 1, 5, and 6.

## 10. Does the thesis prove identity information is useless?

Answer:
No. It shows that the tested native SAM 3 self-pointer and
tracked-competitor pointer additions did not establish incremental POR@30
benefit beyond the learned quality-plus-temporal gate under the frozen
protocol.

Evidence:
Chapters 1, 2, 5, and 6.

## 11. What does B3-R add scientifically?

Answer:
B3-R provides a clean incremental test of competitor-relative native-pointer
information beyond self identity. Its Hard TEST POR@30 difference relative to
B3-S was 0.000 at matched write rate.

Evidence:
Chapters 3, 5, and 6.

## 12. Why video-clustered inference?

Answer:
Multiple events from the same video are dependent. Treating events as
independent would understate uncertainty, especially in videos containing
many related disappearance and reappearance events.

Evidence:
Chapter 3.

## 13. Is B5 an external-method reproduction?

Answer:
No. B5 is a prospectively frozen DMS-lite write-side comparator inspired by
the SAM3-DMS reliability signal. It deliberately uses the thesis physical
write intervention and is not presented as an exact reproduction of official
SAM3-DMS.

Evidence:
Chapters 2, 3, 4, and 6.

## 14. What is the strongest methodological contribution?

Answer:
The protocol makes interpretation conditional on a prospectively frozen write
budget and accepts `RATE_MISMATCH` rather than retuning TEST thresholds after
seeing outcomes. The final primary result demonstrates why that safeguard
matters.

Evidence:
Chapters 1, 3, 5, and 6.

## 15. What is the main empirical conclusion?

Answer:
Within frozen SAM 3 VOS/PVS, learned quality-plus-temporal admission received
positive evidence relative to the manual rule. The tested native self and
relational pointer-identity additions did not establish additional
post-occlusion recovery benefit.

Evidence:
Chapters 1, 5, and 6.

## 16. What is the most important limitation?

Answer:
The primary B3-S versus B2 Hard TEST comparison missed the frozen write-rate
tolerance, so it cannot support the intended matched-rate self-identity claim.
The tested identity representation is also limited to the native SAM 3 pointer
space.

Evidence:
Chapter 6.

## 17. What is the most defensible future-work hypothesis?

Answer:
A frame can be locally informative without necessarily being a useful future
memory write. Direct causal write-utility experiments could test this
prospectively.

Boundary:
This is an interpretation and future-work hypothesis, not a demonstrated
mechanism in the present thesis.

Evidence:
Chapter 6.
