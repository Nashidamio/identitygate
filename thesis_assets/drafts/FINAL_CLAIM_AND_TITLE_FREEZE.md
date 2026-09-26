# Final Claim and Title Freeze Candidate

Status: FINAL WORDING CANDIDATE - FREEZE WHEN COMMITTED

## Recommended title

When Should SAM 3 Write to Memory? A Controlled Study of Signal-Informed Gating for Video Object Segmentation

## Role of SIGMA

SIGMA remains the name of the experimental memory-write gating framework
inside the thesis. It is not used as the leading promise of the thesis title,
because the main contribution is the controlled empirical answer rather than a
claim of a universally improved algorithm.

## Central thesis claim

For frozen SAM 3 memory-write admission, the clearest positive evidence
favored learned quality-plus-temporal gating over a manual rule, while the
tested native pointer-identity additions did not establish incremental
post-occlusion recovery benefit.

## Direct answer to the research question

The experiments provide positive evidence for learned quality-plus-temporal
admission relative to the manual quality/temporal rule. They do not establish
that adding the tested native self-identity or tracked-competitor identity
signals provides additional POR@30 benefit.

This conclusion is specific to the tested frozen SAM 3 VOS/PVS substrate,
feature definitions, routing rules, operating points, dataset cohorts, and
evaluation protocol.

## Primary methodological claim

Memory-write policies must be interpreted under controlled write budgets.
Raw performance differences can otherwise be confounded by how often each
policy writes.

The frozen protocol demonstrated this directly: B3-S had numerically higher
POR@30 than B2 on HARD_TEST80, but their realized write-rate difference
exceeded the prospectively frozen tolerance. Therefore that primary result is:

RATE_MISMATCH_NO_MATCHED_RATE_PRIMARY_INTERPRETATION

The protocol prevented a superficially favorable identity result from being
overclaimed.

## Supported result claims

1. On HARD_TEST80, B2 exceeded B1 by 0.02964 POR@30 at matched TEST write
   rate, with paired video-clustered BCa 95% CI [0.00612, 0.06430].

2. B3-S minus B2 had a POR@30 point difference of 0.00791 with BCa 95% CI
   [-0.00204, 0.02088], but the realized write-rate difference was 0.03270,
   exceeding the frozen 0.02 tolerance. It is not a matched-rate primary
   effect.

3. B3-R minus B3-S produced a POR@30 difference of 0.00000 at matched TEST
   write rate, with BCa 95% CI [-0.01129, 0.01431].

4. On REPRESENTATIVE_TEST40, B2, B3-S, and B3-R each obtained POR@30=0.7237.

## Negative-result interpretation

The tested native SAM 3 pointer-identity signals did not establish incremental
post-occlusion recovery benefit beyond the learned quality-plus-temporal gate.

Do not generalize this to:
- identity information is useless;
- identity-aware memory gating cannot work;
- signal-informed gating does not work;
- no memory-write gate can improve SAM 3.

## Conceptual interpretation

The findings challenge the assumption that adding a semantically
identity-oriented local signal necessarily improves downstream memory utility.

They motivate, but do not directly establish, the distinction:

informative frame != useful memory write

Direct causal write-utility analysis was not part of the frozen final
inference and is therefore future work rather than a thesis conclusion.

## Final contribution framing

1. A controlled comparison of information families for SAM 3 memory-write
   admission rather than an algorithm-only leaderboard claim.

2. A physically verified closed-loop memory-write intervention in frozen
   SAM 3 without fine-tuning, LoRA, detector modification, or SAM architecture
   changes.

3. A budget-controlled evaluation protocol with development-only operating
   point selection, explicit TEST rate-mismatch handling, exact per-video
   neutral controls, and supported write-rate sweeps.

4. POR@30 as the sole primary endpoint with paired video-clustered BCa
   inference using video as the statistical cluster.

5. Empirical evidence that learned quality-plus-temporal gating improved over
   the manual rule under the matched Hard TEST comparison, while the tested
   native self-identity and relational identity additions did not establish
   incremental POR@30 benefit.

6. A reusable interpretation boundary showing why raw performance ordering
   and matched-rate causal interpretation must not be conflated.

## Defense one-sentence version

The thesis does not claim that identity cannot help memory; it shows that,
under a frozen and controlled SAM 3 experiment, learned quality-plus-temporal
gating improved on the manual rule, whereas the tested native pointer-identity
signals did not demonstrate additional recovery benefit, and the matched-rate
protocol prevented us from turning a superficially positive raw result into an
unsupported claim.
