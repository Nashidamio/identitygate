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
