# Amendment A11 - BCa Implementation Freeze

Date: 2026-09-24

Status: FROZEN WHEN THE COMMIT INTRODUCING THIS FILE IS COMMITTED TO main

## 1. Scope and timing

A10 already froze the sole primary endpoint, primary contrast, video cluster,
95 percent BCa inference requirement, and +0.08 practical threshold.

Fresh-DEV point outcomes from EXP051 have now been observed. No EXP052
bootstrap distribution, jackknife acceleration, BCa interval, or CI-based
interpretation has been computed before this amendment.

This amendment only fixes the operational BCa implementation. It does not
change any model, feature, threshold, r_star, split, event definition,
endpoint, contrast, or practical threshold.

## 2. Point estimator

For each run label, POR30 is the pooled event-level recovery proportion over
the frozen 233 Fresh DEV primary events.

The primary statistic is:

    POR30(B3-S) - POR30(B2)

The video remains the statistical cluster for uncertainty.

## 3. Paired video-cluster bootstrap

- Number of Fresh DEV video clusters: 40.
- Bootstrap replicates: 50,000.
- Deterministic seed: 52, fixed from the EXP052 identifier before any BCa
  result is computed.
- Each replicate samples 40 videos with replacement.
- When a video is sampled, all paired event rows from that video are retained.
- A video sampled multiple times contributes its entire event cluster multiple
  times.
- The same sampled video indices are used across methods within a replicate.

## 4. BCa interval

- Confidence level: 95 percent.
- Bias correction z0 uses the midrank bootstrap position relative to the
  observed statistic, including half weight for ties, with finite-sample
  clipping to avoid infinite normal quantiles.
- Acceleration uses the delete-one-video jackknife over the 40 video clusters.
- The adjusted lower and upper probabilities are applied to the empirical
  bootstrap distribution.
- No percentile-only interval may substitute for this BCa interval.

EXP024 remains valid for its historical TRAIN-only percentile cluster
bootstrap purpose, but its percentile quantiles do not implement the A10 BCa
requirement and are not reused as the EXP052 CI calculation.

## 5. Secondary and diagnostic contrasts

A10 secondary contrasts are computed under the same paired video-clustered
BCa procedure.

For B5, all core gated-method comparisons are reported to avoid selective
post-outcome choice.

Gated-versus-B0 practical-reference contrasts are retained as A10 secondary
comparisons.

Matched signal-versus-neutral pairs are reported as diagnostics under the same
clustered BCa machinery. They do not replace the primary contrast.

ITR30 is secondary and uses the same video-clustered BCa machinery.

## 6. Outcome firewall

No p-value target, CI target, bootstrap seed search, replicate-count search,
subgroup selection, threshold change, model change, or split change is
permitted in response to EXP052 results.

TEST remains untouched.
