# Amendment A13 - Final TEST Inference Freeze

Date: 2026-09-26

Status: FROZEN WHEN THIS FILE, EXP054 SCRIPT, AND CONFIG ARE COMMITTED TO main

## Scope

EXP054 performs final HARD_TEST80 paired video-clustered BCa inference.
It does not rerun SAM 3 or change predictions, gates, thresholds, splits,
endpoints, contrasts, the +0.08 practical threshold, seed, or bootstrap count.

## Frozen statistical kernel

The EXP052/A11 numerical implementation is reused unchanged for:
- pooled event-rate differences;
- paired video-cluster resampling;
- 50,000 bootstrap replicates;
- seed 52;
- BCa 95 percent intervals;
- midrank tie handling;
- delete-one-video jackknife acceleration.

EXP054 is derived only after verifying the EXP052 script/config/A11 files are
unchanged from commit 9426e8754507700d6c8ad991940ef81b29b17b01.

## TEST input

Source: experiments/EXP053_final_test/event_outcomes.csv
Scope: HARD_TEST80 headline labels only.
Expected: 80 videos, 506 paired events per label, 11 labels, 5,566 rows.

## Primary rate mismatch

EXP053 established B3-S rate 0.29056540649046503 and B2 rate
0.3232686517229843 on HARD_TEST80. Absolute difference is
0.032703245232519274 > frozen tolerance 0.02.

EXP054 therefore computes the frozen-threshold BCa interval but does not permit
a matched-rate primary-effect interpretation. No TEST threshold retuning is
allowed.

Final primary status when the frozen rate mismatch is verified:
RATE_MISMATCH_NO_MATCHED_RATE_PRIMARY_INTERPRETATION.

## ITR metadata correction

EXP052 incorrectly inherited group=PRIMARY onto the ITR30 B3-S-minus-B2 row.
EXP054 changes reporting metadata only: all ITR30 inferential rows are
secondary, while matched-neutral ITR rows are secondary diagnostics.
The numerical estimator is unchanged.

## Outcome firewall

No seed search, replicate-count search, model change, threshold refinement,
split change, subgroup search, endpoint change, or TEST rerun is permitted.
