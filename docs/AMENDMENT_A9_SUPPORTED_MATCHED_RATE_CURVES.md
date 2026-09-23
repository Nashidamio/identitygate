# Amendment A9 - Supported Matched-Write-Rate Curve Handling

Date: 2026-09-23
Status: FROZEN WHEN COMMITTED TO main

## Trigger

EXP050 executed the frozen A4 write-rate-only procedure on fresh DEV before
any tracking-performance outcome was computed or inspected.

Headline matching succeeded at:

    r_star = 0.30

Frozen headline operating points:

    B1    tau=0.10  rate=0.28367729831144467
    B2    tau=0.10  rate=0.30919324577861160
    B3-S  tau=0.20  rate=0.30393996247654786
    B3-R  tau=0.20  rate=0.29043151969981240
    B5    tau=0.70  rate=0.29812382739212007

All satisfy the frozen absolute tolerance of 0.02.

For the requested 45 full-curve variant-target rows, 19 matched and
26 remained unmatched after the frozen maximum four midpoint refinements.

## Locked consequence

A9 supersedes only the requirement that every requested curve target must
successfully produce a matched operating point.

The requested 0.10-through-0.90 target grid remains mandatory for attempted
mapping and reporting.

For every requested target:

- successful A4 mappings are frozen SUPPORTED operating points;
- failed mappings are reported as UNSUPPORTED;
- no threshold is fabricated;
- no interpolation or extrapolation across unsupported regions is allowed;
- the four-refinement budget is not increased for EXP050;
- the 0.02 tolerance is not relaxed;
- target rates are not changed after tracking outcomes are observed.

Primary comparative inference remains at frozen r_star=0.30.

The EXP050 headline thresholds above remain frozen.

No later POR, ITR, J&F, UAR, contamination, DEV outcome, or TEST outcome may
replace r_star or these headline thresholds.

## Observed rate-support behavior

All variants had pooled write rate 1.0 at tau=0.

At tau=0.00625:

    B1    0.374109
    B2    0.427767
    B3-S  0.437711
    B3-R  0.399625
    B5    0.472045

This is recorded as a sharp observed near-zero threshold discontinuity.

A9 does NOT conclude that unmatched high-rate targets are mathematically
unreachable.

## TEST

TEST receives only thresholds frozen from successful DEV mappings.

No TEST threshold search, interpolation, extrapolation, refinement, or
retuning is allowed.

The A4 RATE_MISMATCH rule and one-touch TEST rule remain unchanged.

## Outcome firewall

At A9 freeze:

- fresh-DEV POR has not been computed;
- fresh-DEV ITR has not been computed;
- fresh-DEV J&F has not been computed;
- fresh-DEV UAR has not been computed;
- fresh-DEV contamination has not been computed;
- TEST has not been evaluated.

## Still open before performance-outcome inspection

- exact ITR denominator;
- F1 endpoint / co-primary-status contradiction.
