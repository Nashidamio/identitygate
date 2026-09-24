# IdentityGate Final Pre-Registration

Date: 2026-09-24

Status: FROZEN WHEN COMMITTED TO main WITH THE EXP053 IMPLEMENTATION

## Research question

"What information should a memory-write gate use — quality signals, temporal signals, or identity signals?"

## Frozen substrate

- SAM 3 VOS/PVS.
- Frozen SAM; no gradient, LoRA, fine-tuning, detector change, or new architecture.
- Gate parameter budget remains <=50K parameters.
- Identity cosine remains FP32 with autocast off, TF32 off, and highest float32 matmul precision.

## Frozen cohorts

- HARD_TEST80: 80 videos / 506 primary events.
- REPRESENTATIVE_TEST40: 40 videos / 76 primary events.
- Fresh DEV: already observed before this pre-registration synthesis.
- Final TEST prediction-derived evidence remains untouched at this freeze.

## Frozen methods

- B0.
- B1.
- B2.
- B3-S.
- B3-R.
- B5.
- Outcome-independent exact-K neutral controls.

## Headline operating point

- r_star = 0.30.
- B1 tau = 0.10.
- B2 tau = 0.10.
- B3-S tau = 0.20.
- B3-R tau = 0.20.
- B5 tau = 0.70.
- B0 remains at native physical write rate.
- TEST thresholds are never retuned.

## Supported curve policy

- Requested target grid remains 0.10 through 0.90.
- A9 freezes exactly 19 SUPPORTED DEV mappings and 26 UNSUPPORTED mappings.
- TEST receives only the 19 frozen SUPPORTED thresholds.
- No TEST search, interpolation, extrapolation, midpoint refinement, or retuning.
- HARD_TEST80 receives all supported curve points.
- REPRESENTATIVE_TEST40 receives final headline methods and exact-K headline neutral controls.

## Endpoint hierarchy

- Sole primary endpoint: POR@30.
- Primary contrast: POR@30(B3-S) - POR@30(B2) at r_star=0.30.
- ITR@30 is secondary.
- B3-R, B1, B5, and B0-reference contrasts remain secondary according to A10.
- Matched-neutral comparisons are diagnostics and cannot replace the primary contrast.

## Event definitions

- Qualifying gap: at least 5 frames.
- Recovery: target IoU > 0.5 within first 30 evaluable GT-visible post-reappearance frames.
- Frozen overlap truncation remains unchanged.
- Theft frame: target IoU < 0.3 AND max-other IoU > 0.5.
- Theft episode: at least 5 consecutive chronological theft frames.
- ITR denominator: qualifying reappearance events under the POR30-aligned interval.

## Matched-rate rule

- Physical write denominator: eligible non-conditioning frames.
- Direct gated TEST contrasts with absolute pooled write-rate difference >0.02 are labelled RATE_MISMATCH and are not described as matched-rate results.
- Signal-versus-neutral controls remain exactly budget matched per video.

## Statistical inference

- Statistical cluster: video.
- Primary uncertainty: paired video-clustered BCa 95% confidence interval.
- Point estimator: pooled event-level POR rate difference.
- Bootstrap replicates: 50,000.
- Seed: 52.
- BCa bias correction: midrank bootstrap position with finite clipping.
- BCa acceleration: delete-one-video jackknife.
- Minimum practically important hard-set POR benefit: +0.08.
- A10/A11 confidence-interval interpretation remains frozen.

## One-touch TEST rule

- Initial final TEST invocation uses EXP053 mode run.
- A persistent campaign marker is created before TEST trajectories begin.
- Engineering interruption may continue only with mode resume under the identical frozen repository commit and config.
- Partial TEST POR/ITR outcomes may not alter any scientific choice.
- Negative, null, harmful, positive, and inconclusive outcomes are all admissible.

## Frozen implementation provenance

- EXP053 config SHA256: 3869eb901f69bdebb854d105ec74c7cf0e596d19dc3436ec2dfd38f10896e791
- EXP053 script SHA256: 875f90da8eae90c1642231dd2b7cb2256690ca90cca370abd426b1ddba1bb84b
- EXP053 tests SHA256: f71fff99c078690075ae0bca0a5466bb055f29a56dedb138c2259bde6a002349
- A12 SHA256: f254e8a4548ce6151295c413e0674b0db3bfa87cda39f5fc86bba26eb8556203

The EXP053 config also pins SHA256 identities for EXP050, EXP051, A9, A10,
A11, final operating points, curve selection, event pool, HARD_TEST80, and
REPRESENTATIVE_TEST40. The Git freeze commit binds this pre-registration,
A12, the config, runner, and tests before any final TEST prediction is generated.

No post-TEST scientific retuning is permitted.
