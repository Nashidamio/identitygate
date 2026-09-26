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
