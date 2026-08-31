# IdentityGate — Experiment Registry

Every meaningful experimental run appended here. One entry per run. Never delete; append corrections as new dated notes.

Fields per entry:
- Experiment ID
- Purpose
- Hypothesis
- Code version (git commit / tag)
- Dataset (name + version + split)
- Checkpoint (model + version)
- Configuration file
- Seed
- Command
- Output location
- Result (raw metrics)
- Status (PLANNED / RUNNING / COMPLETED / FAILED / SUPERSEDED)
- Interpretation (what the evidence supports, honestly)

---

## EXP000 — Environment sanity (bookkeeping only)
- Purpose: Placeholder to verify registry format.
- Status: PLANNED (no run).
- Notes: First real experiment will be a single-video SAM 3.1 smoke test after Task 1 (env + checkpoints) completes.

## EXP013 - Full event-bearing attribute scan
- Purpose: Compute GT-only per-track attributes over every MOSEv2 train video containing at least one qualifying event.
- Hypothesis: Enlarging beyond the 291-video convenience pool will provide enough candidate data for a defensible hard stratum.
- Code version: 2dcd019.
- Dataset: MOSEv2 train, 3,666 videos; event-bearing subset.
- Checkpoint: None; GT-only CPU experiment.
- Configuration file: None.
- Seed: None.
- Command: python scripts/exp013_attrs.py 0
- Output location: experiments/EXP013_attrs.csv
- Result: 1,691 videos; 3,237 tracks; 4,469 events.
- Status: COMPLETED.
- Interpretation: Full event-bearing attribute pool is available and exactly reconciles EXP006 event count.

## EXP014 - Synchronized-reappearance visual audit
- Purpose: Test the EXP011 interpretation that synchronized reappearances represent whole-scene occlusion.
- Hypothesis: Timing synchronization alone is insufficient to establish whole-scene occlusion.
- Code version: milestone working tree based on 2dcd019.
- Dataset: MOSEv2 train; five synchronized-reappearance clusters identified from EXP011.
- Checkpoint: None; GT/raw-frame audit.
- Configuration file: None.
- Seed: None.
- Command: python scripts/exp014_sync_visual_audit.py
- Output location: experiments/EXP014_sync_visual_audit/
- Result: 82 video-flagged events; 78 event-level synchronized events; 4 false inclusions. Five clusters visually audited.
- Status: COMPLETED.
- Interpretation: Whole-scene-occlusion interpretation rejected. Most audited clusters are consistent with camera-induced out-of-view/re-entry; sync timing is retained only as diagnostic evidence.

## EXP015 - Full-pool GT-only lever feasibility
- Purpose: Apply the existing EXP012 GT-only hardening levers to the full EXP013 event-bearing pool.
- Hypothesis: Pool enlargement resolves the sample-size failure seen in EXP012.
- Code version: milestone working tree based on 2dcd019.
- Dataset: MOSEv2 train via experiments/EXP013_attrs.csv.
- Checkpoint: None; GT-only CPU experiment.
- Configuration file: None.
- Seed: None.
- Command: python scripts/exp015_fullpool_levers.py
- Output location: experiments/EXP015_fullpool_levers.csv and experiments/EXP015_fullpool_levers.md
- Result: 4,469 events; 3,237 tracks; 1,691 videos; 31 non-redundant rules; 17 rules retain >=210 videos.
- Status: COMPLETED.
- Interpretation: Raw candidate-pool size is no longer the headroom bottleneck.

## EXP016 - Fresh hard-stratum B0 headroom
- Purpose: Measure vanilla SAM 3.1 recovery headroom on a fresh, score-independent GT-hard sample.
- Hypothesis: A fresh sample satisfying obj_size < 0.005 AND n_frames >= 100 will lower B0 POR_hard_w30 enough to provide intervention headroom.
- Code version: full scientific run PENDING clean milestone commit; parent 2dcd019.
- Dataset: MOSEv2 train; EXP016 frozen headroom-development manifest.
- Checkpoint: SAM 3.1 installed project checkpoint/model builder.
- Configuration file: experiments/EXP016_headroom_manifest.json
- Seed: 42.
- Commands: python scripts/exp016_make_headroom_manifest.py; python scripts/exp016_headroom.py sanity; full command pending.
- Output location: experiments/EXP016_headroom_manifest.json; local sanity artifact experiments/EXP016_sanity/results.json.
- Result so far: 40 fresh selected videos; 60 eligible tracks; 112 hard events. Sanity 3/3 videos completed, 7/7 eligible events matched, max VRAM 5.67 GB.
- Status: RUNNING - manifest and sanity VERIFIED; 40-video B0 run pending.
- Interpretation: Runner/API/event filtering are verified. Sanity POR is not scientific evidence.

### EXP016 completion note - 2026-08-24
- Code version: 37e3ed0b3beec010c43429bb336042d7d85dcd34.
- Command: python scripts/exp016_headroom.py full
- Output location: experiments/EXP016_full/results.json
- Result: 40 videos; 120 total events; 112/112 frozen hard events scored.
- Hard POR: W15=0.5804, W30=0.6071, W60=0.6071.
- All-event POR: W15=0.6083, W30=0.6333, W60=0.6333.
- Theft events: 2 across 1 track.
- Runtime: 19.9 min.
- Peak VRAM: 12.38 GB.
- Status: COMPLETED.
- Interpretation: Predeclared B0 headroom criterion passes because POR_hard_w30=0.6071 <= 0.70. This validates the candidate hard-stratum rule for split construction but does not itself lock final TRAIN/DEV/TEST.

## EXP017 - Final split construction
- Purpose: Freeze leakage-controlled hard-stratum and full-event-bearing evaluation cohorts.
- Hypothesis: GT-only split construction can satisfy sample-size, event-count, headroom, and development-exclusion requirements without model-score cherry-picking.
- Code version: source parent 9e4cac1; generator hashes recorded in manifests where available.
- Dataset: MOSEv2 train partition.
- Checkpoint: None; GT-only split construction.
- Configuration: obj_size < 0.005 AND n_frames >= 100; seed 42.
- Commands: scripts/exp017_make_candidate_splits.py and scripts/exp017_make_full_distribution.py.
- Outputs: experiments/EXP017_split_manifest.json and experiments/EXP017_full_distribution_manifest.json.
- Result: TRAIN=100, DEV=40, hard TEST=118/343 hard events; full-event-bearing TEST=118/295 events; 9-video TEST/Test overlap; 227 unique held-out TEST videos; zero TEST overlap with TRAIN, DEV, or 81 development exclusions.
- Status: COMPLETED / LOCKED.
- Interpretation: Dataset selection and held-out cohort construction are complete. Future model results may not change these splits.



## SUBSTRATE CORRECTION — AMENDMENT A1 — 2026-08-28

Status: **LOCKED / USER-APPROVED**

Effective for all model-based experiments after A1:

- Core model: SAM 3 VOS/PVS.
- Builder: `build_sam3_video_model()`.
- Upstream checkpoint identity: `facebook/sam3/sam3.pt`.
- True SAM 3.1 Object Multiplex is a transfer/extension substrate only.
- SAM 3.1 checkpoint identity:
  `facebook/sam3.1/sam3.1_multiplex.pt`.

Historical registry entries that describe
`build_sam3_video_model()` runs as “SAM 3.1” are retained for provenance but
their model-version label is **SUPERSEDED by Amendment A1**.

In particular:
- EXP016 is a SAM 3 VOS B0 headroom experiment.
- EXP018 is SAM 3 VOS signal-path engineering evidence.
- EXP020 is SAM 3 VOS relational-pointer feasibility evidence.
- GT-only experiments are unaffected.

No historical metric is changed by this correction.

### EXP017 supersession note - 2026-08-31

- Historical EXP017 manifests and measurements are retained unchanged for provenance.
- EXP017 as the final thesis TRAIN/DEV/TEST split is **SUPERSEDED**.
- Its previously exposed TRAIN+DEV videos remain only a development-exposure boundary.
- Final DEV/TEST construction must obey the later locked leakage rules and development exclusions.
- No historical EXP017 metric or manifest is rewritten.

## EXP018 - SAM 3 VOS signal-path engineering probe

- Purpose: Verify current-core VOS signal availability, per-object pointer access, object-score semantics, and the mask-prompt API required for later signal extraction.
- Hypothesis: The SAM 3 VOS path exposes the per-object identity and quality primitives required for IdentityGate development.
- Code version: exact producing commit UNKNOWN; surviving probe files remain untracked as of 2026-08-31.
- Dataset: MOSEv2 train, development-exposed video 7744dc51.
- Checkpoint: SAM 3 VOS/PVS; `facebook/sam3/sam3.pt` under Amendment A1.
- Configuration file: None recorded.
- Seed: None.
- Command: UNKNOWN / not recovered from surviving evidence.
- Output location: `experiments/EXP018_signal_probe_attempt1_failed.txt`; `experiments/EXP018_signal_probe_attempt2.txt`.
- Result: Attempt 1 failed because `add_new_mask` received a NumPy mask instead of a Torch tensor. Attempt 2 verified `hidden_dim=256`, `mem_dim=64`, `max_obj_ptrs_in_encoder=16`, per-object `obj_ptr [1,256]` float32, `object_score_logits [1,1]`, `iou_score [1]`, and four-object `maskmem_features [4,64,72,72]` bf16.
- Status: COMPLETED engineering probe; attempt 1 FAILED, attempt 2 COMPLETED.
- Interpretation: SAM 3 VOS exposes usable per-object identity pointers. No predictive-utility claim is supported by this probe.

## EXP019 - Whole-scene annotation candidate census

- Purpose: Apply the frozen WS-v1 annotation rule over the full event-bearing corpus and create annotation-side whole-scene candidates without using model outcomes.
- Hypothesis: The annotation rule can provide candidates for later independent pixel-side confirmation.
- Code version: 527b8d8.
- Dataset: MOSEv2 train; all 1,691 event-bearing videos and 4,469 qualifying object events.
- Checkpoint: None; GT-only CPU experiment.
- Configuration file: `configs/WS-v1.json`.
- Seed: None recorded.
- Command: `python scripts/exp019_ws_gt_tagging.py`.
- Output location: exact final artifact path UNKNOWN / not recovered in this registry backfill.
- Result: 4,469 raw object events exactly reconstructed; 2,945 annotation-side candidate object events; 2,179 candidate scene events after collapse; 1,524 non-candidate object events; 3,703 analysis events after candidate collapse; 1,281 videos with at least one candidate. Among flagged events, 1,888 had only one reference object.
- Status: COMPLETED / GT_ANNOTATION_CANDIDATES_ONLY_NOT_FINAL_WS_LABELS.
- Interpretation: Annotation timing alone is not a final whole-scene label. Independent pixel cross-check and the predeclared manual audit remain OPEN and mandatory.

## EXP020 - Relational pointer feasibility

- Purpose: Verify that SAM 3 VOS per-object pointers can be aligned exactly with tracked object IDs and compared against tracked competitors.
- Hypothesis: Packed VOS pointer rows preserve the object-ID ordering required for relational identity features.
- Code version: 6646f32.
- Dataset: MOSEv2 train development-exposed video 7744dc51; four frame-0 tracked objects.
- Checkpoint: SAM 3 VOS/PVS; `facebook/sam3/sam3.pt`.
- Configuration file: None recorded.
- Seed: None.
- Command: UNKNOWN / not recovered from surviving evidence.
- Output location: exact artifact path UNKNOWN / not recovered in this registry backfill.
- Result: Four frame-0 anchor pointers were extracted. Packed frame-0 representation had shape `(4,256)` and propagation representation had shape `(4,256)`. Packed row order aligned exactly with the object-ID mapping and matched per-object extraction.
- Status: COMPLETED / TECHNICAL_FEASIBILITY_PASS_SCIENTIFIC_UTILITY_PENDING.
- Interpretation: Tracked-competitor pointer comparison is technically executable on the SAM 3 VOS core. This experiment does not establish predictive utility.

## EXP021 - Relational development-scope freeze

- Purpose: Freeze a leakage-controlled multi-object development scope for relational identity analysis using only already development-exposed videos.
- Hypothesis: Restricting relational work to previously exposed multi-object videos permits utility analysis without touching final TEST.
- Code version: 809ed5d.
- Dataset: Historical EXP017 development-exposed TRAIN+DEV boundary; videos with at least two frame-0 objects.
- Checkpoint: None; scope construction only.
- Configuration file: None recorded.
- Seed: None.
- Command: UNKNOWN / not recovered from surviving evidence.
- Output location: `experiments/EXP021_relational_scope.json`.
- Result: 26 videos total = 18 TRAIN + 8 DEV; 4,066 video frames; expected 22,140 candidate object-frame rows; 1,482 ordered anchor-pair rows; TEST touched = 0. Scope SHA256: `e528ab0b422f57fee371425199f56193c2b54ca103836ae6cf3f364cd8f5e2b2`.
- Status: COMPLETED / FROZEN DEVELOPMENT SCOPE.
- Interpretation: EXP021 defines the no-new-leakage boundary for EXP022 and EXP023 relational development work. It is not a final thesis split.

## EXP022 - Full relational signal census

- Purpose: Characterize self-anchor and tracked-competitor pointer signals, pointer validity, and relational degeneracy across the frozen 26-video development scope.
- Hypothesis: Relational pointer signals are extractable at scale and may contain candidate identity information beyond self similarity.
- Code version: pilot 671e9f7; serialization correction d74f126; full-run freeze 035185d; result commit d370558.
- Dataset: EXP021 relational scope; 26 videos = 18 TRAIN + 8 DEV.
- Checkpoint: SAM 3 VOS/PVS; `facebook/sam3/sam3.pt`; checkpoint SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Configuration file: `configs/EXP022-relational-census-v1.json`.
- Seed: deterministic frozen scope; no stochastic sampling recorded for the full census.
- Command: UNKNOWN / not recovered from surviving evidence.
- Output location: `experiments/EXP022_full/features.csv`; `experiments/EXP022_full/anchor_cosine.csv`; `experiments/EXP022_full/summary.json`.
- Result: 26/26 videos; 22,140 candidate object-frame rows; 1,482 anchor rows; TEST touched = 0. Overall valid-pointer fraction = 0.5567299006. Approximately 71.6% of valid rows had negative self-minus-competitor margin. Pointer validity varied strongly by video.
- Status: COMPLETED / VERIFIED.
- Interpretation: Relational signals exist, but predictive utility is NOT established. `pointer_valid` remains a validity mask, not silently a predictive feature. Negative identity margin is not evidence of theft. B3-R remains KEEP-BUT-NOT-FROZEN.

## EXP023 - TRAIN18 production primitive cache

- Purpose: Cache definition-independent SAM 3 primitives and raw FP32 pointers needed for later B2 vs B3-S vs B3-R utility analysis without manufacturing clean-state-dependent features.
- Hypothesis: The frozen extractor can reproduce verified EXP022 primitives and generate a canonical TRAIN-only development cache without touching DEV or TEST.
- Code version: signal-schema freeze b9ed7b3; production freeze 41a31ea; provenance correction 29f6332; production result commit 801d8ea.
- Dataset: 18 multi-object TRAIN videos from the frozen EXP021 development scope.
- Checkpoint: SAM 3 VOS/PVS; `facebook/sam3/sam3.pt`; checkpoint SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- Configuration file: `configs/EXP023-primitive-cache-v1.json`.
- Seed: deterministic frozen scope; no stochastic sampling recorded for production extraction.
- Command: exact invocation UNKNOWN; execution provenance is preserved in `experiments/EXP023_train18_run.log`.
- Output location: `experiments/EXP023_train18_cache/`; run log `experiments/EXP023_train18_run.log`.
- Result: 18/18 videos; 13,524 primitive data rows; 57 cache files = 18 per-video pointer NPZ + 18 per-video primitive CSV + 18 per-video summary JSON + merged `primitives.csv`, `index.csv`, and `summary.json`. DEV touched = 0; TEST touched = 0.
- Result hashes: merged `primitives.csv` = `c13fddfcc7fe422893e5cfed86100d9a8407abb0421c6296f5ed168a34e92feb`; `index.csv` = `9e73e27506d540aab63dd55e2a07a4b5e7e5c8e18e54b393e0891c4d1fa36db0`; `summary.json` = `145889fc5e26c021e8e083956baa7a1b01333357f137ce3c4da42c33448946ce`; run log = `e3cab3898963f559397da9b7bb35213338273fb05afbe2977ad6fa9b23e56e15`.
- Status: COMPLETED / VERIFIED.
- Interpretation: Canonical TRAIN primitive evidence for the first incremental utility analysis is available. Features 4, 8, 9, and 11 remain DEFERRED; Feature 7 remains OPEN. No IdentityGate model has been trained and no B2/B3 utility claim has been established.

### Registry backfill open item - 2026-08-31

- EXP001 through EXP012 remain missing from this registry.
- Their provenance is recoverable from `docs/RESEARCH_HISTORY.md`.
- Backfilling EXP001-EXP012 is OPEN / NON-BLOCKING and is intentionally deferred so it does not delay current B2/B3 utility work.
