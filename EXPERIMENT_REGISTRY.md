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
