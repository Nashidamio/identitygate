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

