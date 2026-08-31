# RESEARCH_STATE.md

**Project:** IdentityGate — Supervised, Identity-Verified Memory Write Admission for SAM 3 Video Tracking with SAM 3.1 Transfer Validation

## AMENDMENT A1 — SAM 3 CORE / SAM 3.1 TRANSFER — 2026-08-28

### Status

**LOCKED / USER-APPROVED**

Approval phrase:

    APPROVE SAM3 CORE + SAM3.1 TRANSFER

### Verified implementation fact

The previously used non-multiplex VOS/PVS API

    from sam3.model_builder import build_sam3_video_model
    build_sam3_video_model()

does not load the released SAM 3.1 multiplex checkpoint by default.

At installed SAM source commit:

    8f0b7f4d4e7eda2ed606ebde6702c93359ad01da

`build_sam3_video_model()` defaults to the SAM 3 Hugging Face checkpoint:

    repo_id = facebook/sam3
    checkpoint = sam3.pt

The released SAM 3.1 video path is Object Multiplex and uses:

    repo_id = facebook/sam3.1
    checkpoint = sam3.1_multiplex.pt

### Hardware constraint

The available experimental GPU is an RTX 4080 SUPER with 16 GB VRAM.

Prior verified engineering measurements showed that the true SAM 3.1 Object
Multiplex path exceeded the 16 GB core experimental budget, whereas the
non-multiplex VOS/PVS path was operational within the available GPU budget.

### Locked substrate decision

CORE THESIS SUBSTRATE:
- frozen SAM 3
- non-multiplex VOS/PVS
- `build_sam3_video_model()`
- `facebook/sam3/sam3.pt`
- no SAM fine-tuning, LoRA, detector replacement, or external re-ID model

TRANSFER / EXTENSION:
- true SAM 3.1 Object Multiplex
- `facebook/sam3.1/sam3.1_multiplex.pt`
- used only for a predeclared transfer/feasibility validation if hardware
  permits
- does not carry the core statistical claim

### Research question

UNCHANGED / LOCKED:

“What information should a memory-write gate use — quality signals, temporal
signals, or identity signals?”

The IdentityGate intervention, GT-derived labels, matched-write-rate design,
risk-control work, whole-scene protocol, and relational-identity investigation
remain unchanged.

### Historical-result reclassification

The following classes of result are retained, not discarded:

- GT-only experiments (including EXP019 and EXP021): unaffected.
- EXP018 and EXP020 VOS engineering probes: valid SAM 3 VOS evidence.
- Earlier VOS/PVS B0 measurements using `build_sam3_video_model()`: valid SAM 3
  measurements; any prior “SAM 3.1” wording is superseded terminology.
- prior true SAM 3.1 Object Multiplex memory/runtime measurements: retained as
  hardware-feasibility evidence.

Historical files under `docs/audit/` are provenance artifacts and may contain
the superseded pre-A1 terminology. They are not silently rewritten.

### Thesis-positioning consequence

The model minor version is not the novelty claim.

The primary contribution remains the controlled study of supervised memory
WRITE admission in a frozen SAM-generation video tracker, including quality,
temporal, self-identity, and proposed tracked-competitor identity evidence,
with matched write rates and risk-controlled admission.


**Last updated:** 2026-08-18 (end of Week 1 / start of Week 2)
**Source of truth:** v4 FINAL plan + Review (fixes F1-F6, N1, N3 adopted; N2 conditional on audit)

---

## 1. MACHINE

Lab PC #27 - Intel i7-14700K, RTX 4080 SUPER 16 GB, 64 GB RAM, 1 TB SSD
Windows 11 host + WSL2 Ubuntu 22.04.5
Lab access: Saturday and Monday, 1 PM - 1 AM

### Restart checklist
    conda activate identitygate
    cd ~/thesis/identitygate
    git log --oneline -3
    python -c "from sam3.model_builder import build_sam3_video_model; print('OK')"

---

## 2. DIRECTORY MAP

WSL paths:
    ~/thesis/identitygate/          THE PROJECT (git repo)
        scripts/                    exp001-008, count_events.py
        experiments/                results: PNG, JSON, CSV
        docs/audit/                 signal_schema.md, extracted notebook code
        docs/ENVIRONMENT.md
        docs/SAM3_INSTALL.md
        RESEARCH_STATE.md, EXPERIMENT_REGISTRY.md
    ~/thesis/externals/sam3/        SAM 3 repo @ 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
    ~/thesis/checkpoints/sam3.1/    3.3 GB, sam3.1_multiplex.pt
    ~/thesis/checkpoints/sam3/      6.5 GB fallback
    ~/thesis/logs/

Windows paths (for viewing):
    \\wsl.localhost\Ubuntu\home\user2\thesis\identitygate
    D:\thesis_data\mosev2\        79 GB dataset, extracted

GitHub: https://github.com/Nashidamio/identitygate  (private until Week 7 per plan)

---

## 3. ENVIRONMENT (verified working)

Python 3.12.13 | PyTorch 2.10.0+cu128 | CUDA 12.8 | conda env `identitygate`

Required pins beyond SAM 3's declared dependencies (upstream bugs):
| Package     | Constraint | Reason |
|-------------|-----------|--------|
| setuptools  | <82       | SAM3 model_builder.py imports pkg_resources, removed in setuptools 82 |
| einops      | any       | sam3/sam/rope.py imports it; declared only in [notebooks] extra |
| pycocotools | any       | imported at load time; declared only in [dev] extra |
| scipy       | any       | mask ops here, plan section 16 statistics later |

Rebuild order after any env reset:
1. conda create -n identitygate python=3.12
2. pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
3. pip install -e ~/thesis/externals/sam3
4. pip install "setuptools<82" einops pycocotools scipy

---

## 4. SUBSTRATE DECISION (locked, evidence-based)

**Thesis runs in VOS/PVS mode, NOT multiplex/PCS.**

Measured on the same video, same GPU:
| Mode | Peak VRAM | Speed | Fits 16 GB? |
|------|-----------|-------|-------------|
| Multiplex (build_sam3_multiplex_video_predictor) | 21.77 GB | 1.0 it/s | NO |
| VOS (build_sam3_video_model().tracker)           | 5.9-7.5 GB | 6.8-11 it/s | YES |

EXP002 (max_num_objects=8 + offload_video_to_cpu): 21.77 GB, negligible gain.
EXP003 (5 vs 30 frames): 21.57 vs 21.77 GB - VRAM is a fixed per-frame cost.
Plan section 19 states the thesis evaluates PVS, so this is aligned, not a deviation.
Multiplex/PCS deferred to the N2 stretch lever.

### VOS API (exact, verified)
    from sam3.model_builder import build_sam3_video_model
    m = build_sam3_video_model()
    predictor = m.tracker
    predictor.backbone = m.detector.backbone          # REQUIRED
    st = predictor.init_state(video_path=FRAME_DIR)
    predictor.clear_all_points_in_video(st)
    predictor.add_new_mask(inference_state=st, frame_idx=0, obj_id=oid,
                           mask=bool_tensor_2d)
    for out in predictor.propagate_in_video(st, start_frame_idx=0,
            max_frame_num_to_track=N, reverse=False, propagate_preflight=True):
        fidx, obj_ids, low_res, video_res = out[0], out[1], out[2], out[3]

Notes:
- build_sam3_video_model takes NO use_fa3 parameter (FA3 defaults False on this path)
- points, if used, must be NORMALIZED 0-1 torch tensors
- multiplex path bug: Sam3BasePredictor.start_session passes offload_state_to_cpu which
  the multiplex init_state rejects; workaround was calling init_state directly

---

## 5. HOOK TARGET (verified firing)

**Sam3TrackerBase._encode_new_memory** - sam3/model/sam3_tracker_base.py:796

kwargs received: image, current_vision_feats, feat_sizes, pred_masks_high_res,
                 object_score_logits, is_mask_from_pts, output_dict, is_init_cond_frame
returns: (maskmem_features [1,64,72,72] bf16, maskmem_pos_enc)

Verified: fires on EVERY frame (97 encodes / 97 frames).
Per-object slicing exists at sam3_tracking_predictor.py:901
    obj_out["maskmem_features"] = maskmem_features[obj_slice]

**Identity features CONFIRMED:** obj_ptr [1,16,256] bf16 present in memory dict.
=> Plan section 5 features 9-11 (ptr_sim_roll/anchor/ema) are computable
=> B3 and the B2-vs-B3 co-primary are EXECUTABLE, no re-scoping needed
=> Review kill-rule F4 does not trigger

Multiplex-path equivalents (for reference if N2 is pursued):
  Sam3MultiplexBase._tracker_update_memories - sam3_multiplex_base.py:2502
  Track B hook: _post_execution_phase_hook - sam3_video_base.py:1253
  Recovery target: _recondition_masklets - sam3_multiplex_base.py:827

---

## 6. DATASET STATUS

Source: HuggingFace `FudanCVL/MOSEv2` (public). All 4 SHA256 checksums verified OK.
Location: D:\thesis_data\mosev2\ (79 GB archives + 57 GB extracted train)

| Split | Videos | Annotations | Usable? |
|-------|--------|-------------|---------|
| train | 3,666  | per-frame   | YES - primary |
| valid |   433  | FIRST FRAME ONLY | NO - cannot build labels or measure POR |

Mask format: PIL mode 'P', uint8, pixel value = object id, 0 = background.
Metadata correction: plan says ~3,466 train videos; actual is **3,666**.

---

## 7. EVENT COUNT (EXP006 - plan section 17 Week-2 checkpoint, COMPLETE)

Implements plan section 14 definition: visibility = GT mask area > 0;
event = visibility gap of >= 5 consecutive frames followed by visibility.
Full scan of all 3,666 videos. Output: experiments/EXP006_events.csv

| Filter | Videos | Tracks | Events |
|--------|--------|--------|--------|
| any track with >=1 event | 1,691 | 3,237 | 4,469 |
| + video has >=2 objects | 641 | 2,187 | 2,497 |
| + track visible >=20 frames | 543 | 1,452 | 1,751 |
| + video >=60 frames | 291 | 743 | 1,023 |
| + video >=100 frames | 122 | 312 | 491 |

Plan section 17 needs TEST >= 200 events (pref 300-500); splits 100-150/30-50/80-150 videos.
- >=100-frame pool: TEST ~40 videos = ~161 events -> **FAILS the 200 minimum**
- >=60-frame pool:  TEST ~101 videos = ~355 events -> **MEETS requirement**

**PROVISIONAL criteria (NOT LOCKED):** >=2 objects AND track visible >=20 frames
AND video >=60 frames = 291 videos, 1,023 events.

**DISCLOSURE REQUIRED:** plan section 14's 60-frame POR sensitivity row will not be
reportable for the shortest videos in this pool. Supervisor must be informed.

Known data artifact: video 0cbfxuq5 has 17 objects each visible only 2 of 63 frames -
an annotation artifact, not real occlusion. Hence the >=20-visible-frames filter.

---

## 8. FAILURE MODE OBSERVED (EXP007 / EXP008)

Video 6042d64a: 97 frames, 2 objects, 6 GT occlusion events, max gap 18.
Prompted with GT masks on frame 0 (thesis PVS protocol, no click guessing).
Objects are small and hard: obj1 ~1,283 px in a 1680x1080 frame; obj2 is the
camera-wearer's own leg entering/leaving frame.

Measured over 92 object-frames where GT and prediction are BOTH present:
| Metric | Value |
|--------|-------|
| mean IoU | 0.573 |
| IoU >= 0.7 (plan 11 'reliable') | 37 (40%) |
| IoU < 0.3 (plan 11 'unreliable') | 15 (16%) |
| **HALLUCINATIONS (GT absent, prediction present)** | **7** |
| MISSES (GT present, prediction absent) | 3 |
| correctly absent | 92 |

Hallucination frames: 3, 4, 15, 60, 76, 77, 79 (457 to 1,287 px predicted where
GT says the object is absent).

**Significance:** memory is encoded on EVERY frame, so these hallucinated masks DO
enter the memory bank. This is the exact failure mode IdentityGate targets, now
observed in our data on our hardware. Per plan section 11 these frames auto-label
'unreliable'. Visual triptychs (RAW | GT | PREDICTION) in
experiments/EXP008_6042d64a_hallucinations/

**Measurement caution:** correct IoU must EXCLUDE frames where GT area = 0.
Counting those as IoU 0.0 inflates apparent failure - an earlier 0.517 mean was
wrong for this reason.

Throughput measured: ~14 s/video -> ~70 min for all 291 candidate videos.

---

## 9. EXPERIMENTS COMPLETED

| ID | Purpose | Key result |
|----|---------|-----------|
| EXP001 | multiplex hook probe | hook fires; obj_ptr [1,16,256] found |
| EXP002 | VRAM tuning (max_num_objects, offload) | 21.77 GB, negligible gain |
| EXP003 | VRAM vs video length | fixed per-frame cost, not accumulating |
| EXP004 | VOS-path probe | 7.54 GB, hook target identified |
| EXP005 | visual verification | 31/31 memory encodes, overlays saved |
| EXP006 | MOSEv2 event count | 4,469 events across 3,666 videos |
| EXP007 | full pipeline, real MOSEv2 video | mean IoU 0.573, 7 hallucinations |
| EXP008 | hallucination visualization | RAW/GT/PRED triptychs |

---

## 10. OPEN DECISIONS (require supervisor input)

1. **F1** - B2-vs-B3 as co-primary endpoint vs POR-vs-B0 as sole primary.
   v4 sections 14/29 lock the single primary; the Review's F1 wants both.
   Must be settled before the Week-6 freeze.
2. **Split criteria** - adopt the >=60-frame provisional pool? Requires the
   headroom check first.
3. **60-frame sensitivity-row limitation** must be disclosed to supervisor.

---

## 11. SCHEDULE POSITION

- Week 0 (machine bring-up): **DONE**
- Week 1 (audit, hooks, both tracks, signal schema): **DONE**
- Week 2 (data, cache, event count, headroom, labels): **IN PROGRESS**
    - MOSEv2 downloaded/verified/extracted: DONE
    - Event count: DONE
    - Video selection + split lock: NOT DONE
    - Vanilla B0 headroom run: NOT DONE
    - Signal caching: NOT DONE
    - Label creation: NOT DONE
- Weeks 3-8: not started. IdentityGate itself does not exist yet.

---

## 12. NEXT REQUIRED ACTION

**Plan section 17 headroom checkpoint.** Run vanilla SAM 3.1 (B0) over a DEV-scale
sample of the candidate pool and measure POR per plan section 14:
    recovered = 1 if IoU(pred, target_GT) > 0.5 within the first 30 evaluable
                frames after a GT reappearance, else 0

Pre-registered decision rule:
    if vanilla DEV-hard-stratum POR > 0.80:
        tighten difficulty using GT attributes (longer gaps, more distractors,
        smaller objects, more same-category neighbours, longer videos)
        until vanilla DEV-hard-stratum POR <= 0.70

This must pass before TRAIN/DEV/TEST splits can be locked.

## 13. VERIFIED CHECKPOINT - 2026-08-23

This section is the current project state and supersedes Sections 9-12 above
where they conflict.

### Current phase
Week 2 - data/headroom/split preparation. IdentityGate itself is NOT implemented.

### Last verified baseline
Git parent before this milestone: 2dcd019.
SAM 3.1 VOS path remains operational on the RTX 4080 SUPER 16 GB.

### VERIFIED - EXP013
Full event-bearing MOSEv2 train attribute scan completed:
- 1,691 videos
- 3,237 event-bearing tracks
- 4,469 qualifying disappearance/reappearance events
Artifact: experiments/EXP013_attrs.csv

### VERIFIED - EXP014
The EXP011 sync_video heuristic was audited.
- 82 events were swept in by the original video-level flag.
- 78 events actually satisfy the event-level >=5 reappearances within +/-10 frames rule.
- 4 events in yp6926lu were false inclusions from video-level flagging.
- Five synchronized-reappearance clusters were visually audited.
- The historical interpretation whole-scene occlusion is NOT supported.
Most audited clusters are consistent with camera-induced out-of-view/re-entry;
5hcafebs remains mixed/ambiguous between crowd occlusion and out-of-view.

PROVISIONAL RESEARCHER-LED DECISION:
Do not use synchronized reappearance as an exclusion criterion.
Retain it only as a diagnostic attribute unless a defensible event taxonomy is
established independently.

### VERIFIED - EXP015
Existing EXP012 GT-only difficulty levers were evaluated over the full EXP013 pool.
- 31 non-redundant rule combinations tested.
- 17 retain at least 210 videos.
- Raw sample-size blocker from the old 291-video convenience pool is resolved.

Chosen HEADROOM-DEVELOPMENT candidate, not a final split rule:
obj_size < 0.005 AND n_frames >= 100

Before historical EXP009 exclusion:
- 258 videos
- 333 tracks
- 732 events

After excluding all 40 EXP009 exploratory videos:
- 252 fresh videos
- 315 tracks
- 705 events

### VERIFIED - EXP016 MANIFEST
Frozen headroom-development manifest:
- seed 42
- 40 randomly sampled fresh videos
- 60 eligible hard-stratum tracks
- 112 eligible hard-stratum events
- 120 total GT events in the selected videos
- EXP009 overlap = 0

Status: HEADROOM_DEVELOPMENT_ONLY_NOT_FINAL_DEV.

The exact eligible (video, object_id, reappear_frame) keys are frozen in
experiments/EXP016_headroom_manifest.json.

### VERIFIED - EXP016 SANITY
Three-shortest-video B0 runtime sanity passed:
- 3 videos
- 7 manifest hard events expected
- 7 hard events scored
- max peak VRAM 5.67 GB
- runtime 0.7 min
- no missing/extra eligible-event assertion

The sanity POR values are NOT scientific results.

### OPEN / BLOCKING BEFORE FINAL SPLIT LOCK
- Contaminated-video IDs from historical P2 work: UNKNOWN.
- Cache-used-video IDs from historical P2 work: UNKNOWN.
- Therefore final TRAIN/DEV/TEST locking is NOT allowed yet.
- F1 endpoint contradiction remains unresolved.
- VOS-path obj_ptr availability remains unverified.
- Per-object memory-write blocking remains unimplemented/unverified.
- Native memory admission semantics remain unresolved.
- ITR denominator remains unresolved.
- Closed-loop conformal guarantee scope remains unresolved.

### NEXT EXACT SCIENTIFIC ACTION
Commit this verified implementation/provenance milestone, then run EXP016 full
from a clean Git commit.

Headroom checkpoint:
POR_hard_w30 <= 0.70 -> headroom requirement passes.
POR_hard_w30 > 0.70 -> follow the predeclared GT-only tightening procedure;
do not select individual videos by B0 performance.

## 14. VERIFIED CHECKPOINT - EXP016 FULL - 2026-08-24

### VERIFIED - EXP016 FULL B0 HEADROOM

Vanilla SAM 3.1 was evaluated on the frozen EXP016 headroom-development
manifest from clean git commit:

37e3ed0b3beec010c43429bb336042d7d85dcd34

Run integrity:
- 40 / 40 selected videos completed
- 120 total GT reappearance events scored
- 112 / 112 frozen hard-stratum events scored
- no missing or extra hard-event keys
- git_dirty = False
- runtime = 19.9 minutes
- max peak VRAM = 12.38 GB on RTX 4080 SUPER 16 GB

Hard-stratum B0 POR:
- POR15 = 0.5804
- POR30 = 0.6071
- POR60 = 0.6071

All selected-video events:
- POR15 = 0.6083
- POR30 = 0.6333
- POR60 = 0.6333

ITR diagnostic:
- theft events = 2
- tracks with theft = 1

HEADROOM RESULT:
POR_hard_w30 = 0.6071 <= 0.70.
The predeclared headroom requirement PASSES.

OBSERVED:
Only three additional hard events recover between windows 15 and 30;
no additional hard event recovers between windows 30 and 60.

The candidate rule
    obj_size < 0.005 AND n_frames >= 100
has therefore passed the headroom-development checkpoint.

IMPORTANT:
This does NOT yet constitute a locked TRAIN/DEV/TEST split.
Historical contaminated-video and cache-used-video exclusion IDs remain UNKNOWN,
so final DEV/TEST construction remains blocked until those exclusions are
resolved or their provenance is formally adjudicated.

### NEXT EXACT ACTION

Resolve the historical contaminated/cache-used exclusion sets, then construct
candidate TRAIN/DEV/TEST manifests under the passed GT hard-stratum rule and
verify all split-size and TEST-event-count requirements before locking.

## 15. PRE-SPLIT PROVENANCE DECISION - DEVELOPMENT EXCLUSIONS

### Historical P2 reference

v4 contained the requirement:
"P2 contaminated 200-video split and P2 600 cache videos never enter DEV or TEST."

During v5 consolidation this wording was generalized to the broader rule that
contaminated/cache-used videos never enter DEV or TEST. The removal of the P2
specifics was not an independently approved methodological decision.

A repository-history and local provenance search found no surviving definition,
manifest, script, branch, repository, or video-ID set grounding the P2 200/600
reference.

Status:
P2-specific IDs = UNKNOWN / UNGROUNDED IN SURVIVING EVIDENCE.

Decision:
- Do not fabricate replacement P2 IDs.
- Do not claim that the historical P2-specific exclusion has been verified.
- Preserve the general anti-contamination invariant.
- If authentic P2 IDs are recovered before TEST lock, union them into the
  development exclusion set before final split lock.

### VERIFIED IdentityGate development exposure

Repository evidence identifies 81 unique MOSEv2 videos exposed to model outputs
or model-derived development results before final split lock:

- 1 early model-output video from EXP007/EXP008: 6042d64a
- 40 EXP009 exploratory videos subsequently evaluated in EXP010/EXP011
- 40 EXP016 B0 headroom-development videos
- EXP009/EXP016 overlap = 0

EXP014 adds no new unique videos because all of its audited videos are already
members of the EXP009 exploratory set.

These 81 videos are prohibited from final DEV and TEST.

GT-only dataset characterization and candidate construction (EXP006, EXP013,
EXP015) are not classified as model-output exposure and therefore do not by
themselves exclude the corresponding videos.

Canonical exclusion artifact:
experiments/development_exclusions_v1.json

### Split-lock assertion

Before final DEV/TEST lock:

    intersection(DEV, development_exclusions) == empty
    intersection(TEST, development_exclusions) == empty

This assertion is mandatory.

## 16. EXP017 SPLIT LOCK - 2026-08-24

### LOCKED hard-stratum split

Final GT-defined hard rule:
obj_size < 0.005 AND n_frames >= 100

All 258 eligible hard-pool videos are assigned exactly once:
- TRAIN: 100 videos / 260 hard events
- DEV: 40 videos / 129 hard events
- TEST: 118 videos / 343 hard events

TEST satisfies the >=200-event requirement and lies in the preferred
300-500-event range.

All 46 development-exposed videos that intersect the hard pool are confined
to TRAIN. DEV and TEST have zero overlap with the 81-video development
exclusion manifest.

### LOCKED full-event-bearing distribution TEST

F5 full-distribution reporting is operationally defined before any gate/test
prediction is observed as a seed-42 random sample of 118 held-out videos from
the 1,691 MOSEv2 event-bearing videos, without hard-stratum attribute filtering.

- 118 videos
- 295 qualifying reappearance events
- overlap with hard TEST: 9 videos
- combined unique TEST universe: 227 videos
- zero overlap with TRAIN, DEV, or development exclusions

The hard/full TEST overlap is permitted because both cohorts are held out.
The secondary cohort is called the full event-bearing distribution; it is not
claimed to represent the 1,975 MOSEv2 videos with no qualifying POR event.

### TEST handling correction

The v5 Week-2 schedule mentions signals_test_locked.parquet, but prediction-
derived TEST signal caching would conflict with the stronger locked rules that
TEST remains untouched until Week 7 and is touched exactly once.

Therefore:
- pre-freeze prediction-derived caching is TRAIN/DEV only;
- TEST IDs and split-construction GT metadata are frozen now;
- prediction-derived TEST caching/evaluation occurs only during the single
  locked Week-7 TEST execution.

No split may be rerolled or changed in response to future model results.

## CURRENT CHECKPOINT - EXP023 TRAIN18 COMPLETE - 2026-08-31

This checkpoint supersedes earlier `NEXT EXACT ACTION` text where it conflicts.
Historical checkpoints above are retained unchanged for provenance.

### VERIFIED

- Core substrate remains frozen SAM 3 VOS/PVS under Amendment A1:
  `build_sam3_video_model()` with `facebook/sam3/sam3.pt`.
- EXP023 TRAIN18 production primitive cache completed on the frozen 18-video
  TRAIN relational-development scope.
- Production result commit: `801d8ea`.
- 18 / 18 videos completed.
- 13,524 primitive data rows were produced.
- DEV videos touched: 0.
- TEST videos touched: 0.
- Merged artifact SHA256:
  - `primitives.csv`: `c13fddfcc7fe422893e5cfed86100d9a8407abb0421c6296f5ed168a34e92feb`
  - `index.csv`: `9e73e27506d540aab63dd55e2a07a4b5e7e5c8e18e54b393e0891c4d1fa36db0`
  - `summary.json`: `145889fc5e26c021e8e083956baa7a1b01333357f137ce3c4da42c33448946ce`
  - run log: `e3cab3898963f559397da9b7bb35213338273fb05afbe2977ad6fa9b23e56e15`
- EXP018 through EXP023 are backfilled in `EXPERIMENT_REGISTRY.md`.
- `docs/RESEARCH_HISTORY.md` is the recovered canonical chronology document.

### OPEN / NOT ESTABLISHED

- EXP001 through EXP012 registry backfill remains OPEN / NON-BLOCKING.
- Whole-scene final labels remain OPEN pending the independent pixel cross-check
  and predeclared manual audit.
- Missing-identity fallback remains OPEN; `pointer_valid` is a validity mask,
  not silently a predictive feature.
- Features 4, 8, 9, and 11 remain DEFERRED; Feature 7 final definition remains OPEN.
- Matched-write-rate denominator remains OPEN.
- ITR denominator remains OPEN.
- No IdentityGate model has been trained.
- B3-S > B2 and B3-R > B3-S are NOT ESTABLISHED.
- Closed-loop write blocking is NOT YET VERIFIED.
- Final TEST remains untouched.

### NEXT EXACT ACTION

Run the first TRAIN-only incremental utility analysis preparation for
B2 vs B3-S vs B3-R using the EXP023 primitive cache.

The analysis must:
- use dual outcomes: drift (`target_iou < 0.3`) and theft
  (`max_other_iou > 0.5`);
- evaluate identity utility first on the valid-pointer conditional subset;
- keep `pointer_valid` as a validity mask rather than a predictive feature;
- treat video as the statistical cluster;
- make no DEV or TEST access;
- make no claim from identity-margin sign alone.

Before writing the analysis implementation, inspect only the cached schema,
row count, and aggregate label/validity counts. Do not dump the full CSV.
