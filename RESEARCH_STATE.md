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

## 2026-09-10 - EXP024 utility inference complete

EXECUTED / VERIFIED:
- TRAIN-only B2-core vs B3-S vs B3-R leave-one-video-out utility probe completed.
- Common complete-case pointer-valid population: 7,948 rows.
- Paired 5,000-replicate video-cluster bootstrap completed.
- B3-S and B3-R did not improve over B2-core; theft comparisons versus B2-core were worse with 95% cluster-bootstrap CIs below zero.
- Theft evidence is cluster-sparse: 143 positives from 5 videos; 16 bootstrap replicates invalid.
- DEV touched: 0.
- TEST touched: 0.

NOT CONCLUDED:
- No closed-loop SAM3 improvement has yet been demonstrated.
- No final gate comparison has yet been run.
- No final TEST evidence exists.

IMPLEMENTATION FACT NOW VERIFIED:
- propagate_in_video stores newly inferred non-conditioning output before yielding it.
- Missing non-conditioning memory entries are safely skipped by subsequent memory retrieval.
- Therefore a blocked write can be implemented without modifying frozen SAM3 by evicting the just-yielded frame from global and per-object non-conditioning memory dictionaries before requesting the next generator frame.

NEXT EXACT ACTION:
- EXP025 closed-loop deterministic write-block mechanism sanity on TRAIN-exposed data.
- Then wire B2 and produce inspectable B0-vs-B2 occlusion recovery outputs.

## 2026-09-10 - EXP025 attempt 1 off-by-one correction

EXECUTED / OBSERVED:
- Frozen EXP025 attempt 1 completed both B0 and deterministic BLOCK propagation but yielded 61 frames when 60 were intended.
- Run stopped at the post-run B0 frame-count assertion; no closed-loop mechanism conclusion was drawn.
- Failed run log SHA256: c59ca19d882f4b87d019a989d50f60edcfc519116e4d3bfda5fccf6a947cc0ac.

VERIFIED DEFECT:
- SAM3 forward processing uses an inclusive end index.
- With start_frame_idx=0 and max_frame_num_to_track=60, frames 0..60 are yielded.
- EXP025 must pass n_frames - 1 to obtain exactly frames 0..59.

STATUS:
- Engineering correction only; scientific mechanism result remains OPEN.

## 2026-09-10 - EXP025 closed-loop mechanism verified

EXECUTED / VERIFIED:
- Frozen SAM3 closed-loop memory intervention successfully executed.
- A just-yielded non-conditioning memory frame can be physically removed before the next inference step.
- Intervention-frame B0/BLOCK predictions were identical.
- Subsequent tracking masks diverged beginning at frame 2.
- Closed-loop write-block mechanism is therefore demonstrated.
- Peak VRAM remained below 6.27 GB.
- DEV touched: 0.
- TEST touched: 0.

OBSERVED:
- Deterministic blocking of every frame 1..30 slightly worsened descriptive mean target IoU by -0.0011057824.
- This stress test is not a learned/selective gate result.

OPEN:
- Final gate must make per-object, per-frame admission decisions.
- B2 closed-loop post-occlusion improvement versus B0 remains untested.

NEXT EXACT ACTION:
- Verify whether vanilla per-object singleton execution reproduces batched B0 closely enough to provide independent per-object memory banks.
- If verified, use that path for true per-object B2 write admission and occlusion recovery evaluation.

## 2026-09-10 - EXP026 singleton route rejected

EXECUTED / VERIFIED:
- Batched B0 versus independent per-object singleton B0 was tested on the frozen EXP026 sanity scope.
- Only 22/120 object-frame binary masks were exactly equal.
- Total disagreement was 3365 pixels; minimum mask IoU was 0.6569468268.
- DEV touched: 0.
- TEST touched: 0.

CONCLUDED:
- The frozen exact-equivalence acceptance rule failed.
- Singleton execution is rejected as the per-object gating route.
- No post-hoc equivalence threshold will be introduced.

OPEN:
- Per-object IdentityGate intervention must retain batched SAM3 execution.

NEXT EXACT ACTION:
- Inspect the batched memory write/read representation required to determine whether one packed object memory payload can be blocked without changing the other objects.

## 2026-09-10 - EXP027 attempt 1 runtime patch indentation defect

EXECUTED / OBSERVED:
- Vanilla batched B0 completed for EXP027 attempt 1.
- Runtime installation of the IdentityGate attention filter then failed before PATCHED_NO_BLOCK execution.
- Failure: IndentationError while compiling the dynamically generated replacement for _prepare_memory_conditioned_features.
- Failed run log SHA256: 869b3e40e33c29815cf253f483e7d12a32442adacc696535591b14d8c52394e4.

VERIFIED DEFECT:
- The generated if line was manually indented eight spaces after textwrap.dedent(), while the replacement position already retained the method-body indentation.
- The generated if line therefore had unexpected extra indentation.

SCIENTIFIC STATUS:
- EXP027 per-object filter mechanism remains OPEN.
- No patched no-block or selective-block result was produced.
- Frozen acceptance criteria are unchanged.

## 2026-09-10 - EXP027 attention-mask route rejected

EXECUTED / VERIFIED:
- Pinned SAM3: 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da.
- EXP027 patched NO-BLOCK execution reached completion.
- Selective memory masking failed in the active TransformerDecoderLayerv2.forward_pre path because memory_key_padding_mask is required to be None.
- DEV touched: 0.
- TEST touched: 0.

CONCLUDED:
- The memory_key_padding_mask Track-B route is rejected.
- No scientific B2/B3 performance conclusion follows from EXP027.

VERIFIED PINNED RUNTIME FLAGS:
- use_memory_selection=True
- non_overlap_masks_for_mem_enc=False
- num_maskmem=7
- memory_temporal_stride_for_eval=1
- max_obj_ptrs_in_encoder=16
- compile_all_components=False
- model-build CUDA allocation observed: 3.491 GB; this is not a tracking peak.

OPEN:
- ROWWISE hybrid feasibility is not yet established because pinned SAM3 uses memory selection.
- Exact interaction between per-object filtered output dictionaries and the existing memory-selection policy must be preserved before using ROWWISE for B2.

NEXT EXACT ACTION:
- Inspect only the pinned use_memory_selection branch and valid_indices construction, then either implement ROWWISE control or reject it.

## 2026-09-10 - EXP028 attempt 1 scalar BFloat16 hash defect

EXECUTED / OBSERVED:
- Frozen commit: 1d50e41 EXP028: freeze rowwise hybrid sanity.
- EXP028 attempt 1 entered VANILLA_BATCHED_B0 and failed at frame 1 before any ROWWISE control execution.
- Failure: tensor_hash attempted a direct torch.uint8 view of a 0-D BFloat16 eff_iou_score tensor.
- No EXP028 scientific comparison result was produced.
- DEV touched: 0.
- TEST touched: 0.

ENGINEERING FIX:
- Preserve tensor dtype and exact bit representation.
- Reshape scalar tensors to a one-dimensional buffer before torch.uint8 byte view.
- Scientific configuration and frozen acceptance criteria are unchanged.

REPRODUCIBILITY NOTE:
- Immediately after the failed run, the reported log SHA256 was c440d2ba922658f4a789c890087a038625b9727356c225b2f61eee8acb11cf70.
- Before archival, the same path had SHA256 9d92651bf10628699d0e44e66785e88b00da981a5fe75cf48e5a8b8054a9e8b1.
- Cause of the hash change is UNKNOWN.
- The archived current artifact is authoritative for the preserved file.

## 2026-09-10 - EXP028 rejects ROWWISE hybrid

EXECUTED / VERIFIED:
- EXP028 completed normally with status ROWWISE_CONTROL_FAIL.
- Full-memory control performed 59 row recomputations with zero memory omissions.
- Binary masks matched vanilla for 79/120 object-frame rows.
- Tracked signals matched for 65/120 rows.
- Global eff_iou_score matched for 58/60 frames.
- Selective blocking was not executed.
- DEV touched: 0.
- TEST touched: 0.

CONCLUDED:
- B=1 rowwise memory-fusion recomputation is not behaviorally equivalent to vanilla batched B0.
- The frozen exact-equivalence criterion failed.
- ROWWISE hybrid is rejected.
- No B2/B3 performance conclusion follows.

NEXT:
- Use the already-verified EXP025 whole-frame physical write-block mechanism as the implementation fallback.
- Append and freeze the resulting frame-level intervention amendment before closed-loop gate experiments.

## AMENDMENT A3 - FRAME-LEVEL PHYSICAL WRITE INTERVENTION - 2026-09-10

LOCK CANDIDATE:
- EXP025 whole-frame physical eviction is the closed-loop intervention.
- EXP026 singleton, EXP027 attention-mask, and EXP028 rowwise routes are
  rejected and will not be relaxed or retuned.
- Object-level gate scores are aggregated by a fixed ALL-SAFE rule:
  frame_score = minimum tracked-object admission score.
- The physical intervention candidate is one non-conditioning frame.
- Conditioning/prompt frames are never blocked.
- The final matched-write-rate denominator/reference/control remains OPEN per
  THESIS_RULES open item 6 and is NOT frozen by A3.
- Direct matching to vanilla B0 physical admission would force no blocking
  because B0 retains every eligible frame under this mechanism.
- A separate outcome-independent matched-budget protocol must be frozen before
  headline B1/B2/B3/B5 matched-rate comparisons.
- Full write-rate sweep curves remain mandatory.
- Vanilla B0 remains the ungated practical reference.
- The final method must be described as frame-level physical memory-write
  admission derived from object-level signals, not per-object physical write
  blocking.
- Research question, POR@30, video-clustered bootstrap, +8 pp practical-effect
  criterion, J&F protection, whole-scene control, frozen SAM3, and one-touch
  TEST remain unchanged.
- Full text: docs/AMENDMENT_A3_FRAME_LEVEL_WRITE_INTERVENTION.md

NEXT:
- Freeze Amendment A3 before using the whole-frame fallback in closed-loop gate sanity experiments.
- Resolve and freeze the matched-budget protocol separately before headline matched-rate DEV/TEST comparisons.

## 2026-09-12 - EXP029 B2-core development weights trained

IMPLEMENTED / EXECUTED / VERIFIED:
- Actual B2-core neural gate weights were trained on TRAIN only.
- Five currently verified quality/temporal features were used.
- Two failure-typed MLP heads were trained: drift and theft.
- Combined learned parameter count: 4,994.
- DEV touched: 0.
- TEST touched: 0.
- Model SHA256: 6160a6be9da2058808c16182abe03443014806273df46fe59a86411ffc869ecf.

INTERPRETATION:
- EXP029 establishes a reusable learned B2-core development model.
- Training loss decreased for both heads, but this is not evidence of
  generalization or tracking improvement.
- Final B2 remains OPEN because deferred/open feature definitions and the final
  dual-head admission composition are not yet frozen.

NEXT:
- Build EXP030 TRAIN-only closed-loop learned-gate sanity using frozen EXP029
  weights, A3 ALL-SAFE frame aggregation, and the verified EXP025 physical
  whole-frame eviction mechanism.

## 2026-09-12 - EXP030 learned gate controls closed-loop writes

IMPLEMENTED / EXECUTED / VERIFIED:
- Frozen EXP029 B2-core neural weights were evaluated live inside frozen SAM3 tracking.
- Live quality/temporal features drove drift/theft failure-head probabilities.
- Development-only object-safe scores were aggregated with A3 ALL-SAFE.
- BLOCK decisions used the verified EXP025 whole-frame physical eviction mechanism.
- 55/59 eligible non-conditioning frames were blocked at the pre-fixed tau=0.5.
- Every blocked entry existed before eviction and was absent afterward.
- The first blocked-frame prediction matched B0, and later masks changed beginning at frame 3.
- DEV touched: 0.
- TEST touched: 0.

OBSERVED / NOT A PERFORMANCE CONCLUSION:
- B0 descriptive visible-row mean target IoU: 0.8243956364947009.
- gated descriptive visible-row mean target IoU: 0.8062881782959369.
- delta: -0.018107458198764026.
- The negative TRAIN-exposed delta is retained.
- The 6.78% admit rate demonstrates that an uncontrolled threshold comparison is not scientifically interpretable.

NEXT:
- Freeze the neutral matched-budget protocol before headline B1/B2/B3 comparisons.
- Then run broader closed-loop development evaluation with full write-rate curves and video-clustered inference.

## 2026-09-12 - Amendment A4 matched-rate protocol frozen

LOCKED / VERIFIED:
- Amendment A4 frozen at commit e6164a5.
- Matched-write-rate protocol previously OPEN under A3 is now RESOLVED.
- B0 is the ungated practical reference, not the matched-budget reference.
- Headline gated comparisons use an outcome-blind common DEV rate with absolute pooled write-rate tolerance 0.02.
- Neutral matched-budget controls copy each signal methods admitted-frame count exactly per video.
- Full write-rate curves remain mandatory.
- TEST thresholds are not retuned.
- POR@30 inference remains paired video-clustered BCa bootstrap.
- Minimum practically important hard-set POR effect remains +0.08.

OPEN:
- ITR denominator.
- F1 endpoint/co-primary contradiction.
- Missing-identity fallback.
- Deferred final feature definitions.
- Final gate/calibration choices.

NEXT:
- Implement the broader closed-loop development evaluator for B1/B2/B3-S/B3-R under A3 and A4 without touching TEST.

## 2026-09-12 - EXP031 B3 development weights trained

IMPLEMENTED / EXECUTED / VERIFIED:
- Actual B3-S and B3-R neural gate weights were trained on TRAIN only.
- Both variants use the same 7,948 pointer-valid complete identity rows.
- pointer_valid remains a routing and availability mask, not a predictive feature.
- B3-S learned parameter count: 5,122.
- B3-R learned parameter count: 5,250.
- DEV touched: 0.
- TEST touched: 0.
- Model SHA256: 235076b86aa0975b3ec624aae703575fd4f030381b6af4008c3808f2db71847a.

INTERPRETATION:
- Actual identity-augmented development models now exist.
- Low TRAIN loss is not held-out evidence.
- EXP024 held-out utility results remain the current evidence about incremental identity discrimination.
- B3-S greater than B2 and B3-R greater than B3-S remain NOT ESTABLISHED.

OPEN:
- Missing-identity deployment fallback must be frozen before B3 closed-loop evaluation.
- Final B3 calibration and thresholds remain unfrozen.

NEXT:
- Freeze an outcome-independent missing-identity routing rule, then wire B3-S and B3-R into the A3 frame-level closed-loop evaluator.

## 2026-09-12 - Amendment A5 missing-identity routing frozen

LOCKED / VERIFIED:
- Amendment A5 frozen at commit 9fb8464.
- Missing-identity routing for B3-S/B3-R is now RESOLVED.
- pointer_valid is an availability/routing mask only and is not a predictive feature.
- B3-S routes to B2 when self identity is unavailable.
- B3-R routes hierarchically: B3-R -> B3-S -> B2 as relational/self identity becomes unavailable.
- Single-object B3-R routes to B3-S when self identity is available.
- Identity missingness never removes an object from A3 ALL-SAFE frame aggregation.
- Unexpected required identity numeric failure must STOP rather than silently fallback.
- DEV outcomes and TEST outcomes do not participate in routing.

NOT CONCLUDED:
- B3-S improves over B2.
- B3-R improves over B3-S.
- Identity improves closed-loop tracking.

OPEN:
- Final B2/B3 feature completion.
- Final dual-head composition/calibration.
- B2-core signal missingness handling.
- ITR denominator.
- F1 endpoint/co-primary contradiction.

NEXT:
- Resolve the exact B1 rule from the canonical repository record, then implement B1 and the unified B1/B2/B3-S/B3-R A3/A4 closed-loop development evaluator without touching TEST.

## 2026-09-12 - EXP032 B1 closed-loop mechanism verified

IMPLEMENTED / EXECUTED / VERIFIED:
- Frozen A6 B1 manual rule was executed closed-loop through the A3 physical memory-write intervention.
- 59 eligible non-conditioning frames: 11 admitted and 48 blocked at mechanism-sanity tau_B1=0.5.
- Every blocked frame was present before eviction and absent afterward.
- frames_already_tracked bookkeeping remained retained.
- The first blocked-frame prediction equaled B0 and downstream predictions changed starting at frame 3.
- DEV touched: 0.
- TEST touched: 0.
- B1 peak VRAM was 6.166836261749268 GB.

INTERPRETATION:
- B1 is now implemented and physically controls frozen SAM3 memory writes.
- The descriptive TRAIN-exposed IoU delta of 0.003185102237101445 is not performance evidence.
- Final B1 threshold selection remains controlled by A4 DEV write rate only.

NEXT:
- Implement the unified B1/B2/B3-S/B3-R closed-loop development evaluator under A3, A4, A5, and A6 without touching TEST.

## 2026-09-12 - EXP033 unified closed-loop gate mechanism verified

IMPLEMENTED / EXECUTED / VERIFIED:
- Unified B1, B2-core, B3-S, and B3-R closed-loop gate execution passed on TRAIN-exposed video 0442a954.
- All four variants physically controlled the A3 SAM3 memory-write intervention.
- B3-S used its identity model on 105 live object-rows.
- B3-R used its relational identity model on 104 live object-rows.
- Invalid/unavailable identity routed according to A5 or failed closed when base features were non-finite.
- Fresh DEV touched: 0.
- TEST touched: 0.
- All observed peak VRAM values remained below 6.3 GB.

INTERPRETATION:
- The unified gate mechanism is operational on frozen SAM3.
- The observed TRAIN-only IoU deltas are descriptive only and are not evidence that any gate is better.
- A4 matched-write-rate development evaluation is still required before comparative conclusions.
- EXP029 remains B2-core rather than final B2.

COVERAGE NOTE:
- B3-R relational routing was live-exercised on this two-object video.
- The single-object B3-R -> B3-S fallback was CPU-smoke-tested but not live-exercised in EXP033.

NEXT:
- Close the remaining pre-DEV gate/split protocol items, then build the DEV-ready A4 matched-write-rate evaluator without touching TEST.


## 2026-09-12 - EXP034 whole-scene pixel sanity verified

IMPLEMENTED / EXECUTED / VERIFIED:
- Frozen EXP034 commit b8cd9c37bb7c37ec6b430d56c238c0a6a275e9a1 ran on development-exposed video 0nrb9vzx.
- One EXP019 candidate scene was processed end-to-end.
- camera-cut positives: 0; global-MAD positives: 0; pixel-confirmed: 0; manual-adjudication-required: 1.
- Peak allocated VRAM: 0.1648869514465332 GB; runtime: 7.353137016296387 s.
- SAM predictions: 0; gate predictions: 0; POR outcomes: 0; TEST gate evaluation: 0.

INTERPRETATION:
- EXP034 engineering sanity passes.
- The pixel-negative sanity outcome does not justify threshold changes.
- Final whole-scene labels remain OPEN pending full cross-check, manual disagreements, and frozen 30-scene audit.

NEXT:
- Commit this sanity record, then run the unchanged frozen full EXP034 pixel cross-check.


## 2026-09-12 - EXP034 full pixel cross-check complete

EXECUTED / OBSERVED / VERIFIED:
- Frozen EXP034 full pixel cross-check completed at commit 98f613ab7efdaf25f868f79b44e948771a6db595.
- 1281 videos and all 2179 EXP019 collapsed annotation-side candidate scenes were processed.
- camera-cut positives: 322.
- global-MAD positives: 226.
- pixel-confirmed scenes: 361.
- manual-adjudication-required scenes: 1818.
- Frozen audit sample rows with pixel statistics: 30.
- Peak allocated VRAM: 0.1648869514465332 GB.
- Runtime: 5693.0291039943695 seconds.
- SAM predictions: 0; gate predictions: 0; POR outcomes: 0; TEST gate evaluation: 0.

INTERPRETATION:
- Automated whole-scene pixel preprocessing is complete.
- Thresholds remain frozen after full-output inspection.
- Final whole-scene labels remain OPEN pending manual disagreement adjudication and the fixed 30-scene manual audit.

NEXT:
- Generate deterministic visual-review packs for audit30 and manual-adjudication scenes.


## 2026-09-19 - EXP037 matched-rate evaluator integration sanity

EXECUTED / OBSERVED / VERIFIED:
- Frozen EXP037 evaluator executed from commit 020705b32f92325c6f3d6a38d0bf122a01cfcc3d.
- TRAIN-exposed video 0442a954, first 60 frames, object IDs [1, 2].
- B0, B1, B2, B3-S, and B3-R completed successfully.
- Each gated variant had 59 eligible non-conditioning frame-write opportunities.
- B1: 11 admitted / 48 blocked, write rate 0.1864406779661017.
- B2: 4 admitted / 55 blocked, write rate 0.06779661016949153.
- B3-S: 18 admitted / 41 blocked, write rate 0.3050847457627119.
- B3-R: 16 admitted / 43 blocked, write rate 0.2711864406779661.
- Physical write-block integrity passed for every gated variant.
- POR@30 endpoint was exercised with 1 qualifying event; B0/B1/B2/B3-S/B3-R each recovered that event.
- Maximum observed peak allocated VRAM: 6.266373634338379 GB.
- Frozen SAM3 commit: 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da.
- Fresh final DEV touched: 0.
- TEST touched: 0.

INTERPRETATION:
- Unified closed-loop gate intervention, write-rate accounting, and POR@30 scoring are integrated and executable.
- The single qualifying event is sufficient for endpoint integration sanity only.
- POR@30 = 1.0 for all variants in this sanity run is not comparative performance evidence.
- Tau 0.5 remains an engineering sanity threshold, not an A4 matched-rate operating point.
- No statistical inference or thesis performance conclusion is supported by EXP037.

ARTIFACTS:
- operating_points.csv SHA256: 41fb68238994d3fb0e6df9c4a117dac66d1a6c530f051af61d09ba684959ab10
- por30_events.csv SHA256: 6231647b9528d0774120bbe15e77d93629aaccbcac47243a1adb2b98b7c36843
- summary.json SHA256: f182356c6d8ec6f758d9a55a7b33b52dfbe216d4fc486b0497736e6b7323fe9d
- ignored run log SHA256: 54990e40e957891f17ca6a3eede2b9c524b1c173bbe10256dd49abfabadb6cff

NEXT:
- Record EXP037, then continue implementation toward the full A4 matched-rate evaluator without touching fresh final DEV or TEST.

## 2026-09-19 - EXP038 A4 rate-selector sanity

EXECUTED / OBSERVED / VERIFIED:
- Frozen EXP038 selector was originally committed as 59395074dc98b38d0c1feeca18a205c48b4f52a8.
- The first execution failed before a scientific tracker result because Runner did not retain the loaded exp037 dependency.
- Minimal dependency-wiring fix was committed separately as 4a5f44fc226be88a9c074cb67b69572b22cffda3; no A4 protocol, threshold, target-order, model, or data-scope rule changed.
- A subsequent foreground execution was manually interrupted and produced no final scientific result.
- The successful retry executed from commit 4a5f44fc226be88a9c074cb67b69572b22cffda3.
- Scope: TRAIN-exposed video 0442a954, first 60 frames, object IDs [1, 2].
- Variants: B1, B2, B3-S, B3-R.
- First A4 target tested: 0.5.
- B1 matched at tau 0.35 with realized write rate 0.4915254237288136.
- B2 matched at tau 0.1875 with realized write rate 0.5084745762711864.
- B3-S matched at tau 0.2 with realized write rate 0.5084745762711864.
- B3-R matched at tau 0.25 with realized write rate 0.4915254237288136.
- Absolute write-rate error for every variant was 0.008474576271186418, within the locked A4 tolerance of 0.02.
- Total executed tau points: 49.
- Maximum observed peak allocated VRAM: 6.266784191131592 GB.
- Frozen SAM3 commit: 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da.
- Fresh final DEV touched: 0.
- TEST touched: 0.
- Final status: EXP038_A4_SELECTOR_SANITY_PASS.

INTERPRETATION:
- The implemented A4 write-rate-only common-target selector is executable in closed loop for B1/B2/B3-S/B3-R.
- Coarse-grid selection and deterministic midpoint refinement were exercised.
- The value 0.5 is subset_common_target_not_final_r_star only.
- EXP038 cannot define final r_star because it is TRAIN-exposed and B5 is not included.
- EXP038 provides no comparative gate-performance result and no final statistical inference.

ARTIFACTS:
- config SHA256: e53a1fdce71e11bd1fff8fcd0bd8b70a5f1291e87f39df8581ab529d84be1636
- fixed script SHA256: bf81cea8e531b08d3ef432d985f4f09971d24fef51338c61896b616c88ffdb9f
- executed_tau_points.csv SHA256: db93bd1dd30698908f4f777ceee043e4cefb3a8518c239253b11c36049e5b1cb
- summary.json SHA256: 1f313044e894f0533ca669a90544ad7eb324db92e40440b556a2fa3271bcf0f6
- target_selection.csv SHA256: b306642d55b3cb4d1a59b14f6275aec00b8a6c87808932852e8a9ef13c5275a6
- ignored original failed-run log SHA256: c91d498e74bce5d3335a18ac6cb87eb394276bdc01ae8f52118cfc00e7d06e4c
- ignored interrupted fixed-run log SHA256: 9b1e0c638351003f469ae9614a4a2ce3fccc83560184feede23e1e136a32f3c1
- ignored successful retry log SHA256: 6e8ba8cc0492bcf555d4e709ae16c93e33a40d604ee891c3fd41c8ab5a9ddb24

NEXT:
- Record EXP038 artifacts, then continue toward the remaining final-evaluation blockers without touching fresh final DEV or TEST.

## EXP039 - B5 DMS-lite write-side comparator sanity

STATUS:
- EXECUTED / VERIFIED PASS.
- TRAIN-exposed engineering sanity only.
- No comparative performance evidence.
- Fresh final DEV touched: false.
- TEST touched: false.

FROZEN IMPLEMENTATION:
- Commit: 4b8b54f058d2187e665d4ea5c5c44015933a2d8a
- Config SHA256: e48a09e27bc6807b7d84a7eb7b517171ace93fbdba339fede109d6a95ee5f015
- Script SHA256: a5380f390ec3559b6a44abc14927d99ed4db594cd1056096d68dcb77ce3d8635
- A8 SHA256: 199b5630fb48bf3f1285683c3b1075476beb2e67d974b4d8199e1f4a7df80252
- SAM3 commit: 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da

SCOPE:
- TRAIN-exposed video: 0442a954.
- Frames: 60.
- Object IDs: [1, 2].
- Variant: B5 DMS-lite write-side comparator.
- B5 is not claimed to reproduce official SAM3-DMS.

OBSERVED:
- Final status: EXP039_B5_DMS_LITE_SANITY_PASS.
- First matched sanity target: 0.5.
- Selected tau: 0.775.
- Realized physical write rate: 0.4915254237288136.
- Absolute rate error: 0.008474576271186418.
- Midpoint refinements: 2.
- Executed tau points: 13.
- Formula rows checked: 1534.
- Actual nonfinite/fail-closed rows observed: 0.
- Synthetic positive-formula check: PASS.
- Synthetic nonpositive-occurrence-zero check: PASS.
- Synthetic NaN FAIL_CLOSED check: PASS.
- Synthetic Inf FAIL_CLOSED check: PASS.
- Patched EXP033 NaN FAIL_CLOSED path: PASS.
- Exact B5 formula checks on executed rows: PASS.
- A3 frame-min aggregation checks: PASS.
- A3 action-rule checks: PASS.
- Physical block-integrity checks: PASS.
- Maximum observed peak allocated VRAM: 6.266374588012695 GB.

INTERPRETATION:
- Frozen A8 B5 scoring is executable with the existing frozen SAM3 outputs.
- B5 operates through the frozen A3 whole-frame physical write intervention.
- Frozen A4 write-rate-only selection can obtain a matched operating point for B5 in this TRAIN-exposed sanity scope.
- The 0.5 target and tau 0.775 are sanity-only and are not final r_star.
- No final DEV, TEST, comparative gate-performance, or statistical conclusion is supported by EXP039.

ARTIFACTS:
- summary.json SHA256: 0849204aab608f9e8424d002c00e591847f6576e0c80f44368fc56a95ac8f8e7
- contract_checks.json SHA256: 5f46691162cc8e7f0da7599283c370afd86ece2aa7bb22a70fd278b77667e5a7
- target_selection.csv SHA256: 3ff941aff44317c208b950622d91e55b22a7d368e8b935600d7e90e26db82e3e
- executed_tau_points.csv SHA256: 5937845489dd414a7fdf60bd28650ce99eb280c0c9638f21ce9e1ac5fb1ba2c9
- ignored execution log SHA256: 26e193f69cc6a0e9f64c66cf9f933e229cf975517da7299393eb35af673082a5

NEXT:
- Close EXP039 provenance, then continue the remaining final-evaluation blockers without touching fresh final DEV or TEST.

## EXP036 - Whole-scene manual adjudication complete

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Frozen EXP034 disagreement population fully manually adjudicated.
- Fresh final DEV touched: false.
- TEST touched: false.
- Separate 3-scene blinded-audit replacement requirement remains OPEN.

OBSERVED:
- Total disagreement scenes: 1818.
- Filled manual labels: 1818.
- Remaining manual labels: 0.
- Unique scene IDs: 1818.
- Scene-ID set matched frozen EXP034 disagreement CSV: true.
- NORMAL_OCCLUSION: 1629.
- WHOLE_SCENE: 189.
- camera_cut_or_global_scene_switch: 14.
- global_obstruction_or_scene_wide_collapse: 175.
- local_object_specific_event: 235.
- continuous_camera_motion_scene_visible: 1394.

ARTIFACTS:
- canonical review_labels.csv SHA256: 9b76ae3bc722db7c6138cb1f2b4ab567990e6921ec2be7caf6dc0d2b239e51a4
- source manual_adjudication_required.csv SHA256: 1f6fa7ad1e10662ff0333331e6d1d5a1127e89c710b0e6d9768cd206a34d63d3
- EXP036 config SHA256: 32dcd845a9bae9cfa12dd9d90d1c9578db2134bf09c7f3a5995b4f30c22b6102
- EXP036 script SHA256: 955e53b572d825fe060ba5122114a1fee6789a6946dbf5ae991c4562973bcf47
- result summary: experiments/EXP036_ws_keyboard_adjudication/final_summary.json

## EXP040 - Blinded audit replacement sample executed

STATUS:
- EXECUTED / VERIFIED.
- Prospective replacement rule was frozen before replacement IDs were revealed.
- Manual final-audit labeling is still PENDING.
- Fresh final DEV touched: false.
- TEST touched: false.

OBSERVED:
- Original audit size: 30.
- Compromised pilot-only scenes: 3.
- Uncompromised original scenes retained: 27.
- Replacement eligible population: 2149.
- Replacement seed: 34035.
- Replacement scenes: 5r6uxga7:WS001, of2thxpc:WS001, 0fc00006:WS001.
- Final valid blinded-audit population: 30.
- Replacement selection used pixel outcomes: false.
- Replacement selection used SAM/gate outcomes: false.

ARTIFACTS:
- experiments/EXP040_ws_audit_replacement/final_audit_source.csv
  SHA256: fe0bc24767fc0a2177f37cd8b1fe68ecf32fadde363df7a8cc31a5e750d009be
- experiments/EXP040_ws_audit_replacement/replacement_manifest.json
  SHA256: eb975d4fff3eefb8345360c495c35cc128f4fe83a918215b1afc7da7e1e527af

NEXT:
- Record blinded manual labels for all 30 valid audit scenes.
- The final whole-scene agreement estimate remains OPEN until those labels are complete.

## EXP040 - Final blinded whole-scene audit complete

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Final valid blinded audit size: 30.
- Agreement: 27/30 = 0.900000.
- Mismatches: 3.
- No acceptance threshold was preregistered.
- EXP034 thresholds remain unchanged.
- Fresh final DEV touched: false.
- TEST touched: false.

OBSERVED:
- Manual NORMAL_OCCLUSION: 23.
- Manual WHOLE_SCENE: 7.
- Automatic NORMAL_OCCLUSION: 24.
- Automatic WHOLE_SCENE: 6.
- NORMAL_OCCLUSION -> NORMAL_OCCLUSION: 22.
- NORMAL_OCCLUSION -> WHOLE_SCENE: 1.
- WHOLE_SCENE -> NORMAL_OCCLUSION: 2.
- WHOLE_SCENE -> WHOLE_SCENE: 5.
- Mismatch scene IDs: la1w5eyj:WS003, n0d05tlz:WS001, zpjsz5f5:WS002.

ARTIFACTS:
- review_labels.csv SHA256: a73ee5ac1b7a327fef82c498c5d6ce89f8a621bfc9da288c31f116b389847724
- final_audit_source.csv SHA256: fe0bc24767fc0a2177f37cd8b1fe68ecf32fadde363df7a8cc31a5e750d009be
- EXP034 per_scene.csv SHA256: aefb19804adf093dfc94cc80aa50dc53e69df83bab2c2ab8ba5d6acd603acd2f
- experiments/EXP040_ws_audit_replacement/final_audit_result.json

NEXT:
- Freeze final whole-scene labels from the already-frozen EXP034 automatic-confirmed population plus completed EXP036 manual adjudication.

## EXP041 - Final whole-scene labels frozen

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Final whole-scene candidate labels are frozen.
- Population: 2179 scenes.
- WHOLE_SCENE: 550.
- NORMAL_OCCLUSION: 1629.
- EXP034 auto-confirmed contribution: 361.
- EXP036 manual-adjudicated contribution: 1818.
- Fresh final DEV touched: false.
- TEST touched: false.
- Whole-scene thresholds changed: false.

ARTIFACTS:
- experiments/EXP041_ws_final_labels/final_ws_labels.csv
  SHA256: a14d9c83b0ddbc62c0cf3bae404950867259b96c773a7db491375ec74f37689e
- experiments/EXP041_ws_final_labels/whole_scene_scene_ids.json
  SHA256: 22cd1e9f208183d912c5dc69ec383911e1cf5d93b90aa2c99a5bec2d93edb13f
- experiments/EXP041_ws_final_labels/summary.json
  SHA256: 4873a1cf33a7f6d28f767ffef80ea6a4eeeca9aa9a6ec7e65f5cbbee70dd20d3

NEXT:
- Freeze final development-exclusion / DI-v1 logic before constructing the fresh final DEV/TEST split.

## EXP042 - Development exposure boundary reconstructed

STATUS:
- EXECUTED / VERIFIED.
- Recorded development-exposure union reconstructed from legacy exclusions plus historical EXP017 TRAIN and DEV.
- Final split is NOT yet authorized.
- DI-v1 is NOT yet defined/frozen.
- Fresh final DEV touched: false.
- TEST touched: false.

OBSERVED:
- Legacy exclusions: 81 videos.
- Historical EXP017 TRAIN: 100 videos.
- Historical EXP017 DEV: 40 videos.
- Historical TRAIN+DEV union: 140 videos.
- Legacy overlap with historical TRAIN+DEV: 46 videos.
- Added beyond development_exclusions_v1: 94 videos.
- Reconstructed unique exclusion boundary: 175 videos.
- EXP021 eligible videos outside boundary: 0.
- Later explicit video IDs outside boundary: 0.
- Later scope files checked: 18.
- Later scope files without explicit video IDs: 4.

OPEN:
- Scope provenance must still be resolved for:
  - experiments/EXP024_cluster_bootstrap/summary.json
  - experiments/EXP024_utility_prepare/summary.json
  - experiments/EXP029_b2core_train/summary.json
  - experiments/EXP031_b3_train/summary.json
- Historical unresolved P2 reference remains unresolved; no IDs are fabricated.
- Final exclusion lock, DI-v1, and fresh final DEV/TEST split remain pending.

ARTIFACTS:
- experiments/EXP042_development_exposure/development_exclusions_v2.json
  SHA256: c3346825babb6a9c28858cbf84022cb9719950f02fbdffd77a21c9dfd5e19240
- experiments/EXP042_development_exposure/coverage_report.json
  SHA256: 1514bc83109e3845fa0a97eae8a48b4d245f437ccef765b24170f5245f4b11cf
- experiments/EXP042_development_exposure/summary.json
  SHA256: 66403caa4308bbad7d76e0847737eb70d2a5efe75d4b79241d08ca756633629d

## EXP043 - Final known development exposure lock

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Known development-exposure boundary is locked at 175 unique videos.
- EXP024/EXP029/EXP031 provenance closure: PASS.
- No additional video IDs were supported by provenance closure.
- Historical P2 reference remains UNRESOLVED_UNGROUNDED_REFERENCE; no IDs were fabricated.
- DI-v1 defined: false.
- Final split constructed: false.
- Fresh final DEV touched: false.
- TEST touched: false.

OBSERVED:
- Known excluded videos: 175.
- Provenance train18 videos: 18.
- train18 videos outside locked boundary: 0.
- New video IDs from provenance closure: 0.

ARTIFACTS:
- experiments/EXP043_development_exposure_final_lock/development_exclusions_locked.json
  SHA256: 4f663bf3a6532fcac662e0ef9afa9af04609de1872f36bab4f0608ceca32333d
- experiments/EXP043_development_exposure_final_lock/provenance_closure.json
  SHA256: 4de201f9b8c2a7b8d68dc9aedd5d5e9349c4732ff4092160d0f6b96b600dc23e
- experiments/EXP043_development_exposure_final_lock/summary.json
  SHA256: 08d78028ecdefb780b00b241b6bba1c885f8292652b0c369ee9ac5084b0bd63e

OPEN:
- Historical P2 remains unresolved; authentic recovered IDs before TEST lock require a new amendment and affected split regeneration.
- DI-v1 and the fresh final DEV/TEST split remain pending.

NEXT:
- Freeze DI-v1 and construct the fresh final split without using model outcomes.
