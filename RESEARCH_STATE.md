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

## EXP044 - Final event pool join

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- All 4469 frozen recovery events retained.
- Whole-scene labels joined through recorded EXP019 scene membership.
- Development-exposure exclusion applied as eligibility metadata using the EXP043 175-video lock.
- Whole-scene handling remains event-level.
- DI-v1 frozen: false.
- Final split constructed: false.
- Fresh final DEV evaluated: false.
- TEST evaluated: false.

OBSERVED:
- Event-bearing videos: 1691.
- Primary eligible non-whole-scene events: 2701.
- Primary eligible videos: 1170.
- Whole-scene control eligible events: 1188.
- Whole-scene control videos: 437.
- Exposed non-whole-scene events: 528.
- Exposed whole-scene events: 52.
- Whole-scene events overall: 1240.
- Non-whole-scene events overall: 3229.
- Mixed whole-scene/non-whole-scene videos: 104.
- Known development-exposed videos in event corpus: 175.

ARTIFACTS:
- experiments/EXP044_event_pool/event_pool.csv
  SHA256: d1c8bb0121796138e6f355fbb4a03d86dcc5103bbfff7d6612e40b950105d285
- experiments/EXP044_event_pool/per_video_pool.csv
  SHA256: 957d48e553ac9887cae78fa33b05b069a7fd2df10b7ba4102d6c1569a816d977
- experiments/EXP044_event_pool/summary.json
  SHA256: 098b0af57d300824d2100619df2068b01c4baf0fd6bf9ecee3360f759f54185a

NEXT:
- Construct and freeze the outcome-independent DI-v1 component table over the eligible event/video population before any fresh split is generated.

## EXP045 - GT DI primitive census

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- GT-only primitive census over the frozen EXP044 primary pool.
- Primary events: 2701.
- Primary videos: 1170.
- SAM/gate inference performed: false.
- Pointer cosine computed: false.
- DI-v1 frozen: false.
- Hard set selected: false.
- Final split constructed: false.
- Fresh DEV evaluated: false.
- TEST evaluated: false.

OBSERVED:
- Frame-0 anchorable primary events: 2701.
- Non-anchorable primary events: 0.
- Videos with any anchorable primary event: 1170.
- Videos with all primary events anchorable: 1170.
- Videos with zero anchorable primary events: 0.
- Events with literal pre-10-frame area minimum equal to zero: 554.
- Events with less than 10 frames of available pre-gap history: 858.
- Events with zero target area at disappear_start-1: 0.

ARTIFACTS:
- experiments/EXP045_di_gt_primitives/event_gt_primitives.csv
  SHA256: 4b4cc3c49a419f777145b824ca80cae6c21a16d134ea3aa566aa728d72d7060c
- experiments/EXP045_di_gt_primitives/video_gt_primitives.csv
  SHA256: ab88e8ecc103cb8e23b48a8c91e8e4635b6bfe8b6a2d52ca7613c0e242086994
- experiments/EXP045_di_gt_primitives/summary.json
  SHA256: c4f716fd1ed4dcf3fa55d9168f01e4813f6587617f1ff2e86e0549b415e995de

NEXT:
- Freeze the exact frame-0 pointer-only distractor-pressure preprocessing after interpreting anchor/competitor coverage and the pre-gap target-size diagnostic.

## EXP046 - Frame-0 identity primitives

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Frozen implementation commit: 1860efda483586b9a1c8066b655f666f128b513f.
- GT-clean SAM 3 frame-0 object-pointer identity primitive extraction over the frozen EXP045 primary population.
- Recovery propagation performed: false.
- Gate inference performed: false.
- DI-v1 frozen: false.
- Hard set selected: false.
- Final split constructed: false.
- Fresh DEV evaluated: false.
- TEST evaluated: false.

OBSERVED:
- Primary events: 2701.
- Primary videos: 1170.
- Multi-object frame-0 videos: 378.
- Single-object frame-0 videos: 792.
- Events with a tracked frame-0 competitor: 1279 (0.47352832284339136).
- Events with no tracked frame-0 competitor: 1422 (0.5264716771566087).
- Single-object/no-competitor cases retained explicitly; no raw similarity was fabricated.
- Anchor pair rows: 17889.
- Unique competitor-defined video-object identities: 1079.
- Repeated event rows beyond unique video-object identities: 200.
- Videos with at least one repeated target-event identity measurement: 92.
- Event-vs-anchor semantics cross-check: PASS.
- Production maximum peak allocated VRAM: 10.260851383209229 GB.
- Total multi-object extraction runtime: 1170.549460887909 s.

IDENTITY-PRESSURE DIAGNOSTIC:
- Event-weighted cosine: median 0.996055483818; mean 0.99079794068.
- Unique-object-weighted cosine: median 0.996669888496; mean 0.991415897531.
- Video maximum cosine: median 0.99401023984; mean 0.989591110478.
- Video mean cosine: median 0.992960363626; mean 0.98841552755.
- Identity cosine is highly concentrated near 1 in the multi-object population.
- The same static frame-0 identity primitive can appear in multiple occlusion-event rows; event-row weighting is therefore not treated as an independent identity measurement.
- Exact DI-v1 video-level identity aggregation and structural no-competitor ranking remain OPEN until DI-v1 freeze.

ARTIFACTS:
- experiments/EXP046_frame0_identity/event_identity_primitives.csv
  SHA256: 902347f11bfcff2e91a287453e375a1895f4e42b6ffc70e42766b50ae28d4a70
- experiments/EXP046_frame0_identity/video_identity_primitives.csv
  SHA256: a77fd63405d9c13ff1e08ec27692a01a5d5f6b041675b348ea928459a4f1953c
- experiments/EXP046_frame0_identity/anchor_cosine.csv
  SHA256: 62dbf9a10e391b87e4568e7ce65a34909a9d6fb5492b66bddb74baba9a9c0a8d
- experiments/EXP046_frame0_identity/summary.json
  SHA256: 8780419efb8634cc93b250f945f02abb8bfb2bbced6102bc2373053788e9f4dd

IMPLEMENTATION:
- config SHA256: d3fcf86a29ed0e37eb7f587cd222cd618a0e37981170ef4f066af940272c358e
- script SHA256: 4edd518e6d4f4f9053c9824a7141491c1dd39bbdfc297ca03a08c93d70f120ae
- SAM checkpoint SHA256: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

NEXT:
- Assemble and freeze the outcome-independent DI-v1 video-level component table, explicitly resolving identity aggregation, structural no-competitor handling, and the pre-gap target-size definition before hard-set selection.

## EXP047 - Frozen DINOv2 visual embeddings

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Frozen implementation commit: 50f7f07905cd2ff70bafe49132a0253648f086db.
- External frozen visual-backbone embeddings extracted for all 1170 videos in the frozen primary population.
- Role: visual-diversity control for later hard-set construction only.
- Gate input: false.
- Outcome variable: false.
- Clustering performed: false.
- DI-v1 frozen: false.
- Hard set selected: false.
- Final split constructed: false.
- Fresh DEV evaluated: false.
- TEST evaluated: false.

FROZEN BACKBONE PROVENANCE:
- Backbone: DINOv2 ViT-S/14, dinov2_vits14.
- DINOv2 source commit: 7764ea0f912e53c92e82eb78a2a1631e92725fc8.
- Weight SHA256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9.
- Parameter count: 22056576.
- Strict pretrained state-dict load: 0 missing keys, 0 unexpected keys.
- Parameters frozen during extraction.
- xFormers absence was non-blocking; the current inference path passed CPU and GPU sanity without installing the historical DINOv2 training dependency stack.

SAMPLING / REPRESENTATION:
- One frame per primary video.
- Frame ordering: sorted *.jpg filenames.
- Selected frame index: n_frames // 2.
- Even-length semantics: upper-middle frame.
- No annotations, labels, recovery outcomes, fresh DEV, or TEST were used for frame selection.
- Official pinned DINOv2 classification eval transform:
  resize shorter side to 256 with bicubic interpolation;
  center crop 224;
  tensor conversion;
  ImageNet mean (0.485, 0.456, 0.406);
  ImageNet std (0.229, 0.224, 0.225).
- Representation: model(x) = IdentityHead(x_norm_clstoken).
- Embedding dimensionality: 384.
- Embedding dtype: float32.
- Extra L2 normalization: false.
- Embedding postprocessing: NONE.

SANITY:
- Same-image repeat maximum absolute difference: 0.0.
- model(x) versus forward_features()[x_norm_clstoken] maximum absolute difference: 0.0.
- Frozen three-video GPU sanity: PASS.
- Sanity artifact written: false.

PRODUCTION OBSERVED:
- Primary videos: 1170.
- Embedding shape: (1170, 384).
- Unique video rows: 1170.
- Embedding norm minimum: 41.223283646063756.
- Embedding norm median: 47.046143158972725.
- Embedding norm maximum: 53.952479547818726.
- Embedding norm mean: 46.94078641062315.
- Maximum peak allocated VRAM: 0.09605073928833008 GB.
- Maximum peak reserved VRAM: 0.111328125 GB.
- Runtime: 69.38440942764282 s.
- Planned downstream visual clustering k: 20.
- Planned hard-set maximum fraction per visual cluster: 0.15.
- No clustering or hard-set selection occurred in EXP047.

IMPLEMENTATION:
- config SHA256: 039e784aca2451cd629efa909893e9aabb7d8de80b90d16d4dd64177b4930b5f.
- script SHA256: eb377e4f702bf1d5be9aee1a477f78b2dccbc9cc4484b09476581902a46ef2f3.
- test SHA256: d1df3aa7e3f5ba14967203c296d09d44e6081acd9e5e69b2db62e4e2cc6a1b54.

ARTIFACTS:
- experiments/EXP047_visual_embeddings/embeddings.npy
  SHA256: 66aacaa538f633976c64a702c2a3fe5fce475a45e1eeb0885836627922434776
- experiments/EXP047_visual_embeddings/video_frames.csv
  SHA256: 2ed34109a2aaac86100cf5f5c5a00652e812094cf68df52019d632de3091f27b
- experiments/EXP047_visual_embeddings/summary.json
  SHA256: e69640de8418ed8bc5efe3a8c2e31a46586518d7ccc4a3e5d53162163648a0e8

NEXT:
- Assemble and freeze DI-v1 at video level using the five supervisor-approved outcome-independent components, then apply the frozen visual-diversity clustering constraint before final hard-set and split construction.

## EXP048 - DI-v1 component freeze

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Frozen implementation commit: 10b9292d20fb35942359a6b7006db4515788e5fb.
- DI-v1 component definition and video-level component table are FROZEN / MATERIALIZED.
- Population: frozen EXP044 primary-eligible population only.
- Primary events: 2701.
- Primary videos: 1170.
- Hard-set size frozen: false.
- Hard set selected: false.
- Final split constructed: false.
- Fresh DEV evaluated: false.
- TEST evaluated: false.

FROZEN DI-v1 COMPONENT SEMANTICS:
- Five equal-weight components, each weight 0.2:
  identity pressure;
  gap duration;
  target size;
  crowding;
  reappearance displacement.
- Event-to-video gap duration: maximum gap_len.
- Event-to-video target size: minimum pre10_min_visible_area_fraction; smaller target is harder.
- Event-to-video crowding: maximum crowding_mean_visible_objects.
- Event-to-video reappearance displacement: maximum reappearance_displacement_diag_norm.
- Identity pressure: maximum unique (video, object_id) max_other_anchor_cos_fp32.
- Repeated occlusion-event rows do not duplicate a frame-0 identity measurement.
- Structural no-frame-0-competitor videos form the lowest tied identity-pressure group; no cosine is fabricated.
- Within-pool component ranks: scipy.stats.rankdata(method="average") / N, N=1170.
- DI-v1 score: arithmetic mean of the five percentile ranks.

IDENTITY COVERAGE:
- Competitor-defined videos: 378.
- Structural no-competitor videos: 792.
- Unique competitor-defined video-object identity measurements: 1079.

VISUAL DIVERSITY:
- Input: frozen EXP047 DINOv2 ViT-S/14 384-D embedding.
- No additional L2 normalization.
- scipy.cluster.vq.kmeans2.
- k=20.
- iter=100.
- minit="++".
- RNG seed=42.
- Hard-set maximum fraction per visual cluster: 0.15.
- Visual cluster counts:
  110,45,57,88,89,80,65,26,35,72,54,49,49,15,75,65,101,64,12,19.

DI-v1 OBSERVED DISTRIBUTION:
- Minimum: 0.19367521367521368.
- Median: 0.48619658119658116.
- Maximum: 0.8810256410256411.

COMPONENT CORRELATION:
- Identity vs gap: -0.15552068548485032.
- Identity vs size: 0.09102515455157509.
- Identity vs crowding: 0.7078374476225872.
- Identity vs displacement: -0.0819060178338578.
- Gap vs size: -0.08460772297469671.
- Gap vs crowding: -0.5259040085641359.
- Gap vs displacement: 0.41297032349648993.
- Size vs crowding: 0.16044527760495481.
- Size vs displacement: -0.12160965389482929.
- Crowding vs displacement: -0.31639716947214896.
- Identity/crowding correlation is substantial, but the supervisor-required leave-one-component-out stability criterion passes; no rebalance is required.

LEAVE-ONE-COMPONENT-OUT STABILITY:
- Required survival overlap: >=0.60.
- hard_n=120: 739 events; min overlap 0.6583333333333333; PASS.
- hard_n=160: 907 events; min overlap 0.65625; PASS.
- hard_n=200: 1053 events; min overlap 0.705; PASS.
- hard_n=240: 1162 events; min overlap 0.6958333333333333; PASS.
- hard_n=280: 1234 events; min overlap 0.6892857142857143; PASS.
- hard_n=320: 1321 events; min overlap 0.703125; PASS.
- All tested hard-set sizes satisfy the <=0.15 visual-cluster cap.
- Equal-weight DI-v1 is retained without rebalance.

IMPLEMENTATION:
- config SHA256: 18e564332c23dace531620f440433cd2457fbaca850cdcbd5ad8e003a0ab19e1.
- script SHA256: b86b86448cd01752cc821f9b1943994718fc48f77960d8090e6b9915b300f00c.
- test SHA256: e022e07614baafb09226d7ff9dc057fac017568e56ac1b8ddd21bdc16bab9ed8.

ARTIFACTS:
- experiments/EXP048_di_v1/di_video_components.csv
  SHA256: 42717e0c25b798c60e1be71cfcaafa49904559e104d6f23b09bff95c10275df3
- experiments/EXP048_di_v1/visual_clusters.csv
  SHA256: 4c229d2e0b077b66c6394b6508830b8df32f3ce5ae0988245f397205576a0aae
- experiments/EXP048_di_v1/component_correlations.csv
  SHA256: cb2e05c5d409bf99343b3595875070be48127da88cdda3783d6378034e8b56b4
- experiments/EXP048_di_v1/stability_grid.csv
  SHA256: 9cdb68443d9042b85d87a18daec8dd40f5ec68669c696bd586a6c6577e6f9fef
- experiments/EXP048_di_v1/summary.json
  SHA256: f6af9aa752e9ebab1b7139ac9b4ff9f5adb657d4f1f781ab03152f0411aba257

NEXT:
- Lock hard-set size and final video-level DEV/TEST construction without using gate/model outcomes, while preserving the frozen DI-v1 score and visual-cluster constraint.

## EXP049 - Final split freeze

STATUS:
- EXECUTED / VERIFIED / COMPLETE.
- Frozen implementation commit: dd0f47c50e8fb008ae61d4371cee824225bb43a4.
- Final video-level cohort membership is FROZEN.
- Model outcomes used for split construction: false.
- Gate inference performed: false.
- Fresh DEV evaluated: false.
- TEST evaluated: false.
- TEST membership was materialized as metadata only.

FROZEN POPULATION:
- Source population: 1170 EXP044 primary-eligible, development-unexposed videos.
- Hard pool: 120 videos / 739 primary events.
- Fresh DEV: 40 videos / 233 primary events.
- Hard TEST: 80 videos / 506 primary events.
- Representative TEST: 40 videos / 76 primary events.
- Hard TEST minimum required event count: 200.
- Observed hard TEST event count: 506; PASS.

HARD-POOL CONSTRUCTION:
- Ranking: descending frozen DI-v1 score.
- Tie-break: ascending video ID.
- Maximum selected fraction per frozen visual cluster: 0.15.
- Observed maximum hard-pool cluster count: 18/120 = 0.15.
- Hard pool membership SHA256:
  50285e5ffd6a30d082fec4c945e4769199f7456a0a119e7f1b487fa8cb17cadc.

FRESH DEV / HARD TEST PARTITION:
- DEV allocation: visual-cluster-stratified largest-remainder quotas.
- RNG seed: 42.
- Fresh DEV membership SHA256:
  fde1d5ba4787fa627948301183256a00102a50ab8be2d41a4dd756cd1a859e8d.
- Hard TEST membership SHA256:
  6bf5a059c05d07f82e43fec9bd6b4c4b723bc551a21b72c988912e57b28ce582.
- Fresh DEV and hard TEST are disjoint.
- Their union exactly equals the frozen 120-video hard pool.

REPRESENTATIVE TEST:
- Size: 40 videos.
- Sampling population: current 1170-video primary-eligible population excluding the entire hard pool.
- Sampling: uniform without replacement.
- RNG seed: 42.
- Representative TEST is disjoint from the entire hard pool.
- Representative TEST membership SHA256:
  ba23def8c8d0de7a83af64c6f952544d5f3e44ad6ca018f9e4d2b6cd82ebfb66.

IMPLEMENTATION:
- config SHA256: 65ad6d3484351fabdafdf44498d456b6702f93783765c0d8a663e2198e584692.
- script SHA256: 706da43538afc64442a5b559e93c5a9b72e6437ebdd6fab61839627e39f2386d.
- test SHA256: a278154542611bf3270c72f4a7f214a1c1cdf54b8052410c62ea43d2b4b3900f.

ARTIFACTS:
- experiments/EXP049_final_split/hard_pool.csv
  SHA256: a283bf41141a58a02e3111ad2edffd0384b11380febe12105e5c3400216a1d7a
- experiments/EXP049_final_split/fresh_dev.csv
  SHA256: 5c40f403337aca576242709cc18c75f8a982de0ea8fc1d3bed1c3fad2ee3ffdc
- experiments/EXP049_final_split/hard_test.csv
  SHA256: f0469d9bf5cdc9f438b8262c626f52b4de5fa690ab7734da36052ff53495f881
- experiments/EXP049_final_split/representative_test.csv
  SHA256: cbdf2e2f7896b326ec810ce0dbcd51722b63421f09a878aa89b0fcf2c926fe0b
- experiments/EXP049_final_split/manifest.json
  SHA256: 9b2d4a4b405ed94339b0b1325782c9471c34e1c1438d60be03cc4d5c39c218fc

NEXT:
- Freeze the exact fresh-DEV evaluation implementation/config, then execute fresh DEV once to choose the outcome-independent matched-rate target r_star and final gate thresholds according to the locked A4 protocol. TEST remains untouched.

## EXP050 - Fresh DEV rate-only matched-write-rate selection

STATUS:
- EXECUTED / VERIFIED.
- HEADLINE MATCHED-RATE SELECTION: PASS.
- MANDATORY FULL-CURVE REQUIREMENT: PROTOCOL FAILURE.
- Implementation freeze commit: d74f0f502f99d3e8f6d290773fe4943f8fbd3ab0.
- Fresh DEV was touched by closed-loop tracking for physical write-rate selection only.
- Tracking-performance outcomes were not computed or inspected.
- TEST remained untouched.

SCOPE:
- Frozen fresh DEV: 40 videos / 233 primary events.
- Fresh DEV membership SHA256:
  fde1d5ba4787fa627948301183256a00102a50ab8be2d41a4dd756cd1a859e8d.
- Variants: B1, B2, B3-S, B3-R, B5.
- Physical pooled write rate: sum(K_v) / sum(N_v).
- Absolute matched-rate tolerance: 0.02.
- Locked target order: 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9.
- Maximum midpoint refinements per A4 target: 4.

HEADLINE RESULT:
- First common matched target in the locked order: r_star = 0.3.
- B1: tau=0.1, realized pooled write rate=0.28367729831144467,
  absolute error=0.01632270168855532.
- B2: tau=0.1, realized pooled write rate=0.3091932457786116,
  absolute error=0.009193245778611636.
- B3-S: tau=0.2, realized pooled write rate=0.30393996247654786,
  absolute error=0.003939962476547876.
- B3-R: tau=0.2, realized pooled write rate=0.2904315196998124,
  absolute error=0.009568480300187587.
- B5: tau=0.7, realized pooled write rate=0.29812382739212007,
  absolute error=0.0018761726078799223.
- All five headline operating points satisfy the locked +/-0.02 tolerance.

FULL-CURVE RESULT:
- Requested curve targets: 0.1 through 0.9 in increments of 0.1.
- Requested variant-target rows: 45.
- Matched rows: 19.
- Unmatched rows: 26.
- B1 unmatched targets: 0.4, 0.5, 0.6, 0.7, 0.8, 0.9.
- B2 unmatched targets: 0.5, 0.6, 0.7, 0.8, 0.9.
- B3-S unmatched targets: 0.5, 0.6, 0.7, 0.8, 0.9.
- B3-R unmatched targets: 0.5, 0.6, 0.7, 0.8, 0.9.
- B5 unmatched targets: 0.5, 0.6, 0.7, 0.8, 0.9.
- Each failed row exhausted the frozen maximum of four midpoint refinements.
- Therefore the mandatory full-curve requirement failed under the frozen A4 search procedure.

NEAR-ZERO RATE BEHAVIOR:
- All variants had pooled write rate 1.0 at tau=0.
- At tau=0.00625 the observed pooled rates were:
  B1=0.374109, B2=0.427767, B3-S=0.437711,
  B3-R=0.399625, B5=0.472045.
- This is an observed sharp near-zero threshold discontinuity.
- It is NOT concluded that the unmatched high-rate targets are mathematically unreachable.
- No post-hoc increase in refinement budget, tolerance, target order, or outcome-driven retuning was performed.

EXECUTION:
- Executed pooled points: 79.
- Executed video trajectories: 3160.
- Scratch cache misses: 3160.
- Scratch cache hits: 0.
- Maximum observed peak VRAM: 13.846986293792725 GB.
- Production log SHA256:
  1abfd0d12aa62b75f48ebe0a247d351ab4016a5244b4894311ab1492c7dbc7c6.
- Cache-ledger SHA256:
  749de50a0ad643e15d8000565361d20fa116f1f0f7dcac07286ce3f135da84ef.

OUTCOME FIREWALL:
- POR computed: false.
- ITR computed: false.
- J&F computed: false.
- UAR computed: false.
- Contamination computed: false.
- Tracking outcomes used for threshold selection: false.
- TEST touched: false.

ARTIFACTS:
- common_target_selection.csv
  SHA256: 92837a93aa50ab71fd7868f751d99f96e1617eb32d66b72fc29875795ea2ff6e
- curve_selection.csv
  SHA256: 9977e9589dfd8e22a69ee7812ba1a48cc89a37cfc56bfe087c1a5c40d9d1c0c2
- executed_pooled_points.csv
  SHA256: c104238f29b7509e32bd631d7e5ce9fa427fc5b98d358e16ea739cb4c77ced22
- final_operating_points.json
  SHA256: b924642245b722b8734e111b9ce4c60b24544f7d7e39defb4711120e74bcea46
- per_video_write_counts.csv
  SHA256: db5e47654bcbd08dbc0857f5c892e680930fbd7bc1ed2ecf1aad8cc5946bfc2d
- summary.json
  SHA256: b5d141fd82c7708d9a54a0889ece3cad1e03ba435b16201bcd075116a792cffb

OPEN PROTOCOL ITEM:
- Headline r_star=0.3 is observed and matched under the frozen A4 procedure.
- The mandatory full 0.1-0.9 curve cannot be declared complete under that same procedure.
- Before any tracking-performance outcome is inspected, freeze a narrow amendment specifying how the observed full-curve protocol failure is handled.
- Do not retune models, target order, tolerance, or headline r_star based on outcomes.

NEXT:
- Record and freeze the narrow A4 full-curve-failure handling amendment.
- Close the remaining preregistration ambiguities before inspecting fresh-DEV tracking outcomes.
