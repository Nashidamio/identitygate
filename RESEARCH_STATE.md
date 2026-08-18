# RESEARCH_STATE.md

**Project:** IdentityGate — Supervised, Identity-Verified Memory Write Admission for SAM 3.1 Video Tracking
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
