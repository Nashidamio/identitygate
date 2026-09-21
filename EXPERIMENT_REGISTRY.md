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

## EXP024 - TRAIN-only incremental signal utility probe

Status: COMPLETE / DEVELOPMENT-ONLY / TEST UNTOUCHED

Producing bootstrap freeze HEAD: 3d3cbe7

Input:
- EXP024 OOF predictions SHA256: b622a3e6838ba574ade8394c007585be253abab6cc8c88fee35155361a345bb2
- Common pointer-valid complete-case population: 7,948 rows
- Validation: leave-one-video-out by video
- Model: fixed standardized logistic-regression utility probe; not final gate architecture
- DEV touched: 0
- TEST touched: 0

OOF discrimination:
- Drift B2-core: AP 0.6983904741, AUROC 0.8185834027
- Drift B3-S: AP 0.6571114519, AUROC 0.8123831092
- Drift B3-R: AP 0.5989720387, AUROC 0.8072995379
- Theft B2-core: AP 0.0229407661, AUROC 0.6152775153
- Theft B3-S: AP 0.0187467663, AUROC 0.5475470475
- Theft B3-R: AP 0.0189913995, AUROC 0.5466326466

Paired video-cluster bootstrap, 5,000 replicates:
- Theft B3-S minus B2-core AP 95% CI: [-0.0115334347, -0.0001670046]
- Theft B3-S minus B2-core AUROC 95% CI: [-0.1015720295, -0.0032966214]
- Theft B3-R minus B2-core AP 95% CI: [-0.0096703590, -0.0000805857]
- Theft B3-R minus B2-core AUROC 95% CI: [-0.1015394434, -0.0001109557]
- Drift B3-R minus B3-S AUROC 95% CI: [-0.0110033203, -0.0008245190]
- Other reported drift comparisons cross zero.

Sparse-cluster limitation:
- Theft positives: 143 rows in only 5 videos.
- 16 of 5,000 theft bootstrap replicates had no evaluable positive class and were reported invalid.

Interpretation:
- Tested self-identity and relational-identity additions did not demonstrate incremental utility over B2-core in this TRAIN-only probe.
- Theft results statistically favor B2-core over B3-S and B3-R under the specified clustered bootstrap.
- This is NOT evidence that B2 improves closed-loop SAM3 tracking.
- This is NOT a final TEST result.

Artifacts:
- experiments/EXP024_cluster_bootstrap/summary.json
  SHA256 69cb95590f21c5fc0b527d5a7f54a9491d8dcecdad4820d0496b4252d6800182
- experiments/EXP024_cluster_bootstrap/bootstrap_draws.csv
  SHA256 933a224ad95842769f05b4861f64a688a50aeba9dd9e2497e343ee4c148e2041
- experiments/EXP024_cluster_bootstrap_run.log
  SHA256 69cb95590f21c5fc0b527d5a7f54a9491d8dcecdad4820d0496b4252d6800182

Next exact scientific action:
- Closed-loop memory-write BLOCK mechanism sanity on TRAIN-exposed data, followed by B0-vs-B2 occlusion recovery evaluation.

## EXP025 - Closed-loop write-block mechanism sanity

Status: COMPLETE / MECHANISM VERIFIED / NOT FINAL GATE

Producing commits:
- ce4c664 EXP025: freeze closed-loop write-block sanity
- 79b462d EXP025: correct closed-loop propagation count

Scope:
- MOSEv2 TRAIN-exposed video 0442a954
- Frames 0..59
- Objects: 1, 2
- Deterministic whole-frame non-conditioning-memory eviction on frames 1..30
- DEV touched: 0
- TEST touched: 0

Verified mechanism:
- 30/30 blocked frames were present in global and per-object memory before eviction.
- 30/30 were absent after eviction.
- frames_already_tracked remained present.
- First intervention-frame prediction was identical between B0 and BLOCK.
- First downstream mask difference occurred at frame 2.
- Total downstream B0-vs-BLOCK XOR pixels: 46177.
- Therefore memory-write intervention changes subsequent closed-loop SAM3 tracking behaviour.

Descriptive tracking result:
- B0 mean target IoU on visible rows: 0.8243956364947009
- deterministic BLOCK mean target IoU: 0.8232898540701283
- delta BLOCK minus B0: -0.001105782424572599
- This is NOT evidence of tracking improvement.

Resource:
- B0 peak VRAM: 6.266373634338379 GB
- BLOCK peak VRAM: 6.202365875244141 GB

Core artifact hashes:
- summary.json: 4661337126c15fa919ecb4edbd8dd58ec6a034cd6d5e5153bb58261d733bef02
- frame_metrics.csv: 2c0d2fc57493758233565684f9f5602e36f2dac4dfa948fcce47716391fbf34d
- write_decisions.csv: d5f2e963f76c02f2b9c9dfeb4c74adbcad8d63a709abf1ee1ac4bf638e1b09e1
- run log: a7eb388d38128bbd9b0093b3c2acb817ce7917ccbb896e8bcafeb0c09e0a33f4

Claim boundary:
- Whole-frame deterministic blocking is an engineering mechanism sanity only.
- It is not the final per-object IdentityGate and is not a POR result.

## EXP026 - Batched versus singleton B0 equivalence

Status: COMPLETE / EXACT_EQUIVALENCE_FAIL / SINGLETON ROUTE REJECTED

Producing commit:
- f82904446caec9d901fbb3b8cc5796e0e2e83f6f EXP026: freeze singleton equivalence sanity

Scope:
- TRAIN-exposed video 0442a954
- 60 frames
- 2 objects
- batched vanilla B0 versus one fresh singleton tracker state per object
- DEV touched: 0
- TEST touched: 0

Frozen acceptance rule:
- PASS only if every object-frame binary prediction mask is exactly equal.

Observed:
- comparison rows: 120
- exact rows: 22
- exact fraction: 0.18333333333333332
- first difference: frame 1, object 1, XOR 2 px
- total XOR pixels: 3365
- minimum batched-vs-singleton mask IoU: 0.6569468267581475
- mean target IoU singleton minus batched: -0.001650891938661303

Decision:
- Exact equivalence failed.
- Singleton execution is rejected as the per-object IdentityGate route.
- The acceptance rule will not be relaxed post hoc.
- Cause of the batched/singleton divergence is UNKNOWN and is not required to reject this route.

Artifact hashes:
- summary.json: 3a01aa0eae5914bdf9e0aafdb36c20c6061d4760f74128edea10dc30ce745f69
- frame_object_comparison.csv: f4027c29a3e5756d1796f79b3dcbbde3d0fb387bb0709ca75275fcf97cbf7cc8
- run log: 745a88eadf92f69c03013e622f45c7f9fab87c8135e2c95e9e2bb53f47fd2405

## EXP027 - Per-object attention-filter mechanism

Status: COMPLETE / IMPLEMENTATION ROUTE REJECTED / NO GATE RESULT

Producing commits:
- 6d27569 EXP027: freeze per-object attention filter helper
- 758b8e7 EXP027: freeze per-object filter sanity
- cb077f0 EXP027: fix attention filter patch indentation

Scope:
- TRAIN-exposed video 0442a954
- 60 frames
- batched frozen SAM3 retained
- intended selective object-level memory-attention filtering
- DEV touched: 0
- TEST touched: 0

Observed:
- Vanilla batched B0 executed.
- Patched NO-BLOCK executed after the indentation correction.
- Selective filtering failed when the first non-empty memory_key_padding_mask reached the active TransformerDecoderLayerv2.forward_pre path.
- Active forward_pre asserts that memory_key_padding_mask must be None.

Decision:
- The memory_key_padding_mask Track-B route is rejected for the pinned SAM3 configuration.
- The frozen acceptance criteria were not relaxed.
- No selective-filter tracking result was produced.
- This is an engineering-route result, not evidence for or against B2 tracking performance.

Failed-run hashes:
- attempt 1 indentation failure: 869b3e40e33c29815cf253f483e7d12a32442adacc696535591b14d8c52394e4
- attempt 2 stdout-only diagnostic: 7f755735902957394bd95b24534b080274c0704d81ec400f69b1a2dcf98e19fd
- attempt 3 captured forward_pre rejection: 2e487385ec658ccc4a8d1fb15b774d337f72e7ec29440dee37ea367d8db0ffde

## EXP028 - Rowwise hybrid equivalence sanity

Status: COMPLETE / ROWWISE CONTROL FAIL / IMPLEMENTATION ROUTE REJECTED

Executed code commit:
- 7c932cdf2d3788ff02ec5202d7f707e367ac8db7

Scope:
- MOSEv2 TRAIN-exposed video 0442a954.
- 60 frames.
- Objects [1, 2].
- Replace object ID 1 / row 0.
- Frozen SAM3.
- DEV touched: 0.
- TEST touched: 0.

Frozen control requirement:
- ROWWISE FULL-MEMORY must exactly reproduce VANILLA B0 before selective blocking.
- Exactness covered binary masks, object_score_logits, iou_score, obj_ptr, maskmem_features, and global eff_iou_score.
- No tolerance relaxation was permitted.

Observed control:
- status: ROWWISE_CONTROL_FAIL.
- comparison rows: 120.
- binary-mask exact rows: 79/120.
- signal exact rows: 65/120.
- global eff_iou exact frames: 58/60.
- row recompute calls: 59.
- omission_count: 0.
- omitted_frames: [].
- vanilla peak VRAM: 6.266373634338379 GB.
- control peak VRAM: 6.266663551330566 GB.
- vanilla runtime: 6.997359037399292 s.
- control runtime: 7.866125583648682 s.

Interpretation:
- Because the full-memory control used zero blocked-frame omissions, the observed differences are caused by the rowwise recomputation path itself rather than by a gate decision.
- The frozen exact-equivalence criterion failed.
- ROWWISE hybrid is therefore rejected as a per-object gating implementation route.
- The selective-block condition was not executed.
- This experiment provides no B2/B3 tracking-performance result.

Artifacts:
- summary.json SHA256: 29979f3b736124a8299bad63f82bec994e84fcc690506daed0fc2e1f486b53cc
- comparisons.csv SHA256: 4d3e75ea171621e0e8726e7994cd4bb535f6532f45dd40f95c9654d701107013
- run log SHA256: a951abb04aadfd27228097cf7aa5955a998e231040248962934c5bfcc877bbe2
- comparisons.csv lines: 121

Decision:
- Do not tune or relax ROWWISE equivalence.
- Proceed to the pre-declared whole-frame physical write-block fallback, with the scientific change documented as an amendment before gate-result experiments.

## EXP029 - B2-core development gate training

Status: COMPLETE / TRAIN-ONLY DEVELOPMENT WEIGHTS / NOT FINAL B2

Executed code commit:
- 8722001e2d337e9c96d3a145f2ad9cb2b063d6cf

Scope:
- 18 TRAIN videos only.
- 8,139 rows with all five B2-core features finite.
- Features: mask_conf_iou_head, occ_score_logit, area_norm,
  area_ratio_anchor, temporal_iou_prev.
- Shared TRAIN-only z-score normalization.
- Two independent failure-typed MLP heads: drift and theft.
- Architecture per head: 5 -> 64 -> 32 -> 1.
- Total learned parameters: 4,994.
- DEV touched: 0.
- TEST touched: 0.

Training:
- PyTorch CPU float64 deterministic full-batch.
- AdamW, 1,000 fixed epochs.
- Weighted BCE independently per head.
- No early stopping and no DEV tuning.

Observed:
- Drift rows: 6,995; positives: 1,507; positive-video clusters: 17.
- Theft rows: 8,104; positives: 143; positive-video clusters: 5.
- Drift weighted BCE: 1.0820583613743782 -> 0.3333268393773441.
- Theft weighted BCE: 1.367510637194698 -> 0.2881328066581184.

Claim boundary:
- These are actual learned B2-core development weights.
- Training-set loss is descriptive only.
- This is not held-out evidence, not closed-loop evidence, and not final frozen B2.
- Features 4 and 8 remain deferred; Feature 7 remains open.
- The two failure heads are not yet frozen into one physical admission-score rule.

Artifacts:
- model.json SHA256: 6160a6be9da2058808c16182abe03443014806273df46fe59a86411ffc869ecf
- training_curve.csv SHA256: e7262d73b829290a1e97753faab9558d877eb733c79799637a787cbc025a2ebb
- summary.json SHA256: a80e790c88c5b0d1ebc6c2b8643cf6ebf336fb7a22cef35e7a23e44316a10560
- run log SHA256: a80e790c88c5b0d1ebc6c2b8643cf6ebf336fb7a22cef35e7a23e44316a10560

## EXP030 - Learned B2-core closed-loop sanity

Status: COMPLETE / LEARNED CLOSED-LOOP MECHANISM PASS / NOT PERFORMANCE RESULT

Executed code commit:
- 590b1f5c1bf8da6a7eb724355b76ec3b3fc5a219

Scope:
- TRAIN-exposed MOSEv2 video 0442a954.
- 60 frames; objects [1, 2].
- Frozen EXP029 B2-core model SHA256:
  6160a6be9da2058808c16182abe03443014806273df46fe59a86411ffc869ecf
- A3 ALL-SAFE frame aggregation.
- Development-only dual-head composition and fail-closed missingness.
- Fixed mechanism-sanity tau_admit=0.5.
- DEV touched: 0.
- TEST touched: 0.

Observed mechanism result:
- status: LEARNED_GATE_CLOSED_LOOP_SANITY_PASS.
- eligible non-conditioning frames: 59.
- ADMIT count: 4.
- BLOCK count: 55.
- descriptive admit fraction: 0.06779661016949153.
- first BLOCK frame: 2.
- first BLOCK prediction exactly matched B0: True.
- first downstream changed frame: 3.
- downstream masks changed: True.
- total downstream XOR pixels: 60790.
- every BLOCK present before eviction: True.
- every BLOCK absent after eviction: True.
- frames_already_tracked retained: True.
- non-finite object-feature rows: 6.

Descriptive tracking result:
- B0 mean target IoU on visible rows: 0.8243956364947009.
- gated mean target IoU on visible rows: 0.8062881782959369.
- gated minus B0: -0.018107458198764026.
- This TRAIN-exposed descriptive delta is negative and is retained without threshold tuning.

Resources:
- B0 peak VRAM: 6.266373634338379 GB.
- gated peak VRAM: 6.055308818817139 GB.

Interpretation:
- A learned neural gate now demonstrably controls physical closed-loop SAM3 memory writes and changes subsequent predictions.
- EXP030 does not establish tracking improvement, generalization, calibration, or final B2 performance.
- The fixed tau=0.5 operating point admitted only 4/59 frames, so write-budget control is mandatory before performance comparison.
- No post-hoc threshold tuning is performed on this result.

Artifacts:
- summary.json SHA256: 248a8323e2f299d17de64231d750a9a91a49a190f6483fbd2ed783f922158384
- write_decisions.csv SHA256: 04c7e1fb2dc4902f4d61376acdcad14093de04a40c8f53fffa32f89a3d5b4a6a
- object_gate_scores.csv SHA256: 1ccf7dc778eeb7ed71a6358fa747837165ea116794adcbf129aebd075050d93d
- frame_metrics.csv SHA256: 7165c8a19b5145409a7b9ce8da09118226bdb4431cc95e50af1cf678c211f687
- visual_sha256.txt SHA256: 5e12e9642d7bfe939850c90722fc0549c83e3e38d7446ed3395c4db940947c51
- run log SHA256: 421d0592ff00e724d068b771d771b2c2f7c558d0b6d25e4aa410a66a4c7d149f

## EXP031 - B3-S and B3-R development gate training

Status: COMPLETE / TRAIN-ONLY DEVELOPMENT WEIGHTS / NOT FINAL B3

Executed code commit:
- 5944229ba518ccb924ce36c8d7368f6602ff8dce

Scope:
- 18 TRAIN videos only.
- 7,951 pointer-valid rows observed.
- 7,948 common complete identity rows used for training.
- pointer_valid is an availability mask and is not a predictive feature.
- B3-S inputs: B2-core plus ptr_sim_anchor_fp32.
- B3-R inputs: B3-S plus max_comp_anchor_cos_fp32.
- DEV touched: 0.
- TEST touched: 0.

Architecture:
- Two independent failure-typed heads per variant: drift and theft.
- Hidden architecture per head: input -> 64 -> 32 -> 1.
- B3-S total learned parameters: 5,122.
- B3-R total learned parameters: 5,250.

Training:
- PyTorch CPU float64 deterministic full-batch.
- AdamW, 1,000 fixed epochs.
- Weighted BCE independently per head.
- No early stopping and no DEV tuning.

Observed TRAIN-only losses:
- B3-S drift: 1.1087743738922864 -> 0.2704712555479701.
- B3-S theft: 1.4022949012535892 -> 0.19085894319757699.
- B3-R drift: 1.1239718935598602 -> 0.20031431750105905.
- B3-R theft: 1.398325059932092 -> 0.12216637266336192.

Claim boundary:
- These are actual B3-S and B3-R development weights.
- Training loss and extreme TRAIN probabilities are descriptive only.
- No generalization or tracking-improvement conclusion follows.
- Missing-identity deployment fallback remains OPEN.
- Existing EXP024 held-out utility evidence remains unchanged and is not superseded by TRAIN fit quality.

Artifacts:
- model.json SHA256: 235076b86aa0975b3ec624aae703575fd4f030381b6af4008c3808f2db71847a
- training_curve.csv SHA256: 2c2614ce13884efb0e8fc7006a801f32e78be362c88ffe970f70596b426b9dd0
- summary.json SHA256: 56bb0668696d9cb1f5fd9514ee2e3a356289107255c3be6ab58eda76acaa59eb
- run log SHA256: 56bb0668696d9cb1f5fd9514ee2e3a356289107255c3be6ab58eda76acaa59eb

## EXP032 - B1 manual-rule closed-loop sanity

Status: COMPLETE / CLOSED-LOOP MECHANISM PASS / NOT PERFORMANCE RESULT

Executed code commit:
- 89d6d3291bb1e294f21aa602453fe85efa06a0e6

Scope:
- TRAIN-exposed MOSEv2 video 0442a954.
- 60 frames; objects [1, 2].
- Frozen SAM3.
- Frozen A6 B1 manual scoring rule.
- A3 ALL-SAFE frame aggregation and whole-frame physical memory eviction.
- Fixed mechanism-sanity tau_B1 = 0.5.
- DEV touched: 0.
- TEST touched: 0.

Observed mechanism result:
- status: B1_GATE_CLOSED_LOOP_SANITY_PASS.
- eligible non-conditioning frames: 59.
- ADMIT: 11.
- BLOCK: 48.
- descriptive admit fraction: 0.1864406779661017.
- nonfinite object feature rows: 11.
- first blocked frame: 2.
- first blocked-frame prediction equals B0: True.
- first changed downstream frame: 3.
- downstream prediction changed: True.
- total downstream XOR pixels after first block: 34005.
- every blocked frame present before eviction: True.
- every blocked frame absent after eviction: True.
- frames_already_tracked retained: True.

Descriptive TRAIN-exposed tracking values:
- B0 mean target IoU: 0.8243956364947009.
- B1-gated mean target IoU: 0.8275807387318024.
- gated minus B0 delta: 0.003185102237101445.
- B0 peak VRAM GB: 6.266373634338379.
- B1 peak VRAM GB: 6.166836261749268.

Interpretation:
- B1 demonstrably controls the physical SAM3 memory-write path.
- The positive descriptive IoU delta is not performance evidence.
- tau_B1=0.5 is not a final operating point and was not A4 matched-rate selected.
- No generalization or tracking-improvement conclusion follows.

Artifacts:
- summary.json SHA256: 962801fb812ef6a1516a171768898f864eb361abbdee2da10306a910a75d1ca5
- write_decisions.csv SHA256: fc52991a7ef29e2bff739d09974aeee6ed1449f31a4564df2d62e38e1dc97a1c
- object_gate_scores.csv SHA256: a5899014014154a5f55a5c982f43da5753e5f6b5a600c9dd16fd1a8f6f2f50ed
- frame_metrics.csv SHA256: 97eded925f9556423bc1f57ee602658b2a18445e98f3ac43d02a587f6eafb079
- visuals.sha256 SHA256: f889731f62c7ee735991e34ffb564669726285b1a0bf592ad217f0c0a6632dd4
- run log SHA256: 94a0d0c418df111168ce013b38ca3edb85bc2f3955ae5db794f6a68164c19d04

## EXP033 - Unified B1/B2/B3 closed-loop sanity

Status: COMPLETE / UNIFIED CLOSED-LOOP MECHANISM PASS / NOT PERFORMANCE RESULT

Executed code commit:
- bbeb184

Scope:
- TRAIN-exposed MOSEv2 video 0442a954.
- 60 frames; objects [1, 2].
- Frozen SAM3 commit 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da.
- B1 manual A6 rule.
- B2 EXP029 development B2-core weights.
- B3-S/B3-R EXP031 development weights.
- A5 missing-identity routing.
- A3 whole-frame physical memory eviction.
- Fixed mechanism-sanity tau = 0.5.
- Fresh DEV touched: 0.
- TEST touched: 0.

Observed:
- Overall status: UNIFIED_GATE_CLOSED_LOOP_SANITY_PASS.
- B1: 11 ADMIT / 48 BLOCK; mechanism PASS.
- B2: 4 ADMIT / 55 BLOCK; mechanism PASS.
- B3-S: 18 ADMIT / 41 BLOCK; mechanism PASS.
- B3-R: 16 ADMIT / 43 BLOCK; mechanism PASS.
- Every variant preserved current-frame prediction at the first block and changed downstream predictions.
- Every blocked frame was present before eviction and absent afterward.
- frames_already_tracked bookkeeping remained retained.
- B3-S live routing: B3_S=105, B2=1, FAIL_CLOSED_BASE=12.
- B3-R live routing: B3_R=104, B2=1, FAIL_CLOSED_BASE=13.
- Peak VRAM remained approximately 6.06-6.18 GB for gated variants; B0 peak was 6.266 GB.

Descriptive TRAIN-exposed IoU deltas versus B0:
- B1: +0.003185102237101445.
- B2: -0.018107458198764026.
- B3-S: +0.0038962141239293757.
- B3-R: +0.0012914734621838342.
These are not performance evidence because write rates are unmatched and the scope is one TRAIN-exposed video.

Coverage note:
- The live two-object run exercised the B3-R relational route.
- The single-object B3-R -> B3-S fallback was not live-exercised here; its deterministic routing branch passed the pre-run CPU smoke test.

Artifacts:
- summary.json SHA256: 0de6088159321da12342a955541af8577a6be8a484608a2bce3c88c2f81b4886
- B1_object_scores.csv SHA256: 5440767106c77f7e3f76dd5ff72149d08c9a918930095b319e02463ea83fd46c
- B1_write_decisions.csv SHA256: fff1dc941d6d6b08c2bb866ae7e0b9b07c572e3e347b769c93c96018da0fa98a
- B2_object_scores.csv SHA256: b0ffb61b2d77129a27704837bf830c25271f14d8c446ca1adb9696b3cdd0db73
- B2_write_decisions.csv SHA256: 325260d36fd63b49838a3e297c484cb707d3b0d01a3c26fc32b2ebbc099757bb
- B3_S_object_scores.csv SHA256: 30dd7b45faa0f24902fa479fe308b5fe982c68e531f0d2cc0eacc568a9c2bc93
- B3_S_write_decisions.csv SHA256: 114e8ebcf236ccdaccdce42d0d0f53e98fd375dabb8093256afd529de6b9185d
- B3_R_object_scores.csv SHA256: e9de784bf51a54c0c81593c0d00fae6da74f6ed180696e8626476a508b1f74dd
- B3_R_write_decisions.csv SHA256: d26ba5c20f789a0972f1f868857394aa4bac8fb6bbe1419e1d401dad887d7a83
- ignored run log SHA256: 8808b52452d8e76320cbfe5e794a2c1e87480e49a6e6b1f7e1d32b5f2c831012


## EXP034 - Whole-scene pixel cross-check sanity

Status: SANITY COMPLETE / ENGINEERING PASS / NOT FINAL WHOLE-SCENE LABELS

Frozen execution commit: b8cd9c37bb7c37ec6b430d56c238c0a6a275e9a1
Scope: predeclared development-exposed video 0nrb9vzx; 1 video / 1 EXP019 scene; fresh final DEV 0; TEST gate evaluation 0.

Observed:
- camera-cut positive scenes: 0.
- global-MAD positive scenes: 0.
- pixel-confirmed scenes: 0.
- manual-adjudication-required scenes: 1.
- peak allocated VRAM: 0.1648869514465332 GB.
- runtime: 7.353137016296387 s.
- SAM predictions used: false.
- gate predictions used: false.
- POR used: false.
- TEST gate evaluation performed: false.

Interpretation:
- Frozen EXP034 executes end-to-end and disagreement routing works.
- This one-scene sanity is not evidence about whole-scene prevalence, pixel-rule accuracy, threshold quality, or gate performance.
- Frozen thresholds are unchanged.

Artifacts:
- per_scene.csv SHA256: e320cc49ad139915891cdb4b02bbc3f2ce40271a3b67c6e6dcf14720da63a17f
- manual_adjudication_required.csv SHA256: e320cc49ad139915891cdb4b02bbc3f2ce40271a3b67c6e6dcf14720da63a17f
- summary.json SHA256: a57c2ca13a47b6bfaddc92e0d264da95642a455bf3617657a88caf140765ef23
- ignored run log SHA256: 887a9f3ce9e335c7793f4b7707be6fc4d45ac10bcd4c73da9623c4d043c6b083

Next: commit the sanity record, then execute the unchanged frozen full EXP034 pixel cross-check.


## EXP034 - Full whole-scene pixel cross-check

Status: FULL PIXEL CROSS-CHECK COMPLETE / MANUAL ADJUDICATION PENDING

Executed commit:
- 98f613ab7efdaf25f868f79b44e948771a6db595

Scope:
- EXP019 collapsed annotation-side candidate scenes.
- Videos processed: 1281.
- Scenes processed: 2179.
- Frozen audit scenes resolved with pixel statistics: 30.
- SAM predictions used: false.
- Gate predictions used: false.
- POR used: false.
- TEST gate evaluation performed: false.

Observed:
- camera-cut positive scenes: 322.
- global-MAD positive scenes: 226.
- pixel-confirmed scenes: 361.
- manual-adjudication-required scenes: 1818.
- Peak allocated VRAM: 0.1648869514465332 GB.
- Runtime: 5693.0291039943695 s.

Interpretation:
- Full frozen pixel-side preprocessing completed successfully.
- 361 scenes are automatically confirmed by the frozen pixel rule.
- 1818 annotation-positive / pixel-negative scenes require manual adjudication.
- The frozen 0.5 / 0.15 / 3.0 thresholds remain unchanged after outcome inspection.
- Final whole-scene labels remain OPEN pending manual adjudication and the frozen 30-scene audit.

Artifacts:
- audit_sample_with_pixel.csv SHA256: 3aa2f19cf69726c6c708ac2c3e99f68fdf5d6bf3de377f5b6f20979ead72cbcc
- auto_confirmed.csv SHA256: bb3d8b9fe8c29882736cacdd6d207a3d1dc43d9092e9996560c7a47fcaf4ffe7
- manual_adjudication_required.csv SHA256: 1f6fa7ad1e10662ff0333331e6d1d5a1127e89c710b0e6d9768cd206a34d63d3
- per_scene.csv SHA256: aefb19804adf093dfc94cc80aa50dc53e69df83bab2c2ab8ba5d6acd603acd2f
- summary.json SHA256: f78da9dc9b029750ad37fb448008740f33d06525d6ac970197f46202c87835a8
- ignored run log SHA256: 71f774ee45850546cfc26a64d6d585761b5daf8f3f3d1246706ff7191211e531

Next:
- Freeze and generate deterministic visual-review artifacts for the 30-scene audit and manual-adjudication population.


## EXP037 - Matched-rate evaluator integration sanity

Status: INTEGRATION SANITY PASS / NOT PERFORMANCE EVIDENCE

Executed commit:
- 020705b32f92325c6f3d6a38d0bf122a01cfcc3d

Scope:
- TRAIN-exposed video: 0442a954.
- Frames: 60.
- Object IDs: [1, 2].
- Variants: B0, B1, B2, B3-S, B3-R.
- Engineering tau: 0.5 for gated variants.
- Fresh final DEV touched: false.
- TEST touched: false.

Observed:
- B0 write rate: 1.0.
- B1 write rate: 0.1864406779661017.
- B2 write rate: 0.06779661016949153.
- B3-S write rate: 0.3050847457627119.
- B3-R write rate: 0.2711864406779661.
- Physical write-block integrity passed for every gated variant.
- Qualifying POR@30 events: 1.
- POR@30: 1.0 for B0, B1, B2, B3-S, and B3-R.
- Maximum observed peak allocated VRAM: 6.266373634338379 GB.

Interpretation:
- Closed-loop intervention, write-rate accounting, and POR@30 endpoint integration are verified.
- One qualifying event is insufficient for comparative performance inference.
- Tau 0.5 is not an A4 matched-rate operating point.
- This experiment provides no final gate-performance or statistical-significance conclusion.

Artifacts:
- config SHA256: 087c5175f43ec13c0c9195fe782a1f5b39ba1e466039aa1137ab9b6147cfb932
- script SHA256: 3c64a2ee49f1dc196558578171478e1c072d6fef5317bcacfcc2d536f9758d1a
- operating_points.csv SHA256: 41fb68238994d3fb0e6df9c4a117dac66d1a6c530f051af61d09ba684959ab10
- por30_events.csv SHA256: 6231647b9528d0774120bbe15e77d93629aaccbcac47243a1adb2b98b7c36843
- summary.json SHA256: f182356c6d8ec6f758d9a55a7b33b52dfbe216d4fc486b0497736e6b7323fe9d
- ignored run log SHA256: 54990e40e957891f17ca6a3eede2b9c524b1c173bbe10256dd49abfabadb6cff

Next:
- Extend the verified evaluator toward the frozen A4 matched-write-rate protocol before any fresh final DEV evaluation.

## EXP038 - A4 rate-selector sanity

Status: SELECTOR SANITY PASS / NOT PERFORMANCE EVIDENCE

Frozen implementation commit:
- 59395074dc98b38d0c1feeca18a205c48b4f52a8

Defect-fix and successful executed commit:
- 4a5f44fc226be88a9c074cb67b69572b22cffda3

Scope:
- TRAIN-exposed video: 0442a954.
- Frames: 60.
- Object IDs: [1, 2].
- Variants: B1, B2, B3-S, B3-R.
- A4 target-order search began at 0.5 and all required sanity variants matched there.
- Fresh final DEV touched: false.
- TEST touched: false.
- B5 included: false.

Observed:
- B1: tau 0.35, write rate 0.4915254237288136, absolute error 0.008474576271186418, 1 midpoint refinement.
- B2: tau 0.1875, write rate 0.5084745762711864, absolute error 0.008474576271186418, 3 midpoint refinements.
- B3-S: tau 0.2, write rate 0.5084745762711864, absolute error 0.008474576271186418, 0 midpoint refinements.
- B3-R: tau 0.25, write rate 0.4915254237288136, absolute error 0.008474576271186418, 1 midpoint refinement.
- All four variants satisfied the locked A4 +/-0.02 write-rate tolerance.
- Total executed tau points: 49.
- subset_common_target_not_final_r_star: 0.5.
- Maximum observed peak allocated VRAM: 6.266784191131592 GB.
- Final status: EXP038_A4_SELECTOR_SANITY_PASS.

Execution provenance:
- Original frozen execution failed before a scientific result because Runner did not retain exp037.
- The dependency-only fix was committed separately; A4 scientific rules were unchanged.
- One fixed foreground execution was manually interrupted and produced no final result.
- The subsequent background retry completed successfully.

Interpretation:
- Closed-loop A4 write-rate-only threshold selection is verified for the four gate variants in this TRAIN-exposed sanity scope.
- Coarse thresholds plus deterministic midpoint refinement are operational.
- The selected 0.5 target is not final r_star.
- B5 and fresh DEV are still required before final common-rate selection.
- No comparative performance or final statistical conclusion is supported by EXP038.

Artifacts:
- config SHA256: e53a1fdce71e11bd1fff8fcd0bd8b70a5f1291e87f39df8581ab529d84be1636
- script SHA256: bf81cea8e531b08d3ef432d985f4f09971d24fef51338c61896b616c88ffdb9f
- executed_tau_points.csv SHA256: db93bd1dd30698908f4f777ceee043e4cefb3a8518c239253b11c36049e5b1cb
- summary.json SHA256: 1f313044e894f0533ca669a90544ad7eb324db92e40440b556a2fa3271bcf0f6
- target_selection.csv SHA256: b306642d55b3cb4d1a59b14f6275aec00b8a6c87808932852e8a9ef13c5275a6
- original failed-run log SHA256: c91d498e74bce5d3335a18ac6cb87eb394276bdc01ae8f52118cfc00e7d06e4c
- interrupted fixed-run log SHA256: 9b1e0c638351003f469ae9614a4a2ce3fccc83560184feede23e1e136a32f3c1
- successful retry log SHA256: 6e8ba8cc0492bcf555d4e709ae16c93e33a40d604ee891c3fd41c8ab5a9ddb24

## EXP039 - B5 DMS-lite write-side comparator sanity

Status: B5 SANITY PASS / NOT PERFORMANCE EVIDENCE

Frozen implementation commit:
- 4b8b54f058d2187e665d4ea5c5c44015933a2d8a

Scope:
- TRAIN-exposed video: 0442a954.
- Frames: 60.
- Object IDs: [1, 2].
- Variant: B5 DMS-lite write-side comparator.
- Fresh final DEV touched: false.
- TEST touched: false.

Observed:
- Final status: EXP039_B5_DMS_LITE_SANITY_PASS.
- First matched sanity target: 0.5.
- Selected tau: 0.775.
- Realized physical write rate: 0.4915254237288136.
- Absolute rate error: 0.008474576271186418.
- Midpoint refinements: 2.
- Executed tau points: 13.
- Formula rows checked: 1534.
- Actual fail-closed rows observed: 0.
- Synthetic and patched-path FAIL_CLOSED checks passed.
- Exact executed-row B5 formula checks passed.
- A3 frame-min and action-rule checks passed.
- Physical block-integrity checks passed.
- Maximum observed peak allocated VRAM: 6.266374588012695 GB.

Interpretation:
- Frozen A8 B5 scoring is operational on the frozen SAM3 substrate.
- B5 uses the frozen A3 physical whole-frame write intervention.
- Frozen A4 write-rate-only threshold selection is operational for B5 in this TRAIN-exposed sanity scope.
- Target 0.5 and tau 0.775 are not final r_star.
- EXP039 provides no final performance or statistical inference.
- B5 is DMS-lite write-side adaptation and is not claimed to reproduce official SAM3-DMS.

Artifacts:
- config SHA256: e48a09e27bc6807b7d84a7eb7b517171ace93fbdba339fede109d6a95ee5f015
- script SHA256: a5380f390ec3559b6a44abc14927d99ed4db594cd1056096d68dcb77ce3d8635
- summary.json SHA256: 0849204aab608f9e8424d002c00e591847f6576e0c80f44368fc56a95ac8f8e7
- contract_checks.json SHA256: 5f46691162cc8e7f0da7599283c370afd86ece2aa7bb22a70fd278b77667e5a7
- target_selection.csv SHA256: 3ff941aff44317c208b950622d91e55b22a7d368e8b935600d7e90e26db82e3e
- executed_tau_points.csv SHA256: 5937845489dd414a7fdf60bd28650ce99eb280c0c9638f21ce9e1ac5fb1ba2c9
- ignored execution log SHA256: 26e193f69cc6a0e9f64c66cf9f933e229cf975517da7299393eb35af673082a5

## EXP036 - Whole-scene manual adjudication complete

Status: MANUAL ADJUDICATION COMPLETE / NOT FINAL WS PROTOCOL COMPLETE

Observed:
- Total scenes: 1818.
- Filled: 1818.
- Remaining: 0.
- NORMAL_OCCLUSION: 1629.
- WHOLE_SCENE: 189.
- Scene-ID set matched frozen EXP034 disagreement population.
- Fresh final DEV touched: false.
- TEST touched: false.
- Three blinded replacement audit scenes remain required.

Artifacts:
- review_labels.csv SHA256: 9b76ae3bc722db7c6138cb1f2b4ab567990e6921ec2be7caf6dc0d2b239e51a4
- source CSV SHA256: 1f6fa7ad1e10662ff0333331e6d1d5a1127e89c710b0e6d9768cd206a34d63d3
- config SHA256: 32dcd845a9bae9cfa12dd9d90d1c9578db2134bf09c7f3a5995b4f30c22b6102
- script SHA256: 955e53b572d825fe060ba5122114a1fee6789a6946dbf5ae991c4562973bcf47
- final_summary.json

## EXP040 - Blinded audit replacement sample executed

Status: REPLACEMENT SAMPLE EXECUTED / MANUAL AUDIT PENDING

Observed:
- Replacement eligible population: 2149.
- Replacement seed: 34035.
- Replacement count: 3.
- Replacement IDs: 5r6uxga7:WS001, of2thxpc:WS001, 0fc00006:WS001.
- Final valid audit population: 30.
- Pixel outcomes used for replacement selection: false.
- SAM/gate outcomes used for replacement selection: false.
- Fresh final DEV touched: false.
- TEST touched: false.

Artifacts:
- final_audit_source.csv SHA256: fe0bc24767fc0a2177f37cd8b1fe68ecf32fadde363df7a8cc31a5e750d009be
- replacement_manifest.json SHA256: eb975d4fff3eefb8345360c495c35cc128f4fe83a918215b1afc7da7e1e527af

Open:
- Complete blinded manual labels for all 30 final-audit scenes.

## EXP040 - Final blinded whole-scene audit complete

Status: COMPLETE

Observed:
- Final blinded audit n: 30.
- Agreement: 27/30 = 0.900000.
- Mismatches: 3.
- Manual WHOLE_SCENE: 7.
- Manual NORMAL_OCCLUSION: 23.
- No preregistered audit acceptance threshold.
- No EXP034 threshold changes allowed from this result.
- Fresh final DEV touched: false.
- TEST touched: false.

Artifacts:
- review_labels.csv SHA256: a73ee5ac1b7a327fef82c498c5d6ce89f8a621bfc9da288c31f116b389847724
- final_audit_source.csv SHA256: fe0bc24767fc0a2177f37cd8b1fe68ecf32fadde363df7a8cc31a5e750d009be
- EXP034 per_scene.csv SHA256: aefb19804adf093dfc94cc80aa50dc53e69df83bab2c2ab8ba5d6acd603acd2f
- final_audit_result.json

## EXP041 - Final whole-scene labels frozen

Status: COMPLETE

Observed:
- Total candidate scenes: 2179.
- WHOLE_SCENE: 550.
- NORMAL_OCCLUSION: 1629.
- Auto-confirmed: 361.
- Manual adjudicated: 1818.
- Fresh final DEV touched: false.
- TEST touched: false.
- Thresholds changed: false.

Artifacts:
- final_ws_labels.csv SHA256: a14d9c83b0ddbc62c0cf3bae404950867259b96c773a7db491375ec74f37689e
- whole_scene_scene_ids.json SHA256: 22cd1e9f208183d912c5dc69ec383911e1cf5d93b90aa2c99a5bec2d93edb13f
- summary.json SHA256: 4873a1cf33a7f6d28f767ffef80ea6a4eeeca9aa9a6ec7e65f5cbbee70dd20d3

## EXP042 - Development exposure boundary reconstructed

Status: RECONSTRUCTION COMPLETE / FINAL EXCLUSION LOCK PENDING

Observed:
- Reconstructed exclusion boundary: 175 unique videos.
- Added beyond v1: 94.
- Legacy overlap with EXP017 TRAIN+DEV: 46.
- EXP021 outside boundary: 0.
- Later explicit IDs outside boundary: 0.
- Four later summary files have no explicit video IDs and require provenance resolution.
- Final split constructed: false.
- DI-v1 defined: false.
- Fresh final DEV touched: false.
- TEST touched: false.

Artifacts:
- development_exclusions_v2.json SHA256: c3346825babb6a9c28858cbf84022cb9719950f02fbdffd77a21c9dfd5e19240
- coverage_report.json SHA256: 1514bc83109e3845fa0a97eae8a48b4d245f437ccef765b24170f5245f4b11cf
- summary.json SHA256: 66403caa4308bbad7d76e0847737eb70d2a5efe75d4b79241d08ca756633629d

## EXP043 - Final known development exposure lock

Status: COMPLETE

Observed:
- Known development-exposure boundary: 175 unique videos.
- train18 provenance videos: 18.
- New IDs from provenance closure: 0.
- Provenance closure: PASS.
- P2 status: UNRESOLVED_UNGROUNDED_REFERENCE.
- DI-v1 defined: false.
- Final split constructed: false.
- Fresh final DEV touched: false.
- TEST touched: false.

Artifacts:
- development_exclusions_locked.json SHA256: 4f663bf3a6532fcac662e0ef9afa9af04609de1872f36bab4f0608ceca32333d
- provenance_closure.json SHA256: 4de201f9b8c2a7b8d68dc9aedd5d5e9349c4732ff4092160d0f6b96b600dc23e
- summary.json SHA256: 08d78028ecdefb780b00b241b6bba1c885f8292652b0c369ee9ac5084b0bd63e

## EXP044 - Final event pool join

Status: COMPLETE

Observed:
- All events retained: 4469.
- Event-bearing videos: 1691.
- Primary eligible events: 2701.
- Primary eligible videos: 1170.
- Whole-scene control events: 1188.
- Whole-scene control videos: 437.
- Mixed WS/non-WS videos: 104.
- Development-exposed videos: 175.
- DI-v1 frozen: false.
- Final split constructed: false.
- Fresh final DEV evaluated: false.
- TEST evaluated: false.

Artifacts:
- event_pool.csv SHA256: d1c8bb0121796138e6f355fbb4a03d86dcc5103bbfff7d6612e40b950105d285
- per_video_pool.csv SHA256: 957d48e553ac9887cae78fa33b05b069a7fd2df10b7ba4102d6c1569a816d977
- summary.json SHA256: 098b0af57d300824d2100619df2068b01c4baf0fd6bf9ecee3360f759f54185a
