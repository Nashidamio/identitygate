Haan — ami ekhon shuru theke ekdom current point porjonto IdentityGate thesis-er full research history ekta coherent record-er moto likhchi. Eta short handoff na; eta emon vabe likhchi jate 2–3 week poreo porle bujha jay keno kon decision neya hoisilo, kon experiment ki prove korse, kon result pore correction hoise, kon jinis supersede hoise, ar exactly akhon amra kothay asi.
Important ekta rule throughout: purono plan/Claude handoff-e jodi kono statement pore actual code/terminal evidence diye vul prove hoye thake, ami latest verified evidence-ke canonical dhorchi. Tai ekhane VERIFIED / OBSERVED / LOCKED / SUPERSEDED / OPEN / PROPOSED / UNKNOWN distinction maintain korchi.

IdentityGate Thesis — Full Chronological Research History
Start → Current state, 28 August 2026
1. Thesis-er core idea kothay theke start
Thesis-er central scientific problem holo SAM-family video tracker-er memory. Video object segmentation/tracking-e tracker previous frame-er information memory-te store kore future frame predict kore. Problem holo occlusion, small object, similar-looking distractor, target disappearance/reappearance-er moto situation-e current prediction wrong hole oi wrong prediction memory-te dhukte pare. Tarpor oi corrupted memory future prediction-ke aro wrong direction-e niye jete pare — basically error accumulation, identity drift, ba wrong object-er identity dhore fela.
Ei problem theke IdentityGate-er core research question lock kora hoy:
“What information should a memory-write gate use — quality signals, temporal signals, or identity signals?”
Ei question-ta amader project-er backbone. Eta ekhono unchanged/LOCKED.
Original idea holo SAM-ke retrain na kore SAM-er memory-write path-er shamne ekta choto supervised gate bosano. Gate-ta per object, per frame decide korbe: current prediction-er memory write-ta safe kina. Safe hole write allow; unsafe hole block/filter. SAM frozen thakbe, gate alone train hobe cached signals-er upor. Original plan-e target architecture chilo choto MLP, roughly few-thousand parameters, hard cap <50K.
Scientific comparison-ta initially chilo:
B0 = vanilla SAM baseline.
B1 = manually designed rule-based gate.
B2 = learned quality + temporal signals.
B3 = B2 + identity/pointer information.
Main scientific interest holo simply “gate useful kina” na. More specifically:
B2 vs B3: identity information quality+temporal signal-er beyond e actually extra predictive information dey kina?
Ekta negative result-o acceptable: jodi B3 B2-ke beat na kore, tao thesis answer dite pare je SAM memory admission-e identity features additional information dey na, ba limited context-e dey. Ei controlled scientific framing ta shuru thekei important chilo.

2. Methodology-r original architecture
Historical implementation plan-e IdentityGate-er intended input family chilo 11ta core feature.
Quality/temporal family chilo Feature 1–8: mask confidence/IoU score, visibility/object-presence score, normalized mask area, rolling-clean area ratio, frame-0 anchor area ratio, previous-mask temporal IoU, centroid displacement, frames-since-last-clean-write.
Identity family chilo Feature 9–11: current object pointer similarity with rolling clean reference, permanent frame-0 identity anchor, ar EMA-clean reference.
Historical B2 features 1–8 pabe; B3 features 1–11 pabe. Feature family assignment arbitrary vabe change kora jabe na, karon thesis-er scientific comparison-er meaning etar upor depend kore.
Gate-er output intended:
p_safe = calibrated probability that this memory write is safe.
Then two thresholds:
tau_admit: main memory-te write allow korar threshold.
tau_clean: clean/trusted memory-te promote korar stricter threshold.
Constraint:
tau_clean > tau_admit.
Meaning, ekta prediction native memory-r jonno acceptable hote pare, but identity-reference update korar moto trustworthy na-o hote pare.
Clean buffer-er design-e permanent frame-0 GT anchor, several rolling clean entries, ar later EMA-clean embedding chilo. Eta important because rolling trusted memory polluted hole frame-0 anchor circularity-r against independent reference hisebe thakbe, while EMA-clean long-term appearance change handle korbe. Ei clean-buffer concept-ta entirely novel memory structure bole claim korar plan chilo na; DAM4SAM-er trusted-memory precedent acknowledge kore amader novelty narrower rakha hoy: learned calibrated admission into trusted memory.

3. Research workflow establish kora
Project shuru thekei amra decide kori je eta generic coding project-er moto chalano jabe na. Every significant experiment-er provenance maintain korte hobe: experiment ID, purpose, dataset, split, config, checkpoint, seed if relevant, git commit, command, output, metric, interpretation.
Same sathe ekta strict distinction established hoy:
PLANNED mane plan-e ache, implement hoy nai.
VERIFIED mane actual code/test diye confirm kora.
OBSERVED mane measured result.
OPEN mane decision/verification baki.
LOCKED mane scientific decision freeze.
SUPERSEDED mane historical decision valid provenance but current methodology na.
Eta important chilo because early project-e plan document onek jaygay confident language use korleo actual code later different fact reveal korse.

4. Environment setup — project-er actual engineering beginning
Initial machine:
Windows 11 host, NVIDIA RTX 4080 SUPER 16 GB GPU, 64 GB system RAM.
SAM-family repo Linux-centric bole native Windows-er bodole WSL2 + Ubuntu 22.04.5 use korar decision neya hoy. Reason chilo shell scripts, CUDA ecosystem, package build behavior, ar Linux-first research repo compatibility.
WSL install-er age storage check kora hoy, because WSL virtual disk ext4.vhdx grow kore and C drive unexpectedly fill hole project risk hote pare. C drive-e roughly 597 GB free thakay default WSL placement accept kora hoy.
WSL resources pore tune kora hoy:
56 GB RAM limit,
16 GB swap,
20 processors.
Actual WSL-e around 54 GiB usable memory dekha gechilo.
Miniconda install hoy. Conda-forge strict priority use kora hoy. identitygate namer conda environment create kora hoy.
First major engineering mistake
Initially Python 3.11 + older PyTorch/CUDA combination choose kora hoy — roughly PyTorch 2.5.1 + cu124.
Pore SAM 3 official requirements verify kore dekha gelo eta wrong foundation: SAM 3 Python 3.12+, PyTorch 2.7+, CUDA 12.6+ expect kore.
So amra “somehow make it work” kori nai. Old environment destroy kore clean rebuild kori.
Final verified environment becomes:
Python 3.12.13
PyTorch 2.10.0+cu128
Torch CUDA 12.8
GPU RTX 4080 SUPER
CUDA capability sm_89.
nvidia-smi host driver capability CUDA 13.1 dekhachhilo, but amra correctly interpret kori je eta driver-supported maximum runtime, PyTorch build version na. Tai unnecessarily CUDA 13 chase kori nai.
GPU compute-o explicit matrix multiplication sanity test diye verify kora hoy.

5. Git/GitHub research provenance establish kora
Repo root:
~/thesis/identitygate
Third-party code deliberately repo-r baire:
~/thesis/externals/
Initial directory scaffold:
docs/
configs/
scripts/
identitygate/
tests/
data/raw
data/cache
data/labels
logs/
experiments/
checkpoints/
RESEARCH_STATE.md ar EXPERIMENT_REGISTRY.md project-er persistent research bookkeeping hisebe create kora hoy.
Private GitHub repo set kora hoy. Early commits chilo roughly:
b2d05a8 — repo scaffolding.
1578874 — .gitignore, README, research state/registry populate.
c8af5ed — environment freeze.
Kintu c8af5ed later stale hoy, karon eta destroyed Python-3.11 environment describe korto.
GitHub PAT setup-e kichu problem hoisilo. Ekbar token permissions wrong chilo. Ekbar secret accidentally chat-e paste hoye gechilo; token revoke kora hoy. Final policy: secrets repo/chat-e na, placeholder use, .gitignore secret patterns protect korbe.
GitHub use korar scientific reason-o chilo: later preregistration freeze, exact commit provenance, public timestamp, reproducibility.

6. SAM install — dependencies and upstream problems
Official SAM repository clone kora hoy and exact source commit pin kora hoy:
8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
Editable install use kora hoy.
Installation-e actual upstream/package issues paoa jay.
setuptools 82-te pkg_resources remove hoar karone import fail korto. Fix: setuptools < 82.
einops and pycocotools runtime import hoy but required path-e properly declared chilo na; explicitly install korte hoy.
scipy stats/mask operation-er jonno retained.
FlashAttention-3 available chilo na. GPU Ada architecture; eta core correctness requirement chilo na. VOS path FA3 chara run kore.
Ekta reproducibility issue-o identify kora hoy: ordinary pip freeze SAM-ke sam3==0.1.0 bole record korte pare, but PyPI theke oi exact source reconstruct hobe na. Tai actual reproducible identity holo git commit + editable install, not pip version string. docs/SAM3_INSTALL.md-e eta document kora hoy.

7. Checkpoint/substrate niye first assumption
Early historical plan-er language chilo SAM 3.1 video object segmentation.
Machine-e true SAM 3.1 multiplex checkpoint-o chilo, around 3.3 GB, and older/non-multiplex SAM checkpoint-o chilo.
Initially assumption chilo amra SAM3.1 terminology use kore VOS/PVS substrate-e kaj korchi.
Eta later ekta major correction hoy — ami later section-e exact correction explain korbo. Current thesis core actually SAM 3, not true SAM 3.1 Multiplex.

8. EXP001 — memory interface-er first real probe
EXP001-er objective chilo memory write interface locate kora and identity pointer accessible kina verify kora.
Initially Multiplex path inspect kora hoy.
Ekta code-reading error-o ekhane dhora pore: _tracker_update_memories initially wrong class Sam3MultiplexTrackerPredictor-er method bole dhora hoy and wrong line ~2461 record hoy.
Pore actual source inspect kore correction:
Method belongs to:
Sam3MultiplexBase
roughly line 2502.
Historical docs/audit/signal_schema.md-e wrong line number preserved ache as provenance; later docs supersede kore.
EXP001-e important finding chilo per-object pointer information exists.
Multiplex-side obj_ptr observed:
[1, 16, 256] bf16.
Other observed memory tensors included multiplex maskmem_features [1,256,72,72].
This was a major go/no-go because identity signals unavailable hole original B3 scientific question collapse korto.
Kintu important nuance: EXP001-er pointer evidence multiplex-specific chilo. Later amra actual VOS core-e pointer abar verify kori.

9. EXP002–EXP004 — 16 GB GPU substrate decision
Object Multiplex run kore VRAM problem immediately clear hoy.
EXP002-e VRAM-saving settings, object limits, CPU offload try kora hoy.
Result around:
21.77 GB peak VRAM.
16 GB GPU-r moddhe real fit na. WSL/system-memory spill diye run korte pare, but eta reliable thesis-scale mode na.
EXP003-e dekha hoy shorter clips memory solve kore kina.
5 frames ≈ 21.57 GB.
30 frames ≈ 21.77 GB.
So memory cost basically clip length accumulation issue chilo na. Short video diye true problem solve hobe na.
EXP004-e non-multiplex VOS path test kora hoy.
Peak VRAM roughly:
7.54 GB
Speed roughly:
10–11.5 iterations/s
versus multiplex around 1 it/s.
Tai evidence-based decision neya hoy: VOS/PVS mode thesis core-er engineering substrate. Historical plan etake VOS/PVS locked bole record kore.

10. EXP005 — memory encoder actually koto bar fire kore?
Amader thesis premise-er jonne critical question chilo: SAM memory encoder ki selective, naki every tracking step-e run kore?
Hook Sam3TrackerBase._encode_new_memory-te deya hoy.
EXP005-e 31 propagated frames-e:
31 memory encodes.
Pore 97-frame real video-te:
97/97 memory encodes.
Ekta early report-e only “3 calls” dekhano hoyechilo, but eta model behavior chilo na — logging code only first 3 call print korto. Eta explicitly correct kora hoy.
VOS path-e observed memory feature:
maskmem_features [1,64,72,72] bf16.
Multiplex-er [1,256,72,72] ar VOS-er [1,64,72,72] difference genuine architecture-path difference.
Actual VOS API-r kichu non-obvious requirement-o ekhane discover kora hoy:
predictor.backbone = m.detector.backbone na korle "Image features for frame 0 are not cached".
propagate_preflight=True na dile "No points are provided".
Ei bugs/API requirements memory theke guess kora hoy nai; actual runs diye verified.

11. MOSEv2 dataset acquisition and audit
Primary dataset choose kora hoy MOSEv2.
Dataset D drive-e store kora hoy, WSL path:
/mnt/d/thesis_data/mosev2
Download size around 79 GB; train extracted around 57 GB.
Important correction:
Historical plan-e training videos approximately 3,466 bola chilo.
Actual scan:
3,666 train videos.
MOSEv2 valid partition-e 433 videos ache, but only first-frame masks available. Amader frame-by-frame target IoU, theft labels, POR, reappearance analysis lagbe; tai valid split unusable.
Resulting methodological decision:
all thesis TRAIN/DEV/TEST cohorts must be constructed from MOSEv2 train, while preventing leakage at the video level.
Mask encoding verified:
PIL mode P, uint8.
pixel 0 = background.
non-zero integer = object ID.

12. EXP006 — exact disappearance/reappearance event census
Event semantics formally implement kora hoy.
Visibility:
GT mask area > 0.
Qualifying recovery event:
object once visible hoar por at least 5 consecutive absent frames, tarpor abar visible.
reappear_frame = gap-er pore first visible frame.
gap_len = exact absent run.
Later disappear_start = reappear_frame - gap_len.
Full 3,666-video scan-e final reconciled counts:
7,631 object tracks
4,469 qualifying events
1,691 event-bearing videos
3,237 event-bearing tracks
Historical filter exploration:
event-bearing: 1,691 videos / 4,469 events.
≥2 objects: 641 videos / 2,497 events.
track ≥20 visible frames: 543 / 1,751.
video ≥60 frames: 291 / 1,023.
video ≥100 frames: 122 / 491.
Early event counter-e frame-alignment bug dhora pore and corrected. Eta important because later all subsequent event scripts exact 4,469 historical count reconstruct korte hobe — otherwise event semantics drift dhora hobe.

13. EXP007 — first real MOSEv2 baseline run
Video:
6042d64a
97 frames, 2 tracked objects, 6 qualifying GT recovery events.
Protocol:
frame 0-te GT masks prompt kora hoy. Random click na. Eta PVS/semi-supervised VOS protocol.
Result:
97/97 frames propagated.
97 memory encodes.
Peak VRAM ≈ 5.9 GB.
Speed ≈ 6.81 it/s.
Initial metric report wrong chilo.
Initially GT-absent frames-ke IoU 0 dhore mean kora hoy, resulting mean IoU ~0.517 and extra “failures”.
Eta scientifically wrong because target genuinely absent hole segmentation overlap 0 mane tracker necessarily failure na.
Correction:
GT and prediction both relevant/present evaluation frames-e mean IoU:
0.573.
Approximately:
37 frames IoU ≥0.7.
15 frames IoU <0.3.
7 hallucination frames.
3 misses.
92 correctly-absent object-frame states.
This correction methodology-te permanent hoy: GT absent frames tracking IoU mean-e ordinary zero hisebe count kora jabe na.

14. EXP008 — memory pollution-er visual proof
EXP007-e seven frame paoa jay jekhane GT object absent but tracker nonzero mask predict kore.
Predicted mask sizes roughly 457–1,287 pixels.
EXP008 raw image | GT | prediction triptych generate kore ei cases visually inspect kore.
Ekhane thesis-er motivational failure mode abstract speculation theke observed machine behavior hoy:
tracker object absent frame-e hallucinate korche, ar memory encoder same tracking flow-te execute korche.
Important modern nuance: later amra bujhi “memory encoder execution” ar “valid identity pointer” same jinish na. Absent state memory encode hote pare while pointer becomes no-object sentinel. Kintu EXP007/008 still valid evidence je baseline tracking state can propagate/encode undesirable absent-state predictions.

15. Early visual correction — 6042d64a object identity
EXP009-er historical visual-audit phase-e zoomed crops dekhe bujha hoy object-gulake loosely “cyclists” bola vul chilo.
Object 2 camera-wearer-er leg/knee area.
Object 1 distant rider-er upor very small annotated item.
Eta seemingly minor, but research discipline-er example: visualization chara semantic guess reliable na.

16. EXP009 — frame-0 anchor feasibility
Clean-buffer design permanent frame-0 identity anchor-er upor depend kore. Question chilo: MOSEv2-te every tracked object ki frame 0-te actually present?
Full metadata/annotation check:
all 7,631 tracks first_visible == 0.
Raw masks diye random videos-e independently check kora hoy: frame-0 object ID set full video-r union-er sathe match kore.
Therefore:
permanent GT-prompted frame-0 anchor is genuinely implementable.
No mid-video identity initialization trick required.
Historical v5 plan-o eta VERIFIED hisebe records kore.

17. First provisional development split and EXP010 B0 headroom
Initial candidate pool chilo convenience criteria diye reduced 291-video pool.
Seed 42 diye provisional 40-video DEV sample construct kora hoy.
Historical EXP009 manifest event count 187 bolechilo, but later EXP010 exact propagation/event recomputation 190 events paay. Eta historical minor discrepancy; later dataset redesign-er por eta final thesis split na, so eta preserved as provenance rather than silently normalized.
EXP010 vanilla baseline B0 run:
40 videos.
190 reappearance events.
POR results:
POR@15 = 0.8263
POR@30 = 0.8579
POR@60 = 0.8579
Runtime ≈ 21.2 min.
Peak VRAM ≈ 11.65 GB.
Predeclared headroom rule chilo:
jodi vanilla POR@30 > 0.80 hoy, then dataset too easy for intervention; GT-derived difficulty criteria diye harden korte hobe until approximately ≤0.70.
Observed:
0.8579 > 0.80
so tightening rule fired.
Another interesting observation:
POR30 == POR60 exactly.
Meaning ei sample-e post-reappearance recovery mostly “fast or never” — 30-er pore 60 frames porjonto new recovery hoy nai.
Identity Theft-o sparse chilo:
8 theft events across 6 tracks.
Historical v5 plan-e ei exact observation record ache.

18. POR and theft evaluator-er exact semantics
Ekhane later important hoyeche bole explicit bolchi.
Prediction mask taken from tracker output:
video_res[i,0] > 0.
Prediction and GT shape exact match assert kora hoy.
target_iou = prediction vs target object's GT IoU.
max_other_iou = same prediction-er against other tracked GT objects-er maximum IoU.
Historical theft condition:
target_iou < 0.3 AND max_other_iou > 0.5.
Theft event historically required at least 5 consecutive theft frames.
POR recovery means reappearance-er por evaluable visible frames-er moddhe target IoU >0.5.
30-frame POR primary. 15/60 sensitivity.
Later label building-e target_iou and max_other_iou crucial hoy, but EXP010/016 aggregate JSON-e full per-frame primitives save hoy nai — ei design limitation eventually EXP023 recache-er main reason hoy.

19. EXP011 — synchronized mass-disappearance anomaly
EXP010 DEV result stratify kore dekha hoy 3ta video disproportionately event count and easy recovery drive korche.
3 videos → around 82/190 events, about 43%.
Their POR approximately 0.976.
Remaining normal-looking events POR around 0.769.
Many objects almost simultaneously disappear/reappear.
Initially inference chilo “whole-scene occlusion/camera cut”.
Gap length analysis-o weird hoy:
longest gaps easiest dekhachhilo, because synchronized events long gap bins dominate korto.
This exposed an important statistical problem: one video-r 20+ synchronized object events-ke 20 independent trials dhora pseudoreplication.

20. EXP012 — dataset difficulty vs sample-size collision
Original GT difficulty levers use kore POR ≤0.70 reach korte gele sample size sharply collapse korto.
Synchronized cases exclude korle better difficulty paoa jacchilo, but original 291-video convenience pool enough volume dite parchhilo na.
Therefore initial dataset strategy scientifically unstable hoye jay.
Amra silent threshold change kori nai.
Instead larger event-bearing pool scan and supervisor decision-er direction-e jai.

21. EXP013 — full 1,691-video event-bearing attribute scan
All 1,691 event-bearing train videos process kora hoy.
Final exact output:
1,691 videos
3,237 tracks
4,469 events
Eta EXP006 full event count exactly reproduce kore.
Artifact:
experiments/EXP013_attrs.csv
Later source hash:
a4dedea00a02bc351cb1e52aa253f59cd1f7dad05fc3e7ff66ef8866251ff2a1
Ei stage-e original limited pool-er sample-size problem largely removed: now full event-bearing corpus available.

22. EXP014 — major correction: synchronized timing ≠ whole-scene occlusion
EXP011-er hypothesis blindly accept kora hoy nai.
Five synchronized clusters visually inspect kora hoy.
Result:
82 video-flagged events.
78 actual timing-synchronized events.
4 false inclusions.
Kintu visually many cases true global blackout/cut na; camera motion/out-of-view/re-entry-er moto behavior.
Therefore major correction:
The old EXP011 synchronized-video heuristic CANNOT be reused as a final whole-scene label.
Timing synchronization useful diagnostic, but semantic whole-scene classification-er jonne independent evidence required.
Eta project-er important scientific self-correction.

23. EXP015 — full-pool difficulty lever feasibility
EXP013 full pool-er upor earlier difficulty levers evaluate kora hoy.
Results:
4,469 events.
3,237 tracks.
1,691 videos.
31 non-redundant candidate rules.
17 rules retained at least 210 videos.
Conclusion:
raw candidate-pool size no longer bottleneck.
Ekhon scientifically defensible hard subset design possible.

24. EXP016 — fresh hard-stratum baseline headroom
A fresh, score-independent GT-hard criterion test kora hoy:
obj_size < 0.005
AND
n_frames >= 100
Important: individual video B0 failure dekhe select kora hoy nai.
Seed 42.
Fresh 40-video development sample, historical exposed videos exclude kore.
Sanity first:
3/3 videos complete.
7/7 expected hard events matched.
Peak ≈5.67 GB.
Then full:
40 videos.
120 total events.
112 frozen hard events.
Hard POR:
W15 = 0.5804
W30 = 0.6071
W60 = 0.6071
All-event:
W15 = 0.6083
W30 = 0.6333
W60 = 0.6333
Theft events:
2 across 1 track.
Runtime ≈19.9 min.
Peak VRAM ≈12.38 GB.
So predeclared headroom gate:
POR@30 <= 0.70
PASSED.

25. EXP017 — first “final” split lock
Based on EXP016, a GT-only hard split originally freeze kora hoy.
Rule:
obj_size < 0.005 AND n_frames >= 100.
258 hard-pool videos:
TRAIN = 100 / 260 hard events.
DEV = 40 / 129.
TEST = 118 / 343.
Separate full-event-bearing TEST:
118 videos / 295 events.
No TEST overlap with TRAIN/DEV/development exclusions.
Prediction-derived TEST cache banano hoy nai; TEST one-touch policy preserve kora hoy.
At oi moment eta LOCKED chilo.
Commit approximately:
573a0f5 — EXP017: lock hard and full-event-bearing evaluation cohorts
Kintu current methodology-te EXP017 final split na
Later supervisor-approved dataset amendment methodology change kore.
Therefore EXP017 is now:
SUPERSEDED AS FINAL THESIS SPLIT.
Delete kora hoy nai. Historical provenance hisebe ache.
Its TRAIN+DEV IDs later development-exposed boundary hisebe useful hoy, because oi videos already touched by model development; TEST untouched rakha possible.
Ei distinction khub important.

26. Supervisor dataset decision — whole-scene handling completely formalized
EXP011/014 issues supervisor-er kache neya hoy.
Supervisor whole-scene exclusion approve koren, but strong conditions-er sathe.
Whole-scene events:
delete kora jabe na.
Corpus-e tag thakbe.
Primary POR theke exclude.
Sensitivity analysis-e include kore report korte hobe.
Negative-control stratum hisebe use hobe.
Critical safety requirement:
no gate may reduce whole-scene recovery >2 percentage points vs B0.
Annotation-side candidate rule approved:
preceding 5-frame lookback.
at least 80% reference objects vanish.
vanish synchronization ±2 frames.
at least 80% of vanished objects return.
return synchronization ±2 frames.
But annotation timing alone sufficient na.
Independent pixel-side signal mandatory:
TransNetV2 for camera-cut subtype.
Plus global image/pixel/coverage evidence for blocked-lens/global occlusion subtype.
Annotation and pixel-side disagree korle manual adjudication.
At least 30 randomly sampled flagged events manually audit.
Agreement report korte hobe.
Supervisor also statistical unit lock koren:
video is the sampling cluster.
Mass simultaneous disappearances event counts inflate korte parbe na; scene-level collapse required.
Primary inferential logic:
video-cluster bootstrap.
Pooled recovery-r sathe video-balanced recovery.
McNemar only deduplicated sensitivity-style analysis.

27. Difficulty Index / new dataset-selection protocol
Supervisor full 1,691-video scan and difficulty-based selection approve koren, but B0 failures diye individual video select kora explicitly prohibit koren.
Difficulty selection must be based on outcome-independent observable covariates.
Approved components include:
identity-space distractor pressure,
gap duration,
pre-gap target size,
crowding,
reappearance displacement.
Identity confusability only frame-0 anchor descriptors use korbe so it never depends on tracker recovery outcomes.
Initial combination: within-pool percentile ranks, equal weights.
Diversity protection add kora hoy.
Original semantic category cap later visual-cluster cap-e refined:
external frozen visual backbone, ideally CLIP ViT-B/32 or DINOv2.
k-means k=20.
Selected hard set-er kono visual cluster ideally >15% na.
Leave-one-component-out stability check:
full hard set-er sathe overlap each omission-e ≥0.60.
Jodi kono one-component removal >40% hard set change kore:
STOP and rebalance DI before freeze.
Random representative stratum:
30–40 videos approved, practical default 40.
Hard set-er sathe disjoint.
Development-e untouched.
Final evaluation-e once run.
Baseline only aggregate verification gate, not individual selection criterion.
configs/DI-v1.yaml currently draft/untracked. Eta not final.

28. EXP018 — actual VOS signal probe
EXP017-er development-exposed TRAIN video 7744dc51 use kora hoy.
Attempt 1 fail:
add_new_mask()-e NumPy mask deya hoyechilo.
Expected Torch tensor.
Fix:
torch.from_numpy(a0 == oid).
Attempt 2 succeeds.
Actual Sam3TrackerPredictor values:
hidden_dim = 256
mem_dim = 64
max_obj_ptrs_in_encoder = 16.
Prompt-time per-object:
obj_ptr [1,256] float32 CUDA.
object_score_logits [1,1].
iou_score [1].
eff_iou_score scalar.
pred_masks [1,1,328,328].
pred_masks_high_res [1,1,1152,1152].
Four-object propagation:
pred_masks_high_res [4,1,1008,1008].
Memory:
maskmem_features [4,64,72,72] bf16.
Propagation output had five elements including ordered object IDs, low-res mask, video-res mask, object-score logits.
This finally resolved earlier important uncertainty:
VOS path itself exposes usable per-object obj_ptr.
So early multiplex-only pointer evidence no longer carries the thesis.
EXP018 attempt 3 was considered but never run.
Its historical files remain untracked.

29. Whole-scene EXP019 implementation
Supervisor-approved annotation rule encode kore:
configs/WS-v1.json
and:
scripts/exp019_ws_gt_tagging.py
create kora hoy.
First 10-video sanity:
14 events.
2 flagged object events.
1 collapsed candidate scene.
Exact EXP013 reconstruction = True.
Then full 1,691-video census.
EXP019 full result:
1,691 videos
4,469 raw object events
2,945 annotation-side WS candidate object events
2,179 candidate scene events after collapse
1,524 non-WS candidate object events
3,703 analysis events after candidate collapse
1,281 videos with at least one candidate
Exact historical event reconstruction true.
Status intentionally:
GT_ANNOTATION_CANDIDATES_ONLY_NOT_FINAL_WS_LABELS
because pixel cross-check unfinished.
Commit:
527b8d8 EXP019: record full WS-v1 annotation census
Important EXP019 diagnostic
Candidate rate was extremely high:
roughly 65.9% object events flagged.
Of flagged events:
1,888 had only 1 reference object.
1,057 had reference_objects ≥2.
Meaning single/sparse-object videos make 80% rule trivially permissive.
Therefore annotation candidate ≠ final whole-scene event.
Pixel-side confirmation became non-negotiable.

30. TransNetV2 preparation
Official TransNetV2 source prepared.
Source location roughly:
~/thesis/external/TransNetV2
source commit around 85cef72...
Converted PyTorch checkpoint SHA256:
214d9bada931b04fe9e7ffdc015e745d86232c9ce00e90a03c843ba7dd11d4eb
Conversion tests:
10/10 pass.
But we deliberately did not run full WS pixel classification because exact frozen pixel thresholds/matching logic remained OPEN.
Running first and choosing thresholds later would be post-hoc methodology.
So this is an intentional stop, not unfinished coding by accident.

31. Major novelty audit before relational identity design
At this point amra implementation temporarily stop kore Aug-2026 current literature re-audit kori because identity contribution-er novelty consequential.
Important papers checked included CMR, VOS-Agent, SAM3Dual, RS³-Prune, SENTRY, SAM3-DMS and other memory work.
Key landscape:
SENTRY = training-free trajectory/cycle style write validation.
SAM3-DMS = training-free per-object confidence/reliability memory selection.
Rethinking Memory Design = SAM memory architecture/memory policy study.
CMR = competitor-relative memory evidence, but primarily memory readout side.
Therefore generic claim “nobody filters SAM memory” defensible na.
Stronger niche became:
supervised + write-side + tracked-competitor/native identity + frozen SAM tracker + risk-controlled admission.
Phrase “tracked competitor” use kora important, generic “distractor” na.
Why?
Because amader relational signal only actual currently tracked objects-er anchor embeddings against compare kore; unknown external distractor detect kore na.

32. Claude independent audit and B3 architecture refinement
Claude-ke ekhane independent auditor hisebe use kora hoy, project manager na.
Audit kichu important conceptual issues point kore.
Original identity-theft label relational: wrong target prediction on another GT object.
But historical B3 mostly self-identity similarity use korto.
Tai scientific comparison strengthen korar proposal:
B2 = quality + temporal only.
B3-S = B2 + self identity.
B3-R = B3-S + explicit tracked-competitor identity information.
This structure current relational refinement-er basis.
identity_margin = self_anchor_cos - max_competitor_anchor_cos
useful derived diagnostic, but because eta self and competitor values deterministic combination, eta separate independent “third identity feature” hisebe overclaim kora jabe na.
Quality-failure/drift label particularly important because eta identity-free outcome. Identity feature jodi drift predict kore, result less circular.
Identity-theft label mechanism-specific but label itself relational by definition, so interpretation cautious korte hobe.

33. Risk metric terminology correction
Conformal/risk-control thinking-e denominator ambiguity dhora pore.
From now:
unsafe writes that were admitted / all unsafe candidates:
Unsafe Admission Rate (UAR), or polluted-write leakage.
Unsafe admitted / all admitted:
contamination proportion — descriptive metric.
Ei distinction important because “risk guarantee” kon denominator-er upor define kora holo seta otherwise scientifically ambiguous.

34. EXP020 — relational pointer feasibility on actual SAM VOS
Actual SAM source re-inspect kora hoy.
Tracker state:
obj_id_to_idx
obj_idx_to_id
obj_ids
packed object pointer row order exactly object ID mapping-er sathe align kore.
Per-object slices exact.
Video:
TRAIN 7744dc51.
Frame0 objects tested: 4.
Prompt anchors:
4 × [1,256].
Packed frame0 representation:
(4,256).
Propagation:
(4,256).
Packed vs per-object extraction exact.
This established:
actual SAM VOS per-object pointer relational comparison technically feasible.
Numerical precision issue-o identify hoy.
BF16 cosine reliable enough na.
Later FP32 mandatory kora hoy.
Frame0 different-object cosine unexpectedly very high — roughly .984–.999.
Eta first sign je raw pointer identity space might be highly compressed/confusable.
But high cosine alone utility kill kore na.
Status:
TECHNICAL_FEASIBILITY_PASS_SCIENTIFIC_UTILITY_PENDING.
Commit:
6646f32 EXP020: verify relational pointer feasibility

35. EXP021 — relational development scope freeze
We final TEST touch korte parbo na.
Old EXP017 final split superseded holeo oi experiment-er TRAIN+DEV model-exposed chilo. Tai no-new-leakage principle use kore oi 140 videos-ke development-exposure boundary hisebe dhora hoy.
TRAIN = 100.
DEV = 40.
Among them frame0-te at least 2 objects:
TRAIN 18.
DEV 8.
Total relational eligible:
26 videos.
Eligible total frames:
4,066.
Rule:
previously development-exposed AND n_frame0_objects >= 2.
No performance outcome use kora hoy nai.
Scope SHA:
e528ab0b422f57fee371425199f56193c2b54ca103836ae6cf3f364cd8f5e2b2
Commit:
809ed5d EXP021: freeze relational development scope

36. EXP022 pilot — 20-object stress test
Deterministic stress video selected:
TRAIN sfbg4630.
136 frames.
20 frame0 objects.
Selection performance-based na: maximum frame0 object count → longest → ID tie-break.
Peak GPU allocation roughly 7.47 GB.
Numerical cosine rule
Raw pointer identity cosine-e FP32 use kora mandatory hoy.
Autocast disabled.
CUDA TF32 matrix multiplication disabled.
torch.set_float32_matmul_precision("highest").
Reason:
TF32 diagonal cosine checks-e ~4.33e-4 numerical error introduce korchhilo.
TF32 off korle CPU float64 cross-check closely match kore.
This becomes LOCKED identity numerical rule.

37. EXP022 pilot CSV defect
Initial pilot commit:
671e9f7 EXP022: verify relational feature extraction pilot
But CSV serialization CRLF defect chilo.
Scientific values-er defect na, artifact serialization/provenance defect.
Corrected:
d74f126 EXP022: fix pilot CSV serialization
Historical defective commit preserved, overwritten history na.

38. EXP022-er most important semantic discovery: no-object pointer
Stress video-r early propagation inspect kore dekha gelo:
frames 1–3-e all 20 objects-er pointer same.
Across those frames-o same.
Object score logits negative.
Actual source inspect kore reason paoa jay:
is_obj_appearing = object_score_logits > 0.
Object appearing false hole actual generated pointer replace hoy learned:
no_obj_ptr.
So absent object-er pointer “identity embedding” na.
This produces an important canonical distinction:
baseline memory candidate / memory encoder execution ≠ valid identity pointer.
Current rules:
pointer_valid = object_score_logit > 0.
False hole pointer is learned no-object sentinel.
Relational identity fields invalid/undefined, so mask korte hobe.
But entire object-frame row delete kora jabe na, because baseline memory-state processing still exists.
pointer_valid automatically B3 feature baniye deya jabe na. Eta currently validity mask, predictive feature na.
Missing-identity fallback still OPEN.
Ei conceptual correction future IdentityGate architecture-er jonno central.

39. Major substrate correction — “SAM 3.1 VOS” terminology was wrong
Eta project-er biggest provenance correction-er moddhe ekta.
Actual installed model_builder.py inspect kore dekhte pai:
build_sam3_video_model() default:
load_from_HF=True
and checkpoint resolution goes to:
facebook/sam3/sam3.pt.
Meaning amader VOS/PVS path — EXP004 onward je non-multiplex tracker use kortesilo — true SAM 3.1 checkpoint use kortesilo na.
True SAM 3.1 official path is:
Object Multiplex.
Checkpoint:
facebook/sam3.1/sam3.1_multiplex.pt.
And true Multiplex measured ≈21.77 GB, exceeding 16 GB GPU.
Therefore we had a scientific naming conflict.
We did not hide it.
Research amendment presented and approved:
APPROVE SAM3 CORE + SAM3.1 TRANSFER
Current LOCKED substrate:
CORE THESIS: frozen SAM 3 VOS/PVS using build_sam3_video_model() + facebook/sam3/sam3.pt.
TRANSFER/extension: true SAM 3.1 Object Multiplex, only feasibility/transfer if hardware allows; no core statistical claim.
Old “SAM 3.1 VOS” wording is SUPERSEDED.
GT-only experiments remain unaffected.
EXP018/020 etc remain engineering evidence, but correctly named SAM 3 VOS, not SAM3.1 VOS.
Amendment commit:
c113e31 research: lock SAM3 core substrate amendment
Files included:
RESEARCH_STATE.md
EXPERIMENT_REGISTRY.md
configs/SUBSTRATE-v1.json
Substrate config SHA:
cf48aff216a544409823421993f54b42c0f2b0e9bfd8c6772b21901ba32ddb31
This is why current thesis should no longer casually be called “SAM 3.1 VOS thesis”. The question remains IdentityGate; core substrate is now explicitly SAM 3.

40. EXP022-FULL production freeze
Production relational census config:
configs/EXP022-relational-census-v1.json
Script:
scripts/exp022_relational_feature_full.py
Exact HuggingFace SAM3 revision:
3c879f39826c281e95690f02c7821c4de09afae7
Checkpoint:
facebook/sam3/sam3.pt
Checkpoint SHA:
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
Config SHA:
9feee41b85e574ec54e1493eec2e42f54b550a9fb5130124e299df038bc506ea
Script SHA:
6df00a5d16629c9495758371384744e961bfad721ccdb29d0bbc634393cfcd5b
Expected scope:
26 videos.
18 TRAIN.
8 DEV.
4,066 video frames.
22,140 candidate object-frame rows.
1,482 ordered anchor-pair rows.
TEST = 0.
Freeze commit:
035185d EXP022: freeze relational signal census

41. EXP022-FULL result
Full run completed:
26/26 videos.
Candidate object-frame rows:
22,140.
Anchor rows:
1,482.
TEST videos touched:
0.
EXP022_FULL_PASS.
Merged output hashes include:
features:
4ad377a2492413bbea025a4743ed2935bb8ea329801ec95dfae20a12d9e9ee38
anchors:
0d99c1b064e98fa78c56a39416248b63bcd98291d93f6f69ae8031fb3a2edbd5
summary:
f46c063e9f58d0a7b5216e02620def10fa24214644c29b042fefa28ca1b390dc
run log:
1b0d3e7beb33e825c399085a56ed9ff87cec2af9bde609a1e926877d59e55160
Result commit:
d370558 EXP022: record full relational signal census

42. EXP022 scientific findings
Overall valid identity pointer fraction:
0.5567299006, about 55.7%.
Every one of the 26 videos had at least some valid pointers.
No video had zero valid pointers entirely.
But huge video-to-video heterogeneity.
Examples:
DEV 1feae28e: ~94.9% valid.
DEV 5twvf4ju: only ~1.12%.
DEV r5frlifm: ~3.39%.
Other videos 60%, 80%, 85% etc.
Meaning identity signal availability itself is highly context-dependent.
Among valid pointers:
negative self - competitor margin fraction ≈ 0.716.
Absolute margin near zero:
within .01 ≈18%.
within .02 ≈36.6%.
within .05 ≈78.8%.
So simple rule “self similarity should be larger than competitor similarity” does not hold reliably.
But we explicitly did not conclude competitor identity useless.
Reasons:
raw SAM pointer is not necessarily metric-learning identity embedding.
Anchor similarities themselves extremely high.
Margin sign alone is not utility analysis.
Rows not independent; videos are clusters.
Thus EXP022 conclusion:
signal existence and degeneracy characterized; predictive utility not yet established.

43. Post-EXP022 scientific decision
Another targeted current-literature audit done.
Decision:
B3-R = KEEP, BUT MODIFY/VALIDATE BEFORE FINAL FREEZE.
Not yet final architecture.
What we need to know next:
Does explicit tracked-competitor identity information add predictive information beyond quality/temporal signals and self identity?
Especially for two outcome classes:
drift / quality failure.
identity theft.
We should compare:
B2 quality+temporal.
B3-S self identity added.
B3-R tracked-competitor information added.
Raw margin remains diagnostic.
Missing relational fallback remains open.
EXP022 alone cannot justify architecture freeze.

44. Dual-label methodology
Historical v5 already recognized one serious circularity issue.
Composite “unsafe write” can be bad because:
low target IoU,
or high overlap with another GT object,
or occlusion/reappearance context.
If identity features are evaluated against a label partly defined by wrong-object overlap, positive identity result can appear somewhat circular.
Therefore dual outcomes adopted:
Quality-failure / drift
Unsafe if:
target_iou < 0.3.
This label contains no competitor identity information.
Identity-failure / theft
Unsafe if:
max_other_iou > 0.5.
This is explicitly relational by construction.
Gray zones handled independently.
This gives a failure-typed 2×2:
quality/temporal features → drift/theft.
identity features → drift/theft.
The identity-free drift outcome is especially important for testing whether identity signals genuinely add predictive information rather than just recreate an identity-defined label.

45. EXP023-er age schema/provenance audit keno laglo
EXP022 only relational pointer features save korechilo.
But utility analysis-er jonno amader lagbe:
quality/temporal features,
self identity,
competitor identity,
GT-derived target IoU,
max-other IoU,
labels.
Repo search kore dekha gelo EXP010/EXP016 target_iou and other-object IoU compute kore, but complete per-frame cache save kore nai.
EXP022 output-e relational columns ache, but full B2 quality/temporal feature family nai.
Therefore simply CSV join kore legitimate B2-vs-B3 utility analysis possible chilo na.
Decision:
guess/approximate feature banaibo na.
Instead exact primitive signals recache korbo from frozen SAM3 core.

46. Current SAM3 feature semantics re-audit
Before EXP023 code, actual installed source inspect kora hoy.
Current tracker exposes true decoder IoU-head values.
Historical assumption chilo Feature 1 maybe object score/confidence proxy.
Current source/live probe verify kore:
Feature 1 can be actual IoU-head score.
Object presence independently available through:
object_score_logits.
So current core schema distinguishes:
Feature 1 = IoU-head/mask confidence.
Feature 2 = object-presence/visibility logit.
This is more defensible than silently using same signal for both.
Live probe verifies stored frame outputs and hook output agree exactly on object-score logits.

47. SIGNAL_SCHEMA-v1 — current feature status
Current source-based schema freeze:
Feature 1 — mask confidence / IoU-head: VERIFIED.
Feature 2 — object-presence score: VERIFIED.
Feature 3 — normalized current mask area: VERIFIED.
Feature 4 — ratio vs rolling clean mask: DEFERRED because requires actual gate-defined clean-state dynamics.
Feature 5 — ratio vs permanent frame-0 anchor: VERIFIED.
Feature 6 — temporal IoU with previous prediction: VERIFIED primitive.
Feature 7 — centroid displacement: primitive available, but exact final reference definition still OPEN.
Feature 8 — frames since last clean write: DEFERRED because requires clean-write policy.
Feature 9 — ptr similarity vs rolling clean: DEFERRED.
Feature 10 — ptr similarity vs frame0 anchor: VERIFIED.
Feature 11 — ptr similarity vs EMA-clean: DEFERRED.
Tracked-competitor relation is currently:
PROPOSED / DEVELOPMENT-ONLY, not silently replacing historical features 9–11.
This gets frozen into:
docs/SIGNAL_SCHEMA-v1.md
Commit:
b9ed7b3 EXP023: freeze current-core signal schema

48. Why EXP023 caches primitives, not final features
This is a key design decision.
Features 4, 8, 9, 11 depend on which past writes were considered “clean”.
But “clean” eventually depends on learned/calibrated gate decisions.
If we manufacture them using an arbitrary rule now, we might bake circularity into training data.
Therefore EXP023 primitive cache saves state-independent base evidence first.
Examples include:
IoU-head score.
Object score/logit.
Prediction mask area.
frame0 anchor area ratio.
previous-prediction temporal IoU.
centroid coordinates/primitives.
target GT visibility/area.
target_iou.
max_other_iou.
wrong-object ID.
raw FP32 pointer.
pointer validity.
self-anchor cosine.
Then later gate-state-dependent features can be deterministically derived once policy semantics are frozen.
This is why EXP023 is not model training.

49. EXP023 sanity
Sanity run done on TRAIN video:
7744dc51
102 frames.
7 objects.
Candidate rows:
707.
Pointer array:
[707,256].
Frame-0 anchors:
[7,256].
DEV touched:


TEST touched:


Most important cross-check against already frozen EXP022:
object-score maximum difference:
0.0
pointer-valid match:
exact
valid self-anchor cosine maximum difference:
0.0
Peak memory around 7.40 GB.
Runtime ~31.4 sec.
CSV line count:
708 including header.
Hashes:
config at that sanity freeze originally:
ec1b16fe...
primitive CSV:
d92a56bcdaad95b0812b6f7721b8c1562e91fb83104ba3bae3294813747fff2d
pointers:
1112e410021cd1024d57679c5566a14b89147e94c8bf1a5dc167bbabd1f358c6
Sanity commit:
dc8a00b EXP023: verify primitive cache sanity
Pushed to origin/main.

50. EXP023 TRAIN18 production scope
We deliberately do not immediately cache all development data.
Current next stage uses the 18 eligible TRAIN videos from EXP021 first.
Why?
Because we want feature/model-design decisions on TRAIN before repeatedly shopping DEV.
DEV should later serve confirmation/tuning roles, not become an unlimited feature-selection dataset.
Production preflight independently recomputed:
18 TRAIN videos.
Expected candidate rows:
13,524.
Expected frame0 anchor pointers:
91.
TEST touched:


Full list includes the 20-object sfbg4630 stress video and other previously exposed multi-object TRAIN videos.
The frozen sanity extractor SHA:
e3f78478171083eab8fc5955b711be39337aa2838da7a05bb6c67771a76c536b
was preserved unchanged for production extraction.

51. EXP023 production freeze
Wrapper:
scripts/exp023_train18_full.py
Config updated:
configs/EXP023-primitive-cache-v1.json
Production freeze commit:
41a31ea EXP023: freeze TRAIN18 production cache run
Pushed.
Then we caught a provenance-only defect.
The reused sanity extractor's internal per-video summary would still label status as something like SANITY_PASS, even during production wrapper execution.
Numerics were not wrong, but provenance misleading.
We fixed it before production result so per-video summary gets production role/status while preserving original extractor status field.
Corrective commit:
29f6332 EXP023: correct production artifact provenance
Pushed successfully.
Current latest verified repo:
29f6332 (HEAD -> main, origin/main)
Untracked files still intentionally present:
configs/DI-v1.yaml
experiments/EXP018_signal_probe_attempt1_failed.txt
experiments/EXP018_signal_probe_attempt2.txt
scripts/exp018_signal_probe.py
No unrelated file was accidentally committed.

52. Exactly current status — eta khub important
As of the latest terminal output you sent, full EXP023 TRAIN18 GPU cache run has NOT yet been returned/executed after the provenance correction.
What has happened:
production code frozen.
CPU preflight passed.
expected row counts verified.
config frozen.
wrapper frozen.
provenance fix committed.
origin synchronized.
What has not happened yet:
13,524-row full TRAIN18 primitive cache successful production result has not yet been reported.
So current last scientific milestone is not “EXP023 complete”.
Correct wording:
EXP023 SANITY VERIFIED; TRAIN18 PRODUCTION RUN READY/FROZEN, execution result pending.

53. Are we training the IdentityGate model now?
No.
Eta tomar ager question-er exact answer.
Current EXP023 is frozen SAM3 inference/data generation.
SAM backbone/tracker completely frozen.
No IdentityGate MLP optimizer.
No epochs.
No loss backward pass.
No learned parameters update.
No B2/B3 model fitting yet.
We are creating the canonical training evidence dataset that the eventual gate train korbe.

54. Current architecture-er scientific form
Akhon best current hypothesis ladder:
B0
Vanilla frozen SAM3 core behavior.
B1
Manual rule gate.
B2
Learned gate using quality + temporal evidence only.
B3-S
B2 plus self identity information such as permanent-anchor similarity.
B3-R
B3-S plus explicit tracked-competitor identity evidence.
This last distinction was not part of earliest v4/v5 architecture in this explicit form. Eta later relational analysis/literature/audit-er result.
B3-R not yet final-frozen.
Need utility analysis first.

55. Identity margin-er current status
identity_margin = self_anchor_cos - competitor_anchor_cos
Useful descriptive diagnostic.
But eta neither independent source of information nor automatically a model input.
Reason:
it is deterministic from self and competitor cosine.
EXP022-e margin mostly negative and small.
That does not mean signal bad.
We need evaluate self and competitor terms jointly against actual outcomes.
Therefore current policy:
self similarity explicit.
competitor similarity explicit.
margin diagnostic.

56. Pointer validity and missing-identity problem
EXP022-er biggest architecture challenge:
Only ~55.7% candidate rows have valid generated identity pointer overall.
Some videos near 95%.
Some near 1–3%.
Therefore eventual deployable gate must define what happens when identity unavailable.
Possibilities future-e thakte pare: quality-only fallback, missing-value indicators, model branch, etc.
But none is locked yet.
Important:
We are not allowed to silently make pointer_valid a predictive feature just because convenient.
Current status:
OPEN scientific/architecture decision.
Likely first utility analysis valid-pointer conditional subset-e signal usefulness establish korbe; tarpor deployable missingness strategy design hobe.

57. What we have scientifically established so far
At this point amra several nontrivial facts actually establish korechi.
SAM VOS memory path exists and is inspectable.
Memory state processing occurs per propagation frame.
MOSEv2-r real examples-e hallucinated absent-target predictions occur.
MOSEv2 full event corpus exactly characterized.
Frame0 identity anchor universally feasible in this dataset.
Baseline recovery can be too easy depending on selection, and whole-scene-like synchronized structures can massively distort pooled POR.
Events within video are not independent.
Actual SAM VOS provides 256-dimensional per-object pointers.
Absent objects can carry no_obj_ptr sentinel; so memory activity and identity validity are distinct.
Tracked-competitor similarities can be extracted at scale.
Those similarities show strong degeneracy/compression/heterogeneity.
Actual IoU-head and object-presence signals are available separately.
A reproducible primitive cache design now exists.
This is a substantial amount of de-risking before model training.

58. What we have NOT established yet
Equally important, following things are not results yet:
IdentityGate improves POR — UNKNOWN.
IdentityGate reduces polluted writes — UNKNOWN.
B3-S beats B2 — UNKNOWN.
B3-R beats B3-S — UNKNOWN.
Competitor identity is predictive — UNKNOWN.
Negative margin indicates theft — FALSE assumption; not established.
Clean buffer improves recovery — UNKNOWN.
Write blocking works correctly closed-loop — not yet final verified.
Risk-controlled threshold achieves desired UAR — UNKNOWN.
J&F non-degradation — UNKNOWN.
Final whole-scene classifications — NOT FROZEN.
Final amended TEST split — NOT FROZEN.
Final B1/B2/B3 trained models — DO NOT EXIST yet.
Final thesis TEST results — DO NOT EXIST.
These distinctions prevent us from accidentally writing thesis conclusions before experiments happen.

59. The literature/novelty position right now
Original simplistic novelty claim would have been dangerous:
“Existing systems do not gate memory.”
That is false/too broad.
There are multiple memory selection/filtering systems.
Our more precise current niche:
a supervised write-side admission study that explicitly decomposes quality/temporal/self-identity/tracked-competitor identity under a frozen SAM tracker, evaluates post-occlusion/identity outcomes under matched write rates, and adds risk-controlled admission.
SENTRY occupies generic write validation.
SAM3-DMS occupies confidence-driven per-object selection.
Rethinking Memory Design occupies memory architecture/policies.
CMR introduces competitor-relative evidence but primarily on memory readout side.
So publication potential is not “we invented memory filtering”.
It is the controlled scientific question and specific write-side relational formulation.
This is also why every major architecture freeze gets literature audit, but ordinary implementation steps do not repeatedly restart literature research.

60. Statistical philosophy we have locked
Naive frame-level p-values are not acceptable.
Video is cluster.
Primary comparative inference should respect paired design.
Baseline and IdentityGate evaluate same videos/events.
Cluster bootstrap by video is primary direction.
Confidence intervals/effect sizes matter.
Minimum practical effect was discussed around +8 pp POR in supervisor memo as planning target, not a result.
Whole-scene negative control has ≤2 pp regression bound.
Threshold comparisons should use matched write rate, otherwise a gate could trivially look safer simply because it writes almost nothing.
Final test is one-touch after freeze.
No p-value shopping.
No TEST-based model selection.
Supervisor’s paired-design/power rationale and no-baseline-failure selection rule were explicitly recorded.

61. Why matched write rate matters
Suppose B3 writes 30% of frames but B0 writes 80%.
If B3 has fewer polluted writes, that alone does not prove better admission intelligence — it may simply write less.
Therefore B1/B2/B3 variants must be compared at approximately same effective write rate as baseline/native operating point.
Also report full threshold sweep:
POR vs write rate.
ITR vs write rate.
Unsafe admission vs write rate.
This makes conclusions harder to game.
This remains an important planned part of final evaluation.

62. Why final TEST has stayed untouched
Throughout recent relational experiments:
EXP021 TEST touched = 0.
EXP022 TEST touched = 0.
EXP023 sanity/production design TEST touched = 0.
Old EXP017 TEST IDs were known as metadata but predictions were not generated after split.
After supervisor amendment, old TEST is not current final set anyway.
But the general firewall remains:
prediction-derived final TEST evidence is only generated once after final methodology/model freeze.
This is one of the strongest parts of the research discipline.

63. Git history — important milestone commits
Not every commit from day one ekhane reproduce korchi na, but scientifically important chain roughly:
1578874 — initial populated repository scaffold.
c8af5ed — initial environment freeze, later stale.
b5e9919 — SAM install/environment refresh.
e197025 — Week-1 SAM code audit / historical signal schema.
0aefb38 — EXP001 memory hook/pointer probe.
03cb8c3 — corrected EXP007 IoU accounting/visualization.
711d551 — EXP008 hallucination evidence.
5f89d93 — Week-1 / Week-2 state.
9fbec76 — EXP010 B0 headroom, POR30 .858.
2dcd019 — EXP011–013-era stratification/pool work.
37e3ed0 — EXP013–016 milestone around successful hard headroom.
573a0f5 — EXP017 historical split lock, now superseded as final split.
527b8d8 — EXP019 whole-scene candidate census.
6646f32 — EXP020 relational pointer feasibility.
809ed5d — EXP021 relational development scope.
c113e31 — critical SAM3 core substrate amendment.
671e9f7 — EXP022 pilot, CSV defect.
d74f126 — EXP022 serialization correction.
035185d — EXP022 production census freeze.
d370558 — EXP022 full relational census result.
dc8a00b — EXP023 primitive-cache sanity verified.
b9ed7b3 — current-core signal schema freeze.
41a31ea — EXP023 TRAIN18 production-run freeze.
29f6332 — current HEAD: correct EXP023 production artifact provenance.
Current local and remote:
HEAD -> main
origin/main
both at 29f6332.

64. Historical decisions that have been SUPERSEDED
Eta alada kore mone rakha important.
“SAM 3.1 VOS” terminology
SUPERSEDED.
Current core = SAM3 VOS.
SAM3.1 = Multiplex transfer extension only.
EXP017 as final thesis split
SUPERSEDED.
Still valid historical experiment.
Its TRAIN/DEV are useful as development-exposed boundary only.
EXP011 synchronized timing = whole-scene label
SUPERSEDED.
EXP014 proved timing alone insufficient.
Current WS-v1 requires annotation + pixel evidence + audit.
Historical signal schema v0
SUPERSEDED for current core feature semantics.
Current:
docs/SIGNAL_SCHEMA-v1.md.
Memory encode = valid identity pointer
SUPERSEDED conceptual assumption.
Memory state can exist while pointer is no_obj_ptr.
“identity margin negative = theft”
Never validly established; should not be used.

65. Major failures/mistakes we caught instead of hiding
Environment first created with wrong Python/PyTorch/CUDA.
Old environment destroyed and rebuilt correctly.
pkg_resources broke under setuptools 82.
Missing undeclared einops / pycocotools.
Wrong class/line number for multiplex memory method.
Logging cap made memory hook look like it fired only 3 times.
EXP006 had frame-alignment bug.
EXP007 IoU metric initially counted GT-absent frames wrong.
Large shell heredoc/paste occasionally mangled files.
A stray zero-byte main file once created and removed.
Later EXP023 probing created three accidental zero-byte root files from mangled shell input; inspected and deleted.
EXP018 first attempt passed NumPy instead of Torch mask.
EXP022 first committed pilot had CRLF CSV serialization defect; corrected in a separate provenance-preserving commit.
EXP023 production wrapper initially would mislabel production per-video summaries as sanity; corrected before run.
And biggest scientific naming issue:
we discovered build_sam3_video_model() was loading SAM3, not SAM3.1, and formally amended the thesis substrate rather than ignoring it.
Ei errors thesis weak kore nai; properly logged correction-gula actually reproducibility stronger kore.

66. Current project files that matter most
Repo:
~/thesis/identitygate
SAM:
~/thesis/externals/sam3
MOSE:
/mnt/d/thesis_data/mosev2/train/JPEGImages
/mnt/d/thesis_data/mosev2/train/Annotations
Current substrate:
configs/SUBSTRATE-v1.json
Whole-scene:
configs/WS-v1.json
EXP021 scope:
experiments/EXP021_relational_scope.json
EXP022:
configs/EXP022-relational-census-v1.json
scripts/exp022_relational_feature_full.py
experiments/EXP022_full/features.csv
experiments/EXP022_full/anchor_cosine.csv
experiments/EXP022_full/summary.json
Current signal schema:
docs/SIGNAL_SCHEMA-v1.md
EXP023:
configs/EXP023-primitive-cache-v1.json
scripts/exp023_train18_primitive_cache.py
scripts/exp023_train18_full.py
sanity:
experiments/EXP023_sanity/...
Full TRAIN18 output directory is intended as:
experiments/EXP023_train18_cache
but successful full result is not yet verified.

67. So, from a thesis-phase perspective, amra exactly kothay?
Ami eta 4 layer-e explain korbo.
Layer 1 — Environment/infrastructure: essentially done and verified.
Layer 2 — SAM substrate/API/failure understanding: extensively de-risked. Current SAM3 core checkpoint/source/pointer/memory semantics known.
Layer 3 — Dataset, labels, signals: advanced but not finished. GT event system is solid; relational signals are solid; primitive cache pipeline ready. Final whole-scene pixel labels and clean-state-dependent features remain.
Layer 4 — Actual IdentityGate learning/evaluation: ekhono main experimental work baki. Gate training starts after canonical TRAIN primitives/derived training table are ready.
So amra “halfway through training” na.
Better description:
We have nearly completed the expensive scientific de-risking and data/signal foundation, and are at the transition into the learned-gate phase.

68. Exactly what remains after current point
Immediate current unfinished operation is EXP023 full TRAIN18 primitive cache.
Once verified, scientifically next stages broadly are:
derive/validate TRAIN labels and feature families without leakage/circularity.
test incremental information: B2 → B3-S → B3-R.
decide missing identity fallback based on evidence.
finalize clean-state-dependent features.
train actual learned gate models.
then use DEV for calibration/threshold selection/architecture confirmation.
implement and verify true memory-write blocking/filtering closed-loop.
build clean buffer and recovery.
run matched-write-rate comparisons.
complete whole-scene negative-control / final dataset lock.
freeze final architecture, configs, statistical analysis, preregistration.
then touch final TEST once.
run cluster statistics, efficiency, ablations.
write thesis/results/paper/reproducibility package.
So yes — significant thesis work remains, particularly the part where we actually train and evaluate IdentityGate.
But crucially, we are no longer blindly experimenting against an unknown SAM implementation.

69. One-sentence version of everything so far
Amra ekta “SAM 3.1 memory gate” idea diye shuru kore environment + GPU + official source theke ground-up verify korte giye dataset/evaluation bugs, whole-scene confounding, pointer semantics, relational identity degeneracy, and even a fundamental SAM3-vs-SAM3.1 substrate mismatch discover korechi; prottekta problem experimentally resolve/amend kore ekhon frozen SAM3 VOS core, verified per-object 256-D identity pointers, verified dual quality/identity primitives, 4,469-event MOSEv2 corpus, leakage-controlled development scope, relational EXP022 census, and exact-reproducing EXP023 primitive-cache pipeline-er point-e eshechi — but the actual IdentityGate MLP has not been trained yet.

70. Current canonical snapshot
Core research question: LOCKED.
Core substrate: SAM3 VOS/PVS, frozen.
SAM source: 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da.
Core checkpoint: facebook/sam3/sam3.pt, exact EXP022 checkpoint SHA 9999e234....
Hardware: RTX 4080 SUPER 16 GB.
Dataset: MOSEv2 train, 3,666 videos.
Qualifying recovery corpus: 4,469 events / 1,691 event-bearing videos.
Whole-scene final labels: OPEN, pixel confirmation pending.
Historical EXP017 final split: SUPERSEDED.
Relational development scope: 26 already-exposed multi-object videos, 18 TRAIN + 8 DEV.
EXP022: COMPLETE / VERIFIED.
B3-R: KEEP, but NOT frozen final.
EXP023 sanity: COMPLETE / VERIFIED.
EXP023 TRAIN18 production: CODE/CONFIG FROZEN, run result pending.
IdentityGate model training: NOT STARTED.
Closed-loop gate: NOT YET VERIFIED.
Final TEST: UNTOUCHED.
Current Git HEAD: 29f6332, pushed to origin/main.
Current unrelated/uncommitted research files: the DI-v1 draft and three historical EXP018 files.

Eta holo amar current canonical understanding of our entire thesis journey from machine setup to exactly now. Purono Claude records, v5 plan, supervisor amendment, and newer terminal evidence merge korar shomoy ami intentionally newer evidence diye old mistakes supersede korechi — especially SAM3 vs SAM3.1, EXP017 split, whole-scene labeling, and pointer-valid semantics. Early environment/SAM audit history is independently preserved in the project handoffs, including the measured VOS-vs-Multiplex decision and actual runtime tensor/API findings. Supervisor’s later whole-scene, clustered-statistics, outcome-independent difficulty selection, and one-touch held-out requirements are also explicit amendments rather than assumptions.
Ar current point-er shobcheye important sentence: amra ekhono model train kori nai; amader next resume point holo 29f6332 theke frozen EXP023 TRAIN18 production primitive-cache run execute + verify kora.
======================================================================
71. CORRECTION APPENDED 2026-08-31 — EXP023 TRAIN18 PRODUCTION COMPLETE
======================================================================

Sections 52 and 70 above state that the EXP023 TRAIN18 production run
result was pending. That is now SUPERSEDED.

OBSERVED on the thesis PC:
  experiments/EXP023_train18_cache/  = 9.6 MB, 57 files
  per_video/ = 18 pointers.npz + 18 primitives.csv + 18 summary.json
  primitives.csv = 2.8 MB
  index.csv      = 4.0 KB
  summary.json   = 13 KB
  run log        = 307 KB (ignored by .gitignore:16 "*.log")

Verified SHA256:
  primitives.csv c13fddfcc7fe422893e5cfed86100d9a8407abb0421c6296f5ed168a34e92feb
  index.csv      9e73e27506d540aab63dd55e2a07a4b5e7e5c8e18e54b393e0891c4d1fa36db0
  summary.json   145889fc5e26c021e8e083956baa7a1b01333357f137ce3c4da42c33448946ce
  run log        e3cab3898963f559397da9b7bb35213338273fb05afbe2977ad6fa9b23e56e15

Status: EXP023 TRAIN18 PRODUCTION RUN COMPLETE, RESULT NOT YET COMMITTED.

Outstanding at this point:
  - experiments/EXP023_train18_cache/ is untracked; needs commit.
  - Run log is gitignored; decide force-add with justification.
  - EXPERIMENT_REGISTRY.md has NO entries for EXP018 through EXP023.
    Registry currently stops at EXP017 plus Amendment A1. This is a
    provenance gap that must be closed.
  - RESEARCH_STATE.md "NEXT EXACT ACTION" still shows EXP016-era text
    and is stale.
  - A stray zero-byte file "0" exists in the repo root from an
    accidental shell redirect; delete it.

Next scientific step after commit: B2 vs B3-S vs B3-R utility analysis
on the cached TRAIN primitives.
