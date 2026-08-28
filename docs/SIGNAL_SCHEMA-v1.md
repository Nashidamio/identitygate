# IdentityGate SIGNAL_SCHEMA-v1

Status: CURRENT SAM 3 CORE SIGNAL MAPPING
Date: 2026-08-28

This document supersedes docs/audit/signal_schema.md for the current
IdentityGate SAM 3 CORE substrate only.

The historical audit file is retained unchanged for provenance.

## 1. Substrate

CORE model:
- Model family: SAM 3
- Task mode: VOS/PVS
- Builder: sam3.model_builder.build_sam3_video_model
- Predictor: Sam3TrackerPredictor
- SAM source commit:
  8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
- Checkpoint:
  facebook/sam3/sam3.pt
- HF revision:
  3c879f39826c281e95690f02c7821c4de09afae7
- Checkpoint SHA256:
  9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

Actual runtime verification showed:
- use_memory_selection = True
- object_score_logits are per-object
- iou_score is per-object
- obj_ptr is per-object
- iou_score = max decoder IoU-head prediction across multimask outputs

## 2. Binary mask and GT outcome primitives

Binary prediction:
    video-resolution prediction logit > 0

No additional resize is performed after propagation.

target_iou:
    intersection(pred, target_GT) / union(pred, target_GT)

max_other_iou:
    maximum IoU(pred, other_GT)
    across other frame-0 tracked object IDs

target_visible:
    target GT pixel area > 0

These definitions match the verified historical EXP010/EXP016 evaluator logic.

## 3. Feature mapping

### Feature 1 — mask confidence / IoU-head score

Status: VERIFIED

Source:
    current_out["iou_score"]

SAM semantics:
    iou_score = ious.max(-1)[0]

This is the actual per-object IoU-head confidence and replaces the old
historical proxy assumption that object_score_logits should serve as
Feature 1.

### Feature 2 — occlusion / visibility / object-presence score

Status: VERIFIED

Primary primitive:
    current_out["object_score_logits"]

Also cache:
    sigmoid(object_score_logits)

SAM uses:
    object_score_logits > 0

to determine whether the object is appearing and whether the object pointer
is replaced by the learned no_obj_ptr sentinel.

### Feature 3 — normalized mask area

Status: VERIFIED

Definition:
    predicted binary mask area / frame pixel area

Primitive:
    pred_area_px

### Feature 4 — area ratio vs rolling clean reference

Status: DEFERRED

Reason:
Requires the IdentityGate clean-reference state and tau_clean promotion rule.

Primitive required for later derivation:
    pred_area_px

No oracle or GT-defined clean reference may be substituted.

### Feature 5 — area ratio vs frame-0 anchor

Status: VERIFIED

Definition:
    predicted binary mask area / frame-0 GT-prompt mask area

Frame-0 GT is legal because the protocol is semi-supervised VOS and frame 0
is the clean prompted conditioning frame.

### Feature 6 — temporal IoU with previous mask

Status: VERIFIED PRIMITIVE

Definition:
    IoU(current binary tracker mask, previous tracker binary mask)

For frame 1, the previous mask is the frame-0 prompted GT mask.

### Feature 7 — normalized centroid displacement

Status: OPEN FOR FINAL DERIVATION

Verified cached primitives:
    centroid_x_px
    centroid_y_px
    previous centroid_x_px
    previous centroid_y_px

Open:
The exact reference and normalization rule for the final displacement
feature is not explicitly fixed by the current implementation plan.

Do not silently choose a reference.

### Feature 8 — frames since last clean write

Status: DEFERRED

Reason:
Requires clean-write state and tau_clean.

Frame indices are sufficient to derive this later without another SAM run.

### Feature 9 — ptr_sim_roll

Status: DEFERRED

Reason:
Requires the rolling clean pointer reference.

Raw per-object FP32 pointer vectors are cached for later derivation.

### Feature 10 — ptr_sim_anchor

Status: VERIFIED

Definition:
    cosine(current obj_ptr, frame-0 clean object anchor pointer)

Validity rule:
    object_score_logit > 0

If false, SAM emits the learned no_obj_ptr sentinel and Feature 10 is
undefined/masked.

Numeric rule:
    FP32
    autocast disabled
    CUDA TF32 disabled
    torch float32 matmul precision = highest

### Feature 11 — ptr_sim_ema

Status: DEFERRED

Reason:
Requires the clean pointer EMA state.

Raw per-object FP32 pointer vectors are cached for later derivation.

## 4. Relational tracked-competitor signal

Status: PROPOSED / DEVELOPMENT-ONLY

EXP022 verified availability of:
- self frame-0 anchor cosine
- maximum tracked-competitor frame-0 anchor cosine
- competitor object ID
- identity margin

identity_margin is derived:
    self_anchor_cos - max_tracked_competitor_cos

It is not treated as an additional independently-primary feature.

This relational signal does not silently replace historical Feature 9-11.
Its final role remains subject to incremental utility evidence.

## 5. EXP023 primitive-cache rule

EXP023 caches definition-independent primitives only.

It does NOT yet construct:
- Feature 4
- final Feature 7
- Feature 8
- Feature 9
- Feature 11
- final safety labels

This prevents circular clean-state definitions and avoids another SAM run
after those downstream rules are frozen.

## 6. Statistical / exposure boundary

EXP023 full primitive caching is restricted to the 18 frozen
development-exposed TRAIN multi-object videos from EXP021.

DEV videos touched: 0
TEST videos touched: 0

Frame-level rows are not treated as independent statistical samples.
The video remains the statistical cluster for later inference.
