# signal_schema.md — v0

**Status:** v0 draft, Week-1 Day-1. Written from code reading only.
Every tensor shape, dtype, and per-object addressability claim marked
[unverified] becomes [verified] only after a live tracker run.

## 0. Repository facts (frozen)

- SAM 3 repo: https://github.com/facebookresearch/sam3
- Commit: 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
- Predictor built via: sam3.model_builder.build_sam3_multiplex_video_predictor()
- Checkpoint: ~/thesis/checkpoints/sam3.1/sam3.1_multiplex.pt (3.26 GB)
- Fallback: ~/thesis/checkpoints/sam3/ (SAM 3 Nov 2025, 6.5 GB)

## 1. Predictor class hierarchy

    Sam3MultiplexVideoPredictor(Sam3BasePredictor)
        -- sam3/model/sam3_multiplex_video_predictor.py:22
        routes handle_request / handle_stream_request
      -> Sam3MultiplexPredictorWrapper
            -- sam3/model/sam3_multiplex_base.py:2893
         wraps
        -> Sam3MultiplexTrackerPredictor(nn.Module)
             -- sam3/model/sam3_multiplex_base.py:42
             THIS is where the memory bank lives.
             All Track A/B overrides go here.

## 2. Public API surface (from notebook)

All external calls use one dict-typed method:
- predictor.handle_request(request={"type": <T>, "session_id": ..., ...})
- predictor.handle_stream_request(request={"type": "propagate_in_video", ...})
  -- yields per-frame response

Confirmed type values (may be incomplete):
start_session, add_prompt, remove_object, reset_session, close_session, propagate_in_video.

No memory-related type in the public API. All memory governance must be done by subclassing.

## 3. Memory-write path (Track A insertion point)

File: sam3/model/sam3_multiplex_base.py

- L2461 _tracker_update_memories(self, sam2_inference_states, frame_idx, tracker_metadata, low_res_masks)
  - L2510-2528: interpolate low_res_masks -> high_res_masks at interpol_size
  - L2530-2537: apply native suppression _suppress_object_pw_area_shrinkage(high_res_masks)
  - L2540: derive object_score_logits from mask non-emptiness
    (this is one of our quality features already computed by the tracker)
  - L2543-2560: build per-object index assignment
    (object_id_to_state_i, object_idx_assignment)
  - L2600-2620: WRITE POINT.
    output_dict[storage_key][frame_idx]["maskmem_features"]
    and ["maskmem_pos_enc"] are set from encoded_mem.

IdentityGate Track A insertion: subclass Sam3MultiplexTrackerPredictor,
override _tracker_update_memories, compute per-object features (see section 8),
evaluate p_safe, and either write / block per object at the L2600 store step.
The suppression block above is preserved -- IdentityGate stacks on native heuristics.

## 4. Track B insertion point

File: sam3/model/sam3_video_base.py
- L1253: _post_execution_phase_hook(self, tracker_states_local, tracker_metadata_new) -> pass
- L1232: called from run_tracker_update_execution_phase, AFTER memory is written

Track B override: subclass same class, override _post_execution_phase_hook,
and filter the just-written output_dict[...][frame_idx] entries where p_safe < tau_admit.
This is the SAM 3-provided extension point; it is OOP-clean, no monkey-patching.

## 5. Recovery path (Track-B algorithm target)

File: sam3/model/sam3_multiplex_base.py
- L827: _recondition_masklets(...)
  This is where the plan's tracker.recondition(mem) pseudocode maps.
  To be inspected in detail when Track-B recovery is implemented.

## 6. Native gating we stack on top of

Enumerated from multiplex_base.py:
- _suppress_object_pw_area_shrinkage (called from L2537 in the memory-write path)
- _apply_object_wise_non_overlapping_constraints (referenced as TODO alternative)
- _suppress_overlapping_based_on_recent_occlusion (L1384)
- _get_objects_to_suppress_based_on_most_recently_occluded (L1595)
- Duplicate-track removal via hotstart_dup_thresh (visible in L2461-2500)

These are the native heuristics the plan section 12 references.
IdentityGate stacks on top of them (default mode); replacement mode is a later ablation.

## 7. Per-object addressability -- KEY FINDING

Confirmed at L2543-2560 of _tracker_update_memories:

    object_idx_assignment: dict[int, list[int]] = {}
    all_object_ids: list[int] = []
    object_id_to_state_i: dict[int, int] = {}
    for state_i, sam2_state in enumerate(sam2_inference_states):
        obj_ids = sam2_state["obj_ids"]
        ...

Objects have unique obj_ids and are indexable inside the multiplex bucket.
This unlocks:
- Track A: per-object write blocking (all conditions in v4 plan section 5 are met)
- N2 lever (multiplex-aware per-object gating): architecturally feasible

Remaining unknown (must be verified on a running tracker):
- exact tensor shape/dtype of maskmem_features per object [unverified]
- whether pointer/embedding features are computed per obj_id or per bucket [unverified]
- object-order stability across frames when the multiplex bucket reshuffles [unverified]

## 8. Feature availability check vs plan section 5 spec

Plan section 5 lists 14 features across quality/temporal/identity families.
First-pass availability judgement (to be verified in Week-1 Day-2 tracker probe):

| # | Feature                          | Source in SAM 3 | v0 status |
|---|----------------------------------|-----------------|-----------|
| 1 | mask confidence / IoU-head score | object_score_logits at L2540 | AVAILABLE (proxy) |
| 2 | occlusion / visibility score     | tracker metadata obj_id_to_last_occluded | LIKELY AVAILABLE |
| 3 | mask area normalized             | derivable from high_res_masks | AVAILABLE |
| 4 | area ratio vs rolling clean      | derivable, our bookkeeping | AVAILABLE |
| 5 | area ratio vs frame-0 anchor     | derivable, our bookkeeping | AVAILABLE |
| 6 | temporal IoU with previous mask  | derivable, our bookkeeping | AVAILABLE |
| 7 | centroid displacement normalized | derivable from masks | AVAILABLE |
| 8 | frames since last clean write    | our bookkeeping | AVAILABLE |
| 9 | ptr_sim_roll (pointer cos)       | needs pointer tensor access [unverified] | UNKNOWN |
| 10 | ptr_sim_anchor                  | needs pointer tensor access [unverified] | UNKNOWN |
| 11 | ptr_sim_ema                     | needs pointer tensor access [unverified] | UNKNOWN |
| 12-14 | phi_* features               | stretch, depends on 9-11 | STRETCH |

Kill-rule reminder (plan section 12 Day-5, review F4): if features 9-11 are not
per-object addressable after Week-1 Day 5, the B2-vs-B3 co-primary must be
re-scoped in writing before Week 2.

## 9. Experiment mode decision (interim)

- Multiplex-only checkpoint is what ships. Non-multiplex would require running with
  the SAM 3 (Nov 2025) checkpoint through Sam3VideoPredictor (non-multiplex path).
- Default mode for the thesis: MULTIPLEX (SAM 3.1, mainline).
- Fallback mode: SAM 3 non-multiplex, only if per-object addressability of pointer
  features fails under multiplex (unlikely given section 7 finding).
- This is v0. Frozen mode goes in PREREGISTRATION.md at Week-6 freeze.

## 10. What v1 must add (before Week 1 ends)

1. Run predictor on one MOSEv2 clip, print the actual maskmem_features shape/dtype.
2. Locate pointer/embedding tensor per object (or record inaccessibility).
3. Test the Track-A override by writing a no-op subclass that just logs every
   _tracker_update_memories call -- verify our override actually runs.
4. Same for Track B via _post_execution_phase_hook.
5. Only after 1-4 pass, unblock feature caching (Week 2 work).
