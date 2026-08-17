"""
EXP001 - Tracker probe.
Purpose: verify Track-A hook fires, and record actual tensor shapes/dtypes
         at the SAM 3.1 memory-write point.
Does NOT change tracker behavior - pure observation (calls super()).
"""
import os, sys, json, torch

VIDEO = os.path.expanduser("~/thesis/externals/sam3/assets/videos/0001")
OUT   = os.path.expanduser("~/thesis/experiments/EXP003_vram_len5.json")
PROMPT = "person"
MAX_FRAMES_LOGGED = 5

from sam3.model_builder import build_sam3_multiplex_video_predictor
from sam3.model.sam3_multiplex_base import Sam3MultiplexBase

log = {"calls": [], "errors": []}

def describe(x, depth=0, name=""):
    """Recursively describe an object's structure."""
    if depth > 3:
        return "<max depth>"
    if torch.is_tensor(x):
        return {"type": "Tensor", "shape": list(x.shape),
                "dtype": str(x.dtype), "device": str(x.device)}
    if isinstance(x, dict):
        return {f"key:{k}": describe(v, depth+1, str(k)) for k, v in list(x.items())[:12]}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "len": len(x),
                "first": describe(x[0], depth+1) if len(x) else None}
    if isinstance(x, (int, float, str, bool, type(None))):
        return repr(x)[:80]
    return f"<{type(x).__name__}>"

_orig = Sam3MultiplexBase._tracker_update_memories

def probe_update_memories(self, sam2_inference_states, frame_idx,
                          tracker_metadata, low_res_masks, *a, **kw):
    if len(log["calls"]) < MAX_FRAMES_LOGGED:
        entry = {
            "frame_idx": int(frame_idx),
            "n_states": len(sam2_inference_states),
            "low_res_masks": describe(low_res_masks),
            "tracker_metadata_keys": list(tracker_metadata.keys())
                                     if isinstance(tracker_metadata, dict) else str(type(tracker_metadata)),
        }
        # inspect first inference state
        if len(sam2_inference_states):
            st = sam2_inference_states[0]
            entry["state0_keys"] = list(st.keys()) if isinstance(st, dict) else str(type(st))
            if isinstance(st, dict):
                entry["state0_obj_ids"] = describe(st.get("obj_ids"))
                od = st.get("output_dict")
                if isinstance(od, dict):
                    entry["state0_output_dict_keys"] = list(od.keys())
                    for sk in od:
                        sub = od[sk]
                        if isinstance(sub, dict) and sub:
                            fk = sorted(sub.keys())[-1]
                            entry[f"output_dict[{sk}][{fk}]"] = describe(sub[fk])
                            break
                # hunt for pointer/embedding tensors
                for k in st.keys():
                    if any(t in str(k).lower() for t in ("ptr", "pointer", "embed", "obj_ptr")):
                        entry[f"PTR_CANDIDATE:{k}"] = describe(st[k])
        log["calls"].append(entry)
    return _orig(self, sam2_inference_states, frame_idx,
                 tracker_metadata, low_res_masks, *a, **kw)

Sam3MultiplexBase._tracker_update_memories = probe_update_memories

print("[1] building predictor (loads 3.3GB ckpt, ~1-2 min)...", flush=True)
predictor = build_sam3_multiplex_video_predictor(use_fa3=False, max_num_objects=8)
# use_fa3=False: flash_attn_interface (FA3) not installed; FA3 targets Hopper,
# we are on Ada sm_89. Standard SDPA attention used instead.
print("    built OK", flush=True)

print("[2] start_session (direct init_state; upstream start_session passes "
      "offload_state_to_cpu which multiplex init_state rejects)...", flush=True)
import uuid, time
inference_state = predictor.model.init_state(
    resource_path=VIDEO,
    offload_video_to_cpu=True,
    async_loading_frames=False,
)
sid = str(uuid.uuid4())
predictor._all_inference_states[sid] = {
    "state": inference_state,
    "session_id": sid,
    "start_time": time.time(),
    "last_use_time": time.time(),
}
log["init_state_keys"] = list(inference_state.keys()) if isinstance(inference_state, dict) else str(type(inference_state))
print("    session:", sid, flush=True)
print("    init_state keys:", log["init_state_keys"], flush=True)

print("[3] add_prompt...", flush=True)
r = predictor.handle_request(request=dict(
    type="add_prompt", session_id=sid, frame_index=0, text=PROMPT))
log["add_prompt_response_keys"] = list(r.keys())
print("    prompt OK, response keys:", list(r.keys()), flush=True)

torch.cuda.reset_peak_memory_stats()
print("[4] propagate (this triggers memory writes)...", flush=True)
n = 0
for resp in predictor.handle_stream_request(request=dict(
        type="propagate_in_video", session_id=sid)):
    n += 1
    if n == 1:
        log["first_propagate_response_keys"] = list(resp.keys())
    if n >= 5:
        break
print(f"    propagated {n} frames", flush=True)
log["frames_propagated"] = n

print("[5] peak VRAM:", round(torch.cuda.max_memory_allocated()/1024**3, 2), "GB", flush=True)
log["peak_vram_gb"] = round(torch.cuda.max_memory_allocated()/1024**3, 2)

predictor.handle_request(request=dict(type="close_session", session_id=sid))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(log, f, indent=2, default=str)
print("[6] wrote", OUT, flush=True)
print("HOOK FIRED:", len(log["calls"]), "times", flush=True)
