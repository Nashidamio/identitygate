"""EXP004 - VOS-mode probe (PVS, no detector). Measures VRAM + verifies hook."""
import os, json, torch
VIDEO = os.path.expanduser("~/thesis/externals/sam3/assets/videos/0001")
OUT   = os.path.expanduser("~/thesis/experiments/EXP004_vos.json")

from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_tracker_base import Sam3TrackerBase

log = {"calls": []}

def describe(x, d=0):
    if d > 3: return "<deep>"
    if torch.is_tensor(x):
        return {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    if isinstance(x, dict):
        return {f"k:{k}": describe(v, d+1) for k, v in list(x.items())[:12]}
    if isinstance(x, (list, tuple)):
        return {"len": len(x), "first": describe(x[0], d+1) if len(x) else None}
    if isinstance(x, (int, float, str, bool, type(None))): return repr(x)[:60]
    return f"<{type(x).__name__}>"

_orig = Sam3TrackerBase._encode_new_memory
def probe(self, *a, **kw):
    r = _orig(self, *a, **kw)
    if len(log["calls"]) < 3:
        log["calls"].append({
            "n_args": len(a), "kw": list(kw.keys()),
            "arg_shapes": [describe(x) for x in a[:5]],
            "RETURN": describe(r),
        })
    return r
Sam3TrackerBase._encode_new_memory = probe

print("[1] build_sam3_video_model()...", flush=True)
m = build_sam3_video_model()
predictor = m.tracker
predictor.backbone = m.detector.backbone
print("    built", flush=True)

print("[2] init_state...", flush=True)
st = predictor.init_state(video_path=VIDEO)
predictor.clear_all_points_in_video(st)
print("    ok", flush=True)

print("[3] add_new_points on frame 0...", flush=True)
# points must be NORMALIZED (0-1) torch tensors, per notebook
pts = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
lbl = torch.tensor([1], dtype=torch.int32)
_, ids, lrm, vrm = predictor.add_new_points(
    inference_state=st, frame_idx=0, obj_id=1,
    points=pts, labels=lbl, clear_old_points=False)
print("    obj_ids:", ids, flush=True)

torch.cuda.reset_peak_memory_stats()
print("[4] propagate 30 frames...", flush=True)
n = 0
for out in predictor.propagate_in_video(st, start_frame_idx=0,
                                        max_frame_num_to_track=30, reverse=False, propagate_preflight=True):
    n += 1
print("    frames:", n, flush=True)

pk = round(torch.cuda.max_memory_allocated()/1024**3, 2)
log["peak_vram_gb_propagation"] = pk
log["frames"] = n
print("[5] PEAK VRAM (propagation only):", pk, "GB", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(log, open(OUT, "w"), indent=2, default=str)
print("HOOK FIRED:", len(log["calls"]), "times", flush=True)
