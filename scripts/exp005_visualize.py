"""EXP005 - Visual output + per-frame memory-write audit. Artifacts -> repo."""
import os, json, torch, numpy as np
from PIL import Image

REPO   = os.path.expanduser("~/thesis/identitygate")
VIDEO  = os.path.expanduser("~/thesis/externals/sam3/assets/videos/0001")
OUTDIR = f"{REPO}/experiments/EXP005_masks"
N_FRAMES = 30
os.makedirs(OUTDIR, exist_ok=True)

from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_tracker_base import Sam3TrackerBase

writes = []
cur = {"idx": -1}
_orig = Sam3TrackerBase._encode_new_memory
def probe(self, *a, **kw):
    r = _orig(self, *a, **kw)
    writes.append({"at_frame": cur["idx"],
                   "is_init_cond_frame": bool(kw.get("is_init_cond_frame", False)),
                   "is_mask_from_pts": bool(kw.get("is_mask_from_pts", False))})
    return r
Sam3TrackerBase._encode_new_memory = probe

print("[1] building model...", flush=True)
m = build_sam3_video_model()
predictor = m.tracker
predictor.backbone = m.detector.backbone

print("[2] init_state...", flush=True)
st = predictor.init_state(video_path=VIDEO)
predictor.clear_all_points_in_video(st)

print("[3] prompt frame 0...", flush=True)
pts = torch.tensor([[0.55, 0.62]], dtype=torch.float32)
lbl = torch.tensor([1], dtype=torch.int32)
_, ids, _, _ = predictor.add_new_points(inference_state=st, frame_idx=0, obj_id=1,
                                        points=pts, labels=lbl, clear_old_points=False)
print("    obj_ids:", ids, flush=True)

print("[4] propagate + save overlays...", flush=True)
frames = sorted(os.listdir(VIDEO), key=lambda f: int(os.path.splitext(f)[0]))
per_frame, saved = [], 0
for out in predictor.propagate_in_video(st, start_frame_idx=0,
        max_frame_num_to_track=N_FRAMES, reverse=False, propagate_preflight=True):
    fidx, obj_ids, _, video_res = out[0], out[1], out[2], out[3]
    cur["idx"] = int(fidx)
    mask = (video_res[0, 0] > 0).detach().cpu().numpy()
    per_frame.append({"frame": int(fidx), "n_objs": len(obj_ids), "mask_px": int(mask.sum())})
    if fidx % 5 == 0:
        img = np.array(Image.open(os.path.join(VIDEO, frames[fidx])).convert("RGB"))
        if mask.shape != img.shape[:2]:
            mask = np.array(Image.fromarray((mask*255).astype(np.uint8))
                            .resize((img.shape[1], img.shape[0]))) > 127
        ov = img.copy()
        ov[mask] = (0.45*ov[mask] + 0.55*np.array([255,60,60])).astype(np.uint8)
        Image.fromarray(ov).save(f"{OUTDIR}/frame_{fidx:03d}.png")
        saved += 1

pk = round(torch.cuda.max_memory_allocated()/1024**3, 2)
print(f"    saved {saved} PNGs -> {OUTDIR}", flush=True)
print("[5] MEMORY WRITE AUDIT", flush=True)
print(f"    frames propagated : {len(per_frame)}")
print(f"    memory encodes    : {len(writes)}")
for w in writes[:12]:
    print(f"      frame={w['at_frame']} init_cond={w['is_init_cond_frame']} from_pts={w['is_mask_from_pts']}")
print("    mask px first 10  :", [p['mask_px'] for p in per_frame[:10]])
print("PEAK VRAM:", pk, "GB")

json.dump({"peak_vram_gb": pk, "n_frames": len(per_frame),
           "n_memory_encodes": len(writes), "per_frame": per_frame, "writes": writes},
          open(f"{REPO}/experiments/EXP005_audit.json", "w"), indent=2)
