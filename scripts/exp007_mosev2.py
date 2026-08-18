"""EXP007 - SAM 3.1 on real MOSEv2 video, GT-mask prompt (thesis PVS protocol)."""
import os, json, torch, numpy as np
from PIL import Image

REPO  = os.path.expanduser("~/thesis/identitygate")
ROOT  = "/mnt/d/thesis_data/mosev2/train"
VID   = "6042d64a"
JPG   = f"{ROOT}/JPEGImages/{VID}"
ANN   = f"{ROOT}/Annotations/{VID}"
OUT   = f"{REPO}/experiments/EXP007_{VID}"
os.makedirs(OUT, exist_ok=True)

from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_tracker_base import Sam3TrackerBase

writes, cur = [], {"idx": -1}
_orig = Sam3TrackerBase._encode_new_memory
def probe(self, *a, **kw):
    r = _orig(self, *a, **kw)
    writes.append({"frame": cur["idx"],
                   "from_pts": bool(kw.get("is_mask_from_pts", False))})
    return r
Sam3TrackerBase._encode_new_memory = probe

masks_f = sorted(f for f in os.listdir(ANN) if f.endswith(".png"))
frames_f = sorted(f for f in os.listdir(JPG) if f.endswith(".jpg"))
print(f"[0] video {VID}: {len(frames_f)} frames, {len(masks_f)} masks", flush=True)

gt0 = np.array(Image.open(os.path.join(ANN, masks_f[0])))
obj_ids = sorted(int(x) for x in np.unique(gt0) if x != 0)
print(f"    objects in frame 0: {obj_ids}", flush=True)

print("[1] building model...", flush=True)
m = build_sam3_video_model(); predictor = m.tracker
predictor.backbone = m.detector.backbone

print("[2] init_state...", flush=True)
st = predictor.init_state(video_path=JPG)
predictor.clear_all_points_in_video(st)

print("[3] prompting with GT masks from frame 0...", flush=True)
for oid in obj_ids:
    binmask = torch.from_numpy((gt0 == oid).astype(np.uint8)).to(torch.bool)
    predictor.add_new_mask(inference_state=st, frame_idx=0, obj_id=oid, mask=binmask)
    print(f"    obj {oid}: {int(binmask.sum())} px", flush=True)

torch.cuda.reset_peak_memory_stats()
print("[4] propagating full video...", flush=True)
per_frame = []
for out in predictor.propagate_in_video(st, start_frame_idx=0,
        max_frame_num_to_track=len(frames_f), reverse=False, propagate_preflight=True):
    fidx, out_ids, _, vres = out[0], out[1], out[2], out[3]
    cur["idx"] = int(fidx)
    gt = np.array(Image.open(os.path.join(ANN, masks_f[fidx])))
    rec = {"frame": int(fidx)}
    for i, oid in enumerate(out_ids):
        pred = (vres[i, 0] > 0).detach().cpu().numpy()
        g = (gt == oid)
        inter = np.logical_and(pred, g).sum(); union = np.logical_or(pred, g).sum()
        rec[f"obj{oid}_iou"] = round(float(inter/union), 3) if union else None
        rec[f"obj{oid}_pred_px"] = int(pred.sum())
        rec[f"obj{oid}_gt_px"] = int(g.sum())
    per_frame.append(rec)

    if fidx % 10 == 0:
        img = np.array(Image.open(os.path.join(JPG, frames_f[fidx])).convert("RGB"))
        ov = img.copy()
        cols = [(255,60,60), (60,140,255), (60,255,120), (255,220,60)]
        for i, oid in enumerate(out_ids):
            pm = (vres[i, 0] > 0).detach().cpu().numpy()
            if pm.shape != img.shape[:2]:
                pm = np.array(Image.fromarray((pm*255).astype(np.uint8))
                              .resize((img.shape[1], img.shape[0]))) > 127
            ov[pm] = (0.5*ov[pm] + 0.5*np.array(cols[i % 4])).astype(np.uint8)
        Image.fromarray(ov).save(f"{OUT}/frame_{fidx:03d}.png")

pk = round(torch.cuda.max_memory_allocated()/1024**3, 2)
ious = [v for r in per_frame for k,v in r.items() if k.endswith("_iou") and v is not None]
print(f"[5] frames={len(per_frame)}  memory_encodes={len(writes)}  peakVRAM={pk} GB", flush=True)
print(f"    mean IoU vs GT = {round(sum(ious)/len(ious),3)}  (n={len(ious)})", flush=True)
print(f"    IoU<0.3 frames = {sum(1 for i in ious if i<0.3)} / {len(ious)}", flush=True)
json.dump({"video": VID, "peak_vram_gb": pk, "n_memory_encodes": len(writes),
           "mean_iou": round(sum(ious)/len(ious),3), "per_frame": per_frame},
          open(f"{REPO}/experiments/EXP007_{VID}.json","w"), indent=2)
print(f"    overlays -> {OUT}", flush=True)
