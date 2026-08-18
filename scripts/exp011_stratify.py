"""
EXP011 - stratify DEV POR by GT attributes to choose a tightening lever.
Plan section 17 allows: longer occlusion gaps, more nearby distractors,
smaller object size, more same-category neighbours, longer videos.
Plan section 18 allows using DEV AGGREGATE POR to pick criteria; it forbids
selecting individual videos by their own baseline score. This script only
ever reports aggregates per attribute bin.
CPU only, reads Annotations PNGs for the 40 DEV videos.
"""
import os, json
import numpy as np, pandas as pd
from PIL import Image

REPO = os.path.expanduser("~/thesis/identitygate")
ANN  = "/mnt/d/thesis_data/mosev2/train/Annotations"
r = json.load(open(f"{REPO}/experiments/EXP010_full/results.json"))
N_EVENT = 5

# per-event attributes re-derived from GT
recs = []
cache = {}
for tr in r["per_track"]:
    vid, oid = tr["video"], tr["object_id"]
    if vid not in cache:
        d = f"{ANN}/{vid}"
        pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        cache[vid] = [np.array(Image.open(os.path.join(d, f))) for f in pngs]
    arrs = cache[vid]
    nf = len(arrs)
    H, W = arrs[0].shape
    area = np.array([float((a == oid).sum()) / (H * W) for a in arrs])
    vis = area > 0
    # objects visible per frame (distractor density)
    dens = np.mean([len(np.unique(a)) - 1 for a in arrs])

    # gap length preceding each reappearance
    gaps, run, seen = {}, 0, False
    for t in range(nf):
        if vis[t]:
            if run >= N_EVENT and seen: gaps[t] = run
            run = 0; seen = True
        elif seen:
            run += 1

    for i, rf in enumerate(tr["reappear_frames"]):
        recs.append({
            "video": vid, "object_id": oid, "reappear_frame": rf,
            "gap_len": gaps.get(rf, -1),
            "obj_size": float(area[vis].mean()) if vis.any() else 0.0,
            "n_frames": tr["n_frames"],
            "n_objects": int(max(len(np.unique(a)) - 1 for a in arrs)),
            "mean_density": round(float(dens), 2),
            "por30": tr["por_w30"][i],
        })

e = pd.DataFrame(recs)
assert (e.gap_len > 0).all(), "gap re-derivation failed"

# synchronized occlusion: >=5 objects in the video reappearing within 10 frames
sync = set()
for v, g in e.groupby("video"):
    for rf in g.reappear_frame:
        if ((g.reappear_frame - rf).abs() <= 10).sum() >= 5:
            sync.add(v); break
e["sync_video"] = e.video.isin(sync)

def strat(col, bins, labels):
    b = pd.cut(e[col], bins=bins, labels=labels, include_lowest=True)
    t = e.groupby(b, observed=True).agg(events=("por30","size"),
                                        POR=("por30","mean"),
                                        videos=("video","nunique"))
    t["POR"] = t.POR.round(3)
    return t

out = []
out.append(f"# EXP011 - DEV POR stratified by GT attribute\n")
out.append(f"Overall: {len(e)} events, POR_w30 = {e.por30.mean():.4f}\n")

out.append("\n## Synchronized whole-scene occlusion\n")
t = e.groupby("sync_video").agg(events=("por30","size"), POR=("por30","mean"),
                                videos=("video","nunique")).round(3)
out.append(t.to_markdown())

out.append("\n\n## Occlusion gap length (frames)\n")
out.append(strat("gap_len", [4,9,19,39,10**6], ["5-9","10-19","20-39","40+"]).to_markdown())

out.append("\n\n## Object size (mean GT area fraction when visible)\n")
out.append(strat("obj_size", [0,.005,.02,.05,1.], ["<0.5%","0.5-2%","2-5%",">5%"]).to_markdown())

out.append("\n\n## Video length (frames)\n")
out.append(strat("n_frames", [0,80,120,200,10**6], ["60-80","81-120","121-200","200+"]).to_markdown())

out.append("\n\n## Distractor density (mean objects visible per frame)\n")
out.append(strat("mean_density", [0,2,4,8,10**6], ["<=2","2-4","4-8","8+"]).to_markdown())

txt = "\n".join(out) + "\n"
open(f"{REPO}/experiments/EXP011_stratification.md","w").write(txt)
e.to_csv(f"{REPO}/experiments/EXP011_events.csv", index=False)
print(txt)
print(f"-> experiments/EXP011_stratification.md")
