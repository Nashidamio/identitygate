"""
EXP013 - per-track GT attributes over all event-bearing MOSEv2 train videos.
Adds what EXP006 lacks: object size, distractor density, reappearance frames
(needed for synchronized-occlusion detection). GT only -- no model, no GPU.
Usage: python scripts/exp013_attrs.py [N_VIDEOS]   (0 = all)
"""
import os, sys, csv, time
import numpy as np, pandas as pd
from PIL import Image

ANN  = "/mnt/d/thesis_data/mosev2/train/Annotations"
REPO = os.path.expanduser("~/thesis/identitygate")
OUT  = f"{REPO}/experiments/EXP013_attrs.csv"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_EVENT = 5

ev = pd.read_csv(f"{REPO}/experiments/EXP006_events.csv")
vids = sorted(ev[ev.n_events >= 1].video.unique())
if LIMIT: vids, OUT = vids[:LIMIT], OUT.replace(".csv", "_timing.csv")
print(f"scanning {len(vids)} event-bearing videos", flush=True)

rows, t0 = [], time.time()
for i, vid in enumerate(vids):
    d = f"{ANN}/{vid}"
    pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    arrs = [np.array(Image.open(os.path.join(d, f))) for f in pngs]
    nf = len(arrs); H, W = arrs[0].shape; fa = float(H * W)
    oids = sorted({int(x) for a in arrs for x in np.unique(a)} - {0})
    dens = float(np.mean([len(np.unique(a)) - 1 for a in arrs]))
    for oid in oids:
        area = np.array([float((a == oid).sum()) / fa for a in arrs])
        vis = area > 0
        if not vis.any(): continue
        reap, run, seen = [], 0, False
        for t in range(nf):
            if vis[t]:
                if run >= N_EVENT and seen: reap.append((t, run))
                run = 0; seen = True
            elif seen: run += 1
        if not reap: continue
        rows.append({"video": vid, "object_id": oid, "n_frames": nf,
                     "n_objects": len(oids), "mean_density": round(dens, 2),
                     "obj_size": round(float(area[vis].mean()), 6),
                     "visible_frames": int(vis.sum()), "n_events": len(reap),
                     "reappear_frames": "|".join(str(t) for t, _ in reap),
                     "gap_lens": "|".join(str(g) for _, g in reap)})
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(vids)}  {el:.0f}s  ETA {el/(i+1)*(len(vids)-i-1)/60:.1f} min",
              flush=True)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\ntracks={len(rows)} events={sum(r['n_events'] for r in rows)} "
      f"videos={len({r['video'] for r in rows})} elapsed={(time.time()-t0)/60:.1f} min")
print(f"-> {OUT}")
