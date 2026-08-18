"""
Count occlusion/reappearance events in MOSEv2 per plan section 14.
Visibility source: rasterized GT mask area (visible iff area > 0).
Event: visibility gap of >= N_EVENT consecutive frames, followed by visibility.
Outputs CSV to experiments/ for split planning (plan section 17 checkpoint).
"""
import os, sys, csv, json
import numpy as np
from PIL import Image

ANN   = "/mnt/d/thesis_data/mosev2/train/Annotations"
META  = "/mnt/d/thesis_data/mosev2/meta_train.json"
OUT   = os.path.expanduser("~/thesis/identitygate/experiments/EXP006_events.csv")
N_EVENT = 5          # plan section 14: gap of at least 5 consecutive frames
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = all videos

def events_for_track(vis):
    """vis: list[bool] per frame. Returns (n_events, gap_lengths)."""
    ev, gaps, run = 0, [], 0
    seen_visible = False
    for v in vis:
        if v:
            if run >= N_EVENT and seen_visible:
                ev += 1
                gaps.append(run)
            run = 0
            seen_visible = True
        else:
            if seen_visible:      # only count gaps AFTER first appearance
                run += 1
    return ev, gaps

def main():
    meta = json.load(open(META))["videos"]
    vids = sorted(os.listdir(ANN))
    if LIMIT:
        vids = vids[:LIMIT]
    print(f"scanning {len(vids)} videos...", flush=True)

    rows = []
    for i, vid in enumerate(vids):
        d = os.path.join(ANN, vid)
        pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        if not pngs:
            continue
        # per-object visibility across frames
        n_frames = len(pngs)
        per_frame_present = []
        all_ids = set()
        for f in pngs:
            a = np.array(Image.open(os.path.join(d, f)))
            present = {int(x) for x in np.unique(a).tolist()} - {0}
            per_frame_present.append(present)
            all_ids |= present
        # correct alignment: index i == frame i for every object
        obj_vis = {oid: [oid in pf for pf in per_frame_present] for oid in all_ids}
        for oid, vis in obj_vis.items():
            n_ev, gaps = events_for_track(vis)
            rows.append({
                "video": vid, "object_id": oid, "n_frames": n_frames,
                "n_objects_in_video": len(obj_vis),
                "visible_frames": int(sum(vis)),
                "n_events": n_ev,
                "max_gap": max(gaps) if gaps else 0,
                "mean_gap": round(sum(gaps)/len(gaps), 1) if gaps else 0,
                "first_visible": vis.index(True) if any(vis) else -1,
                "last_visible": len(vis)-1-vis[::-1].index(True) if any(vis) else -1,
            })
        if (i+1) % 50 == 0:
            tot = sum(r["n_events"] for r in rows)
            print(f"  {i+1}/{len(vids)} videos | {tot} events so far", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    tot_ev = sum(r["n_events"] for r in rows)
    vids_with = len({r["video"] for r in rows if r["n_events"] > 0})
    multi_with = len({r["video"] for r in rows
                      if r["n_events"] > 0 and r["n_objects_in_video"] >= 2})
    print(f"\n=== RESULTS ===")
    print(f"videos scanned            : {len(vids)}")
    print(f"object-tracks             : {len(rows)}")
    print(f"TOTAL events (gap>={N_EVENT})   : {tot_ev}")
    print(f"videos with >=1 event     : {vids_with}")
    print(f"  of those, >=2 objects   : {multi_with}")
    print(f"CSV -> {OUT}")

main()
