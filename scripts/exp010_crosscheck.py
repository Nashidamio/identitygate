"""Verify EXP010 event detection == EXP006 event detection on the same tracks."""
import json, os, pandas as pd
REPO = os.path.expanduser("~/thesis/identitygate")
r = json.load(open(f"{REPO}/experiments/EXP010_sanity/results.json"))
d = pd.read_csv(f"{REPO}/experiments/EXP006_events.csv")
vids = [v["video"] for v in r["per_video"]]
new = {(x["video"], x["object_id"]): x for x in r["per_track"]}
old = d[d.video.isin(vids)]
bad = 0
print(f"{'video':10s} {'obj':>4s} {'ev006':>6s} {'ev010':>6s} {'vis006':>7s} {'vis010':>7s}  status")
for _, o in old.iterrows():
    k = (o.video, int(o.object_id)); n = new.get(k)
    if o.n_events == 0 and n is None:
        continue                       # correctly absent from EXP010 (no events)
    if n is None:
        print(f"{o.video:10s} {o.object_id:4d} {o.n_events:6d} {'MISSING':>6s} "
              f"{o.visible_frames:7d} {'-':>7s}  FAIL"); bad += 1; continue
    ok = (o.n_events == n["n_events"]) and (o.visible_frames == n["visible_frames"])
    bad += (not ok)
    print(f"{o.video:10s} {o.object_id:4d} {o.n_events:6d} {n['n_events']:6d} "
          f"{o.visible_frames:7d} {n['visible_frames']:7d}  {'ok' if ok else 'FAIL'}")
print(f"\nVERDICT: {'PASS - event definitions agree' if bad==0 else f'FAIL ({bad} mismatches)'}")
