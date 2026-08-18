"""
EXP009 step 1: verify the frame-0 visibility claim, then draw provisional DEV.

Substep A -- independent verification. EXP006's CSV says every candidate track
first appears at frame 0. Re-derive this from the raw PNGs for a random sample
of pool videos, and check the claim across ALL 7,631 tracks in the dataset,
not only the filtered pool. A uniform result can also be a bug signature.

Substep B -- if A passes, draw a seeded video-level DEV sample. DEV is fixed
before any POR is measured (plan section 17 / risk R9).
"""
import os, json, random
import numpy as np, pandas as pd
from PIL import Image

ANN  = "/mnt/d/thesis_data/mosev2/train/Annotations"
BASE = os.path.expanduser("~/thesis/identitygate/experiments")
CSV  = f"{BASE}/EXP006_events.csv"
SEED = 42
N_DEV = 40

d = pd.read_csv(CSV)

# ---------- Substep A1: whole-dataset check ----------
print("=== A1. first_visible across ALL tracks in EXP006 ===")
print(d.first_visible.value_counts().head(10).to_string())
print(f"tracks total          : {len(d)}")
print(f"first_visible == 0    : {(d.first_visible == 0).sum()}")
print(f"first_visible >  0    : {(d.first_visible >  0).sum()}")

pool = d[(d.n_objects_in_video >= 2) & (d.visible_frames >= 20) &
         (d.n_frames >= 60) & (d.n_events >= 1)].copy()
pool_vids = sorted(pool.video.unique())

# ---------- Substep A2: raster re-derivation from PNGs ----------
print("\n=== A2. raw-PNG re-derivation on 5 random pool videos ===")
rng = random.Random(SEED)
ok = True
for vid in rng.sample(pool_vids, 5):
    dd = os.path.join(ANN, vid)
    pngs = sorted(f for f in os.listdir(dd) if f.endswith(".png"))
    ids0 = {int(x) for x in np.unique(np.array(Image.open(os.path.join(dd, pngs[0])))).tolist()} - {0}
    idsA = set()
    for f in pngs:
        idsA |= {int(x) for x in np.unique(np.array(Image.open(os.path.join(dd, f)))).tolist()} - {0}
    late = idsA - ids0
    status = "OK" if not late else f"MISMATCH late-appearing ids {sorted(late)}"
    if late: ok = False
    print(f"  {vid}: frame0 ids={sorted(ids0)} all ids={sorted(idsA)} -> {status}")

print(f"\nA2 verdict: {'PASS' if ok else 'FAIL'}")

# ---------- Substep B: draw provisional DEV ----------
if not ok:
    raise SystemExit("A2 failed -- stopping before drawing splits.")

rng2 = random.Random(SEED)
shuf = pool_vids[:]           # already sorted -> deterministic input order
rng2.shuffle(shuf)
dev_vids = sorted(shuf[:N_DEV])
rest     = sorted(shuf[N_DEV:])

dev  = pool[pool.video.isin(dev_vids)]
rst  = pool[pool.video.isin(rest)]

manifest = {
    "experiment": "EXP009",
    "purpose": "provisional DEV sample for the plan section 17 headroom checkpoint",
    "seed": SEED,
    "source_csv": "experiments/EXP006_events.csv",
    "pool_criteria": {"n_objects_in_video>=": 2, "visible_frames>=": 20,
                      "n_frames>=": 60, "n_events>=": 1},
    "pool_videos": len(pool_vids),
    "dev_videos": dev_vids,
    "dev_n_videos": len(dev_vids),
    "dev_n_tracks": int(len(dev)),
    "dev_n_events": int(dev.n_events.sum()),
    "remaining_videos": len(rest),
    "remaining_n_tracks": int(len(rst)),
    "remaining_n_events": int(rst.n_events.sum()),
    "status": "PROVISIONAL -- splits not locked; TRAIN/TEST drawn only after headroom",
}
with open(f"{BASE}/EXP009_dev_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

print("\n=== B. provisional DEV drawn ===")
for k in ("seed","pool_videos","dev_n_videos","dev_n_tracks","dev_n_events",
          "remaining_videos","remaining_n_events"):
    print(f"  {k:20s}: {manifest[k]}")
print(f"\nmanifest -> {BASE}/EXP009_dev_manifest.json")
