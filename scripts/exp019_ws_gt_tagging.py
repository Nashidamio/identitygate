from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path.home() / "thesis" / "identitygate"
ANN = Path("/mnt/d/thesis_data/mosev2/train/Annotations")
ATTRS = REPO / "experiments" / "EXP013_attrs.csv"
CFG = REPO / "configs" / "WS-v1.json"

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cfg = json.load(CFG.open())
rule = cfg["annotation_rule"]

N_EVENT = int(cfg["event_gap_min_frames"])
LOOKBACK = int(rule["reference_lookback_frames"])
VANISH_SHARE = float(rule["vanish_share_min"])
VANISH_W = int(rule["vanish_window_frames"])
RETURN_SHARE = float(rule["return_share_min"])
RETURN_W = int(rule["return_window_frames"])

attrs = pd.read_csv(ATTRS)
videos = sorted(attrs.video.unique())
if LIMIT:
    videos = videos[:LIMIT]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def derive_events(vis):
    out = []
    run = 0
    seen = False
    for t, visible in enumerate(vis):
        if visible:
            if seen and run >= N_EVENT:
                out.append((t - run, t, run))
            run = 0
            seen = True
        elif seen:
            run += 1
    return out

expected = set()
for r in attrs.itertuples():
    reaps = [int(x) for x in str(r.reappear_frames).split("|")]
    gaps = [int(x) for x in str(r.gap_lens).split("|")]
    assert len(reaps) == len(gaps) == int(r.n_events)
    if r.video not in videos:
        continue
    for rf, gap in zip(reaps, gaps):
        expected.add((str(r.video), int(r.object_id), rf - gap, rf, gap))

rows = []
t0 = time.time()

for vi, vid in enumerate(videos):
    pngs = sorted((ANN / vid).glob("*.png"))
    assert pngs, vid

    present = []
    all_ids = set()

    for p in pngs:
        a = np.array(Image.open(p))
        ids = {int(x) for x in np.unique(a)} - {0}
        present.append(ids)
        all_ids |= ids

    nf = len(present)
    vis = {
        oid: np.array([oid in s for s in present], dtype=bool)
        for oid in sorted(all_ids)
    }

    vanish_transitions = {}
    return_transitions = {}
    derived = []

    for oid, v in vis.items():
        vanish_transitions[oid] = [
            t for t in range(1, nf) if v[t - 1] and not v[t]
        ]
        return_transitions[oid] = [
            t for t in range(1, nf) if not v[t - 1] and v[t]
        ]

        for ds, rf, gap in derive_events(v):
            derived.append((vid, oid, ds, rf, gap))

    expected_video = {x for x in expected if x[0] == vid}
    assert set(derived) == expected_video, (
        vid, len(derived), len(expected_video)
    )

    for _, oid, ds, rf, gap in derived:
        complete_lookback = ds >= LOOKBACK

        if complete_lookback:
            reference = set()
            for t in range(ds - LOOKBACK, ds):
                reference |= present[t]
        else:
            reference = set()

        vanished = {
            x for x in reference
            if any(abs(t - ds) <= VANISH_W
                   for t in vanish_transitions[x])
        }

        returned = {
            x for x in vanished
            if any(abs(t - rf) <= RETURN_W
                   for t in return_transitions[x])
        }

        n_ref = len(reference)
        vanish_share = len(vanished) / n_ref if n_ref else 0.0
        return_share = len(returned) / len(vanished) if vanished else 0.0

        flag = (
            complete_lookback
            and vanish_share >= VANISH_SHARE
            and return_share >= RETURN_SHARE
        )

        rows.append({
            "video": vid,
            "object_id": oid,
            "disappear_start": ds,
            "reappear_frame": rf,
            "gap_len": gap,
            "lookback_complete": complete_lookback,
            "reference_objects": n_ref,
            "vanished_objects": len(vanished),
            "returned_objects": len(returned),
            "vanish_share": round(vanish_share, 6),
            "return_share": round(return_share, 6),
            "ws_gt_candidate": bool(flag),
        })

    if (vi + 1) % 25 == 0 or vi + 1 == len(videos):
        print(
            f"{vi+1}/{len(videos)} videos | "
            f"{len(rows)} events | "
            f"{sum(x['ws_gt_candidate'] for x in rows)} flagged",
            flush=True,
        )

df = pd.DataFrame(rows)

assert len(df) == len(expected)
assert len(set(zip(
    df.video,
    df.object_id,
    df.disappear_start,
    df.reappear_frame,
    df.gap_len,
))) == len(df)

# Collapse synchronized flagged object-events into scene events.
scene_rows = []
scene_map = {}

for vid, g in df[df.ws_gt_candidate].groupby("video"):
    recs = g.sort_values(
        ["disappear_start", "reappear_frame", "object_id"]
    ).to_dict("records")

    groups = []
    current = []

    for r in recs:
        trial = current + [r]

        ds = [x["disappear_start"] for x in trial]
        rf = [x["reappear_frame"] for x in trial]

        fits = (
            max(ds) - min(ds)
            <= cfg["scene_collapse"]["max_disappearance_spread_frames"]
            and
            max(rf) - min(rf)
            <= cfg["scene_collapse"]["max_reappearance_spread_frames"]
        )

        if current and not fits:
            groups.append(current)
            current = [r]
        else:
            current = trial

    if current:
        groups.append(current)

    for gi, group in enumerate(groups, 1):
        sid = f"{vid}:WS{gi:03d}"

        for r in group:
            scene_map[
                (r["video"], r["object_id"], r["reappear_frame"])
            ] = sid

        scene_rows.append({
            "scene_event_id": sid,
            "video": vid,
            "n_object_events": len(group),
            "n_unique_objects": len({
                int(x["object_id"]) for x in group
            }),
            "disappear_min": min(x["disappear_start"] for x in group),
            "disappear_max": max(x["disappear_start"] for x in group),
            "reappear_min": min(x["reappear_frame"] for x in group),
            "reappear_max": max(x["reappear_frame"] for x in group),
        })

df["ws_scene_event_id"] = [
    scene_map.get((v, int(o), int(rf)), "")
    for v, o, rf in zip(
        df.video, df.object_id, df.reappear_frame
    )
]

scenes = pd.DataFrame(scene_rows)

raw_events = len(df)
flagged_object_events = int(df.ws_gt_candidate.sum())
normal_object_events = raw_events - flagged_object_events
scene_events = len(scenes)
collapsed_event_count = normal_object_events + scene_events

suffix = f"_sanity{LIMIT}" if LIMIT else ""

out_events = REPO / "experiments" / f"EXP019_ws_gt_events{suffix}.csv"
out_scenes = REPO / "experiments" / f"EXP019_ws_gt_scenes{suffix}.csv"
out_summary = REPO / "experiments" / f"EXP019_ws_gt_summary{suffix}.json"

df.to_csv(out_events, index=False)
scenes.to_csv(out_scenes, index=False)

summary = {
    "experiment": "EXP019",
    "status": "GT_ANNOTATION_CANDIDATES_ONLY_NOT_FINAL_WS_LABELS",
    "videos_scanned": len(videos),
    "raw_object_events": raw_events,
    "ws_gt_flagged_object_events": flagged_object_events,
    "ws_gt_scene_events_after_collapse": scene_events,
    "non_ws_object_events": normal_object_events,
    "analysis_event_count_after_ws_collapse": collapsed_event_count,
    "videos_with_ws_gt_candidate": int(
        df.loc[df.ws_gt_candidate, "video"].nunique()
    ),
    "exp013_exact_event_reconstruction": True,
    "config": str(CFG.relative_to(REPO)),
    "config_sha256": sha256(CFG),
    "source_attrs_sha256": sha256(ATTRS),
    "runtime_sec": round(time.time() - t0, 2),
}

out_summary.write_text(json.dumps(summary, indent=2) + "\n")

print()
print("=== EXP019 SUMMARY ===")
for k, v in summary.items():
    print(f"{k} = {v}")
print("events ->", out_events)
print("scenes ->", out_scenes)
print("summary ->", out_summary)
