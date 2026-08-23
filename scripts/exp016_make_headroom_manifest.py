from pathlib import Path
import json
import random

import pandas as pd

REPO = Path.home() / "thesis" / "identitygate"
ATTRS = REPO / "experiments" / "EXP013_attrs.csv"
OLD_MANIFEST = REPO / "experiments" / "EXP009_dev_manifest.json"
OUT = REPO / "experiments" / "EXP016_headroom_manifest.json"

SEED = 42
N_SAMPLE = 40

attrs = pd.read_csv(ATTRS)

with OLD_MANIFEST.open(encoding="utf-8") as f:
    old_manifest = json.load(f)

old_videos = set(old_manifest["dev_videos"])

assert len(attrs) == 3237
assert attrs.video.nunique() == 1691
assert int(attrs.n_events.sum()) == 4469
assert len(old_videos) == 40

candidate_tracks = attrs[
    (attrs.obj_size < 0.005) &
    (attrs.n_frames >= 100)
].copy()

candidate_videos = set(candidate_tracks.video.unique())

assert len(candidate_tracks) == 333
assert int(candidate_tracks.n_events.sum()) == 732
assert len(candidate_videos) == 258

fresh_tracks = candidate_tracks[
    ~candidate_tracks.video.isin(old_videos)
].copy()

fresh_videos = sorted(fresh_tracks.video.unique())

assert len(fresh_tracks) == 315
assert int(fresh_tracks.n_events.sum()) == 705
assert len(fresh_videos) == 252

rng = random.Random(SEED)
selected_videos = sorted(rng.sample(fresh_videos, N_SAMPLE))

assert len(selected_videos) == N_SAMPLE
assert not (set(selected_videos) & old_videos)

selected_tracks = fresh_tracks[
    fresh_tracks.video.isin(selected_videos)
].copy()

eligible_events = []

for _, r in selected_tracks.iterrows():
    reaps = [int(x) for x in str(r.reappear_frames).split("|")]

    assert len(reaps) == int(r.n_events)

    for rf in reaps:
        eligible_events.append({
            "video": str(r.video),
            "object_id": int(r.object_id),
            "reappear_frame": rf,
        })

eligible_events = sorted(
    eligible_events,
    key=lambda x: (
        x["video"],
        x["object_id"],
        x["reappear_frame"],
    ),
)

all_selected = attrs[attrs.video.isin(selected_videos)]

manifest = {
    "experiment": "EXP016",
    "purpose": "fresh B0 hard-stratum headroom development sample",
    "status": "HEADROOM_DEVELOPMENT_ONLY_NOT_FINAL_DEV",
    "seed": SEED,
    "source_attrs": "experiments/EXP013_attrs.csv",
    "historical_exclusion_source":
        "experiments/EXP009_dev_manifest.json",
    "historical_exclusion_count": len(old_videos),
    "candidate_rule": {
        "obj_size": "<0.005",
        "n_frames": ">=100",
    },
    "candidate_rule_scope":
        "track level for obj_size and video level for n_frames",
    "metric_scope":
        "hard-stratum POR includes only eligible_events listed here",
    "inference_scope":
        "all frame-0 objects in selected videos remain active during tracking",
    "candidate_pool_before_exp009_exclusion": {
        "videos": len(candidate_videos),
        "tracks": len(candidate_tracks),
        "events": int(candidate_tracks.n_events.sum()),
    },
    "fresh_candidate_pool": {
        "videos": len(fresh_videos),
        "tracks": len(fresh_tracks),
        "events": int(fresh_tracks.n_events.sum()),
    },
    "headroom_sample": {
        "videos": len(selected_videos),
        "eligible_tracks": len(selected_tracks),
        "eligible_events": len(eligible_events),
        "all_gt_events_in_selected_videos":
            int(all_selected.n_events.sum()),
    },
    "selected_videos": selected_videos,
    "eligible_events": eligible_events,
    "final_split_exclusions": {
        "contaminated_video_ids": "UNKNOWN",
        "cache_used_video_ids": "UNKNOWN",
        "final_split_lock_allowed": False,
    },
}

OUT.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("EXP016 MANIFEST CREATED")
print("seed =", SEED)
print("rule = obj_size < 0.005 AND n_frames >= 100")
print()
print("candidate videos =", len(candidate_videos))
print("candidate tracks =", len(candidate_tracks))
print("candidate events =", int(candidate_tracks.n_events.sum()))
print()
print("fresh videos =", len(fresh_videos))
print("fresh tracks =", len(fresh_tracks))
print("fresh events =", int(fresh_tracks.n_events.sum()))
print()
print("sample videos =", len(selected_videos))
print("sample eligible tracks =", len(selected_tracks))
print("sample eligible events =", len(eligible_events))
print(
    "sample all GT events =",
    int(all_selected.n_events.sum()),
)
print()
print("EXP009 overlap =", len(set(selected_videos) & old_videos))
print()
print("SELECTED VIDEOS")
for v in selected_videos:
    print(v)
print()
print("->", OUT)
