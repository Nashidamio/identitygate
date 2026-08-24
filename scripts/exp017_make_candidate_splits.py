from pathlib import Path
import hashlib
import json
import random
import subprocess

import pandas as pd

REPO = Path.home() / "thesis" / "identitygate"
ATTRS = REPO / "experiments" / "EXP013_attrs.csv"
EXCLUSIONS = REPO / "experiments" / "development_exclusions_v1.json"
OUT = REPO / "experiments" / "EXP017_split_manifest.json"

SEED = 42
N_TRAIN = 100
N_DEV = 40

attrs = pd.read_csv(ATTRS)
exc = json.load(EXCLUSIONS.open(encoding="utf-8"))
excluded = set(exc["video_ids"])

assert len(attrs) == 3237
assert attrs.video.nunique() == 1691
assert int(attrs.n_events.sum()) == 4469
assert len(excluded) == 81

hard = attrs[
    (attrs.obj_size < 0.005) &
    (attrs.n_frames >= 100)
].copy()

hard_videos = set(hard.video.unique())
exposed_hard = sorted(hard_videos & excluded)
clean_hard = sorted(hard_videos - excluded)

assert len(hard_videos) == 258
assert len(hard) == 333
assert int(hard.n_events.sum()) == 732
assert len(exposed_hard) == 46
assert len(clean_hard) == 212

# Exposure is an evaluation exclusion, not a TRAIN exclusion.
# Put every exposed hard-pool video into TRAIN.
n_clean_train = N_TRAIN - len(exposed_hard)
assert n_clean_train == 54

rng = random.Random(SEED)
shuf = clean_hard[:]
rng.shuffle(shuf)

train_videos = sorted(exposed_hard + shuf[:n_clean_train])
dev_videos = sorted(shuf[n_clean_train:n_clean_train + N_DEV])
test_videos = sorted(shuf[n_clean_train + N_DEV:])

assert len(train_videos) == 100
assert len(dev_videos) == 40
assert len(test_videos) == 118

train_set = set(train_videos)
dev_set = set(dev_videos)
test_set = set(test_videos)

assert not (train_set & dev_set)
assert not (train_set & test_set)
assert not (dev_set & test_set)
assert train_set | dev_set | test_set == hard_videos

assert not (dev_set & excluded)
assert not (test_set & excluded)
assert set(exposed_hard).issubset(train_set)


def event_keys(df):
    out = []
    for r in df.itertuples():
        reaps = [int(x) for x in str(r.reappear_frames).split("|")]
        assert len(reaps) == int(r.n_events)
        for rf in reaps:
            out.append({
                "video": str(r.video),
                "object_id": int(r.object_id),
                "reappear_frame": rf,
            })

    out.sort(
        key=lambda x: (
            x["video"],
            x["object_id"],
            x["reappear_frame"],
        )
    )

    keys = {
        (x["video"], x["object_id"], x["reappear_frame"])
        for x in out
    }
    assert len(keys) == len(out)
    return out


def split_record(videos):
    hard_tracks = hard[hard.video.isin(videos)].copy()
    all_tracks = attrs[attrs.video.isin(videos)].copy()

    hard_events = event_keys(hard_tracks)
    all_events = event_keys(all_tracks)

    return {
        "n_videos": len(videos),
        "videos": videos,
        "hard_eligible_tracks": len(hard_tracks),
        "hard_eligible_events": len(hard_events),
        "all_event_tracks": len(all_tracks),
        "all_qualifying_events_in_selected_videos": len(all_events),
        "hard_event_keys": hard_events,
        "all_event_keys": all_events,
    }


splits = {
    "TRAIN": split_record(train_videos),
    "DEV": split_record(dev_videos),
    "TEST": split_record(test_videos),
}

assert splits["TEST"]["hard_eligible_events"] >= 200

git_parent = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO,
    text=True,
).strip()

generator_sha256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()

manifest = {
    "experiment": "EXP017",
    "purpose": "candidate final TRAIN/DEV/TEST construction",
    "status": "LOCKED",
    "seed": SEED,
    "source_git_parent": git_parent,
    "generator": "scripts/exp017_make_candidate_splits.py",
    "generator_sha256": generator_sha256,
    "source_attrs": "experiments/EXP013_attrs.csv",
    "development_exclusions":
        "experiments/development_exclusions_v1.json",
    "candidate_rule": {
        "obj_size": "<0.005",
        "n_frames": ">=100",
    },
    "candidate_rule_scope":
        "video eligible iff it contains >=1 track satisfying the hard rule",
    "selection_algorithm": (
        "All development-exposed hard-pool videos are assigned to TRAIN. "
        "Remaining clean hard-pool videos are shuffled once with seed 42; "
        "54 enter TRAIN, next 40 enter DEV, remaining 118 enter TEST. "
        "Seed is not rerolled based on event counts."
    ),
    "pool": {
        "videos": len(hard_videos),
        "tracks": len(hard),
        "hard_events": int(hard.n_events.sum()),
        "development_exposed_hard_videos": len(exposed_hard),
        "clean_hard_videos": len(clean_hard),
    },
    "assertions": {
        "splits_pairwise_disjoint": True,
        "all_258_hard_videos_assigned_once": True,
        "dev_development_exclusion_overlap": 0,
        "test_development_exclusion_overlap": 0,
        "test_hard_events_minimum": 200,
        "locked_test_predictions_inspected": False,
    },
    "dual_reporting_note": (
        "Manifest records both hard-event keys and all qualifying event keys "
        "within each selected-video split. Full-distribution TEST is frozen separately "
        "in experiments/EXP017_full_distribution_manifest.json."
    ),
    "splits": splits,
}

OUT.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("EXP017 CANDIDATE CREATED")
print("seed =", SEED)
print("rule = obj_size < 0.005 AND n_frames >= 100")
print()
print("hard pool videos =", len(hard_videos))
print("development-exposed hard =", len(exposed_hard))
print("clean hard =", len(clean_hard))
print()
for name in ("TRAIN", "DEV", "TEST"):
    s = splits[name]
    print(
        "{}: videos={} hard_tracks={} hard_events={} "
        "all_tracks={} all_events={}".format(
            name,
            s["n_videos"],
            s["hard_eligible_tracks"],
            s["hard_eligible_events"],
            s["all_event_tracks"],
            s["all_qualifying_events_in_selected_videos"],
        )
    )

print()
print("DEV exclusion overlap =", len(dev_set & excluded))
print("TEST exclusion overlap =", len(test_set & excluded))
print("TEST >=200 hard events =", splits["TEST"]["hard_eligible_events"] >= 200)
print()
print("->", OUT)
