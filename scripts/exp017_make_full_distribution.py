import json
import random
from pathlib import Path
import pandas as pd

REPO = Path.home() / "thesis" / "identitygate"
ATTRS = REPO / "experiments" / "EXP013_attrs.csv"
HARD = REPO / "experiments" / "EXP017_split_manifest.json"
EXC = REPO / "experiments" / "development_exclusions_v1.json"
OUT = REPO / "experiments" / "EXP017_full_distribution_manifest.json"

SEED = 42
N_TEST = 118

attrs = pd.read_csv(ATTRS)
hard = json.load(HARD.open())
exc = set(json.load(EXC.open())["video_ids"])

train = set(hard["splits"]["TRAIN"]["videos"])
dev = set(hard["splits"]["DEV"]["videos"])
hard_test = set(hard["splits"]["TEST"]["videos"])

population = sorted(set(attrs.video))
assert len(population) == 1691

eligible = sorted(set(population) - exc - train - dev)

rng = random.Random(SEED)
full_test = sorted(rng.sample(eligible, N_TEST))

assert len(full_test) == 118
assert not (set(full_test) & train)
assert not (set(full_test) & dev)
assert not (set(full_test) & exc)

tracks = attrs[attrs.video.isin(full_test)].copy()
hard_tracks = tracks[
    (tracks.obj_size < 0.005) &
    (tracks.n_frames >= 100)
].copy()

manifest = {
    "experiment": "EXP017",
    "cohort": "FULL_EVENT_BEARING_DISTRIBUTION_TEST",
    "scope_label": "full event-bearing distribution",
    "status": "LOCKED",
    "seed": SEED,
    "population_definition":
        "all 1691 MOSEv2 train videos containing at least one qualifying reappearance event",
    "selection_rule":
        "random held-out sample without hard-stratum attribute filtering",
    "n_population_videos": len(population),
    "n_eligible_after_exclusions": len(eligible),
    "n_test_videos": len(full_test),
    "n_event_tracks": len(tracks),
    "n_qualifying_events": int(tracks.n_events.sum()),
    "n_hard_tracks_inside_full_test": len(hard_tracks),
    "n_hard_events_inside_full_test": int(hard_tracks.n_events.sum()),
    "hard_test_overlap": len(set(full_test) & hard_test),
    "train_overlap": len(set(full_test) & train),
    "dev_overlap": len(set(full_test) & dev),
    "development_exclusion_overlap": len(set(full_test) & exc),
    "test_videos": full_test,
    "evaluation_policy": {
        "no_training": True,
        "no_calibration": True,
        "no_threshold_tuning": True,
        "same_frozen_models_as_hard_test": True,
        "run_once_with_union_of_test_cohorts": True,
        "hard_dev_conformal_guarantee_automatically_transfers": False
    }
}

OUT.write_text(json.dumps(manifest, indent=2) + "\n")

print("EXP017 FULL-DISTRIBUTION MANIFEST CREATED")
print("videos =", len(full_test))
print("events =", int(tracks.n_events.sum()))
print("hard events inside =", int(hard_tracks.n_events.sum()))
print("hard TEST overlap =", len(set(full_test) & hard_test))
print("TRAIN overlap =", len(set(full_test) & train))
print("DEV overlap =", len(set(full_test) & dev))
print("development exclusion overlap =", len(set(full_test) & exc))
print("->", OUT)
