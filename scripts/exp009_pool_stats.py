"""
EXP009 pre-flight: quantify the cost of requiring frame-0 visibility.

Plan section 7 defines the clean-buffer anchor as a GT-prompted FRAME-0 entry.
In MOSEv2 many objects first appear later than frame 0. This script measures
how much of the candidate pool survives a strict frame-0 requirement, so the
anchor policy can be decided on evidence rather than convenience.

Reads EXP006_events.csv only. No GPU, no model, seconds to run.
"""
import os
import pandas as pd

CSV = os.path.expanduser("~/thesis/identitygate/experiments/EXP006_events.csv")
OUT = os.path.expanduser("~/thesis/identitygate/experiments/EXP009_pool_stats.md")

d = pd.read_csv(CSV)

# Provisional candidate criteria from EXP006 (NOT yet locked):
#   video has >=2 objects, track visible >=20 frames, video >=60 frames,
#   and the track has at least one qualifying event.
pool = d[(d.n_objects_in_video >= 2) &
         (d.visible_frames >= 20) &
         (d.n_frames >= 60) &
         (d.n_events >= 1)].copy()

def summarize(df, label):
    return {
        "filter": label,
        "videos": df.video.nunique(),
        "tracks": len(df),
        "events": int(df.n_events.sum()),
    }

rows = [summarize(pool, "provisional pool (all tracks)")]

# Option (a): every event-bearing track must be visible at frame 0
a_tracks = pool[pool.first_visible == 0]
rows.append(summarize(a_tracks, "(a) event-track visible at frame 0"))

# Stricter reading of (a): the WHOLE video is usable only if EVERY object
# in it is visible at frame 0, because we must prompt all objects to have
# a valid multi-object scene.
all_objs = d[d.video.isin(pool.video.unique())]
ok_vids = (all_objs.groupby("video").first_visible.max() == 0)
ok_vids = set(ok_vids[ok_vids].index)
a_strict = pool[pool.video.isin(ok_vids)]
rows.append(summarize(a_strict, "(a-strict) ALL objects in video visible at frame 0"))

summary = pd.DataFrame(rows)

# Distribution of first-appearance frame among pool tracks
fv = pool.first_visible
dist = pd.DataFrame({
    "first_visible == 0":      [(fv == 0).sum()],
    "1..9":                    [((fv >= 1) & (fv <= 9)).sum()],
    "10..29":                  [((fv >= 10) & (fv <= 29)).sum()],
    "30+":                     [(fv >= 30).sum()],
})

txt = []
txt.append("# EXP009 pre-flight — anchor policy cost\n")
txt.append("Source: experiments/EXP006_events.csv (no new computation).\n")
txt.append("## Pool survival under frame-0 anchor requirement\n")
txt.append(summary.to_markdown(index=False))
txt.append("\n\n## First-visible frame distribution (pool tracks)\n")
txt.append(dist.to_markdown(index=False))
txt.append("\n")

open(OUT, "w").write("\n".join(txt))
print("\n".join(txt))
print(f"\nwritten -> {OUT}")
