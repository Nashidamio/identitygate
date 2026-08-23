from pathlib import Path
import os
import itertools
import pandas as pd

REPO = os.path.expanduser("~/thesis/identitygate")
SRC = f"{REPO}/experiments/EXP013_attrs.csv"
OUT_CSV = f"{REPO}/experiments/EXP015_fullpool_levers.csv"
OUT_MD = f"{REPO}/experiments/EXP015_fullpool_levers.md"

TRACKS = pd.read_csv(SRC)

# Expand EXP013 track rows to one row per qualifying
# disappearance/reappearance event.
events = []

for _, r in TRACKS.iterrows():
    reaps = [int(x) for x in str(r.reappear_frames).split("|")]
    gaps = [int(x) for x in str(r.gap_lens).split("|")]

    assert len(reaps) == len(gaps) == int(r.n_events), (
        f"event expansion mismatch: {r.video} obj={r.object_id}"
    )

    for j, (rf, gap) in enumerate(zip(reaps, gaps), start=1):
        events.append({
            "video": r.video,
            "object_id": int(r.object_id),
            "event_index": j,
            "reappear_frame": rf,
            "gap_len": gap,
            "obj_size": float(r.obj_size),
            "n_frames": int(r.n_frames),
            "n_objects": int(r.n_objects),
            "mean_density": float(r.mean_density),
        })

e = pd.DataFrame(events)

# EXP006 / EXP013 integrity invariants.
assert len(e) == 4469, f"expected 4469 events, got {len(e)}"
assert e.video.nunique() == 1691, (
    f"expected 1691 videos, got {e.video.nunique()}"
)
assert len(e.drop_duplicates(
    ["video", "object_id", "reappear_frame"]
)) == 4469, "duplicate event keys found"

LEV = {
    "gap>=10":   lambda d: d.gap_len >= 10,
    "gap>=20":   lambda d: d.gap_len >= 20,
    "size<2%":   lambda d: d.obj_size < 0.02,
    "size<0.5%": lambda d: d.obj_size < 0.005,
    "len>=100":  lambda d: d.n_frames >= 100,
    "dens>=3":   lambda d: d.mean_density >= 3,
}

# Nested rules that are logically redundant when combined.
NESTED_PAIRS = [
    {"gap>=10", "gap>=20"},
    {"size<2%", "size<0.5%"},
]

rows = []

for k in (1, 2, 3):
    for combo in itertools.combinations(LEV.keys(), k):
        names = set(combo)

        if any(pair.issubset(names) for pair in NESTED_PAIRS):
            continue

        mask = pd.Series(True, index=e.index)

        for criterion in combo:
            mask &= LEV[criterion](e)

        s = e[mask]

        if s.empty:
            continue

        per_video = s.groupby("video").size().sort_values(
            ascending=False
        )

        rows.append({
            "criteria": " AND ".join(combo),
            "events": len(s),
            "tracks": s[
                ["video", "object_id"]
            ].drop_duplicates().shape[0],
            "videos": s.video.nunique(),
            "events_per_video_mean": round(
                len(s) / s.video.nunique(), 3
            ),
            "max_events_one_video": int(per_video.iloc[0]),
            "top5_video_event_share": round(
                per_video.head(5).sum() / len(s), 4
            ),
            "meets_min_210_videos": s.video.nunique() >= 210,
            "within_210_350_videos":
                210 <= s.video.nunique() <= 350,
        })

res = pd.DataFrame(rows)

# Most selective candidate pools first.
res = res.sort_values(
    ["meets_min_210_videos", "videos", "events"],
    ascending=[False, True, True],
).reset_index(drop=True)

res.to_csv(OUT_CSV, index=False)

candidate = res[res.meets_min_210_videos].copy()

lines = [
    "# EXP015 - full-pool GT-only lever feasibility",
    "",
    "Purpose: determine whether the existing EXP012 GT-only "
    "difficulty criteria retain sufficient sample size after "
    "enlarging to all EXP013 event-bearing videos.",
    "",
    "This experiment uses no SAM predictions and does not "
    "select videos by B0 performance.",
    "",
    f"Expanded events: {len(e)}",
    f"Unique videos: {e.video.nunique()}",
    f"Unique tracks: {e[["video", "object_id"]].drop_duplicates().shape[0]}",
    "",
    "## Candidate rules retaining at least 210 videos",
    "",
]

if len(candidate):
    lines.append(candidate.to_markdown(index=False))
else:
    lines.append("NONE")

lines += [
    "",
    "## All non-redundant EXP012 rule combinations",
    "",
    res.to_markdown(index=False),
    "",
    "Note: retained-video count is a feasibility measure, not "
    "evidence that a rule is harder. Baseline POR must be measured "
    "on a fresh stratum-development sample before any final "
    "difficulty rule is locked.",
]

Path(OUT_MD).write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("EXP015 COMPLETE")
print(f"events={len(e)}")
print(f"tracks={e[["video", "object_id"]].drop_duplicates().shape[0]}")
print(f"videos={e.video.nunique()}")
print(f"rules_tested={len(res)}")
print(f"rules_ge_210_videos={len(candidate)}")
print()
print("MOST SELECTIVE RULES THAT STILL RETAIN >=210 VIDEOS:")
if len(candidate):
    print(candidate.head(15)[
        ["criteria", "events", "tracks", "videos",
         "max_events_one_video", "top5_video_event_share"]
    ].to_string(index=False))
else:
    print("NONE")
print()
print(f"-> {OUT_CSV}")
print(f"-> {OUT_MD}")
