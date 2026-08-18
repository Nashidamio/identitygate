"""
EXP012 - can the section-17-enumerated levers alone reach DEV POR <= 0.70?
Enumerated: longer gaps, more distractors, smaller objects, longer videos.
Reports aggregate POR + event count for every combination. No video is ever
selected by its own score (section 18).
Also reports the same grid with synchronized-occlusion videos removed, purely
as a comparison for the pending methodology decision.
"""
import os, itertools, pandas as pd
REPO = os.path.expanduser("~/thesis/identitygate")
e = pd.read_csv(f"{REPO}/experiments/EXP011_events.csv")

LEV = {
    "gap>=10":      lambda d: d.gap_len >= 10,
    "gap>=20":      lambda d: d.gap_len >= 20,
    "size<2%":      lambda d: d.obj_size < 0.02,
    "size<0.5%":    lambda d: d.obj_size < 0.005,
    "len>=100":     lambda d: d.n_frames >= 100,
    "dens>=3":      lambda d: d.mean_density >= 3,
}

def grid(df, tag):
    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(LEV, k):
            m = pd.Series(True, index=df.index)
            for c in combo: m &= LEV[c](df)
            s = df[m]
            if len(s) < 20: continue
            rows.append({"criteria": " AND ".join(combo), "events": len(s),
                         "videos": s.video.nunique(), "POR": round(s.por30.mean(), 3)})
    t = pd.DataFrame(rows).sort_values("POR").reset_index(drop=True)
    return f"\n## {tag}\n(min 20 events shown; sorted by POR)\n\n" + t.head(25).to_markdown(index=False)

out  = "# EXP012 - reachable DEV POR under section-17 levers\n"
out += f"\nBaseline: {len(e)} events, POR {e.por30.mean():.3f}\n"
out += grid(e, "A. All DEV videos (strict section-17 compliance)")
ns = e[~e.sync_video]
out += f"\n\n(non-sync subset: {len(ns)} events, POR {ns.por30.mean():.3f})\n"
out += grid(ns, "B. Synchronized-occlusion videos removed (requires plan amendment)")
open(f"{REPO}/experiments/EXP012_levers.md","w").write(out+"\n")
print(out)
