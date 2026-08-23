# EXP016 - vanilla SAM 3.1 B0 on fresh hard-stratum headroom sample.
# Derived from the verified EXP010 evaluator.
#
# All frame-0 GT objects remain active during inference.
# Hard POR is scored only on eligible event keys frozen in the EXP016 manifest.
#
# Usage:
#   python scripts/exp016_headroom.py sanity
#   python scripts/exp016_headroom.py full

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = os.path.expanduser("~/thesis/identitygate")
ROOT = "/mnt/d/thesis_data/mosev2/train"
MANIFEST = f"{REPO}/experiments/EXP016_headroom_manifest.json"

MODE = sys.argv[1] if len(sys.argv) > 1 else "sanity"

if MODE not in {"sanity", "full"}:
    raise ValueError("MODE must be sanity or full")

N_EVENT = 5
WINDOWS = (15, 30, 60)
IOU_REC = 0.5
THEFT_SELF = 0.3
THEFT_OTHER = 0.5
THEFT_RUN = 5

OUT = f"{REPO}/experiments/EXP016_{MODE}"
os.makedirs(OUT, exist_ok=True)

with open(MANIFEST, encoding="utf-8") as f:
    man = json.load(f)

selected = list(man["selected_videos"])

eligible_keys = {
    (
        str(x["video"]),
        int(x["object_id"]),
        int(x["reappear_frame"]),
    )
    for x in man["eligible_events"]
}

assert len(selected) == 40
assert len(eligible_keys) == 112

lens = {
    v: len([
        f for f in os.listdir(f"{ROOT}/Annotations/{v}")
        if f.endswith(".png")
    ])
    for v in selected
}

srt = sorted(selected, key=lambda v: lens[v])

if MODE == "sanity":
    vids = srt[:3]
else:
    vids = srt

active_expected_hard = {
    k for k in eligible_keys
    if k[0] in set(vids)
}

git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO,
    text=True,
).strip()

git_dirty = bool(
    subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        text=True,
    ).strip()
)

print("EXP016 B0 HEADROOM")
print("MODE={}".format(MODE))
print("git_commit={}".format(git_commit))
print("git_dirty={}".format(git_dirty))
print("selected_videos={}".format(len(vids)))
print("expected_hard_events={}".format(len(active_expected_hard)))
print(
    "frame_counts: min={} med={} max={}".format(
        lens[vids[0]],
        lens[vids[len(vids) // 2]],
        lens[vids[-1]],
    )
)
print(
    "videos={}".format(
        [(v, lens[v]) for v in vids]
    )
)
print()

from sam3.model_builder import build_sam3_video_model

print("building model...", flush=True)

m = build_sam3_video_model()
pred_or = m.tracker
pred_or.backbone = m.detector.backbone


def events_for_track(vis):
    reap = []
    run = 0
    seen = False

    for t, v in enumerate(vis):
        if v:
            if run >= N_EVENT and seen:
                reap.append(t)
            run = 0
            seen = True
        else:
            if seen:
                run += 1

    return reap


def score_track(vis, ious, reap):
    n = len(vis)
    gap_starts = []
    run = 0
    seen = False

    for t, v in enumerate(vis):
        if v:
            run = 0
            seen = True
        else:
            if seen:
                run += 1
                if run == N_EVENT:
                    gap_starts.append(t - N_EVENT + 1)

    res = {}

    for W in WINDOWS:
        rec = []

        for r in reap:
            stop = next(
                (g for g in gap_starts if g > r),
                n,
            )

            ev = 0
            hit = 0

            for t in range(r, stop):
                if not vis[t]:
                    continue

                ev += 1

                if (
                    ious[t] is not None
                    and ious[t] > IOU_REC
                ):
                    hit = 1
                    break

                if ev >= W:
                    break

            rec.append(hit)

        res[f"por_w{W}"] = rec

    return res


rows = []
per_video = []
seen_hard = set()
t_all = time.time()

for vi, vid in enumerate(vids):
    JPG = f"{ROOT}/JPEGImages/{vid}"
    ANN = f"{ROOT}/Annotations/{vid}"

    pngs = sorted(
        f for f in os.listdir(ANN)
        if f.endswith(".png")
    )
    jpgs = sorted(
        f for f in os.listdir(JPG)
        if f.endswith(".jpg")
    )

    assert len(pngs) == len(jpgs)

    assert [
        os.path.splitext(a)[0] for a in pngs
    ] == [
        os.path.splitext(b)[0] for b in jpgs
    ]

    nf = len(pngs)

    gt0 = np.array(
        Image.open(os.path.join(ANN, pngs[0]))
    )

    oids = sorted(
        int(x) for x in np.unique(gt0)
        if x != 0
    )

    st = pred_or.init_state(video_path=JPG)
    pred_or.clear_all_points_in_video(st)

    for oid in oids:
        mask0 = torch.from_numpy(
            (gt0 == oid).astype(np.uint8)
        ).to(torch.bool)

        pred_or.add_new_mask(
            inference_state=st,
            frame_idx=0,
            obj_id=oid,
            mask=mask0,
        )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    vis = {
        o: [False] * nf
        for o in oids
    }
    iou = {
        o: [None] * nf
        for o in oids
    }
    xiou = {
        o: [0.0] * nf
        for o in oids
    }

    for out in pred_or.propagate_in_video(
        st,
        start_frame_idx=0,
        max_frame_num_to_track=nf,
        reverse=False,
        propagate_preflight=True,
    ):
        fi = int(out[0])
        ids = out[1]
        vres = out[3]

        gt = np.array(
            Image.open(os.path.join(ANN, pngs[fi]))
        )

        gmask = {
            o: (gt == o)
            for o in oids
        }

        for i, oid in enumerate(ids):
            oid = int(oid)

            if oid not in gmask:
                continue

            p = (
                vres[i, 0] > 0
            ).detach().cpu().numpy()

            assert p.shape == gt.shape

            g = gmask[oid]
            vis[oid][fi] = bool(g.sum())

            union = np.logical_or(p, g).sum()

            iou[oid][fi] = (
                float(
                    np.logical_and(p, g).sum()
                    / union
                )
                if union
                else None
            )

            best = 0.0

            for o2 in oids:
                if o2 == oid:
                    continue

                g2 = gmask[o2]
                union2 = np.logical_or(p, g2).sum()

                if union2:
                    best = max(
                        best,
                        float(
                            np.logical_and(p, g2).sum()
                            / union2
                        ),
                    )

            xiou[oid][fi] = best

    dt = time.time() - t0
    pk = (
        torch.cuda.max_memory_allocated()
        / 1024 ** 3
    )

    n_all_events = 0
    n_hard_events = 0

    for oid in oids:
        reap = events_for_track(vis[oid])

        if not reap:
            continue

        sc = score_track(
            vis[oid],
            iou[oid],
            reap,
        )

        hard_indices = [
            i for i, r in enumerate(reap)
            if (vid, oid, r) in eligible_keys
        ]

        hard_reap = [
            reap[i] for i in hard_indices
        ]

        for r in hard_reap:
            seen_hard.add((vid, oid, r))

        hard_sc = {
            f"por_w{W}": [
                sc[f"por_w{W}"][i]
                for i in hard_indices
            ]
            for W in WINDOWS
        }

        theft = [
            1
            if (
                iou[oid][t] is not None
                and iou[oid][t] < THEFT_SELF
                and xiou[oid][t] > THEFT_OTHER
            )
            else 0
            for t in range(nf)
        ]

        run = 0
        theft_events = 0

        for x in theft:
            run = run + 1 if x else 0

            if run == THEFT_RUN:
                theft_events += 1

        n_all_events += len(reap)
        n_hard_events += len(hard_reap)

        row = {
            "video": vid,
            "object_id": oid,
            "n_frames": nf,
            "n_events": len(reap),
            "reappear_frames": reap,
            "hard_n_events": len(hard_reap),
            "hard_reappear_frames": hard_reap,
            "theft_events": theft_events,
            "theft_frames": int(sum(theft)),
            "visible_frames": int(sum(vis[oid])),
        }

        for k, v in sc.items():
            row[k] = v

        for k, v in hard_sc.items():
            row[f"hard_{k}"] = v

        rows.append(row)

    per_video.append({
        "video": vid,
        "n_frames": nf,
        "n_objects": len(oids),
        "all_events": n_all_events,
        "hard_events": n_hard_events,
        "sec": round(dt, 1),
        "peak_vram_gb": round(pk, 2),
    })

    print(
        "[{}/{}] {}  {}f {}obj  all={} hard={}  {:.0f}s  {:.2f}GB".format(
            vi + 1,
            len(vids),
            vid,
            nf,
            len(oids),
            n_all_events,
            n_hard_events,
            dt,
            pk,
        ),
        flush=True,
    )

    del st
    torch.cuda.empty_cache()


missing = sorted(active_expected_hard - seen_hard)
extra = sorted(seen_hard - active_expected_hard)

assert not missing, (
    "eligible hard events missing from tracker scoring: {}"
    .format(missing)
)

assert not extra, (
    "unexpected hard events scored: {}"
    .format(extra)
)

all_event_count = sum(
    r["n_events"] for r in rows
)

hard_event_count = sum(
    r["hard_n_events"] for r in rows
)

assert hard_event_count == len(active_expected_hard)

if MODE == "full":
    expected_all = int(
        man["headroom_sample"][
            "all_gt_events_in_selected_videos"
        ]
    )

    expected_hard = int(
        man["headroom_sample"][
            "eligible_events"
        ]
    )

    assert all_event_count == expected_all, (
        "expected {} all events, got {}"
        .format(expected_all, all_event_count)
    )

    assert hard_event_count == expected_hard, (
        "expected {} hard events, got {}"
        .format(expected_hard, hard_event_count)
    )


all_tot = {
    f"por_w{W}": [
        hit
        for r in rows
        for hit in r[f"por_w{W}"]
    ]
    for W in WINDOWS
}

hard_tot = {
    f"por_w{W}": [
        hit
        for r in rows
        for hit in r[f"hard_por_w{W}"]
    ]
    for W in WINDOWS
}

summary = {
    "experiment": "EXP016",
    "mode": MODE,
    "manifest_status": man["status"],
    "git_commit": git_commit,
    "git_dirty": git_dirty,
    "candidate_rule": man["candidate_rule"],
    "videos": len(vids),
    "tracks_with_events": len(rows),
    "all_events": all_event_count,
    "hard_eligible_events_manifest":
        len(active_expected_hard),
    "hard_events_scored": hard_event_count,
    "theft_events": sum(
        r["theft_events"] for r in rows
    ),
    "tracks_with_theft": sum(
        1 for r in rows
        if r["theft_events"] > 0
    ),
    "runtime_min": round(
        (time.time() - t_all) / 60,
        1,
    ),
    "max_peak_vram_gb": max(
        v["peak_vram_gb"]
        for v in per_video
    ),
}

for W in WINDOWS:
    all_hits = all_tot[f"por_w{W}"]
    hard_hits = hard_tot[f"por_w{W}"]

    summary[f"POR_all_w{W}"] = (
        round(
            sum(all_hits) / len(all_hits),
            4,
        )
        if all_hits
        else None
    )

    summary[f"n_events_all_w{W}"] = len(
        all_hits
    )

    summary[f"POR_hard_w{W}"] = (
        round(
            sum(hard_hits) / len(hard_hits),
            4,
        )
        if hard_hits
        else None
    )

    summary[f"n_events_hard_w{W}"] = len(
        hard_hits
    )


with open(
    f"{OUT}/results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "summary": summary,
            "per_video": per_video,
            "per_track": rows,
        },
        f,
        indent=2,
    )


print()
print("=== SUMMARY ===")

for k, v in summary.items():
    print("  {:30s}: {}".format(k, v))

print()
print("-> {}/results.json".format(OUT))
