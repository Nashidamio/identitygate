#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG_REL = "configs/EXP045-di-gt-primitives-v1.json"
SCRIPT_REL = "scripts/exp045_build_di_gt_primitives.py"

EVENT_FIELDS = [
    "video",
    "object_id",
    "disappear_start",
    "reappear_frame",
    "gap_len",
    "frame_height",
    "frame_width",
    "pre10_start_frame",
    "pre10_end_frame",
    "pre10_frame_count",
    "pre10_visible_frame_count",
    "pre10_min_visible_area_fraction",
    "pre10_min_all_area_fraction",
    "pre10_last_frame_area_fraction",
    "crowding_mean_visible_objects",
    "last_seen_frame",
    "last_seen_centroid_x_norm",
    "last_seen_centroid_y_norm",
    "reappear_centroid_x_norm",
    "reappear_centroid_y_norm",
    "reappearance_displacement_diag_norm",
    "target_visible_frame0",
    "frame0_object_count",
    "frame0_competitor_count",
]

VIDEO_FIELDS = [
    "video",
    "primary_event_count",
    "frame0_object_count",
    "anchorable_primary_event_count",
    "nonanchorable_primary_event_count",
    "all_primary_events_anchorable",
    "has_any_anchorable_primary_event",
    "crowding_mean_visible_objects",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
    ).strip()


def centroid(mask):
    ys, xs = np.nonzero(mask)

    require(
        len(xs) > 0,
        "Cannot compute centroid of empty mask",
    )

    return float(xs.mean()), float(ys.mean())


def compute_event(
    mask_by_frame,
    event,
    height,
    width,
    crowding,
    frame0_ids,
    gap_min,
):
    oid = int(event["object_id"])
    disappear = int(event["disappear_start"])
    reappear = int(event["reappear_frame"])
    gap = int(event["gap_len"])

    require(
        gap >= gap_min,
        "Event gap shorter than frozen minimum",
    )

    require(
        reappear - disappear == gap,
        "gap_len != reappear_frame - disappear_start",
    )

    require(
        disappear >= 1,
        "disappear_start has no preceding frame",
    )

    start = max(
        0,
        disappear - 10,
    )

    pre_frames = list(
        range(start, disappear)
    )

    require(
        pre_frames,
        "Empty pre-disappearance window",
    )

    required = set(pre_frames)
    required.add(reappear)

    require(
        required <= set(mask_by_frame),
        "Missing required annotation frame",
    )

    denom = float(
        height * width
    )

    areas = []

    for frame in pre_frames:
        target = (
            mask_by_frame[frame] == oid
        )

        areas.append(
            float(target.sum()) / denom
        )

    positive_areas = [
        value
        for value in areas
        if value > 0.0
    ]

    require(
        positive_areas,
        "Target never visible in pre-gap window",
    )

    last_seen_frame = disappear - 1

    last_mask = (
        mask_by_frame[last_seen_frame]
        == oid
    )

    require(
        last_mask.any(),
        "Target absent at disappear_start - 1",
    )

    reappear_mask = (
        mask_by_frame[reappear]
        == oid
    )

    require(
        reappear_mask.any(),
        "Target absent at reappear_frame",
    )

    last_x, last_y = centroid(
        last_mask
    )

    reap_x, reap_y = centroid(
        reappear_mask
    )

    diagonal = math.hypot(
        width,
        height,
    )

    require(
        diagonal > 0.0,
        "Invalid image diagonal",
    )

    displacement = (
        math.hypot(
            reap_x - last_x,
            reap_y - last_y,
        )
        / diagonal
    )

    target_frame0 = (
        oid in frame0_ids
    )

    if target_frame0:
        competitor_count = (
            len(frame0_ids) - 1
        )
    else:
        competitor_count = (
            len(frame0_ids)
        )

    return {
        "video": event["video"],
        "object_id": oid,
        "disappear_start": disappear,
        "reappear_frame": reappear,
        "gap_len": gap,
        "frame_height": height,
        "frame_width": width,
        "pre10_start_frame": start,
        "pre10_end_frame": disappear - 1,
        "pre10_frame_count": len(pre_frames),
        "pre10_visible_frame_count": len(
            positive_areas
        ),
        "pre10_min_visible_area_fraction": min(
            positive_areas
        ),
        "pre10_min_all_area_fraction": min(
            areas
        ),
        "pre10_last_frame_area_fraction": areas[-1],
        "crowding_mean_visible_objects": crowding,
        "last_seen_frame": last_seen_frame,
        "last_seen_centroid_x_norm": (
            last_x / width
        ),
        "last_seen_centroid_y_norm": (
            last_y / height
        ),
        "reappear_centroid_x_norm": (
            reap_x / width
        ),
        "reappear_centroid_y_norm": (
            reap_y / height
        ),
        "reappearance_displacement_diag_norm": displacement,
        "target_visible_frame0": int(
            target_frame0
        ),
        "frame0_object_count": len(
            frame0_ids
        ),
        "frame0_competitor_count": competitor_count,
    }


def self_test():
    height = 8
    width = 8

    frames = {}

    for frame in range(16):
        arr = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        if frame <= 9 or frame == 15:
            arr[1:3, 1:3] = 1

        arr[5:7, 5:7] = 2

        frames[frame] = arr

    event = {
        "video": "fixture",
        "object_id": "1",
        "disappear_start": "10",
        "reappear_frame": "15",
        "gap_len": "5",
    }

    row = compute_event(
        frames,
        event,
        height,
        width,
        2.0,
        {1, 2},
        5,
    )

    require(
        row["target_visible_frame0"] == 1,
        "Synthetic frame0 failure",
    )

    require(
        row["frame0_competitor_count"] == 1,
        "Synthetic competitor failure",
    )

    require(
        row["pre10_frame_count"] == 10,
        "Synthetic history-window failure",
    )

    require(
        row["pre10_visible_frame_count"] == 10,
        "Synthetic visible-history failure",
    )

    require(
        row[
            "pre10_min_visible_area_fraction"
        ] == 4 / 64,
        "Synthetic visible-area failure",
    )

    require(
        row[
            "pre10_min_all_area_fraction"
        ] == 4 / 64,
        "Synthetic literal-area failure",
    )

    require(
        row[
            "reappearance_displacement_diag_norm"
        ] == 0.0,
        "Synthetic displacement failure",
    )

    bad = dict(event)
    bad["gap_len"] = "4"

    rejected = False

    try:
        compute_event(
            frames,
            bad,
            height,
            width,
            2.0,
            {1, 2},
            5,
        )

    except RuntimeError:
        rejected = True

    require(
        rejected,
        "Synthetic short gap was not rejected",
    )

    print(
        "EXP045_SYNTHETIC_TESTS=PASS"
    )


def write_csv(path, rows, fields):
    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    require(
        not git("status", "--porcelain"),
        "STOP: execution requires a clean committed tree",
    )

    tracked = set(
        git("ls-files").splitlines()
    )

    require(
        CONFIG_REL in tracked,
        "STOP: EXP045 config must be committed first",
    )

    require(
        SCRIPT_REL in tracked,
        "STOP: EXP045 script must be committed first",
    )

    cfg_path = ROOT / CONFIG_REL

    cfg = json.loads(
        cfg_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        cfg["experiment"] == "EXP045",
        "Wrong EXP045 configuration",
    )

    subprocess.check_call([
        "git",
        "-C",
        str(ROOT),
        "merge-base",
        "--is-ancestor",
        cfg["source_commit"],
        "HEAD",
    ])

    event_path = (
        ROOT
        / cfg["event_pool"]["path"]
    )

    require(
        sha256(event_path)
        == cfg["event_pool"]["sha256"],
        "EXP044 event-pool SHA mismatch",
    )

    annotation_root = Path(
        cfg["annotations_root"]
    )

    require(
        annotation_root.is_dir(),
        "Annotation root missing",
    )

    output_root = (
        ROOT / cfg["output_dir"]
    )

    require(
        not output_root.exists(),
        "STOP: EXP045 output exists; do not overwrite",
    )

    with event_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        all_events = list(
            csv.DictReader(f)
        )

    primary = [
        row
        for row in all_events
        if int(
            row["primary_pool_eligible"]
        ) == 1
    ]

    expected = cfg["expected"]

    require(
        len(primary)
        == expected["primary_events"],
        "Primary event count mismatch",
    )

    by_video = defaultdict(list)

    event_keys = set()

    for row in primary:
        require(
            int(row["development_exposed"]) == 0,
            "Exposed event entered primary pool",
        )

        require(
            int(row["is_final_whole_scene"]) == 0,
            "Whole-scene event entered primary pool",
        )

        key = (
            row["video"],
            int(row["object_id"]),
            int(row["reappear_frame"]),
        )

        require(
            key not in event_keys,
            "Duplicate primary event",
        )

        event_keys.add(key)

        by_video[
            row["video"]
        ].append(row)

    require(
        len(by_video)
        == expected["primary_videos"],
        "Primary video count mismatch",
    )

    output_rows = []
    video_rows = []

    events_literal_zero = 0
    events_short_history = 0
    anchorable_events = 0

    for video_index, video in enumerate(
        sorted(by_video),
        start=1,
    ):
        ann_dir = (
            annotation_root / video
        )

        pngs = sorted(
            ann_dir.glob("*.png")
        )

        require(
            pngs,
            "No annotations for " + video,
        )

        required_frames = {0}

        for event in by_video[video]:
            disappear = int(
                event["disappear_start"]
            )

            reappear = int(
                event["reappear_frame"]
            )

            required_frames.update(
                range(
                    max(0, disappear - 10),
                    disappear,
                )
            )

            required_frames.add(
                reappear
            )

        require(
            max(required_frames) < len(pngs),
            "Required frame exceeds video length: "
            + video,
        )

        masks = {}
        crowding_counts = []

        height = None
        width = None

        for frame_index, png in enumerate(
            pngs
        ):
            with Image.open(png) as image:
                arr = np.array(image)

            require(
                arr.ndim == 2,
                "Annotation must be single-channel: "
                + str(png),
            )

            if height is None:
                height, width = arr.shape

            else:
                require(
                    arr.shape
                    == (height, width),
                    "Annotation dimensions changed: "
                    + video,
                )

            ids = {
                int(x)
                for x in np.unique(arr)
                if int(x) != 0
            }

            crowding_counts.append(
                len(ids)
            )

            if frame_index in required_frames:
                masks[
                    frame_index
                ] = arr

        require(
            required_frames
            <= set(masks),
            "Required mask extraction incomplete: "
            + video,
        )

        frame0_ids = {
            int(x)
            for x in np.unique(
                masks[0]
            )
            if int(x) != 0
        }

        crowding = float(
            np.mean(
                crowding_counts
            )
        )

        current_video_rows = []

        for event in by_video[video]:
            row = compute_event(
                masks,
                event,
                height,
                width,
                crowding,
                frame0_ids,
                expected[
                    "event_gap_min_frames"
                ],
            )

            output_rows.append(row)
            current_video_rows.append(row)

            anchorable_events += (
                row[
                    "target_visible_frame0"
                ]
            )

            if (
                row[
                    "pre10_min_all_area_fraction"
                ]
                == 0.0
            ):
                events_literal_zero += 1

            if (
                row[
                    "pre10_frame_count"
                ]
                < 10
            ):
                events_short_history += 1

        n_anchor = sum(
            row[
                "target_visible_frame0"
            ]
            for row
            in current_video_rows
        )

        video_rows.append({
            "video": video,
            "primary_event_count": len(
                current_video_rows
            ),
            "frame0_object_count": len(
                frame0_ids
            ),
            "anchorable_primary_event_count": n_anchor,
            "nonanchorable_primary_event_count": (
                len(current_video_rows)
                - n_anchor
            ),
            "all_primary_events_anchorable": int(
                n_anchor
                == len(current_video_rows)
            ),
            "has_any_anchorable_primary_event": int(
                n_anchor > 0
            ),
            "crowding_mean_visible_objects": crowding,
        })

        if (
            video_index % 100 == 0
            or video_index == len(by_video)
        ):
            print(
                "PROGRESS={}/{}".format(
                    video_index,
                    len(by_video),
                ),
                flush=True,
            )

    require(
        len(output_rows)
        == len(primary),
        "Primary-event loss or multiplication",
    )

    output_rows.sort(
        key=lambda row: (
            row["video"],
            int(row["object_id"]),
            int(row["reappear_frame"]),
        )
    )

    video_rows.sort(
        key=lambda row: row["video"]
    )

    zero_last_frame = sum(
        row[
            "pre10_last_frame_area_fraction"
        ] == 0.0
        for row in output_rows
    )

    require(
        zero_last_frame == 0,
        "Target absent at disappear_start-1",
    )

    anchorable_videos = sum(
        row[
            "has_any_anchorable_primary_event"
        ]
        for row in video_rows
    )

    all_anchorable_videos = sum(
        row[
            "all_primary_events_anchorable"
        ]
        for row in video_rows
    )

    zero_anchor_videos = sum(
        row[
            "anchorable_primary_event_count"
        ] == 0
        for row in video_rows
    )

    frame0_hist = Counter(
        row[
            "frame0_object_count"
        ]
        for row in video_rows
    )

    visible_pre_hist = Counter(
        row[
            "pre10_visible_frame_count"
        ]
        for row in output_rows
    )

    output_root.mkdir(
        parents=True
    )

    event_out = (
        output_root
        / "event_gt_primitives.csv"
    )

    video_out = (
        output_root
        / "video_gt_primitives.csv"
    )

    write_csv(
        event_out,
        output_rows,
        EVENT_FIELDS,
    )

    write_csv(
        video_out,
        video_rows,
        VIDEO_FIELDS,
    )

    summary = {
        "experiment": "EXP045",
        "status": (
            "GT_DI_PRIMITIVE_CENSUS_COMPLETE_"
            "NOT_DI_FREEZE"
        ),
        "repo_commit_at_execution": git(
            "rev-parse",
            "HEAD",
        ),
        "config_sha256": sha256(
            cfg_path
        ),
        "script_sha256": sha256(
            ROOT / SCRIPT_REL
        ),
        "primary_events": len(
            output_rows
        ),
        "primary_videos": len(
            video_rows
        ),
        "anchorable_primary_events": (
            anchorable_events
        ),
        "nonanchorable_primary_events": (
            len(output_rows)
            - anchorable_events
        ),
        "videos_with_any_anchorable_primary_event": (
            anchorable_videos
        ),
        "videos_with_all_primary_events_anchorable": (
            all_anchorable_videos
        ),
        "videos_with_zero_anchorable_primary_events": (
            zero_anchor_videos
        ),
        "events_with_literal_pre10_min_zero": (
            events_literal_zero
        ),
        "events_with_shorter_than_10_frame_history": (
            events_short_history
        ),
        "events_with_zero_area_at_disappear_start_minus_1": (
            zero_last_frame
        ),
        "frame0_object_count_histogram": {
            str(k): v
            for k, v
            in sorted(
                frame0_hist.items()
            )
        },
        "pre10_visible_frame_count_histogram": {
            str(k): v
            for k, v
            in sorted(
                visible_pre_hist.items()
            )
        },
        "sam_or_gate_inference_performed": False,
        "pointer_cosine_computed": False,
        "di_v1_frozen": False,
        "hard_set_selected": False,
        "final_split_constructed": False,
        "fresh_dev_evaluated": False,
        "test_evaluated": False,
        "claim_boundary": cfg[
            "claim_boundary"
        ],
        "output_hashes": {
            "event_gt_primitives.csv": sha256(
                event_out
            ),
            "video_gt_primitives.csv": sha256(
                video_out
            ),
        },
    }

    summary_out = (
        output_root
        / "summary.json"
    )

    summary_out.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "EXP045_GT_PRIMITIVE_CENSUS=PASS"
    )

    for output in (
        event_out,
        video_out,
        summary_out,
    ):
        print(
            "SHA256={} {}".format(
                sha256(output),
                output.relative_to(ROOT),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()

        else:
            main()

    except Exception as exc:
        print(
            "EXP045_STOP="
            + str(exc),
            file=sys.stderr,
        )
        raise
