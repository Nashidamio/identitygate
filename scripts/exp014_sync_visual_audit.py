from pathlib import Path
import csv
import math
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/mnt/d/thesis_data/mosev2/train")
EVENTS = Path("experiments/EXP011_events.csv")
OUT = Path("experiments/EXP014_sync_visual_audit")

PM = 10
MIN_NEIGHBORS = 5


def load_events():
    rows = []
    with EVENTS.open(newline="") as f:
        for r in csv.DictReader(f):
            r["reappear_frame"] = int(r["reappear_frame"])
            r["gap_len"] = int(r["gap_len"])
            r["por30"] = int(r["por30"])
            r["start_absent"] = r["reappear_frame"] - r["gap_len"]
            rows.append(r)
    return rows


def mark_sync_events(rows):
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)

    for video, group in by_video.items():
        for r in group:
            rf = r["reappear_frame"]
            r["neighbor_count"] = sum(
                abs(x["reappear_frame"] - rf) <= PM for x in group
            )
            r["sync_event"] = r["neighbor_count"] >= MIN_NEIGHBORS

    return by_video


def make_clusters(sync_rows):
    by_video = defaultdict(list)
    for r in sync_rows:
        by_video[r["video"]].append(r)

    clusters = []

    for video in sorted(by_video):
        group = sorted(by_video[video], key=lambda r: r["reappear_frame"])

        current = [group[0]]
        previous = group[0]["reappear_frame"]

        for r in group[1:]:
            if r["reappear_frame"] - previous <= PM:
                current.append(r)
            else:
                clusters.append((video, current))
                current = [r]
            previous = r["reappear_frame"]

        clusters.append((video, current))

    return clusters


def mask_info(mask):
    ids = sorted(int(x) for x in np.unique(mask) if int(x) != 0)
    return ids


def overlay_gt(image, mask):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()

    if mask.shape != (arr.shape[0], arr.shape[1]):
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize((arr.shape[1], arr.shape[0]), Image.Resampling.NEAREST)
        mask = np.asarray(mask_img)

    for oid in mask_info(mask):
        region = mask == oid
        color = np.array([
            (37 * oid + 53) % 256,
            (97 * oid + 101) % 256,
            (173 * oid + 29) % 256,
        ], dtype=np.float32)
        arr[region] = 0.55 * arr[region] + 0.45 * color

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_sheet(video, frames, output_path, use_overlay):
    jpg_dir = ROOT / "JPEGImages" / video
    ann_dir = ROOT / "Annotations" / video

    tiles = []
    font = ImageFont.load_default()

    for frame in frames:
        jpg = jpg_dir / f"{frame:05d}.jpg"
        ann = ann_dir / f"{frame:05d}.png"

        image = Image.open(jpg).convert("RGB")
        mask = np.asarray(Image.open(ann))
        ids = mask_info(mask)

        shown = overlay_gt(image, mask) if use_overlay else image

        tile_w = 320
        ratio = tile_w / shown.width
        tile_h = max(1, int(shown.height * ratio))
        shown = shown.resize((tile_w, tile_h), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (tile_w, tile_h + 34), "white")
        canvas.paste(shown, (0, 34))

        draw = ImageDraw.Draw(canvas)
        label = f"frame {frame} | GT objects={len(ids)}"
        draw.text((6, 5), label, fill="black", font=font)

        tiles.append(canvas)

    cols = 4
    rows = math.ceil(len(tiles) / cols)
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")

    for i, tile in enumerate(tiles):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(tile, (x, y))

    sheet.save(output_path, quality=92)


def write_events_csv(path, rows):
    fields = [
        "video",
        "object_id",
        "start_absent",
        "reappear_frame",
        "gap_len",
        "neighbor_count",
        "por30",
    ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def write_timeline(path, video, cluster_rows):
    ann_dir = ROOT / "Annotations" / video

    cluster_ids = {int(r["object_id"]) for r in cluster_rows}

    first = max(0, min(r["start_absent"] for r in cluster_rows) - 2)
    last = max(r["reappear_frame"] for r in cluster_rows) + 2

    n_frames = len(list(ann_dir.glob("*.png")))
    last = min(last, n_frames - 1)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame",
            "all_visible_object_count",
            "cluster_visible_object_count",
            "all_visible_object_ids",
            "cluster_visible_object_ids",
        ])

        for frame in range(first, last + 1):
            mask = np.asarray(Image.open(ann_dir / f"{frame:05d}.png"))
            ids = mask_info(mask)
            cluster_visible = [x for x in ids if x in cluster_ids]

            w.writerow([
                frame,
                len(ids),
                len(cluster_visible),
                "|".join(map(str, ids)),
                "|".join(map(str, cluster_visible)),
            ])


def choose_keyframes(video, cluster_rows):
    ann_dir = ROOT / "Annotations" / video
    n_frames = len(list(ann_dir.glob("*.png")))

    starts = sorted({r["start_absent"] for r in cluster_rows})
    returns = sorted({r["reappear_frame"] for r in cluster_rows})

    frames = set(starts + returns)

    if starts:
        frames.add(max(0, min(starts) - 1))

    if starts and returns:
        middle = int(round((max(starts) + min(returns)) / 2))
        frames.add(max(0, min(middle, n_frames - 1)))

    if returns:
        frames.add(min(n_frames - 1, max(returns) + 1))

    return sorted(frames)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_events()
    mark_sync_events(rows)

    sync_rows = [r for r in rows if r["sync_event"]]
    clusters = make_clusters(sync_rows)

    summary_fields = [
        "video",
        "cluster_id",
        "sync_events",
        "objects",
        "start_absent_min",
        "start_absent_max",
        "reappear_min",
        "reappear_max",
    ]

    summary_rows = []

    for cluster_number, (video, group) in enumerate(clusters, 1):
        rf_min = min(r["reappear_frame"] for r in group)
        rf_max = max(r["reappear_frame"] for r in group)

        folder = OUT / f"{video}_cluster{cluster_number}_{rf_min}-{rf_max}"
        folder.mkdir(parents=True, exist_ok=True)

        write_events_csv(folder / "events.csv", group)
        write_timeline(folder / "timeline.csv", video, group)

        keyframes = choose_keyframes(video, group)

        make_sheet(
            video,
            keyframes,
            folder / "keyframes_raw.jpg",
            use_overlay=False,
        )

        make_sheet(
            video,
            keyframes,
            folder / "keyframes_gt_overlay.jpg",
            use_overlay=True,
        )

        summary_rows.append({
            "video": video,
            "cluster_id": cluster_number,
            "sync_events": len(group),
            "objects": len({r["object_id"] for r in group}),
            "start_absent_min": min(r["start_absent"] for r in group),
            "start_absent_max": max(r["start_absent"] for r in group),
            "reappear_min": rf_min,
            "reappear_max": rf_max,
        })

    with (OUT / "cluster_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    lines = [
        "# EXP014 synchronized-reappearance visual audit",
        "",
        "Purpose: visually inspect GT-timing synchronized reappearance clusters before any dataset exclusion or reclassification.",
        "",
        "Important: these clusters are candidates only. No cluster is classified as whole-scene occlusion by this script.",
        "",
        f"Event-level criterion: at least {MIN_NEIGHBORS} reappearance events within +/-{PM} frames.",
        f"Event-level synchronized events: {len(sync_rows)}.",
        f"Candidate clusters: {len(clusters)}.",
        "",
        "Each cluster directory contains:",
        "- events.csv: qualifying event rows",
        "- timeline.csv: per-frame GT visibility counts",
        "- keyframes_raw.jpg: raw visual evidence",
        "- keyframes_gt_overlay.jpg: same frames with GT masks overlaid",
        "",
        "Classification is intentionally left OPEN pending human inspection.",
    ]

    (OUT / "README.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")

    print(f"Created {OUT}")
    print(f"sync_events={len(sync_rows)}")
    print(f"clusters={len(clusters)}")

    for r in summary_rows:
        print(
            "{} cluster={} events={} objects={} absent_starts={}-{} reappear={}-{}".format(
                r["video"],
                r["cluster_id"],
                r["sync_events"],
                r["objects"],
                r["start_absent_min"],
                r["start_absent_max"],
                r["reappear_min"],
                r["reappear_max"],
            )
        )


if __name__ == "__main__":
    main()
