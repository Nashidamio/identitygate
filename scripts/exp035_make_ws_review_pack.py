import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import PIL
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP035-ws-review-v1.json"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_clean_committed_tree():
    required = [
        "configs/EXP035-ws-review-v1.json",
        "scripts/exp035_make_ws_review_pack.py",
    ]
    for rel in required:
        subprocess.check_call(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError("EXP035 requires a clean committed tree")


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda x: x["scene_event_id"])


def get_frame_paths(frame_root, video):
    d = frame_root / video
    if not d.is_dir():
        raise RuntimeError("Missing video directory: {}".format(d))
    paths = sorted(
        [x for x in d.iterdir() if x.suffix.lower() == ".jpg"],
        key=lambda x: int(x.stem),
    )
    stems = [int(x.stem) for x in paths]
    if stems != list(range(len(paths))):
        raise RuntimeError("Non-contiguous frames: {}".format(video))
    return paths


def selected_indices(row, n_frames, cfg):
    ctx = int(cfg["visual_context"]["transition_context_frames"])
    gap_n = int(cfg["visual_context"]["gap_sample_count"])

    d0 = int(row["disappear_min"])
    d1 = int(row["disappear_max"])
    r0 = int(row["reappear_min"])
    r1 = int(row["reappear_max"])

    idx = set()

    for i in range(max(0, d0 - ctx), min(n_frames - 1, d1 + ctx) + 1):
        idx.add(i)

    for i in range(max(0, r0 - ctx), min(n_frames - 1, r1 + ctx) + 1):
        idx.add(i)

    lo = d1 + 1
    hi = r0 - 1
    count = hi - lo + 1

    if count > 0 and gap_n > 0:
        k = min(gap_n, count)
        if k == 1:
            samples = [(lo + hi) // 2]
        else:
            samples = [
                round(lo + j * (hi - lo) / (k - 1))
                for j in range(k)
            ]
        idx.update(samples)

    out = sorted(idx)
    if not out:
        raise RuntimeError("No review frames selected")

    return out


def resized_frame(path, width):
    with Image.open(path) as im:
        im = im.convert("RGB")
        height = max(1, round(im.height * width / im.width))
        return im.resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )


def labeled_tile(path, frame_idx, width):
    image = resized_frame(path, width)
    bar_h = 24
    tile = Image.new(
        "RGB",
        (image.width, image.height + bar_h),
        "white",
    )
    tile.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (6, 5),
        "frame {:05d}".format(frame_idx),
        fill="black",
    )
    return tile


def make_contact_sheet(scene_id, paths, indices, width, out_path):
    tiles = [
        labeled_tile(paths[i], i, width)
        for i in indices
    ]

    cols = min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    title_h = 32
    tile_h = max(x.height for x in tiles)

    sheet = Image.new(
        "RGB",
        (cols * width, title_h + rows * tile_h),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        scene_id,
        fill="black",
    )

    for j, tile in enumerate(tiles):
        x = (j % cols) * width
        y = title_h + (j // cols) * tile_h
        sheet.paste(tile, (x, y))

    sheet.save(out_path)


def make_gif(scene_id, paths, indices, width, duration, loop, out_path):
    frames = []

    for i in indices:
        image = resized_frame(paths[i], width)
        bar_h = 24
        frame = Image.new(
            "RGB",
            (image.width, image.height + bar_h),
            "white",
        )
        frame.paste(image, (0, bar_h))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (6, 5),
            "{}  frame {:05d}".format(scene_id, i),
            fill="black",
        )
        frames.append(frame)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pack",
        choices=["audit30", "disagreements"],
        required=True,
    )
    args = ap.parse_args()

    require_clean_committed_tree()

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

    if sha256sum(CFG_PATH) != "f88f34449f7ead0efecec01114d0a445f7ccc6c871041914d7dfd3614a030bad":
        raise RuntimeError("Unexpected EXP035 config SHA")

    blind = cfg["review_blinding"]
    required_blinding = [
        "hide_transnet_scores",
        "hide_mad_scores",
        "hide_component_decisions",
        "hide_automatic_status",
    ]
    if not all(blind[x] is True for x in required_blinding):
        raise RuntimeError("Blinding contract failure")

    pack_cfg = cfg["packs"][args.pack]

    if args.pack == "audit30":
        source_rel = cfg["source"]["audit_csv"]
        expected_sha = cfg["source"]["audit_csv_sha256"]
    else:
        source_rel = cfg["source"]["disagreement_csv"]
        expected_sha = cfg["source"]["disagreement_csv_sha256"]

    source_path = ROOT / source_rel

    if sha256sum(source_path) != expected_sha:
        raise RuntimeError("Source CSV SHA mismatch")

    rows = load_rows(source_path)

    if len(rows) != int(pack_cfg["expected_scenes"]):
        raise RuntimeError(
            "Unexpected scene count: {}".format(len(rows))
        )

    frame_root = Path(cfg["frame_source"])
    output_root = Path(cfg["output_root"]).expanduser()
    pack_root = output_root / args.pack

    if pack_root.exists():
        raise RuntimeError(
            "Output pack already exists: {}".format(pack_root)
        )

    pack_root.mkdir(parents=True)

    batch_size = int(pack_cfg["batch_size"])
    thumb_width = int(
        cfg["visual_context"]["contact_sheet_thumbnail_width"]
    )
    gif_duration = int(
        cfg["visual_context"]["animated_gif_frame_duration_ms"]
    )
    gif_loop = int(
        cfg["visual_context"]["animated_gif_loop"]
    )

    label_rows = []
    index_rows = []

    start = time.time()

    for row_no, row in enumerate(rows):
        scene_id = row["scene_event_id"]
        video = row["video"]

        batch = row_no // batch_size + 1
        batch_dir = pack_root / "batch_{:04d}".format(batch)
        safe_id = scene_id.replace(":", "__")
        scene_dir = batch_dir / safe_id
        scene_dir.mkdir(parents=True)

        paths = get_frame_paths(frame_root, video)
        indices = selected_indices(
            row,
            len(paths),
            cfg,
        )

        make_contact_sheet(
            scene_id,
            paths,
            indices,
            thumb_width,
            scene_dir / "contact_sheet.png",
        )

        make_gif(
            scene_id,
            paths,
            indices,
            thumb_width,
            gif_duration,
            gif_loop,
            scene_dir / "review.gif",
        )

        label_rows.append(
            {
                "scene_event_id": scene_id,
                "manual_label": "",
                "reason": "",
                "reviewer_notes": "",
            }
        )

        index_rows.append(
            {
                "scene_event_id": scene_id,
                "video": video,
                "relative_folder": str(scene_dir.relative_to(pack_root)),
                "selected_frame_count": len(indices),
                "selected_frame_indices": ";".join(str(x) for x in indices),
            }
        )

        print(
            "SCENE {}/{} {}".format(
                row_no + 1,
                len(rows),
                scene_id,
            ),
            flush=True,
        )

    with open(pack_root / "review_labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_event_id",
                "manual_label",
                "reason",
                "reviewer_notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(label_rows)

    with open(pack_root / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_event_id",
                "video",
                "relative_folder",
                "selected_frame_count",
                "selected_frame_indices",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(index_rows)

    manifest = {
        "experiment": "EXP035",
        "pack": args.pack,
        "status": "VISUAL_REVIEW_PACK_GENERATED",
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "config_sha256": sha256sum(CFG_PATH),
        "source_csv": source_rel,
        "source_csv_sha256": expected_sha,
        "scene_count": len(rows),
        "pillow_version": PIL.__version__,
        "blinded": True,
        "scores_exposed": False,
        "automatic_status_exposed": False,
        "runtime_sec": time.time() - start,
    }

    (pack_root / "pack_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
