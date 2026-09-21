#!/usr/bin/env python3

import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import PIL

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP040-ws-audit-replacement-v1.json"

sys.path.insert(0, str(ROOT / "scripts"))

from exp035_make_ws_review_pack import (
    get_frame_paths,
    make_contact_sheet,
    make_gif,
    selected_indices,
)


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_clean_committed_tree():
    required = [
        "configs/EXP040-ws-audit-replacement-v1.json",
        "scripts/exp040_make_ws_audit_replacement.py",
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
        raise RuntimeError("EXP040 requires a clean committed tree")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    require_clean_committed_tree()

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

    if cfg["experiment"] != "EXP040":
        raise RuntimeError("Unexpected experiment ID")

    source_cfg = cfg["source_population"]
    source_path = ROOT / source_cfg["path"]

    if sha256sum(source_path) != source_cfg["sha256"]:
        raise RuntimeError("EXP019 source SHA mismatch")

    rows = read_csv(source_path)

    expected_fields = [
        "scene_event_id",
        "video",
        "n_object_events",
        "n_unique_objects",
        "disappear_min",
        "disappear_max",
        "reappear_min",
        "reappear_max",
    ]

    if not rows:
        raise RuntimeError("Empty EXP019 source")

    if list(rows[0].keys()) != expected_fields:
        raise RuntimeError("Unexpected EXP019 source schema")

    if len(rows) != int(source_cfg["expected_scene_count"]):
        raise RuntimeError("Unexpected EXP019 scene count")

    by_id = {r["scene_event_id"]: r for r in rows}

    if len(by_id) != len(rows):
        raise RuntimeError("Duplicate EXP019 scene_event_id")

    population_ids = sorted(by_id)

    audit_cfg = cfg["original_audit"]
    audit_path = ROOT / audit_cfg["manifest_path"]

    if sha256sum(audit_path) != audit_cfg["manifest_sha256"]:
        raise RuntimeError("Original audit manifest SHA mismatch")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    original_ids = list(audit["sampled_scene_event_ids"])

    if len(original_ids) != int(audit_cfg["sample_size"]):
        raise RuntimeError("Original audit count mismatch")

    if len(set(original_ids)) != len(original_ids):
        raise RuntimeError("Duplicate original audit IDs")

    if not set(original_ids).issubset(set(population_ids)):
        raise RuntimeError("Original audit ID outside EXP019 population")

    compromised = list(cfg["compromised_scene_event_ids"])

    if len(compromised) != 3 or len(set(compromised)) != 3:
        raise RuntimeError("Compromised ID contract failure")

    if not set(compromised).issubset(set(original_ids)):
        raise RuntimeError("Compromised ID is not in original audit")

    valid_original_ids = [
        x for x in original_ids
        if x not in set(compromised)
    ]

    if len(valid_original_ids) != 27:
        raise RuntimeError("Expected 27 uncompromised original items")

    original_set = set(original_ids)

    eligible_ids = [
        x for x in population_ids
        if x not in original_set
    ]

    if len(eligible_ids) != 2149:
        raise RuntimeError("Replacement eligibility count mismatch")

    repl_cfg = cfg["replacement_sampling"]
    seed = int(repl_cfg["seed"])
    sample_size = int(repl_cfg["sample_size"])

    rng = random.Random(seed)
    replacement_ids = rng.sample(
        eligible_ids,
        sample_size,
    )

    if set(replacement_ids) & original_set:
        raise RuntimeError("Replacement overlaps original audit")

    final_ids = valid_original_ids + replacement_ids

    if len(final_ids) != 30 or len(set(final_ids)) != 30:
        raise RuntimeError("Final audit ID contract failure")

    final_rows = sorted(
        [by_id[x] for x in final_ids],
        key=lambda r: r["scene_event_id"],
    )

    repo_out = ROOT / "experiments/EXP040_ws_audit_replacement"
    visual_root = Path(cfg["output_root"]).expanduser()

    if repo_out.exists():
        raise RuntimeError("Repository output already exists")

    if visual_root.exists():
        raise RuntimeError("Visual output already exists")

    repo_out.mkdir(parents=True)
    visual_root.mkdir(parents=True)

    source_out = repo_out / "final_audit_source.csv"

    write_csv(
        source_out,
        final_rows,
        expected_fields,
    )

    frame_root = Path(cfg["frame_source"])
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

    for row_no, row in enumerate(final_rows, start=1):
        scene_id = row["scene_event_id"]
        video = row["video"]
        safe_id = scene_id.replace(":", "__")
        scene_dir = visual_root / safe_id
        scene_dir.mkdir(parents=True)

        paths = get_frame_paths(
            frame_root,
            video,
        )

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
                "relative_folder": safe_id,
                "selected_frame_count": len(indices),
                "selected_frame_indices": ";".join(
                    str(x) for x in indices
                ),
            }
        )

        print(
            "SCENE {}/{} {}".format(
                row_no,
                len(final_rows),
                scene_id,
            ),
            flush=True,
        )

    write_csv(
        visual_root / "review_labels.csv",
        label_rows,
        [
            "scene_event_id",
            "manual_label",
            "reason",
            "reviewer_notes",
        ],
    )

    write_csv(
        visual_root / "index.csv",
        index_rows,
        [
            "scene_event_id",
            "video",
            "relative_folder",
            "selected_frame_count",
            "selected_frame_indices",
        ],
    )

    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    replacement_manifest = {
        "experiment": "EXP040",
        "status": "PROSPECTIVE_REPLACEMENT_SAMPLE_EXECUTED",
        "repo_commit_at_execution": head,
        "source_population_count": len(population_ids),
        "source_population_sha256": sha256sum(source_path),
        "original_audit_manifest_sha256": sha256sum(audit_path),
        "original_audit_count": len(original_ids),
        "compromised_count": len(compromised),
        "uncompromised_original_count": len(valid_original_ids),
        "replacement_eligible_count": len(eligible_ids),
        "replacement_seed": seed,
        "replacement_count": len(replacement_ids),
        "replacement_scene_event_ids": replacement_ids,
        "final_valid_audit_count": len(final_ids),
        "final_audit_source_sha256": sha256sum(source_out),
        "config_sha256": sha256sum(CFG_PATH),
        "script_sha256": sha256sum(Path(__file__)),
        "replacement_selection_used_pixel_outcomes": False,
        "replacement_selection_used_sam_or_gate_outcomes": False,
        "fresh_final_dev_touched": False,
        "test_touched": False,
    }

    manifest_path = repo_out / "replacement_manifest.json"
    manifest_path.write_text(
        json.dumps(
            replacement_manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    pack_manifest = {
        "experiment": "EXP040",
        "status": "BLINDED_FINAL_AUDIT30_PACK_GENERATED",
        "repo_commit_at_execution": head,
        "scene_count": len(final_rows),
        "uncompromised_original_count": len(valid_original_ids),
        "replacement_count": len(replacement_ids),
        "blinded": True,
        "scores_exposed": False,
        "automatic_status_exposed": False,
        "source": "EXP019 annotation-side rows only",
        "source_csv_sha256": sha256sum(source_out),
        "config_sha256": sha256sum(CFG_PATH),
        "script_sha256": sha256sum(Path(__file__)),
        "pillow_version": PIL.__version__,
        "runtime_sec": time.time() - start,
    }

    (visual_root / "pack_manifest.json").write_text(
        json.dumps(
            pack_manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("EXP040_REPLACEMENT_AND_PACK=PASS")
    print("ELIGIBLE_COUNT="+str(len(eligible_ids)))
    print("REPLACEMENT_COUNT="+str(len(replacement_ids)))
    print("FINAL_AUDIT_COUNT="+str(len(final_ids)))
    print("REPLACEMENT_IDS=")
    for x in replacement_ids:
        print(x)
    print("FINAL_SOURCE_SHA256="+sha256sum(source_out))
    print("REPLACEMENT_MANIFEST_SHA256="+sha256sum(manifest_path))
    print("REVIEW_LABELS_SHA256="+sha256sum(visual_root / "review_labels.csv"))
    print("FRESH_FINAL_DEV_TOUCHED=0")
    print("TEST_TOUCHED=0")


if __name__ == "__main__":
    main()
