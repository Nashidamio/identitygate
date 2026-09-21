#!/usr/bin/env python3

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP042-development-exposure-v2.json"

SCALAR_VIDEO_KEYS = {
    "video",
    "video_id",
}

LIST_VIDEO_KEYS = {
    "videos",
    "video_ids",
    "train_videos",
    "dev_videos",
}


def sha256sum(path):
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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def require_clean_committed_tree():
    required = [
        "configs/EXP042-development-exposure-v2.json",
        "scripts/exp042_build_development_exposure.py",
    ]

    for rel in required:
        subprocess.check_call(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--error-unmatch",
                rel,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    status = git("status", "--porcelain")

    require(
        not status,
        "EXP042 requires a clean committed tree",
    )


def validate_video_list(values, name, expected=None):
    require(
        isinstance(values, list),
        name + " must be a list",
    )

    require(
        all(
            isinstance(x, str)
            and x
            and x == x.strip()
            for x in values
        ),
        name + " contains invalid video IDs",
    )

    require(
        len(values) == len(set(values)),
        name + " contains duplicates",
    )

    if expected is not None:
        require(
            len(values) == expected,
            "{} expected {} IDs but found {}".format(
                name,
                expected,
                len(values),
            ),
        )

    return set(values)


def explicit_json_video_ids(document):
    found = defaultdict(list)

    def visit(node, location):
        if isinstance(node, dict):
            for key, value in node.items():
                here = location + "/" + key

                if (
                    key in SCALAR_VIDEO_KEYS
                    and isinstance(value, str)
                    and value
                ):
                    found[value].append(here)

                elif (
                    key in LIST_VIDEO_KEYS
                    and isinstance(value, list)
                ):
                    for i, item in enumerate(value):
                        if isinstance(item, str) and item:
                            found[item].append(
                                here + "/" + str(i)
                            )

                if isinstance(value, (dict, list)):
                    visit(value, here)

        elif isinstance(node, list):
            for i, value in enumerate(node):
                if isinstance(value, (dict, list)):
                    visit(
                        value,
                        location + "/" + str(i),
                    )

    visit(document, "$")

    return dict(found)


def explicit_csv_video_ids(path):
    found = defaultdict(list)

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        fields = list(reader.fieldnames or [])

        keys = [
            k
            for k in fields
            if k in SCALAR_VIDEO_KEYS
        ]

        for row_no, row in enumerate(
            reader,
            start=2,
        ):
            for key in keys:
                value = (row.get(key) or "").strip()

                if value:
                    found[value].append(
                        "row={}/{}".format(
                            row_no,
                            key,
                        )
                    )

    return dict(found)


def inspect_scope_file(path, boundary):
    if path.suffix.lower() == ".json":
        found = explicit_json_video_ids(
            read_json(path)
        )

    elif path.suffix.lower() == ".csv":
        found = explicit_csv_video_ids(path)

    else:
        raise RuntimeError(
            "Unsupported later-scope file: {}".format(path)
        )

    ids = set(found)
    outside = sorted(ids - boundary)

    if outside:
        status = "EXPLICIT_IDS_OUTSIDE_BOUNDARY"

    elif ids:
        status = "EXPLICIT_IDS_COVERED"

    else:
        status = "NO_EXPLICIT_VIDEO_IDS"

    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256sum(path),
        "status": status,
        "explicit_video_count": len(ids),
        "outside_boundary_ids": outside,
        "id_evidence": found,
    }


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main():
    require_clean_committed_tree()

    cfg = read_json(CFG_PATH)

    require(
        cfg["experiment"] == "EXP042",
        "Unexpected experiment ID",
    )

    subprocess.check_call(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            cfg["source_git_head"],
            "HEAD",
        ]
    )

    docs = {}

    for key, item in cfg["inputs"].items():
        path = ROOT / item["path"]

        require(
            path.is_file(),
            "Missing input: {}".format(path),
        )

        actual = sha256sum(path)

        require(
            actual == item["sha256"],
            "Input SHA mismatch for {}".format(path),
        )

        docs[key] = read_json(path)

    expected = cfg["expected"]

    legacy = validate_video_list(
        docs["legacy_exclusions"]["video_ids"],
        "legacy exclusions",
        expected["legacy_exclusions"],
    )

    require(
        docs["legacy_exclusions"]["counts"][
            "total_unique_excluded_videos"
        ] == len(legacy),
        "Legacy exclusion count disagrees with video_ids",
    )

    splits = docs["historical_split"]["splits"]

    train = validate_video_list(
        splits["TRAIN"]["videos"],
        "EXP017 TRAIN",
        expected["exp017_train"],
    )

    dev = validate_video_list(
        splits["DEV"]["videos"],
        "EXP017 DEV",
        expected["exp017_dev"],
    )

    require(
        splits["TRAIN"]["n_videos"] == len(train),
        "EXP017 TRAIN declared count mismatch",
    )

    require(
        splits["DEV"]["n_videos"] == len(dev),
        "EXP017 DEV declared count mismatch",
    )

    require(
        not (train & dev),
        "EXP017 TRAIN/DEV overlap",
    )

    boundary = legacy | train | dev

    exp021 = docs["relational_scope"]

    require(
        exp021["test_videos_touched"] == 0,
        "EXP021 reports TEST videos touched",
    )

    rel_rows = exp021["eligible_videos"]

    require(
        isinstance(rel_rows, list),
        "EXP021 eligible_videos is not a list",
    )

    rel_ids = validate_video_list(
        [r["video"] for r in rel_rows],
        "EXP021 eligible videos",
        expected["exp021_eligible"],
    )

    require(
        exp021["n_eligible_videos"] == len(rel_ids),
        "EXP021 eligible count mismatch",
    )

    require(
        not (rel_ids - boundary),
        "EXP021 contains videos outside reconstructed boundary",
    )

    for row in rel_rows:
        split = row["split"]
        video = row["video"]

        require(
            split in ("TRAIN", "DEV"),
            "Unexpected EXP021 split: {}".format(split),
        )

        expected_set = (
            train
            if split == "TRAIN"
            else dev
        )

        require(
            video in expected_set,
            "EXP021 video not in declared historical split: "
            + video,
        )

    evidence = defaultdict(list)

    for video in sorted(boundary):
        if video in legacy:
            evidence[video].append(
                "development_exclusions_v1.video_ids"
            )

        if video in train:
            evidence[video].append(
                "EXP017.splits.TRAIN.videos"
            )

        if video in dev:
            evidence[video].append(
                "EXP017.splits.DEV.videos"
            )

        if video in rel_ids:
            evidence[video].append(
                "EXP021.eligible_videos"
            )

    scope_checks = []

    for rel in cfg["later_scope_files"]:
        path = ROOT / rel

        require(
            path.is_file(),
            "Missing later-scope file: {}".format(path),
        )

        scope_checks.append(
            inspect_scope_file(
                path,
                boundary,
            )
        )

    outside_ids = sorted(
        {
            video
            for item in scope_checks
            for video in item["outside_boundary_ids"]
        }
    )

    no_explicit_scope = [
        item["path"]
        for item in scope_checks
        if item["status"] == "NO_EXPLICIT_VIDEO_IDS"
    ]

    counts = {
        "legacy_exclusions": len(legacy),
        "exp017_train": len(train),
        "exp017_dev": len(dev),
        "exp017_train_dev_union": len(train | dev),
        "legacy_overlap_with_exp017_train_dev": len(
            legacy & (train | dev)
        ),
        "added_beyond_v1": len(
            boundary - legacy
        ),
        "total_unique_excluded_videos": len(boundary),
        "exp021_eligible": len(rel_ids),
        "exp021_outside_boundary": len(
            rel_ids - boundary
        ),
        "later_scope_files_checked": len(scope_checks),
        "later_explicit_ids_outside_boundary": len(
            outside_ids
        ),
        "later_scope_files_without_explicit_video_ids": len(
            no_explicit_scope
        ),
    }

    out_dir = ROOT / cfg["output_dir"]

    require(
        not out_dir.exists(),
        "Output directory already exists; do not overwrite",
    )

    out_dir.mkdir(parents=True)

    exclusion_records = [
        {
            "video_id": video,
            "exclude_from": [
                "DEV",
                "TEST",
            ],
            "evidence": evidence[video],
        }
        for video in sorted(boundary)
    ]

    exclusions = {
        "experiment": "EXP042",
        "version": 2,
        "status": (
            "DEVELOPMENT_EXPOSURE_BOUNDARY_RECONSTRUCTED"
        ),
        "policy": docs["legacy_exclusions"]["policy"],
        "historical_p2_reference": docs[
            "legacy_exclusions"
        ].get("historical_p2_reference"),
        "video_ids": sorted(boundary),
        "records": exclusion_records,
        "counts": counts,
        "claim_boundary": cfg["claim_boundary"],
        "fresh_final_dev_touched": False,
        "test_touched": False,
    }

    coverage = {
        "experiment": "EXP042",
        "status": "LATER_SCOPE_EXPLICIT_ID_CHECK_COMPLETE",
        "checks": scope_checks,
        "outside_boundary_ids": outside_ids,
        "files_without_explicit_video_ids": no_explicit_scope,
        "automatic_boundary_expansion_performed": False,
        "note": (
            "Only explicit video-ID fields were checked. "
            "No ID was inferred from metrics, counts or filenames."
        ),
    }

    exclusion_path = (
        out_dir
        / "development_exclusions_v2.json"
    )

    coverage_path = (
        out_dir
        / "coverage_report.json"
    )

    write_json(
        exclusion_path,
        exclusions,
    )

    write_json(
        coverage_path,
        coverage,
    )

    summary = {
        "experiment": "EXP042",
        "status": "DEVELOPMENT_EXPOSURE_RECONSTRUCTION_COMPLETE",
        "repo_commit_at_execution": git(
            "rev-parse",
            "HEAD",
        ),
        "config_sha256": sha256sum(CFG_PATH),
        "script_sha256": sha256sum(Path(__file__)),
        "counts": counts,
        "outside_boundary_ids": outside_ids,
        "files_without_explicit_video_ids": no_explicit_scope,
        "final_split_constructed": False,
        "di_v1_defined": False,
        "sam_or_gate_inference_performed": False,
        "fresh_final_dev_touched": False,
        "test_touched": False,
        "output_hashes": {
            "development_exclusions_v2.json": sha256sum(
                exclusion_path
            ),
            "coverage_report.json": sha256sum(
                coverage_path
            ),
        },
    }

    summary_path = out_dir / "summary.json"

    write_json(
        summary_path,
        summary,
    )

    print("EXP042_BOUNDARY_BUILD=PASS")
    print(
        "COUNTS="
        + json.dumps(
            counts,
            sort_keys=True,
        )
    )

    print(
        "OUTSIDE_BOUNDARY_IDS="
        + json.dumps(outside_ids)
    )

    print(
        "NO_EXPLICIT_SCOPE_FILES="
        + json.dumps(no_explicit_scope)
    )

    for item in scope_checks:
        print(
            "SCOPE_CHECK={} status={} explicit_ids={} outside={}".format(
                item["path"],
                item["status"],
                item["explicit_video_count"],
                len(item["outside_boundary_ids"]),
            )
        )

    print("FINAL_SPLIT_CONSTRUCTED=0")
    print("DI_V1_DEFINED=0")
    print("FRESH_FINAL_DEV_TOUCHED=0")
    print("TEST_TOUCHED=0")

    for path in (
        exclusion_path,
        coverage_path,
        summary_path,
    ):
        print(
            "SHA256={} {}".format(
                sha256sum(path),
                path.relative_to(ROOT),
            )
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            "EXP042_STOP={}".format(exc),
            file=sys.stderr,
        )
        raise
