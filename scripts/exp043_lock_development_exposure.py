#!/usr/bin/env python3

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_REL = "configs/EXP043-development-exposure-final-lock-v1.json"
SCRIPT_REL = "scripts/exp043_lock_development_exposure.py"


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


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
    ).strip()


def csv_video_ids(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])

        candidates = [
            x
            for x in (
                "video_id",
                "video",
                "video_name",
            )
            if x in fields
        ]

        require(
            candidates,
            "No recognized video column in {}: {}".format(
                path,
                fields,
            ),
        )

        key = candidates[0]

        ids = {
            row[key].strip()
            for row in reader
            if row.get(key)
            and row[key].strip()
        }

    return ids


def require_source_unchanged(source_commit, rel):
    subprocess.check_call(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
            source_commit,
            "HEAD",
            "--",
            rel,
        ]
    )


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
    require(
        not git("status", "--porcelain"),
        "STOP: execution requires a clean working tree",
    )

    tracked = set(
        git("ls-files").splitlines()
    )

    require(
        CFG_REL in tracked,
        "EXP043 config must be committed first",
    )

    require(
        SCRIPT_REL in tracked,
        "EXP043 script must be committed first",
    )

    cfg_path = ROOT / CFG_REL
    cfg = read_json(cfg_path)

    require(
        cfg["experiment"] == "EXP043",
        "Wrong experiment config",
    )

    source_commit = cfg["source_result_commit"]

    subprocess.check_call(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            source_commit,
            "HEAD",
        ]
    )

    source_paths = []

    for item in cfg["exp042_inputs"].values():
        source_paths.append(item["path"])

    source_paths.extend(
        cfg["closure_files"].values()
    )

    for rel in source_paths:
        path = ROOT / rel

        require(
            path.is_file(),
            "Missing source file: " + rel,
        )

        require(
            rel in tracked,
            "Source file is not tracked: " + rel,
        )

        require_source_unchanged(
            source_commit,
            rel,
        )

    exp042_docs = {}

    for key, item in cfg["exp042_inputs"].items():
        path = ROOT / item["path"]
        digest = sha256(path)

        require(
            digest == item["sha256"],
            "EXP042 hash mismatch: " + item["path"],
        )

        exp042_docs[key] = read_json(path)

    exclusions = exp042_docs["exclusions"]
    coverage = exp042_docs["coverage"]
    summary42 = exp042_docs["summary"]

    boundary = set(
        exclusions["video_ids"]
    )

    require(
        len(boundary)
        == cfg["expected"]["excluded_videos"],
        "Unexpected development-exclusion count",
    )

    require(
        exclusions["counts"][
            "total_unique_excluded_videos"
        ] == len(boundary),
        "EXP042 exclusion count mismatch",
    )

    require(
        summary42["counts"][
            "total_unique_excluded_videos"
        ] == len(boundary),
        "EXP042 summary count mismatch",
    )

    require(
        coverage["outside_boundary_ids"] == [],
        "EXP042 recorded outside-boundary IDs",
    )

    expected_unresolved = {
        "experiments/EXP024_cluster_bootstrap/summary.json",
        "experiments/EXP024_utility_prepare/summary.json",
        "experiments/EXP029_b2core_train/summary.json",
        "experiments/EXP031_b3_train/summary.json",
    }

    require(
        set(
            coverage[
                "files_without_explicit_video_ids"
            ]
        )
        == expected_unresolved,
        "Unexpected EXP042 unresolved scope set",
    )

    f = {
        key: ROOT / rel
        for key, rel
        in cfg["closure_files"].items()
    }

    index_ids = csv_video_ids(
        f["exp023_index"]
    )

    primitive_ids = csv_video_ids(
        f["exp023_primitives"]
    )

    utility_ids = csv_video_ids(
        f["exp024_utility_csv"]
    )

    require(
        len(index_ids)
        == cfg["expected"]["train18_videos"],
        "Unexpected EXP023 index video count",
    )

    require(
        index_ids == primitive_ids == utility_ids,
        "EXP023/EXP024 train18 video sets disagree",
    )

    utility_cfg = read_json(
        f["exp024_utility_config"]
    )

    utility_summary = read_json(
        f["exp024_utility_summary"]
    )

    require(
        ROOT / utility_cfg["input_csv"]
        == f["exp023_primitives"],
        "EXP024 utility config input mismatch",
    )

    require(
        sha256(f["exp023_primitives"])
        == utility_summary["input_sha256"],
        "EXP024 utility summary input hash mismatch",
    )

    require(
        set(
            utility_summary["per_video"].keys()
        )
        == utility_ids,
        "EXP024 utility per-video set mismatch",
    )

    require(
        utility_summary["videos"]
        == len(utility_ids),
        "EXP024 utility video count mismatch",
    )

    model_utility_cfg = read_json(
        f["exp024_model_utility_config"]
    )

    require(
        ROOT / model_utility_cfg["input_csv"]
        == f["exp024_utility_csv"],
        "EXP024 model-utility input mismatch",
    )

    oof_ids = csv_video_ids(
        f["exp024_oof_predictions"]
    )

    require(
        oof_ids == utility_ids,
        "EXP024 OOF video set differs from train18",
    )

    bootstrap_cfg = read_json(
        f["exp024_bootstrap_config"]
    )

    bootstrap_summary = read_json(
        f["exp024_bootstrap_summary"]
    )

    require(
        ROOT / bootstrap_cfg["input_csv"]
        == f["exp024_oof_predictions"],
        "EXP024 bootstrap config input mismatch",
    )

    require(
        sha256(f["exp024_oof_predictions"])
        == bootstrap_summary["input_sha256"],
        "EXP024 bootstrap input hash mismatch",
    )

    cfg29 = read_json(
        f["exp029_config"]
    )

    sum29 = read_json(
        f["exp029_summary"]
    )

    require(
        ROOT / cfg29["input_csv"]
        == f["exp024_utility_csv"],
        "EXP029 input path mismatch",
    )

    require(
        sha256(f["exp024_utility_csv"])
        == sum29["input_sha256"],
        "EXP029 input hash mismatch",
    )

    require(
        sum29["videos"] == len(utility_ids),
        "EXP029 video count mismatch",
    )

    cfg31 = read_json(
        f["exp031_config"]
    )

    sum31 = read_json(
        f["exp031_summary"]
    )

    require(
        ROOT / cfg31["input_csv"]
        == f["exp024_utility_csv"],
        "EXP031 input path mismatch",
    )

    require(
        sha256(f["exp024_utility_csv"])
        == sum31["input_sha256"],
        "EXP031 input hash mismatch",
    )

    require(
        sum31["videos"] == len(utility_ids),
        "EXP031 video count mismatch",
    )

    require(
        utility_ids <= boundary,
        "train18 contains video outside exclusion boundary",
    )

    require(
        oof_ids <= boundary,
        "EXP024 OOF contains video outside exclusion boundary",
    )

    p2 = exclusions.get(
        "historical_p2_reference"
    )

    require(
        isinstance(p2, dict),
        "Missing historical P2 record",
    )

    require(
        p2.get("status")
        == "UNRESOLVED_UNGROUNDED_REFERENCE",
        "Historical P2 status unexpectedly changed",
    )

    file_hashes = {
        rel: sha256(ROOT / rel)
        for rel in source_paths
    }

    out_dir = ROOT / cfg["output_dir"]

    require(
        not out_dir.exists(),
        "STOP: output directory exists; do not overwrite",
    )

    out_dir.mkdir(parents=True)

    provenance = {
        "experiment": "EXP043",
        "status": "PROVENANCE_CLOSURE_PASS",
        "train18_video_count": len(utility_ids),
        "train18_video_ids": sorted(utility_ids),
        "train18_outside_locked_boundary": sorted(
            utility_ids - boundary
        ),
        "chains": {
            "EXP024_utility_prepare": (
                "EXP023_train18_cache/primitives.csv -> "
                "EXP024_utility_prepare/train18_utility.csv"
            ),
            "EXP024_cluster_bootstrap": (
                "EXP024_utility_prepare/train18_utility.csv -> "
                "EXP024_model_utility/oof_predictions.csv -> "
                "EXP024_cluster_bootstrap"
            ),
            "EXP029_b2core_train": (
                "EXP024_utility_prepare/train18_utility.csv -> "
                "EXP029_b2core_train"
            ),
            "EXP031_b3_train": (
                "EXP024_utility_prepare/train18_utility.csv -> "
                "EXP031_b3_train"
            )
        },
        "source_hashes": file_hashes,
        "new_video_ids_supported": [],
    }

    provenance_path = (
        out_dir
        / "provenance_closure.json"
    )

    write_json(
        provenance_path,
        provenance,
    )

    locked = {
        "experiment": "EXP043",
        "version": 1,
        "status": (
            "KNOWN_DEVELOPMENT_EXPOSURE_LOCKED_"
            "WITH_DOCUMENTED_P2_LIMITATION"
        ),
        "development_exposure_lock_complete": True,
        "known_excluded_video_count": len(boundary),
        "video_ids": sorted(boundary),
        "records": exclusions["records"],
        "counts": exclusions["counts"],
        "policy": exclusions["policy"],
        "historical_p2_reference": p2,
        "p2_handling": cfg["p2_handling"],
        "claim_boundary": cfg["claim_boundary"],
        "provenance_closure": {
            "status": "PASS",
            "train18_video_count": len(
                utility_ids
            ),
            "new_video_ids_supported": [],
        },
        "di_v1_defined": False,
        "final_split_constructed": False,
        "fresh_final_dev_touched": False,
        "test_touched": False,
    }

    locked_path = (
        out_dir
        / "development_exclusions_locked.json"
    )

    write_json(
        locked_path,
        locked,
    )

    summary = {
        "experiment": "EXP043",
        "status": (
            "DEVELOPMENT_EXPOSURE_LOCK_COMPLETE"
        ),
        "repo_commit_at_execution": git(
            "rev-parse",
            "HEAD",
        ),
        "source_result_commit": source_commit,
        "config_sha256": sha256(cfg_path),
        "script_sha256": sha256(
            ROOT / SCRIPT_REL
        ),
        "known_excluded_video_count": len(
            boundary
        ),
        "train18_video_count": len(
            utility_ids
        ),
        "new_video_ids_from_provenance_closure": 0,
        "p2_status": p2["status"],
        "development_exposure_lock_complete": True,
        "di_v1_defined": False,
        "final_split_constructed": False,
        "fresh_final_dev_touched": False,
        "test_touched": False,
        "output_hashes": {
            "development_exclusions_locked.json":
                sha256(locked_path),
            "provenance_closure.json":
                sha256(provenance_path),
        },
    }

    summary_path = out_dir / "summary.json"

    write_json(
        summary_path,
        summary,
    )

    print("EXP043_LOCK_BUILD=PASS")
    print(
        "KNOWN_EXCLUDED_VIDEO_COUNT="
        + str(len(boundary))
    )
    print(
        "TRAIN18_VIDEO_COUNT="
        + str(len(utility_ids))
    )
    print(
        "NEW_VIDEO_IDS_FROM_CLOSURE=0"
    )
    print(
        "TRAIN18_OUTSIDE_LOCKED_BOUNDARY=[]"
    )
    print(
        "P2_STATUS="
        + p2["status"]
    )
    print(
        "DEVELOPMENT_EXPOSURE_LOCK_COMPLETE=1"
    )
    print("DI_V1_DEFINED=0")
    print("FINAL_SPLIT_CONSTRUCTED=0")
    print("FRESH_FINAL_DEV_TOUCHED=0")
    print("TEST_TOUCHED=0")

    for path in (
        locked_path,
        provenance_path,
        summary_path,
    ):
        print(
            "SHA256={} {}".format(
                sha256(path),
                path.relative_to(ROOT),
            )
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            "EXP043_STOP={}".format(exc),
            file=sys.stderr,
        )
        raise
