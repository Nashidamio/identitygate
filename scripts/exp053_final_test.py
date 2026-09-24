import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP053-final-test-v1.json"
OUT = ROOT / "experiments/EXP053_final_test"
STAGE_OUT = ROOT.parent / ".EXP053_final_test_staging"

REQUIRED_TRACKED = [
    "configs/EXP053-final-test-v1.json",
    "docs/AMENDMENT_A12_FINAL_TEST_EXECUTION_FREEZE.md",
    "PREREGISTRATION.md",
    "scripts/exp053_final_test.py",
    "tests/test_exp053_final_test.py",
]

HEADLINE_LABELS = [
    "B0",
    "B1",
    "B1_NEUTRAL",
    "B2",
    "B2_NEUTRAL",
    "B3_S",
    "B3_S_NEUTRAL",
    "B3_R",
    "B3_R_NEUTRAL",
    "B5",
    "B5_NEUTRAL",
]


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv_atomic(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load module: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_frozen_clean_tree(
    allow_final_output=False,
):
    for rel in REQUIRED_TRACKED:
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

    tracked_dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
        ],
        check=False,
    ).returncode != 0

    staged_dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--cached",
            "--quiet",
        ],
        check=False,
    ).returncode != 0

    if tracked_dirty or staged_dirty:
        raise RuntimeError(
            "EXP053 requires no tracked or staged changes"
        )

    untracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
        ],
        text=True,
    ).splitlines()

    output_prefix = (
        str(OUT.relative_to(ROOT)) + "/"
    )

    unexpected = []

    for rel in untracked:
        allowed = (
            allow_final_output
            and rel.startswith(output_prefix)
        )

        if not allowed:
            unexpected.append(rel)

    if unexpected:
        raise RuntimeError(
            "EXP053 unexpected untracked files: "
            + ", ".join(unexpected)
        )

    head = git_head()

    origin = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "origin/main",
        ],
        text=True,
    ).strip()

    if head != origin:
        raise RuntimeError(
            "EXP053 requires HEAD == origin/main"
        )


def verify_dependencies(cfg):
    for name, item in cfg["source_dependencies"].items():
        actual = sha256sum(ROOT / item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(
                "{} SHA mismatch: {} != {}".format(
                    name,
                    actual,
                    item["sha256"],
                )
            )


def load_cfg():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    verify_dependencies(cfg)

    if cfg["experiment"] != "EXP053":
        raise RuntimeError("Wrong experiment config")

    boundary = cfg["boundary"]
    if boundary["test_touched_before_execution"] is not False:
        raise RuntimeError("TEST untouched boundary violated")
    if boundary["test_threshold_search_allowed"] is not False:
        raise RuntimeError("TEST threshold search must be disabled")
    if boundary["test_retuning_allowed"] is not False:
        raise RuntimeError("TEST retuning must be disabled")

    if int(cfg["expected_supported_curve_points"]) != 19:
        raise RuntimeError("Supported curve-point count changed")
    if int(cfg["expected_unsupported_curve_points"]) != 26:
        raise RuntimeError("Unsupported curve-point count changed")

    return cfg


def load_runtime(cfg):
    exp051 = load_module(
        "exp051_for_exp053",
        ROOT / cfg["source_dependencies"]["exp051_script"]["path"],
    )
    exp050, runtime = exp051.load_runtime(cfg)
    return exp051, exp050, runtime


def event_key(row):
    return (
        row["video"],
        int(row["object_id"]),
        int(row["reappear_frame"]),
        int(row["disappear_start"]),
    )


def event_key_hash(rows):
    keys = sorted(event_key(row) for row in rows)
    payload = json.dumps(keys, separators=(",", ":"))
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def load_scope(cfg, exp050, scope_label):
    spec = cfg["scopes"][scope_label]
    manifest_path = ROOT / spec["manifest_csv"]

    if sha256sum(manifest_path) != spec["manifest_csv_sha256"]:
        raise RuntimeError(
            "{} manifest SHA mismatch".format(scope_label)
        )

    manifest_rows = read_csv(manifest_path)

    if len(manifest_rows) != int(spec["expected_videos"]):
        raise RuntimeError(
            "{} video-count mismatch".format(scope_label)
        )

    videos = [row["video"] for row in manifest_rows]

    if videos != sorted(videos):
        raise RuntimeError(
            "{} manifest is not video-sorted".format(scope_label)
        )

    if len(set(videos)) != len(videos):
        raise RuntimeError(
            "{} duplicate video IDs".format(scope_label)
        )

    if any(
        row["partition"] != spec["partition"]
        for row in manifest_rows
    ):
        raise RuntimeError(
            "{} partition mismatch".format(scope_label)
        )

    manifest_events = sum(
        int(row["primary_event_count"])
        for row in manifest_rows
    )

    if manifest_events != int(spec["expected_primary_events"]):
        raise RuntimeError(
            "{} manifest event-count mismatch".format(scope_label)
        )

    membership = exp050.membership_hash(
        spec["membership_hash_label"],
        videos,
    )

    if membership != spec["membership_sha256"]:
        raise RuntimeError(
            "{} membership hash mismatch".format(scope_label)
        )

    event_pool_path = ROOT / cfg["scope"]["event_pool_csv"]

    if (
        sha256sum(event_pool_path)
        != cfg["scope"]["event_pool_csv_sha256"]
    ):
        raise RuntimeError("Event-pool SHA mismatch")

    video_set = set(videos)
    pool_rows = read_csv(event_pool_path)

    primary = [
        row
        for row in pool_rows
        if row["video"] in video_set
        and int(row["primary_pool_eligible"]) == 1
    ]

    if len(primary) != int(spec["expected_primary_events"]):
        raise RuntimeError(
            "{} primary-event count mismatch".format(scope_label)
        )

    expected_by_video = {
        row["video"]: int(row["primary_event_count"])
        for row in manifest_rows
    }

    actual_by_video = defaultdict(int)
    keys = set()

    for row in primary:
        key = event_key(row)

        if key in keys:
            raise RuntimeError(
                "{} duplicate primary event key: {}".format(
                    scope_label,
                    key,
                )
            )

        keys.add(key)
        actual_by_video[row["video"]] += 1

    if dict(actual_by_video) != expected_by_video:
        raise RuntimeError(
            "{} per-video event counts mismatch".format(
                scope_label
            )
        )

    primary_by_video = defaultdict(list)

    for row in primary:
        primary_by_video[row["video"]].append(row)

    for video in primary_by_video:
        primary_by_video[video].sort(
            key=lambda row: (
                int(row["object_id"]),
                int(row["reappear_frame"]),
                int(row["disappear_start"]),
            )
        )

    return {
        "scope_label": scope_label,
        "spec": spec,
        "manifest_rows": manifest_rows,
        "videos": videos,
        "primary_by_video": primary_by_video,
        "event_keys": keys,
    }


def load_test_scopes(cfg, exp050):
    hard = load_scope(cfg, exp050, "HARD_TEST80")
    representative = load_scope(
        cfg,
        exp050,
        "REPRESENTATIVE_TEST40",
    )

    overlap = (
        set(hard["videos"])
        & set(representative["videos"])
    )

    if overlap:
        raise RuntimeError(
            "Hard/representative TEST overlap: {}".format(
                sorted(overlap)
            )
        )

    return {
        "HARD_TEST80": hard,
        "REPRESENTATIVE_TEST40": representative,
    }


def curve_code(target_rate):
    return "R{:03d}".format(
        int(round(float(target_rate) * 100))
    )


def curve_run_labels(target_rate, variant):
    if abs(float(target_rate) - 0.3) < 1e-12:
        return variant, variant + "_NEUTRAL"

    base = "CURVE_{}_{}".format(
        curve_code(target_rate),
        variant,
    )

    return base, base + "_NEUTRAL"


def build_specs(cfg, scope_label):
    specs = [
        {
            "run_label": "B0",
            "kind": "B0",
            "source_variant": "B0",
            "tracker_variant": "B0",
            "tau": None,
            "target_rate": None,
            "neutral_source_run_label": None,
        }
    ]

    for variant in cfg["variants"]:
        tau = float(cfg["headline"]["taus"][variant])

        specs.append(
            {
                "run_label": variant,
                "kind": "HEADLINE_SIGNAL",
                "source_variant": variant,
                "tracker_variant": variant,
                "tau": tau,
                "target_rate": float(
                    cfg["headline"]["r_star"]
                ),
                "neutral_source_run_label": None,
            }
        )

        specs.append(
            {
                "run_label": variant + "_NEUTRAL",
                "kind": "HEADLINE_NEUTRAL",
                "source_variant": variant,
                "tracker_variant": "B0",
                "tau": None,
                "target_rate": float(
                    cfg["headline"]["r_star"]
                ),
                "neutral_source_run_label": variant,
            }
        )

    if scope_label == "HARD_TEST80":
        extras = [
            point
            for point in cfg["supported_curve_points"]
            if abs(
                float(point["target_rate"]) - 0.3
            ) >= 1e-12
        ]

        order = {
            variant: index
            for index, variant in enumerate(cfg["variants"])
        }

        extras.sort(
            key=lambda point: (
                float(point["target_rate"]),
                order[point["variant"]],
            )
        )

        for point in extras:
            target = float(point["target_rate"])
            variant = point["variant"]

            signal_label, neutral_label = (
                curve_run_labels(target, variant)
            )

            specs.append(
                {
                    "run_label": signal_label,
                    "kind": "CURVE_SIGNAL",
                    "source_variant": variant,
                    "tracker_variant": variant,
                    "tau": float(point["tau"]),
                    "target_rate": target,
                    "neutral_source_run_label": None,
                }
            )

            specs.append(
                {
                    "run_label": neutral_label,
                    "kind": "CURVE_NEUTRAL",
                    "source_variant": variant,
                    "tracker_variant": "B0",
                    "tau": None,
                    "target_rate": target,
                    "neutral_source_run_label":
                        signal_label,
                }
            )

    labels = [spec["run_label"] for spec in specs]

    if len(labels) != len(set(labels)):
        raise RuntimeError(
            "{} duplicate run labels".format(scope_label)
        )

    if scope_label == "HARD_TEST80":
        expected = int(
            cfg["reproducibility"][
                "expected_hard_trajectories_per_video"
            ]
        )
    else:
        expected = int(
            cfg["reproducibility"][
                "expected_representative_trajectories_per_video"
            ]
        )

    if len(specs) != expected:
        raise RuntimeError(
            "{} spec count {} != {}".format(
                scope_label,
                len(specs),
                expected,
            )
        )

    return specs


def cache_path(
    cfg,
    scope_label,
    run_label,
    video,
    sanity=False,
):
    base = Path(cfg["runtime"]["scratch_dir"])

    if sanity:
        base = Path(str(base) + "_sanity")

    return (
        base
        / scope_label
        / run_label
        / (video + ".json")
    )


def validate_cached(row, expected, primary_rows):
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(
                "Stale EXP053 cache field {}: {} != {}".format(
                    key,
                    row.get(key),
                    value,
                )
            )

    eligible = int(
        row["eligible_nonconditioning_frames"]
    )
    admit = int(row["admit_count"])
    block = int(row["block_count"])

    if eligible <= 0:
        raise RuntimeError(
            "Invalid cached eligible-frame count"
        )

    if not 0 <= admit <= eligible:
        raise RuntimeError(
            "Invalid cached admit count"
        )

    if admit + block != eligible:
        raise RuntimeError(
            "Cached write accounting mismatch"
        )

    rate = float(row["write_rate"])

    if not math.isfinite(rate):
        raise RuntimeError(
            "Cached write rate is not finite"
        )

    if abs(rate - admit / eligible) > 1e-12:
        raise RuntimeError(
            "Cached write-rate mismatch"
        )

    event_rows = row.get("event_rows")

    if not isinstance(event_rows, list):
        raise RuntimeError("Cached event_rows missing")

    if len(event_rows) != len(primary_rows):
        raise RuntimeError(
            "Cached event denominator mismatch"
        )

    expected_keys = {
        event_key(item)
        for item in primary_rows
    }

    actual_keys = {
        event_key(item)
        for item in event_rows
    }

    if len(actual_keys) != len(event_rows):
        raise RuntimeError(
            "Duplicate cached event rows"
        )

    if actual_keys != expected_keys:
        raise RuntimeError(
            "Cached event-key set mismatch"
        )

    for event in event_rows:
        if event["run_label"] != expected["run_label"]:
            raise RuntimeError(
                "Cached event run-label mismatch"
            )

        if int(event["recovered_w30"]) not in (0, 1):
            raise RuntimeError(
                "Cached POR endpoint is not binary"
            )

        if int(event["theft_event_w30"]) not in (0, 1):
            raise RuntimeError(
                "Cached ITR endpoint is not binary"
            )

    for key in (
        "por30",
        "itr30",
        "peak_vram_gb",
        "runtime_sec",
    ):
        if not math.isfinite(float(row[key])):
            raise RuntimeError(
                "Cached {} is not finite".format(key)
            )

    if (
        row["test_touched"]
        != expected["test_touched"]
    ):
        raise RuntimeError(
            "Cached test_touched mismatch"
        )

    return row


def run_or_load(
    cfg,
    exp051,
    exp050,
    runtime,
    predictor,
    scope_label,
    video,
    primary_rows,
    spec,
    neutral_k=None,
    test_touched=True,
    sanity=False,
):
    expected = {
        "experiment": "EXP053",
        "repo_commit": git_head(),
        "config_sha256": sha256sum(CFG_PATH),
        "script_sha256": sha256sum(Path(__file__)),
        "scope_label": scope_label,
        "scope_manifest_sha256": (
            "TRAIN_EXPOSED_SANITY"
            if sanity
            else cfg["scopes"][scope_label][
                "manifest_csv_sha256"
            ]
        ),
        "run_label": spec["run_label"],
        "video": video,
        "kind": spec["kind"],
        "source_variant": spec["source_variant"],
        "tracker_variant": spec["tracker_variant"],
        "tau": spec["tau"],
        "target_rate": spec["target_rate"],
        "neutral_source_run_label":
            spec["neutral_source_run_label"],
        "neutral_source_k": neutral_k,
        "primary_event_keys_sha256":
            event_key_hash(primary_rows),
        "test_touched": bool(test_touched),
    }

    path = cache_path(
        cfg,
        scope_label,
        spec["run_label"],
        video,
        sanity=sanity,
    )

    if path.exists():
        cached = json.loads(
            path.read_text(encoding="utf-8")
        )
        return (
            validate_cached(
                cached,
                expected,
                primary_rows,
            ),
            True,
        )

    ctx, gt_cache, visibility = exp051.context_gt(
        runtime,
        exp050,
        cfg,
        video,
    )

    if spec["tau"] is None:
        effective_tau = float(
            cfg["headline"]["taus"]["B2"]
        )
    else:
        effective_tau = float(spec["tau"])

    run_cfg = exp051.make_run_cfg(
        runtime,
        exp050,
        cfg,
        ctx,
        effective_tau,
    )

    if neutral_k is None:
        tracked = runtime["exp033"].run_tracker(
            predictor=predictor,
            jpg_dir=ctx["jpg_dir"],
            gt0=ctx["gt0"],
            object_ids=ctx["object_ids"],
            cfg=run_cfg,
            variant=spec["tracker_variant"],
            packs=runtime["packs"],
            deps=runtime["deps"],
        )

        eligible = int(ctx["n_frames"]) - 1

        if spec["tracker_variant"] == "B0":
            admit_count = eligible
            block_count = 0
        else:
            decisions = tracked["decisions"]

            if len(decisions) != eligible:
                raise RuntimeError(
                    "Signal decision-count mismatch"
                )

            admit_count = sum(
                row["action"] == "ADMIT"
                for row in decisions
            )

            block_count = sum(
                row["action"] == "BLOCK"
                for row in decisions
            )

            if admit_count + block_count != eligible:
                raise RuntimeError(
                    "Signal action-count mismatch"
                )

            if not runtime[
                "exp037"
            ].block_integrity_pass(
                decisions,
                len(ctx["object_ids"]),
            ):
                raise RuntimeError(
                    "Signal block-integrity failure"
                )

    else:
        eligible = int(ctx["n_frames"]) - 1

        admit_frames = exp051.neutral_admit_frames(
            eligible,
            int(neutral_k),
        )

        tracked = exp051.run_neutral(
            predictor=predictor,
            jpg_dir=ctx["jpg_dir"],
            gt0=ctx["gt0"],
            object_ids=ctx["object_ids"],
            run_cfg=run_cfg,
            admit_frames=admit_frames,
            runtime=runtime,
        )

        admit_count = int(neutral_k)
        block_count = eligible - admit_count

    event_rows = exp051.score_primary_events(
        run_label=spec["run_label"],
        tau=spec["tau"],
        predictions=tracked["predictions"],
        gt_cache=gt_cache,
        visibility=visibility,
        object_ids=ctx["object_ids"],
        primary_rows=primary_rows,
        endpoint=cfg["endpoint"],
    )

    if len(event_rows) != len(primary_rows):
        raise RuntimeError(
            "Primary event denominator mismatch"
        )

    result = {
        **expected,
        "n_frames": int(ctx["n_frames"]),
        "eligible_nonconditioning_frames":
            eligible,
        "admit_count": int(admit_count),
        "block_count": int(block_count),
        "write_rate":
            float(admit_count / eligible),
        "event_count": len(event_rows),
        "por30": float(
            sum(
                int(row["recovered_w30"])
                for row in event_rows
            )
            / len(event_rows)
        ),
        "itr30": float(
            sum(
                int(row["theft_event_w30"])
                for row in event_rows
            )
            / len(event_rows)
        ),
        "peak_vram_gb":
            float(tracked["peak_vram_gb"]),
        "runtime_sec":
            float(tracked["runtime_sec"]),
        "event_rows": event_rows,
    }

    validate_cached(
        result,
        expected,
        primary_rows,
    )

    write_json_atomic(path, result)

    del tracked
    gc.collect()
    torch.cuda.empty_cache()

    return result, False


def campaign_identity(cfg):
    return {
        "experiment": "EXP053",
        "repo_commit": git_head(),
        "config_sha256": sha256sum(CFG_PATH),
        "script_sha256":
            sha256sum(Path(__file__)),
        "hard_test_manifest_sha256":
            cfg["scopes"]["HARD_TEST80"][
                "manifest_csv_sha256"
            ],
        "hard_test_membership_sha256":
            cfg["scopes"]["HARD_TEST80"][
                "membership_sha256"
            ],
        "representative_test_manifest_sha256":
            cfg["scopes"]["REPRESENTATIVE_TEST40"][
                "manifest_csv_sha256"
            ],
        "representative_test_membership_sha256":
            cfg["scopes"]["REPRESENTATIVE_TEST40"][
                "membership_sha256"
            ],
    }


def start_or_resume_campaign(cfg, mode):
    marker = Path(cfg["runtime"]["touch_marker"])
    identity = campaign_identity(cfg)

    if mode == "run":
        if marker.exists():
            raise RuntimeError(
                "TEST campaign marker already exists; "
                "use resume only with the same frozen "
                "commit/config"
            )

        scratch = Path(
            cfg["runtime"]["scratch_dir"]
        )

        if scratch.exists():
            stale = list(
                scratch.rglob("*.json")
            )

            if stale:
                raise RuntimeError(
                    "TEST scratch contains cache files "
                    "without a campaign marker"
                )

        marker.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            **identity,
            "status": "RUNNING",
            "test_touched": True,
            "initial_mode": "run",
        }

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
        )

        fd = os.open(
            marker,
            flags,
            0o644,
        )

        try:
            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        payload,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
        except Exception:
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
            raise

        return payload

    if not marker.exists():
        raise RuntimeError(
            "resume requires an existing "
            "TEST campaign marker"
        )

    payload = json.loads(
        marker.read_text(encoding="utf-8")
    )

    for key, value in identity.items():
        if payload.get(key) != value:
            raise RuntimeError(
                "TEST campaign identity mismatch "
                "for {}".format(key)
            )

    if payload.get("status") != "RUNNING":
        raise RuntimeError(
            "TEST campaign is not resumable: "
            "status={}".format(
                payload.get("status")
            )
        )

    if payload.get("test_touched") is not True:
        raise RuntimeError(
            "TEST marker test_touched mismatch"
        )

    return payload


def classify_finalization_state(
    mode,
    final_exists,
    staging_exists,
):
    if mode not in ("run", "resume"):
        raise ValueError(
            "Invalid finalization mode"
        )

    if mode == "run":
        if final_exists:
            raise RuntimeError(
                "Final TEST output already exists"
            )

        if staging_exists:
            raise RuntimeError(
                "Staging output exists before initial TEST run"
            )

        return "EXECUTE"

    if final_exists and staging_exists:
        raise RuntimeError(
            "Both final and staging outputs exist"
        )

    if final_exists:
        return "RECOVER_FINAL_OUTPUT"

    if staging_exists:
        return "REBUILD_STAGING"

    return "EXECUTE"


def finish_campaign(cfg):
    validate_output_dir(
        cfg,
        OUT,
    )

    marker = Path(
        cfg["runtime"]["touch_marker"]
    )

    payload = json.loads(
        marker.read_text(encoding="utf-8")
    )

    if payload.get("status") != "RUNNING":
        raise RuntimeError(
            "Cannot complete non-running "
            "TEST campaign"
        )

    payload["status"] = "COMPLETE"
    payload["output_dir"] = (
        cfg["runtime"]["output_dir"]
    )
    payload[
        "artifact_manifest_sha256"
    ] = sha256sum(
        OUT / "artifact_manifest.json"
    )

    write_json_atomic(
        marker,
        payload,
    )


def aggregate_label(
    rows,
    expected_videos,
    expected_events,
):
    if len(rows) != expected_videos:
        raise RuntimeError(
            "Trajectory video-count mismatch"
        )

    events = [
        event
        for row in rows
        for event in row["event_rows"]
    ]

    if len(events) != expected_events:
        raise RuntimeError(
            "Trajectory event-count mismatch"
        )

    sum_k = sum(
        int(row["admit_count"])
        for row in rows
    )

    sum_n = sum(
        int(
            row[
                "eligible_nonconditioning_frames"
            ]
        )
        for row in rows
    )

    return {
        "run_label": rows[0]["run_label"],
        "kind": rows[0]["kind"],
        "source_variant":
            rows[0]["source_variant"],
        "tau": rows[0]["tau"],
        "target_rate":
            rows[0]["target_rate"],
        "videos": len(rows),
        "events": len(events),
        "por30": float(
            sum(
                int(event["recovered_w30"])
                for event in events
            )
            / len(events)
        ),
        "itr30": float(
            sum(
                int(event["theft_event_w30"])
                for event in events
            )
            / len(events)
        ),
        "sum_admit_count": int(sum_k),
        "sum_eligible_count": int(sum_n),
        "pooled_write_rate":
            float(sum_k / sum_n),
        "max_peak_vram_gb": float(
            max(
                float(row["peak_vram_gb"])
                for row in rows
            )
        ),
        "runtime_sec": float(
            sum(
                float(row["runtime_sec"])
                for row in rows
            )
        ),
    }


def aggregate_scope(
    cfg,
    scope_label,
    run_rows,
):
    scope_cfg = cfg["scopes"][scope_label]
    specs = build_specs(
        cfg,
        scope_label,
    )

    labels = [
        spec["run_label"]
        for spec in specs
    ]

    by_label_rows = {
        label: [
            row
            for row in run_rows
            if row["run_label"] == label
        ]
        for label in labels
    }

    by_label = {
        label: aggregate_label(
            by_label_rows[label],
            int(scope_cfg["expected_videos"]),
            int(
                scope_cfg[
                    "expected_primary_events"
                ]
            ),
        )
        for label in labels
    }

    rows_by_key = {
        (
            row["run_label"],
            row["video"],
        ): row
        for row in run_rows
    }

    if len(rows_by_key) != len(run_rows):
        raise RuntimeError(
            "Duplicate trajectory row"
        )

    for spec in specs:
        source_label = (
            spec[
                "neutral_source_run_label"
            ]
        )

        if source_label is None:
            continue

        neutral_rows = (
            by_label_rows[
                spec["run_label"]
            ]
        )

        for neutral in neutral_rows:
            video = neutral["video"]
            signal = rows_by_key[
                (source_label, video)
            ]

            if (
                int(neutral["admit_count"])
                != int(signal["admit_count"])
                or int(
                    neutral[
                        "eligible_nonconditioning_frames"
                    ]
                )
                != int(
                    signal[
                        "eligible_nonconditioning_frames"
                    ]
                )
            ):
                raise RuntimeError(
                    "{} exact per-video neutral "
                    "budget mismatch: {} {}".format(
                        scope_label,
                        spec["run_label"],
                        video,
                    )
                )

    return by_label


def curve_rows_for_scope(
    cfg,
    scope_label,
    by_label,
):
    if scope_label != "HARD_TEST80":
        return []

    rows = []

    for point in cfg["supported_curve_points"]:
        target = float(point["target_rate"])
        variant = point["variant"]

        signal_label, neutral_label = (
            curve_run_labels(
                target,
                variant,
            )
        )

        signal = by_label[signal_label]
        neutral = by_label[neutral_label]

        rows.append(
            {
                "scope": scope_label,
                "target_rate": target,
                "variant": variant,
                "tau": float(point["tau"]),
                "dev_realized_write_rate":
                    float(
                        point[
                            "dev_realized_write_rate"
                        ]
                    ),
                "test_realized_write_rate":
                    signal[
                        "pooled_write_rate"
                    ],
                "test_absolute_target_error":
                    abs(
                        signal[
                            "pooled_write_rate"
                        ]
                        - target
                    ),
                "signal_por30":
                    signal["por30"],
                "signal_itr30":
                    signal["itr30"],
                "neutral_por30":
                    neutral["por30"],
                "neutral_itr30":
                    neutral["itr30"],
                "neutral_realized_write_rate":
                    neutral[
                        "pooled_write_rate"
                    ],
                "signal_neutral_exact_budget_match":
                    True,
            }
        )

    if len(rows) != 19:
        raise RuntimeError(
            "Hard TEST curve row-count mismatch"
        )

    return rows


def rate_difference_exceeds_tolerance(
    sum_k_a,
    sum_n_a,
    sum_k_b,
    sum_n_b,
    tolerance,
):
    k_a = int(sum_k_a)
    n_a = int(sum_n_a)
    k_b = int(sum_k_b)
    n_b = int(sum_n_b)

    if n_a <= 0 or n_b <= 0:
        raise ValueError(
            "Write-rate denominators must be positive"
        )

    if not 0 <= k_a <= n_a:
        raise ValueError(
            "Invalid write count for method A"
        )

    if not 0 <= k_b <= n_b:
        raise ValueError(
            "Invalid write count for method B"
        )

    tol = Fraction(str(tolerance))

    if tol < 0:
        raise ValueError(
            "Write-rate tolerance must be non-negative"
        )

    difference_numerator = abs(
        k_a * n_b - k_b * n_a
    )
    difference_denominator = n_a * n_b

    return (
        difference_numerator
        * tol.denominator
        >
        tol.numerator
        * difference_denominator
    )


def rate_status_rows(
    cfg,
    scope_label,
    by_label,
):
    tolerance = float(
        cfg["rate_mismatch"][
            "absolute_pooled_rate_tolerance"
        ]
    )

    rows = []

    for comparison in (
        cfg[
            "headline_direct_gated_comparisons"
        ]
    ):
        a = by_label[comparison["a"]]
        b = by_label[comparison["b"]]

        k_a = int(a["sum_admit_count"])
        n_a = int(a["sum_eligible_count"])
        k_b = int(b["sum_admit_count"])
        n_b = int(b["sum_eligible_count"])

        rate_a = float(a["pooled_write_rate"])
        rate_b = float(b["pooled_write_rate"])

        expected_rate_a = k_a / n_a
        expected_rate_b = k_b / n_b

        if abs(
            rate_a - expected_rate_a
        ) > 1e-12:
            raise RuntimeError(
                "Method A pooled write-rate "
                "accounting mismatch"
            )

        if abs(
            rate_b - expected_rate_b
        ) > 1e-12:
            raise RuntimeError(
                "Method B pooled write-rate "
                "accounting mismatch"
            )

        difference = abs(
            expected_rate_a - expected_rate_b
        )

        exceeds = (
            rate_difference_exceeds_tolerance(
                k_a,
                n_a,
                k_b,
                n_b,
                tolerance,
            )
        )

        if exceeds:
            status = (
                cfg["rate_mismatch"][
                    "status_if_exceeded"
                ]
            )
        else:
            status = (
                cfg["rate_mismatch"][
                    "status_if_within"
                ]
            )

        rows.append(
            {
                "scope": scope_label,
                "group":
                    comparison["group"],
                "comparison":
                    comparison["name"],
                "a": comparison["a"],
                "b": comparison["b"],
                "rate_a": rate_a,
                "rate_b": rate_b,
                "absolute_rate_difference":
                    difference,
                "tolerance": tolerance,
                "rate_status": status,
            }
        )

    return rows


def write_outputs(
    cfg,
    all_rows,
    scope_summaries,
    curve_rows,
    rate_rows,
    stats,
    target_dir,
):
    target_dir = Path(target_dir)

    if target_dir.exists():
        raise RuntimeError(
            "STOP: EXP053 target output directory already exists"
        )

    event_rows = []
    trajectory_rows = []
    headline_rows = []

    for scope_label, rows in all_rows.items():
        for row in rows:
            trajectory_rows.append(
                {
                    "scope": scope_label,
                    "run_label":
                        row["run_label"],
                    "kind": row["kind"],
                    "source_variant":
                        row["source_variant"],
                    "tracker_variant":
                        row["tracker_variant"],
                    "tau": (
                        ""
                        if row["tau"] is None
                        else row["tau"]
                    ),
                    "target_rate": (
                        ""
                        if row["target_rate"]
                        is None
                        else row["target_rate"]
                    ),
                    "video": row["video"],
                    "eligible_nonconditioning_frames":
                        row[
                            "eligible_nonconditioning_frames"
                        ],
                    "admit_count":
                        row["admit_count"],
                    "block_count":
                        row["block_count"],
                    "write_rate":
                        row["write_rate"],
                    "event_count":
                        row["event_count"],
                    "por30": row["por30"],
                    "itr30": row["itr30"],
                    "peak_vram_gb":
                        row["peak_vram_gb"],
                    "runtime_sec":
                        row["runtime_sec"],
                }
            )

            for event in row["event_rows"]:
                event_rows.append(
                    {
                        "scope": scope_label,
                        **event,
                    }
                )

        for label in HEADLINE_LABELS:
            headline_rows.append(
                {
                    "scope": scope_label,
                    **scope_summaries[
                        scope_label
                    ][label],
                }
            )

    write_csv_atomic(
        target_dir / "event_outcomes.csv",
        [
            "scope",
            "run_label",
            "tau",
            "video",
            "object_id",
            "reappear_frame",
            "disappear_start",
            "recovered_w30",
            "first_recovery_frame",
            "evaluable_visible_frames",
            "theft_event_w30",
            "first_theft_episode_start",
            "max_consecutive_theft_frames",
        ],
        event_rows,
    )

    write_csv_atomic(
        target_dir / "trajectory_summary.csv",
        [
            "scope",
            "run_label",
            "kind",
            "source_variant",
            "tracker_variant",
            "tau",
            "target_rate",
            "video",
            "eligible_nonconditioning_frames",
            "admit_count",
            "block_count",
            "write_rate",
            "event_count",
            "por30",
            "itr30",
            "peak_vram_gb",
            "runtime_sec",
        ],
        trajectory_rows,
    )

    write_csv_atomic(
        target_dir / "headline_summary.csv",
        [
            "scope",
            "run_label",
            "kind",
            "source_variant",
            "tau",
            "target_rate",
            "videos",
            "events",
            "por30",
            "itr30",
            "sum_admit_count",
            "sum_eligible_count",
            "pooled_write_rate",
            "max_peak_vram_gb",
            "runtime_sec",
        ],
        headline_rows,
    )

    write_csv_atomic(
        target_dir / "curve_summary.csv",
        [
            "scope",
            "target_rate",
            "variant",
            "tau",
            "dev_realized_write_rate",
            "test_realized_write_rate",
            "test_absolute_target_error",
            "signal_por30",
            "signal_itr30",
            "neutral_por30",
            "neutral_itr30",
            "neutral_realized_write_rate",
            "signal_neutral_exact_budget_match",
        ],
        curve_rows,
    )

    write_csv_atomic(
        target_dir / "headline_rate_status.csv",
        [
            "scope",
            "group",
            "comparison",
            "a",
            "b",
            "rate_a",
            "rate_b",
            "absolute_rate_difference",
            "tolerance",
            "rate_status",
        ],
        rate_rows,
    )

    total_trajectories = sum(
        len(rows)
        for rows in all_rows.values()
    )

    expected_total = int(
        cfg["reproducibility"][
            "expected_total_trajectories"
        ]
    )

    if total_trajectories != expected_total:
        raise RuntimeError(
            "Final trajectory count mismatch: "
            "{} != {}".format(
                total_trajectories,
                expected_total,
            )
        )

    summary = {
        "experiment": "EXP053",
        "status":
            "FINAL_TEST_CAMPAIGN_COMPLETE",
        "repo_commit": git_head(),
        "config_sha256":
            sha256sum(CFG_PATH),
        "script_sha256":
            sha256sum(Path(__file__)),
        "test_touched": True,
        "test_campaign_complete": True,
        "threshold_search_performed": False,
        "threshold_retuning_performed": False,
        "endpoint_roles": {
            "POR30":
                "SOLE_PRIMARY_ENDPOINT",
            "ITR30":
                "SECONDARY_ENDPOINT",
        },
        "primary_contrast":
            "B3_S_minus_B2",
        "r_star":
            cfg["headline"]["r_star"],
        "rate_mismatch_tolerance":
            cfg["rate_mismatch"][
                "absolute_pooled_rate_tolerance"
            ],
        "supported_curve_points": 19,
        "unsupported_curve_points": 26,
        "trajectory_count":
            total_trajectories,
        "expected_trajectory_count":
            expected_total,
        "scratch_cache_hits":
            stats["cache_hits"],
        "scratch_cache_misses":
            stats["cache_misses"],
        "max_peak_vram_gb": max(
            float(row["peak_vram_gb"])
            for rows in all_rows.values()
            for row in rows
        ),
        "scopes": {},
    }

    for scope_label in all_rows:
        summary["scopes"][
            scope_label
        ] = {
            "videos":
                cfg["scopes"][
                    scope_label
                ]["expected_videos"],
            "primary_events":
                cfg["scopes"][
                    scope_label
                ][
                    "expected_primary_events"
                ],
            "headline": {
                label:
                    scope_summaries[
                        scope_label
                    ][label]
                for label
                in HEADLINE_LABELS
            },
            "headline_rate_status": [
                row
                for row in rate_rows
                if row["scope"]
                == scope_label
            ],
        }

    write_json_atomic(
        target_dir / "summary.json",
        summary,
    )

    write_artifact_manifest(
        cfg,
        target_dir,
    )


OUTPUT_DATA_FILES = [
    "event_outcomes.csv",
    "trajectory_summary.csv",
    "headline_summary.csv",
    "curve_summary.csv",
    "headline_rate_status.csv",
    "summary.json",
]


def csv_data_row_count(path):
    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:
        return sum(
            1
            for _ in csv.DictReader(f)
        )


def write_artifact_manifest(
    cfg,
    out_dir,
):
    out_dir = Path(out_dir)

    files = {
        name: sha256sum(out_dir / name)
        for name in OUTPUT_DATA_FILES
    }

    manifest = {
        "experiment": "EXP053",
        "repo_commit": git_head(),
        "config_sha256":
            sha256sum(CFG_PATH),
        "script_sha256":
            sha256sum(Path(__file__)),
        "files": files,
    }

    write_json_atomic(
        out_dir / "artifact_manifest.json",
        manifest,
    )


def verify_artifact_manifest(
    cfg,
    out_dir,
):
    out_dir = Path(out_dir)

    manifest_path = (
        out_dir
        / "artifact_manifest.json"
    )

    if not manifest_path.is_file():
        raise RuntimeError(
            "EXP053 artifact manifest missing"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if manifest.get("experiment") != "EXP053":
        raise RuntimeError(
            "Artifact-manifest experiment mismatch"
        )

    if manifest.get("repo_commit") != git_head():
        raise RuntimeError(
            "Artifact-manifest commit mismatch"
        )

    if (
        manifest.get("config_sha256")
        != sha256sum(CFG_PATH)
    ):
        raise RuntimeError(
            "Artifact-manifest config mismatch"
        )

    if (
        manifest.get("script_sha256")
        != sha256sum(Path(__file__))
    ):
        raise RuntimeError(
            "Artifact-manifest script mismatch"
        )

    files = manifest.get("files")

    if set(files or {}) != set(
        OUTPUT_DATA_FILES
    ):
        raise RuntimeError(
            "Artifact-manifest file-set mismatch"
        )

    for name in OUTPUT_DATA_FILES:
        actual = sha256sum(
            out_dir / name
        )

        if actual != files[name]:
            raise RuntimeError(
                "Artifact hash mismatch: {}".format(
                    name
                )
            )

    return True


def validate_output_dir(
    cfg,
    out_dir,
):
    out_dir = Path(out_dir)

    if not out_dir.is_dir():
        raise RuntimeError(
            "EXP053 output directory missing"
        )

    expected_files = sorted(
        OUTPUT_DATA_FILES
        + ["artifact_manifest.json"]
    )

    actual_files = sorted(
        item.name
        for item in out_dir.iterdir()
        if item.is_file()
    )

    actual_dirs = sorted(
        item.name
        for item in out_dir.iterdir()
        if item.is_dir()
    )

    if actual_dirs:
        raise RuntimeError(
            "Unexpected output directories: "
            + ", ".join(actual_dirs)
        )

    if actual_files != expected_files:
        raise RuntimeError(
            "EXP053 output file-set mismatch"
        )

    verify_artifact_manifest(
        cfg,
        out_dir,
    )

    expected_trajectories = int(
        cfg["reproducibility"][
            "expected_total_trajectories"
        ]
    )

    expected_event_rows = (
        int(
            cfg["scopes"]["HARD_TEST80"][
                "expected_primary_events"
            ]
        )
        * int(
            cfg["reproducibility"][
                "expected_hard_trajectories_per_video"
            ]
        )
        + int(
            cfg["scopes"][
                "REPRESENTATIVE_TEST40"
            ]["expected_primary_events"]
        )
        * int(
            cfg["reproducibility"][
                "expected_representative_trajectories_per_video"
            ]
        )
    )

    expected_counts = {
        "event_outcomes.csv":
            expected_event_rows,
        "trajectory_summary.csv":
            expected_trajectories,
        "headline_summary.csv":
            len(HEADLINE_LABELS)
            * len(cfg["scopes"]),
        "curve_summary.csv":
            int(
                cfg[
                    "expected_supported_curve_points"
                ]
            ),
        "headline_rate_status.csv":
            len(
                cfg[
                    "headline_direct_gated_comparisons"
                ]
            )
            * len(cfg["scopes"]),
    }

    for name, expected in (
        expected_counts.items()
    ):
        actual = csv_data_row_count(
            out_dir / name
        )

        if actual != expected:
            raise RuntimeError(
                "{} row count {} != {}".format(
                    name,
                    actual,
                    expected,
                )
            )

    summary = json.loads(
        (
            out_dir
            / "summary.json"
        ).read_text(encoding="utf-8")
    )

    required_equal = {
        "experiment": "EXP053",
        "status":
            "FINAL_TEST_CAMPAIGN_COMPLETE",
        "repo_commit": git_head(),
        "config_sha256":
            sha256sum(CFG_PATH),
        "script_sha256":
            sha256sum(Path(__file__)),
        "test_touched": True,
        "test_campaign_complete": True,
        "threshold_search_performed": False,
        "threshold_retuning_performed": False,
        "trajectory_count":
            expected_trajectories,
        "expected_trajectory_count":
            expected_trajectories,
        "supported_curve_points": 19,
        "unsupported_curve_points": 26,
    }

    for key, expected in (
        required_equal.items()
    ):
        if summary.get(key) != expected:
            raise RuntimeError(
                "Summary field mismatch: {}".format(
                    key
                )
            )

    if (
        int(summary["scratch_cache_hits"])
        + int(summary["scratch_cache_misses"])
        != expected_trajectories
    ):
        raise RuntimeError(
            "Final cache accounting mismatch"
        )

    roles = summary.get(
        "endpoint_roles",
        {},
    )

    if (
        roles.get("POR30")
        != "SOLE_PRIMARY_ENDPOINT"
    ):
        raise RuntimeError(
            "POR30 endpoint-role mismatch"
        )

    if (
        roles.get("ITR30")
        != "SECONDARY_ENDPOINT"
    ):
        raise RuntimeError(
            "ITR30 endpoint-role mismatch"
        )

    if (
        summary.get("primary_contrast")
        != "B3_S_minus_B2"
    ):
        raise RuntimeError(
            "Primary contrast mismatch"
        )

    if set(
        summary.get("scopes", {})
    ) != set(cfg["scopes"]):
        raise RuntimeError(
            "Final summary scope mismatch"
        )

    for scope_label, spec in (
        cfg["scopes"].items()
    ):
        observed = summary["scopes"][
            scope_label
        ]

        if int(
            observed["videos"]
        ) != int(
            spec["expected_videos"]
        ):
            raise RuntimeError(
                "{} video-count mismatch".format(
                    scope_label
                )
            )

        if int(
            observed["primary_events"]
        ) != int(
            spec["expected_primary_events"]
        ):
            raise RuntimeError(
                "{} event-count mismatch".format(
                    scope_label
                )
            )

    return True


def finalize_outputs(
    cfg,
    all_rows,
    scope_summaries,
    curve_rows,
    rate_rows,
    stats,
):
    if OUT.exists():
        raise RuntimeError(
            "Final EXP053 output already exists"
        )

    if STAGE_OUT.exists():
        raise RuntimeError(
            "EXP053 staging output already exists"
        )

    if (
        os.stat(STAGE_OUT.parent).st_dev
        != os.stat(OUT.parent).st_dev
    ):
        raise RuntimeError(
            "Staging and final outputs are "
            "not on the same filesystem"
        )

    write_outputs(
        cfg,
        all_rows,
        scope_summaries,
        curve_rows,
        rate_rows,
        stats,
        STAGE_OUT,
    )

    validate_output_dir(
        cfg,
        STAGE_OUT,
    )

    os.replace(
        STAGE_OUT,
        OUT,
    )

    validate_output_dir(
        cfg,
        OUT,
    )


def plan():
    cfg = load_cfg()
    exp051, exp050, runtime = (
        load_runtime(cfg)
    )

    del exp051
    del runtime

    scopes = load_test_scopes(
        cfg,
        exp050,
    )

    hard_specs = build_specs(
        cfg,
        "HARD_TEST80",
    )

    representative_specs = build_specs(
        cfg,
        "REPRESENTATIVE_TEST40",
    )

    total = (
        len(hard_specs)
        * len(
            scopes[
                "HARD_TEST80"
            ]["videos"]
        )
        + len(representative_specs)
        * len(
            scopes[
                "REPRESENTATIVE_TEST40"
            ]["videos"]
        )
    )

    if total != int(
        cfg["reproducibility"][
            "expected_total_trajectories"
        ]
    ):
        raise RuntimeError(
            "Planned trajectory total mismatch"
        )

    print("hard_test_videos=80")
    print("hard_test_primary_events=506")
    print(
        "hard_trajectories_per_video={}".format(
            len(hard_specs)
        )
    )
    print(
        "representative_test_videos=40"
    )
    print(
        "representative_test_primary_events=76"
    )
    print(
        "representative_trajectories_per_video={}".format(
            len(representative_specs)
        )
    )
    print("supported_curve_points=19")
    print("unsupported_curve_points=26")
    print(
        "total_trajectories={}".format(
            total
        )
    )
    print(
        "test_predictions_generated=false"
    )
    print("EXP053_PLAN=PASS")


def selftest():
    cfg = load_cfg()

    hard_specs = build_specs(
        cfg,
        "HARD_TEST80",
    )

    representative_specs = build_specs(
        cfg,
        "REPRESENTATIVE_TEST40",
    )

    assert len(hard_specs) == 39
    assert len(representative_specs) == 11

    hard_labels = {
        item["run_label"]
        for item in hard_specs
    }

    for variant in cfg["variants"]:
        assert variant in hard_labels
        assert (
            variant + "_NEUTRAL"
            in hard_labels
        )

    assert curve_run_labels(
        0.3,
        "B2",
    ) == (
        "B2",
        "B2_NEUTRAL",
    )

    assert curve_run_labels(
        0.1,
        "B2",
    ) == (
        "CURVE_R010_B2",
        "CURVE_R010_B2_NEUTRAL",
    )

    tolerance = float(
        cfg["rate_mismatch"][
            "absolute_pooled_rate_tolerance"
        ]
    )

    assert tolerance == 0.02

    assert not rate_difference_exceeds_tolerance(
        30,
        100,
        319,
        1000,
        tolerance,
    )

    assert not rate_difference_exceeds_tolerance(
        30,
        100,
        32,
        100,
        tolerance,
    )

    assert rate_difference_exceeds_tolerance(
        30,
        100,
        321,
        1000,
        tolerance,
    )

    assert (
        classify_finalization_state(
            "run",
            False,
            False,
        )
        == "EXECUTE"
    )

    assert (
        classify_finalization_state(
            "resume",
            True,
            False,
        )
        == "RECOVER_FINAL_OUTPUT"
    )

    assert (
        classify_finalization_state(
            "resume",
            False,
            True,
        )
        == "REBUILD_STAGING"
    )

    print("EXP053_SELFTEST=PASS")


def sanity():
    require_frozen_clean_tree()

    cfg = load_cfg()

    exp051, exp050, runtime = (
        load_runtime(cfg)
    )

    scopes = load_test_scopes(
        cfg,
        exp050,
    )

    sanity_video = cfg["sanity"]["video"]

    final_test_videos = {
        video
        for scope in scopes.values()
        for video in scope["videos"]
    }

    if sanity_video in final_test_videos:
        raise RuntimeError(
            "Sanity video overlaps final TEST"
        )

    pool_rows = read_csv(
        ROOT
        / cfg["scope"]["event_pool_csv"]
    )

    primary_rows = [
        row
        for row in pool_rows
        if row["video"] == sanity_video
        and int(
            row["primary_pool_eligible"]
        ) == 1
    ]

    if not primary_rows:
        raise RuntimeError(
            "Sanity video has no "
            "frozen primary events"
        )

    primary_rows.sort(
        key=lambda row: (
            int(row["object_id"]),
            int(row["reappear_frame"]),
            int(row["disappear_start"]),
        )
    )

    model, predictor = (
        exp050.build_predictor(runtime)
    )

    del model

    signal_spec = {
        "run_label": "SANITY_B3_S",
        "kind": "SANITY_SIGNAL",
        "source_variant": "B3_S",
        "tracker_variant": "B3_S",
        "tau": float(
            cfg["headline"]["taus"]["B3_S"]
        ),
        "target_rate": float(
            cfg["headline"]["r_star"]
        ),
        "neutral_source_run_label": None,
    }

    signal, signal_hit = run_or_load(
        cfg,
        exp051,
        exp050,
        runtime,
        predictor,
        "TRAIN_EXPOSED_SANITY",
        sanity_video,
        primary_rows,
        signal_spec,
        neutral_k=None,
        test_touched=False,
        sanity=True,
    )

    neutral_spec = {
        "run_label":
            "SANITY_B3_S_NEUTRAL",
        "kind": "SANITY_NEUTRAL",
        "source_variant": "B3_S",
        "tracker_variant": "B0",
        "tau": None,
        "target_rate": float(
            cfg["headline"]["r_star"]
        ),
        "neutral_source_run_label":
            "SANITY_B3_S",
    }

    neutral, neutral_hit = run_or_load(
        cfg,
        exp051,
        exp050,
        runtime,
        predictor,
        "TRAIN_EXPOSED_SANITY",
        sanity_video,
        primary_rows,
        neutral_spec,
        neutral_k=int(
            signal["admit_count"]
        ),
        test_touched=False,
        sanity=True,
    )

    if int(signal["admit_count"]) != int(
        neutral["admit_count"]
    ):
        raise RuntimeError(
            "Sanity neutral budget mismatch"
        )

    if signal["test_touched"] is not False:
        raise RuntimeError(
            "Sanity touched TEST flag"
        )

    if neutral["test_touched"] is not False:
        raise RuntimeError(
            "Sanity neutral touched TEST flag"
        )

    print(
        "sanity_video={}".format(
            sanity_video
        )
    )
    print(
        "sanity_events={}".format(
            len(primary_rows)
        )
    )
    print(
        "signal_cache_hit={}".format(
            int(signal_hit)
        )
    )
    print(
        "neutral_cache_hit={}".format(
            int(neutral_hit)
        )
    )
    print(
        "signal_k={}".format(
            signal["admit_count"]
        )
    )
    print(
        "neutral_k={}".format(
            neutral["admit_count"]
        )
    )
    print(
        "max_peak_vram_gb={}".format(
            max(
                signal["peak_vram_gb"],
                neutral["peak_vram_gb"],
            )
        )
    )
    print("test_touched=false")
    print("EXP053_SANITY=PASS")


def run_campaign(mode):
    finalization_state = (
        classify_finalization_state(
            mode,
            OUT.exists(),
            STAGE_OUT.exists(),
        )
    )

    require_frozen_clean_tree(
        allow_final_output=(
            mode == "resume"
            and OUT.exists()
        )
    )

    cfg = load_cfg()

    exp051, exp050, runtime = (
        load_runtime(cfg)
    )

    scopes = load_test_scopes(
        cfg,
        exp050,
    )

    if (
        finalization_state
        == "RECOVER_FINAL_OUTPUT"
    ):
        start_or_resume_campaign(
            cfg,
            mode,
        )

        validate_output_dir(
            cfg,
            OUT,
        )

        finish_campaign(cfg)

        print(
            "finalization_recovered_from_final_output=true"
        )
        print("test_touched=true")
        print("test_campaign_complete=true")
        print("EXP053_RUN=PASS")
        return

    model, predictor = (
        exp050.build_predictor(runtime)
    )

    del model

    start_or_resume_campaign(
        cfg,
        mode,
    )

    if (
        finalization_state
        == "REBUILD_STAGING"
    ):
        shutil.rmtree(STAGE_OUT)

        print(
            "stale_finalization_staging_rebuild=true",
            flush=True,
        )

    all_rows = {}
    scope_summaries = {}
    curve_rows = []
    rate_rows = []

    stats = {
        "cache_hits": 0,
        "cache_misses": 0,
    }

    for scope_label in (
        "HARD_TEST80",
        "REPRESENTATIVE_TEST40",
    ):
        scope = scopes[scope_label]

        specs = build_specs(
            cfg,
            scope_label,
        )

        run_rows = []

        for index, video in enumerate(
            scope["videos"],
            start=1,
        ):
            primary_rows = (
                scope[
                    "primary_by_video"
                ][video]
            )

            print(
                "EXP053 {} video {}/{} "
                "{} events={}".format(
                    scope_label,
                    index,
                    len(scope["videos"]),
                    video,
                    len(primary_rows),
                ),
                flush=True,
            )

            results_for_video = {}

            for spec in specs:
                source_label = (
                    spec[
                        "neutral_source_run_label"
                    ]
                )

                neutral_k = None

                if source_label is not None:
                    if (
                        source_label
                        not in results_for_video
                    ):
                        raise RuntimeError(
                            "Neutral source has not "
                            "run: {}".format(
                                source_label
                            )
                        )

                    neutral_k = int(
                        results_for_video[
                            source_label
                        ]["admit_count"]
                    )

                row, cache_hit = (
                    run_or_load(
                        cfg,
                        exp051,
                        exp050,
                        runtime,
                        predictor,
                        scope_label,
                        video,
                        primary_rows,
                        spec,
                        neutral_k=neutral_k,
                        test_touched=True,
                        sanity=False,
                    )
                )

                results_for_video[
                    spec["run_label"]
                ] = row

                run_rows.append(row)

                stats["cache_hits"] += int(
                    cache_hit
                )

                stats[
                    "cache_misses"
                ] += int(
                    not cache_hit
                )

                print(
                    "  {} cache_hit={}".format(
                        spec["run_label"],
                        int(cache_hit),
                    ),
                    flush=True,
                )

        by_label = aggregate_scope(
            cfg,
            scope_label,
            run_rows,
        )

        all_rows[scope_label] = run_rows
        scope_summaries[
            scope_label
        ] = by_label

        curve_rows.extend(
            curve_rows_for_scope(
                cfg,
                scope_label,
                by_label,
            )
        )

        rate_rows.extend(
            rate_status_rows(
                cfg,
                scope_label,
                by_label,
            )
        )

    finalize_outputs(
        cfg,
        all_rows,
        scope_summaries,
        curve_rows,
        rate_rows,
        stats,
    )

    finish_campaign(cfg)

    print(
        "trajectory_count={}".format(
            sum(
                len(rows)
                for rows
                in all_rows.values()
            )
        )
    )

    print(
        "scratch_cache_hits={}".format(
            stats["cache_hits"]
        )
    )

    print(
        "scratch_cache_misses={}".format(
            stats["cache_misses"]
        )
    )

    print("test_touched=true")
    print("test_campaign_complete=true")
    print("EXP053_RUN=PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "selftest",
            "plan",
            "sanity",
            "run",
            "resume",
        ],
    )

    args = parser.parse_args()

    if args.mode == "selftest":
        selftest()
    elif args.mode == "plan":
        plan()
    elif args.mode == "sanity":
        sanity()
    else:
        run_campaign(args.mode)


if __name__ == "__main__":
    main()
