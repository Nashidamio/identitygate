import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP037-matched-rate-evaluator-sanity-v1.json"
OUT = ROOT / "experiments/EXP037_matched_rate_evaluator_sanity"


def sha256sum(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)

    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load module: {}".format(path))

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def require_clean_committed_tree():
    required = [
        "configs/EXP037-matched-rate-evaluator-sanity-v1.json",
        "scripts/exp037_matched_rate_evaluator_sanity.py",
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
        raise RuntimeError("EXP037 requires a clean committed tree")


def verify_dependencies(cfg):
    for name, item in cfg["source_dependencies"].items():
        path = ROOT / item["path"]
        actual = sha256sum(path)

        if actual != item["sha256"]:
            raise RuntimeError(
                "{} SHA mismatch: {}".format(name, actual)
            )


def sorted_frames(path, suffix):
    return sorted(
        [
            x
            for x in os.listdir(path)
            if x.endswith(suffix)
        ],
        key=lambda x: int(Path(x).stem),
    )


def configure_scope(base_cfg, cfg, tau):
    run_cfg = copy.deepcopy(base_cfg)

    run_cfg["scope"]["dataset_root"] = cfg["scope"]["dataset_root"]
    run_cfg["scope"]["video"] = cfg["scope"]["video"]
    run_cfg["scope"]["n_frames"] = cfg["scope"]["n_frames"]
    run_cfg["scope"]["expected_object_ids"] = cfg["scope"]["expected_object_ids"]

    run_cfg["frame_intervention"]["tau"] = float(tau)
    run_cfg["frame_intervention"]["tau_status"] = (
        cfg["engineering_tau_status"]
    )

    return run_cfg


def build_gt_cache(ann_dir, pngs, object_ids, n_frames):
    gt_cache = []
    visibility = {
        oid: [False] * n_frames
        for oid in object_ids
    }

    for frame_idx in range(n_frames):
        gt = np.array(
            Image.open(
                ann_dir / pngs[frame_idx]
            )
        )

        gt_cache.append(gt)

        for oid in object_ids:
            visibility[oid][frame_idx] = bool(
                np.any(gt == oid)
            )

    return gt_cache, visibility


def event_structure(vis, min_gap):
    reappear = []
    gap_starts = []

    run = 0
    seen = False

    for frame_idx, visible in enumerate(vis):
        if visible:
            if run >= min_gap and seen:
                reappear.append(frame_idx)

            run = 0
            seen = True

        elif seen:
            run += 1

            if run == min_gap:
                gap_starts.append(
                    frame_idx - min_gap + 1
                )

    return reappear, gap_starts


def score_por30(
    variant,
    tau,
    predictions,
    gt_cache,
    visibility,
    object_ids,
    deps,
    endpoint,
):
    n_frames = len(gt_cache)
    min_gap = int(endpoint["qualifying_gap_frames"])
    window = int(endpoint["por_window"])
    threshold = float(
        endpoint["recovery_iou_strictly_greater_than"]
    )

    rows = []

    for oid in object_ids:
        vis = visibility[oid]

        ious = [None] * n_frames

        for frame_idx in range(n_frames):
            if not vis[frame_idx]:
                continue

            target = (
                gt_cache[frame_idx]
                == oid
            )

            ious[frame_idx] = deps[
                "exp032"
            ].target_iou(
                predictions[frame_idx][oid],
                target,
            )

        reappear, gap_starts = event_structure(
            vis,
            min_gap,
        )

        for reappear_frame in reappear:
            stop = next(
                (
                    g
                    for g in gap_starts
                    if g > reappear_frame
                ),
                n_frames,
            )

            evaluable = 0
            recovered = 0
            first_recovery_frame = ""

            for frame_idx in range(
                reappear_frame,
                stop,
            ):
                if not vis[frame_idx]:
                    continue

                evaluable += 1

                value = ious[frame_idx]

                if (
                    value is not None
                    and value > threshold
                ):
                    recovered = 1
                    first_recovery_frame = frame_idx
                    break

                if evaluable >= window:
                    break

            rows.append(
                {
                    "variant": variant,
                    "tau": tau,
                    "object_id": oid,
                    "reappear_frame": reappear_frame,
                    "recovered_w30": recovered,
                    "evaluable_frames_examined": evaluable,
                    "first_recovery_frame": first_recovery_frame,
                }
            )

    return rows


def pooled_por(rows):
    if not rows:
        return None

    return float(
        sum(
            int(r["recovered_w30"])
            for r in rows
        )
        / len(rows)
    )


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def block_integrity_pass(decisions, n_objects):
    blocks = [
        row
        for row in decisions
        if row["action"] == "BLOCK"
    ]

    return all(
        int(row["global_present_before"]) == 1
        and int(row["global_present_after"]) == 0
        and int(row["per_object_present_before"]) == n_objects
        and int(row["per_object_present_after"]) == 0
        and int(row["frames_already_tracked_retained"]) == 1
        for row in blocks
    )


def main():
    cfg = json.loads(
        CFG_PATH.read_text(encoding="utf-8")
    )

    require_clean_committed_tree()
    verify_dependencies(cfg)

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(
                OUT
            )
        )

    exp033_item = cfg[
        "source_dependencies"
    ]["exp033_script"]

    exp033 = load_module(
        "exp033_frozen",
        ROOT / exp033_item["path"],
    )

    base_cfg_path = (
        ROOT
        / cfg["source_dependencies"][
            "exp033_config"
        ]["path"]
    )

    base_cfg = json.loads(
        base_cfg_path.read_text(
            encoding="utf-8"
        )
    )

    deps = exp033.load_dependencies(
        base_cfg
    )

    packs = exp033.load_model_packs(
        base_cfg,
        deps,
    )

    sam_root = (
        Path.home()
        / "thesis/externals/sam3"
    )

    sam_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(sam_root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    if sam_commit != cfg[
        "substrate"
    ]["expected_sam_commit"]:
        raise RuntimeError("SAM3 commit mismatch")

    data_root = Path(
        cfg["scope"]["dataset_root"]
    )

    video = cfg["scope"]["video"]
    n_frames = int(
        cfg["scope"]["n_frames"]
    )

    jpg_dir = (
        data_root
        / "JPEGImages"
        / video
    )

    ann_dir = (
        data_root
        / "Annotations"
        / video
    )

    jpgs = sorted_frames(
        jpg_dir,
        ".jpg",
    )

    pngs = sorted_frames(
        ann_dir,
        ".png",
    )

    if len(jpgs) != len(pngs):
        raise RuntimeError(
            "JPEG/annotation count mismatch"
        )

    if len(jpgs) < n_frames:
        raise RuntimeError(
            "Video shorter than sanity scope"
        )

    gt0 = np.array(
        Image.open(
            ann_dir / pngs[0]
        )
    )

    object_ids = sorted(
        int(x)
        for x in np.unique(gt0)
        if int(x) != 0
    )

    if object_ids != cfg[
        "scope"
    ]["expected_object_ids"]:
        raise RuntimeError(
            "Object ID mismatch: {}".format(
                object_ids
            )
        )

    gt_cache, visibility = build_gt_cache(
        ann_dir,
        pngs,
        object_ids,
        n_frames,
    )

    OUT.mkdir(parents=True)

    repo_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    print(
        "video={} frames={} objects={}".format(
            video,
            n_frames,
            object_ids,
        ),
        flush=True,
    )

    print(
        "building frozen SAM3...",
        flush=True,
    )

    model = exp033.build_sam3_video_model()

    predictor = model.tracker
    predictor.backbone = (
        model.detector.backbone
    )

    reference_tau = float(
        cfg["engineering_taus"][0]
    )

    b0_cfg = configure_scope(
        base_cfg,
        cfg,
        reference_tau,
    )

    print("running B0...", flush=True)

    b0 = exp033.run_tracker(
        predictor=predictor,
        jpg_dir=jpg_dir,
        gt0=gt0,
        object_ids=object_ids,
        cfg=b0_cfg,
        variant="B0",
        packs=packs,
        deps=deps,
    )

    b0_por_rows = score_por30(
        variant="B0",
        tau="",
        predictions=b0["predictions"],
        gt_cache=gt_cache,
        visibility=visibility,
        object_ids=object_ids,
        deps=deps,
        endpoint=cfg["endpoint"],
    )

    all_por_rows = list(
        b0_por_rows
    )

    operating_points = [
        {
            "variant": "B0",
            "tau": "",
            "eligible_nonconditioning_frames": n_frames - 1,
            "admit_count": n_frames - 1,
            "block_count": 0,
            "write_rate": 1.0,
            "por30_events": len(b0_por_rows),
            "por30": pooled_por(
                b0_por_rows
            ),
            "block_integrity_pass": 1,
            "peak_vram_gb": b0["peak_vram_gb"],
            "runtime_sec": b0["runtime_sec"],
        }
    ]

    for tau in cfg["engineering_taus"]:
        tau = float(tau)

        run_cfg = configure_scope(
            base_cfg,
            cfg,
            tau,
        )

        for variant in cfg["variants"]:
            print(
                "running {} tau={}...".format(
                    variant,
                    tau,
                ),
                flush=True,
            )

            gated = exp033.run_tracker(
                predictor=predictor,
                jpg_dir=jpg_dir,
                gt0=gt0,
                object_ids=object_ids,
                cfg=run_cfg,
                variant=variant,
                packs=packs,
                deps=deps,
            )

            decisions = gated["decisions"]

            if len(decisions) != n_frames - 1:
                raise RuntimeError(
                    "{} unexpected decision count {}".format(
                        variant,
                        len(decisions),
                    )
                )

            admit_count = sum(
                row["action"] == "ADMIT"
                for row in decisions
            )

            block_count = sum(
                row["action"] == "BLOCK"
                for row in decisions
            )

            if (
                admit_count + block_count
                != n_frames - 1
            ):
                raise RuntimeError(
                    "{} invalid action count".format(
                        variant
                    )
                )

            por_rows = score_por30(
                variant=variant,
                tau=tau,
                predictions=gated["predictions"],
                gt_cache=gt_cache,
                visibility=visibility,
                object_ids=object_ids,
                deps=deps,
                endpoint=cfg["endpoint"],
            )

            if len(por_rows) != len(
                b0_por_rows
            ):
                raise RuntimeError(
                    "{} POR denominator mismatch".format(
                        variant
                    )
                )

            integrity = block_integrity_pass(
                decisions,
                len(object_ids),
            )

            if not integrity:
                raise RuntimeError(
                    "{} physical write-block integrity failed".format(
                        variant
                    )
                )

            all_por_rows.extend(
                por_rows
            )

            row = {
                "variant": variant,
                "tau": tau,
                "eligible_nonconditioning_frames": n_frames - 1,
                "admit_count": admit_count,
                "block_count": block_count,
                "write_rate": float(
                    admit_count
                    / (n_frames - 1)
                ),
                "por30_events": len(
                    por_rows
                ),
                "por30": pooled_por(
                    por_rows
                ),
                "block_integrity_pass": int(
                    integrity
                ),
                "peak_vram_gb": gated[
                    "peak_vram_gb"
                ],
                "runtime_sec": gated[
                    "runtime_sec"
                ],
            }

            operating_points.append(
                row
            )

            print(
                "{} tau={} rate={:.4f} events={} POR30={}".format(
                    variant,
                    tau,
                    row["write_rate"],
                    row["por30_events"],
                    row["por30"],
                ),
                flush=True,
            )

    por_fields = [
        "variant",
        "tau",
        "object_id",
        "reappear_frame",
        "recovered_w30",
        "evaluable_frames_examined",
        "first_recovery_frame",
    ]

    op_fields = [
        "variant",
        "tau",
        "eligible_nonconditioning_frames",
        "admit_count",
        "block_count",
        "write_rate",
        "por30_events",
        "por30",
        "block_integrity_pass",
        "peak_vram_gb",
        "runtime_sec",
    ]

    write_csv(
        OUT / "por30_events.csv",
        por_fields,
        all_por_rows,
    )

    write_csv(
        OUT / "operating_points.csv",
        op_fields,
        operating_points,
    )

    b0_event_count = len(
        b0_por_rows
    )

    status = (
        "EXP037_INTEGRATION_SANITY_PASS"
        if b0_event_count > 0
        else "EXP037_INTEGRATION_SANITY_PASS_ZERO_EVENTS_ENDPOINT_NOT_EXERCISED"
    )

    result = {
        "experiment": "EXP037",
        "status": status,
        "repo_commit": repo_commit,
        "config_sha256": sha256sum(
            CFG_PATH
        ),
        "script_sha256": sha256sum(
            Path(__file__)
        ),
        "sam_commit": sam_commit,
        "video": video,
        "frames": n_frames,
        "object_ids": object_ids,
        "b2_model_sha256": packs[
            "b2_sha256"
        ],
        "b3_model_sha256": packs[
            "b3_sha256"
        ],
        "b0_por30_event_count": b0_event_count,
        "operating_points": operating_points,
        "claim_boundary": cfg[
            "claim_boundary"
        ],
        "fresh_dev_touched": 0,
        "test_touched": 0,
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
