import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "configs/WS-pixel-v1.json"
AUDIT_PATH = ROOT / "experiments/EXP034_ws_pixel_audit_manifest.json"
SCENE_PATH = ROOT / "experiments/EXP019_ws_gt_scenes.csv"

DATA_ROOT = Path("/mnt/d/thesis_data/mosev2/train/JPEGImages")

SANITY_OUT = ROOT / "experiments/EXP034_ws_pixel_sanity"
FULL_OUT = ROOT / "experiments/EXP034_ws_pixel_full"


def sha256sum(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require_repo_clean_and_committed():
    required = [
        "configs/WS-pixel-v1.json",
        "experiments/EXP034_ws_pixel_audit_manifest.json",
        "scripts/exp034_ws_pixel_crosscheck.py",
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

    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
        ],
        text=True,
    ).strip()

    if status:
        raise RuntimeError(
            "EXP034 requires a clean committed IdentityGate tree"
        )


def load_external_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load module: {}".format(path)
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_transnet(cfg, device):
    tcfg = cfg["transnetv2"]

    source = Path(
        tcfg["source_path"]
    ).expanduser()

    checkpoint = Path(
        tcfg["checkpoint_path"]
    ).expanduser()

    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    if commit != tcfg["source_commit"]:
        raise RuntimeError(
            "TransNetV2 source commit mismatch"
        )

    tracked_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()

    if tracked_status:
        raise RuntimeError(
            "Tracked TransNetV2 source is dirty"
        )

    actual_checkpoint_sha = sha256sum(
        checkpoint
    )

    if actual_checkpoint_sha != (
        tcfg["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "TransNetV2 checkpoint SHA mismatch"
        )

    module_path = (
        source / tcfg["module"]
    )

    module = load_external_module(
        "exp034_transnetv2",
        module_path,
    )

    model = module.TransNetV2()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()
    model.to(device)

    return model, commit, actual_checkpoint_sha


def frame_paths(video):
    frame_dir = DATA_ROOT / video

    if not frame_dir.is_dir():
        raise RuntimeError(
            "Missing frame directory: {}".format(
                frame_dir
            )
        )

    paths = sorted(
        [
            x
            for x in frame_dir.iterdir()
            if x.suffix.lower() == ".jpg"
        ],
        key=lambda x: int(x.stem),
    )

    if not paths:
        raise RuntimeError(
            "No JPEG frames for {}".format(video)
        )

    stems = [
        int(x.stem)
        for x in paths
    ]

    if stems != list(range(len(paths))):
        raise RuntimeError(
            "Non-contiguous frame numbering for {}".format(
                video
            )
        )

    return paths


def load_lowres_rgb(paths, cfg):
    width = cfg["transnetv2"][
        "input"
    ]["resize_width"]

    height = cfg["transnetv2"][
        "input"
    ]["resize_height"]

    if not hasattr(Image, "Resampling"):
        raise RuntimeError(
            "Pillow Resampling API unavailable"
        )

    frames = []

    for path in paths:
        with Image.open(path) as im:
            arr = np.asarray(
                im.convert("RGB").resize(
                    (width, height),
                    resample=(
                        Image.Resampling.BILINEAR
                    ),
                ),
                dtype=np.uint8,
            )

        if arr.shape != (
            height,
            width,
            3,
        ):
            raise RuntimeError(
                "Unexpected resized frame shape"
            )

        frames.append(arr)

    return np.stack(
        frames,
        axis=0,
    )


def predict_transnet_official(
    model,
    frames,
    device,
):
    if (
        frames.ndim != 4
        or tuple(frames.shape[1:])
        != (27, 48, 3)
        or frames.dtype != np.uint8
    ):
        raise RuntimeError(
            "Unexpected TransNet input"
        )

    n_frames = len(frames)

    no_pad_start = 25

    remainder = n_frames % 50

    no_pad_end = (
        25
        + 50
        - (
            remainder
            if remainder != 0
            else 50
        )
    )

    padded = np.concatenate(
        [
            np.repeat(
                frames[0:1],
                no_pad_start,
                axis=0,
            ),
            frames,
            np.repeat(
                frames[-1:],
                no_pad_end,
                axis=0,
            ),
        ],
        axis=0,
    )

    predictions = []

    ptr = 0

    with torch.no_grad():
        while ptr + 100 <= len(padded):
            inp = torch.from_numpy(
                padded[
                    ptr:ptr + 100
                ][None]
            ).to(device)

            logits, _ = model(inp)

            single = (
                torch.sigmoid(logits)[
                    0,
                    25:75,
                    0,
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if single.shape != (50,):
                raise RuntimeError(
                    "Unexpected TransNet output slice"
                )

            predictions.append(single)
            ptr += 50

    if not predictions:
        raise RuntimeError(
            "No TransNet prediction windows"
        )

    out = np.concatenate(
        predictions,
        axis=0,
    )[:n_frames]

    if out.shape != (n_frames,):
        raise RuntimeError(
            "Prediction length mismatch"
        )

    if not np.all(np.isfinite(out)):
        raise RuntimeError(
            "Non-finite TransNet predictions"
        )

    return out


def adjacent_rgb_mad(paths):
    values = np.full(
        len(paths),
        np.nan,
        dtype=np.float64,
    )

    previous = None
    previous_shape = None

    for frame_idx, path in enumerate(paths):
        with Image.open(path) as im:
            current = np.asarray(
                im.convert("RGB"),
                dtype=np.float32,
            )

        if previous is not None:
            if current.shape != previous_shape:
                raise RuntimeError(
                    "Frame shape changed within video"
                )

            values[frame_idx] = float(
                np.mean(
                    np.abs(
                        current - previous
                    )
                )
                / 255.0
            )

        previous = current
        previous_shape = current.shape

    if len(paths) > 1:
        if not np.all(
            np.isfinite(
                values[1:]
            )
        ):
            raise RuntimeError(
                "Non-finite adjacent MAD"
            )

    return values


def max_inclusive(values, lo, hi):
    lo = max(
        0,
        int(lo),
    )

    hi = min(
        len(values) - 1,
        int(hi),
    )

    if hi < lo:
        raise RuntimeError(
            "Empty inclusive window"
        )

    x = values[
        lo:hi + 1
    ]

    finite = x[
        np.isfinite(x)
    ]

    if len(finite) == 0:
        raise RuntimeError(
            "No finite values in window"
        )

    return float(
        np.max(finite)
    )


def evaluate_scene(
    row,
    transnet,
    mad,
    cfg,
):
    d0 = int(
        row["disappear_min"]
    )
    d1 = int(
        row["disappear_max"]
    )

    r0 = int(
        row["reappear_min"]
    )
    r1 = int(
        row["reappear_max"]
    )

    n_frames = len(transnet)

    if len(mad) != n_frames:
        raise RuntimeError(
            "Pixel arrays length mismatch"
        )

    if not (
        0 <= d0 <= d1 < n_frames
        and 0 <= r0 <= r1 < n_frames
    ):
        raise RuntimeError(
            "EXP019 scene frame outside video"
        )

    ccfg = cfg[
        "transnetv2"
    ]["camera_cut"]

    w = int(
        ccfg[
            "matching_window_frames"
        ]
    )

    t_disappear = max_inclusive(
        transnet,
        d0 - w,
        d1 + w,
    )

    t_reappear = max_inclusive(
        transnet,
        r0 - w,
        r1 + w,
    )

    threshold = float(
        ccfg["threshold"]
    )

    camera_positive = (
        t_disappear > threshold
        or t_reappear > threshold
    )

    if d0 < 5:
        raise RuntimeError(
            "WS candidate lacks five-frame reference"
        )

    # Exact five pre-disappearance frames:
    # d0-5 ... d0-1
    # Four adjacent transitions end at d0-4 ... d0-1.
    reference = mad[
        d0 - 4:d0
    ]

    if (
        len(reference) != 4
        or not np.all(
            np.isfinite(reference)
        )
    ):
        raise RuntimeError(
            "Invalid MAD reference window"
        )

    reference_median = float(
        np.median(reference)
    )

    mad_disappear = max_inclusive(
        mad,
        max(1, d0 - w),
        d1 + w,
    )

    mad_reappear = max_inclusive(
        mad,
        max(1, r0 - w),
        r1 + w,
    )

    event_max_mad = max(
        mad_disappear,
        mad_reappear,
    )

    gcfg = cfg[
        "global_frame_difference"
    ]

    floor = float(
        gcfg["reference_floor"]
    )

    denominator = max(
        reference_median,
        floor,
    )

    ratio = (
        event_max_mad
        / denominator
    )

    global_positive = (
        event_max_mad
        >= float(
            gcfg[
                "absolute_mad_min"
            ]
        )
        and ratio
        >= float(
            gcfg[
                "reference_multiplier_min"
            ]
        )
    )

    pixel_positive = (
        camera_positive
        or global_positive
    )

    auto_status = (
        "AUTO_WS_CONFIRMED"
        if pixel_positive
        else "MANUAL_ADJUDICATION_REQUIRED"
    )

    return {
        "scene_event_id": (
            row["scene_event_id"]
        ),
        "video": row["video"],
        "n_object_events": int(
            row["n_object_events"]
        ),
        "n_unique_objects": int(
            row["n_unique_objects"]
        ),
        "disappear_min": d0,
        "disappear_max": d1,
        "reappear_min": r0,
        "reappear_max": r1,

        "transnet_disappear_max_prob": (
            t_disappear
        ),
        "transnet_reappear_max_prob": (
            t_reappear
        ),
        "camera_cut_positive": int(
            camera_positive
        ),

        "reference_median_mad": (
            reference_median
        ),
        "disappear_max_mad": (
            mad_disappear
        ),
        "reappear_max_mad": (
            mad_reappear
        ),
        "event_max_mad": (
            event_max_mad
        ),
        "event_to_reference_mad_ratio": (
            ratio
        ),
        "global_mad_positive": int(
            global_positive
        ),

        "pixel_confirmation_positive": int(
            pixel_positive
        ),
        "automatic_status": auto_status,
    }


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            "Refusing to write empty CSV: {}".format(
                path
            )
        )

    fields = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sanity",
        action="store_true",
    )
    args = ap.parse_args()

    require_repo_clean_and_committed()

    cfg = json.loads(
        CFG_PATH.read_text()
    )

    audit = json.loads(
        AUDIT_PATH.read_text()
    )

    config_sha = sha256sum(
        CFG_PATH
    )

    if audit[
        "config_sha256"
    ] != config_sha:
        raise RuntimeError(
            "Audit manifest/config SHA mismatch"
        )

    if config_sha != (
        "24d177e6af760aad5719829b191d8da533a35b8248e691ad4a7d3e28c76a0da2"
    ):
        raise RuntimeError(
            "Unexpected frozen WS-pixel config SHA"
        )

    parent = ROOT / cfg[
        "parent_annotation_rule"
    ]["path"]

    if sha256sum(parent) != cfg[
        "parent_annotation_rule"
    ]["sha256"]:
        raise RuntimeError(
            "Parent WS config SHA mismatch"
        )

    with open(
        SCENE_PATH,
        newline="",
    ) as f:
        scenes = list(
            csv.DictReader(f)
        )

    if len(scenes) != cfg[
        "input_population"
    ]["annotation_candidate_scene_count"]:
        raise RuntimeError(
            "EXP019 scene count mismatch"
        )

    if len(
        {
            r["scene_event_id"]
            for r in scenes
        }
    ) != len(scenes):
        raise RuntimeError(
            "Duplicate scene_event_id"
        )

    if args.sanity:
        selected_videos = [
            audit["sanity_video"]
        ]

        out_dir = SANITY_OUT

        expected_mode = (
            "SANITY_DEVELOPMENT_EXPOSED_ONLY"
        )

    else:
        selected_videos = sorted(
            {
                r["video"]
                for r in scenes
            }
        )

        out_dir = FULL_OUT
        expected_mode = (
            "FULL_PIXEL_CROSSCHECK"
        )

    if out_dir.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(
                out_dir
            )
        )

    selected_set = set(
        selected_videos
    )

    selected_scenes = [
        r
        for r in scenes
        if r["video"]
        in selected_set
    ]

    if not selected_scenes:
        raise RuntimeError(
            "No selected EXP019 scenes"
        )

    if args.sanity:
        if selected_videos != [
            audit["sanity_video"]
        ]:
            raise RuntimeError(
                "Sanity-video selection changed"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for EXP034 execution"
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision(
        "highest"
    )

    device = torch.device(
        "cuda"
    )

    torch.cuda.reset_peak_memory_stats()

    start = time.time()

    model, tn_commit, checkpoint_sha = (
        load_transnet(
            cfg,
            device,
        )
    )

    out_rows = []

    by_video = {}

    for row in selected_scenes:
        by_video.setdefault(
            row["video"],
            [],
        ).append(row)

    for video_index, video in enumerate(
        sorted(by_video),
        start=1,
    ):
        print(
            "VIDEO",
            video_index,
            "/",
            len(by_video),
            video,
            flush=True,
        )

        paths = frame_paths(
            video
        )

        lowres = load_lowres_rgb(
            paths,
            cfg,
        )

        transnet = (
            predict_transnet_official(
                model,
                lowres,
                device,
            )
        )

        mad = adjacent_rgb_mad(
            paths
        )

        for row in sorted(
            by_video[video],
            key=lambda x: x[
                "scene_event_id"
            ],
        ):
            out_rows.append(
                evaluate_scene(
                    row,
                    transnet,
                    mad,
                    cfg,
                )
            )

        del lowres
        del transnet
        del mad

        gc.collect()
        torch.cuda.empty_cache()

    out_rows = sorted(
        out_rows,
        key=lambda x: x[
            "scene_event_id"
        ],
    )

    out_dir.mkdir(
        parents=True
    )

    write_csv(
        out_dir / "per_scene.csv",
        out_rows,
    )

    confirmed = [
        r
        for r in out_rows
        if r[
            "pixel_confirmation_positive"
        ] == 1
    ]

    disagreements = [
        r
        for r in out_rows
        if r[
            "pixel_confirmation_positive"
        ] == 0
    ]

    if confirmed:
        write_csv(
            out_dir
            / "auto_confirmed.csv",
            confirmed,
        )

    if disagreements:
        write_csv(
            out_dir
            / "manual_adjudication_required.csv",
            disagreements,
        )

    audit_ids = set(
        audit[
            "sampled_scene_event_ids"
        ]
    )

    audit_rows = [
        r
        for r in out_rows
        if r[
            "scene_event_id"
        ] in audit_ids
    ]

    if not args.sanity:
        if len(out_rows) != 2179:
            raise RuntimeError(
                "Full-run scene count mismatch"
            )

        if len(audit_rows) != 30:
            raise RuntimeError(
                "Frozen audit sample did not resolve to 30 rows"
            )

        write_csv(
            out_dir
            / "audit_sample_with_pixel.csv",
            audit_rows,
        )

    torch.cuda.synchronize()

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    camera_positive = sum(
        int(
            r[
                "camera_cut_positive"
            ]
        )
        for r in out_rows
    )

    global_positive = sum(
        int(
            r[
                "global_mad_positive"
            ]
        )
        for r in out_rows
    )

    pixel_positive = len(
        confirmed
    )

    result = {
        "experiment": "EXP034",
        "mode": expected_mode,
        "status": (
            "SANITY_COMPLETE_NOT_FINAL_WS_LABELS"
            if args.sanity
            else "FULL_PIXEL_CROSSCHECK_COMPLETE_MANUAL_ADJUDICATION_PENDING"
        ),

        "identitygate_commit_at_execution": (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip()
        ),

        "config_sha256": (
            config_sha
        ),
        "audit_manifest_sha256": (
            sha256sum(
                AUDIT_PATH
            )
        ),
        "scene_csv_sha256": (
            sha256sum(
                SCENE_PATH
            )
        ),

        "transnet_source_commit": (
            tn_commit
        ),
        "transnet_checkpoint_sha256": (
            checkpoint_sha
        ),

        "pillow_version": (
            PIL.__version__
        ),
        "torch_version": (
            torch.__version__
        ),
        "cuda_version": (
            torch.version.cuda
        ),
        "gpu": (
            torch.cuda.get_device_name(
                0
            )
        ),

        "videos_processed": len(
            by_video
        ),
        "scenes_processed": len(
            out_rows
        ),
        "camera_cut_positive_scenes": (
            camera_positive
        ),
        "global_mad_positive_scenes": (
            global_positive
        ),
        "pixel_confirmation_positive_scenes": (
            pixel_positive
        ),
        "manual_adjudication_required_scenes": (
            len(disagreements)
        ),

        "audit_rows_resolved_in_this_run": (
            len(audit_rows)
        ),

        "peak_vram_gb": (
            peak_vram
        ),
        "runtime_sec": (
            time.time() - start
        ),

        "sam_predictions_used": False,
        "gate_predictions_used": False,
        "por_used": False,
        "test_gate_evaluation_performed": False,

        "claim_boundary": (
            cfg[
                "claim_boundary"
            ]
        ),
    }

    (
        out_dir / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
