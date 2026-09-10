import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP026-singleton-equivalence-v1.json"
DATA_ROOT = Path("/mnt/d/thesis_data/mosev2/train")
OUT = ROOT / "experiments/EXP026_singleton_equivalence"


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def target_iou(pred, gt):
    if not gt.any():
        return None
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum() / union)


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def add_mask_prompt(predictor, state, gt0, oid):
    mask0 = torch.from_numpy(
        (gt0 == oid).astype(np.uint8)
    ).to(torch.bool)

    predictor.add_new_mask(
        inference_state=state,
        frame_idx=0,
        obj_id=oid,
        mask=mask0,
    )


def propagate(predictor, state, n_frames, expected_ids):
    predictions = {}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for out in predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=n_frames - 1,
        reverse=False,
        propagate_preflight=True,
    ):
        frame_idx = int(out[0])
        ids = [int(x) for x in out[1]]
        video_res = out[3]

        if set(ids) != set(expected_ids):
            raise RuntimeError(
                "Unexpected object IDs at frame {}: {}".format(
                    frame_idx,
                    ids,
                )
            )

        predictions[frame_idx] = {}

        for i, oid in enumerate(ids):
            predictions[frame_idx][oid] = (
                (video_res[i, 0] > 0)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

    sec = float(time.time() - t0)
    peak = float(torch.cuda.max_memory_allocated() / 1024 ** 3)

    if len(predictions) != n_frames:
        raise RuntimeError(
            "Expected {} frames, got {}".format(
                n_frames,
                len(predictions),
            )
        )

    return predictions, peak, sec


def run_batched(predictor, jpg_dir, gt0, object_ids, n_frames):
    state = predictor.init_state(video_path=str(jpg_dir))
    predictor.clear_all_points_in_video(state)

    for oid in object_ids:
        add_mask_prompt(predictor, state, gt0, oid)

    predictions, peak, sec = propagate(
        predictor,
        state,
        n_frames,
        object_ids,
    )

    del state
    torch.cuda.empty_cache()

    return predictions, peak, sec


def run_singleton(predictor, jpg_dir, gt0, oid, n_frames):
    state = predictor.init_state(video_path=str(jpg_dir))
    predictor.clear_all_points_in_video(state)

    add_mask_prompt(predictor, state, gt0, oid)

    predictions, peak, sec = propagate(
        predictor,
        state,
        n_frames,
        [oid],
    )

    del state
    torch.cuda.empty_cache()

    return predictions, peak, sec


def main():
    cfg = json.loads(CFG_PATH.read_text())

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(OUT)
        )

    OUT.mkdir(parents=True)

    jpg_dir = DATA_ROOT / "JPEGImages" / cfg["video"]
    ann_dir = DATA_ROOT / "Annotations" / cfg["video"]

    jpgs = sorted(
        [x for x in os.listdir(jpg_dir) if x.endswith(".jpg")],
        key=lambda x: int(Path(x).stem),
    )
    pngs = sorted(
        [x for x in os.listdir(ann_dir) if x.endswith(".png")],
        key=lambda x: int(Path(x).stem),
    )

    if len(jpgs) != len(pngs):
        raise RuntimeError("JPEG and annotation counts differ")

    if len(jpgs) < cfg["n_frames"]:
        raise RuntimeError("Video shorter than configured range")

    if [
        Path(x).stem for x in jpgs[:cfg["n_frames"]]
    ] != [
        Path(x).stem for x in pngs[:cfg["n_frames"]]
    ]:
        raise RuntimeError("Frame stems are misaligned")

    gt0 = np.array(Image.open(ann_dir / pngs[0]))

    object_ids = sorted(
        int(x)
        for x in np.unique(gt0)
        if int(x) != 0
    )

    if len(object_ids) < 2:
        raise RuntimeError(
            "Need at least two objects for batching equivalence test"
        )

    print("video={}".format(cfg["video"]), flush=True)
    print("frames={}".format(cfg["n_frames"]), flush=True)
    print("object_ids={}".format(object_ids), flush=True)

    print("building frozen SAM3...", flush=True)
    model = build_sam3_video_model()
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    print("running batched B0...", flush=True)
    batched, batched_peak, batched_sec = run_batched(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg["n_frames"],
    )

    singleton = {}
    singleton_peaks = {}
    singleton_secs = {}

    for oid in object_ids:
        print("running singleton B0 object {}...".format(oid), flush=True)

        pred, peak, sec = run_singleton(
            predictor,
            jpg_dir,
            gt0,
            oid,
            cfg["n_frames"],
        )

        singleton[oid] = pred
        singleton_peaks[oid] = peak
        singleton_secs[oid] = sec

    rows = []
    exact_rows = 0
    total_xor_px = 0
    first_difference = None
    min_mask_iou = 1.0

    batched_target_ious = []
    singleton_target_ious = []

    per_object = {
        oid: {
            "rows": 0,
            "exact_rows": 0,
            "total_xor_px": 0,
            "min_mask_iou": 1.0,
        }
        for oid in object_ids
    }

    for frame_idx in range(cfg["n_frames"]):
        gt = np.array(Image.open(ann_dir / pngs[frame_idx]))

        for oid in object_ids:
            pb = batched[frame_idx][oid]
            ps = singleton[oid][frame_idx][oid]

            equal = bool(np.array_equal(pb, ps))
            xor_px = int(np.logical_xor(pb, ps).sum())
            between_iou = mask_iou(pb, ps)

            gt_mask = gt == oid
            ib = target_iou(pb, gt_mask)
            isg = target_iou(ps, gt_mask)

            if gt_mask.any():
                batched_target_ious.append(ib)
                singleton_target_ious.append(isg)

            if equal:
                exact_rows += 1

            total_xor_px += xor_px
            min_mask_iou = min(min_mask_iou, between_iou)

            po = per_object[oid]
            po["rows"] += 1
            po["exact_rows"] += int(equal)
            po["total_xor_px"] += xor_px
            po["min_mask_iou"] = min(
                po["min_mask_iou"],
                between_iou,
            )

            if not equal and first_difference is None:
                first_difference = {
                    "frame_idx": frame_idx,
                    "object_id": oid,
                    "xor_px": xor_px,
                    "mask_iou": between_iou,
                }

            rows.append({
                "frame_idx": frame_idx,
                "object_id": oid,
                "exact_equal": int(equal),
                "xor_px": xor_px,
                "batched_vs_singleton_mask_iou": between_iou,
                "target_visible": int(gt_mask.any()),
                "batched_target_iou": "" if ib is None else ib,
                "singleton_target_iou": "" if isg is None else isg,
            })

    total_rows = cfg["n_frames"] * len(object_ids)
    exact_all = exact_rows == total_rows

    summary = {
        "status": (
            "EXACT_EQUIVALENCE_PASS"
            if exact_all
            else "EXACT_EQUIVALENCE_FAIL"
        ),
        "video": cfg["video"],
        "frames": cfg["n_frames"],
        "object_ids": object_ids,
        "comparison_rows": total_rows,
        "exact_equal_rows": exact_rows,
        "exact_equal_fraction": float(exact_rows / total_rows),
        "exact_all_masks_equal": exact_all,
        "first_difference": first_difference,
        "total_xor_px": total_xor_px,
        "minimum_batched_vs_singleton_mask_iou": min_mask_iou,
        "batched_mean_target_iou_visible_rows": float(
            np.mean(batched_target_ious)
        ),
        "singleton_mean_target_iou_visible_rows": float(
            np.mean(singleton_target_ious)
        ),
        "descriptive_target_iou_delta_singleton_minus_batched": float(
            np.mean(singleton_target_ious)
            - np.mean(batched_target_ious)
        ),
        "per_object": per_object,
        "batched_peak_vram_gb": batched_peak,
        "batched_runtime_sec": batched_sec,
        "singleton_peak_vram_gb": singleton_peaks,
        "singleton_runtime_sec": singleton_secs,
        "acceptance_rule": cfg["acceptance_rule"],
        "claim_boundary": cfg["claim_boundary"],
        "dev_touched": 0,
        "test_touched": 0,
    }

    write_csv(
        OUT / "frame_object_comparison.csv",
        [
            "frame_idx",
            "object_id",
            "exact_equal",
            "xor_px",
            "batched_vs_singleton_mask_iou",
            "target_visible",
            "batched_target_iou",
            "singleton_target_iou",
        ],
        rows,
    )

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
