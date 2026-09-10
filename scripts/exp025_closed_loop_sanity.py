import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_video_model


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP025-closed-loop-sanity-v1.json"
DATA_ROOT = Path("/mnt/d/thesis_data/mosev2/train")
OUT = ROOT / "experiments/EXP025_closed_loop_sanity"


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def gt_iou(pred, gt):
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


def overlay(raw, masks, object_ids):
    out = raw.copy()
    palette = [
        np.array([255, 64, 64], dtype=np.float32),
        np.array([64, 220, 64], dtype=np.float32),
        np.array([64, 128, 255], dtype=np.float32),
        np.array([255, 190, 64], dtype=np.float32),
        np.array([210, 64, 255], dtype=np.float32),
        np.array([64, 230, 230], dtype=np.float32),
    ]

    for j, oid in enumerate(object_ids):
        m = masks.get(oid)
        if m is None:
            continue
        m = m.astype(bool)
        if not m.any():
            continue
        color = palette[j % len(palette)]
        out[m] = (
            0.45 * out[m].astype(np.float32)
            + 0.55 * color
        ).astype(np.uint8)

    return out


def make_visual(path, raw, gt_labels, b0_masks, block_masks, object_ids):
    gt_masks = {oid: gt_labels == oid for oid in object_ids}

    panels = [
        raw,
        overlay(raw, gt_masks, object_ids),
        overlay(raw, b0_masks, object_ids),
        overlay(raw, block_masks, object_ids),
    ]
    names = ["RAW", "GT", "B0", "WHOLE_FRAME_BLOCK"]

    h, w = raw.shape[:2]
    top = 28
    canvas = Image.new("RGB", (4 * w, h + top), (255, 255, 255))

    for i, arr in enumerate(panels):
        canvas.paste(Image.fromarray(arr), (i * w, top))

    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(names):
        draw.text((i * w + 8, 7), name, fill=(0, 0, 0))

    canvas.save(path)


def run_tracker(predictor, jpg_dir, gt0, object_ids, cfg, block):
    state = predictor.init_state(video_path=str(jpg_dir))
    predictor.clear_all_points_in_video(state)

    for oid in object_ids:
        mask0 = torch.from_numpy(
            (gt0 == oid).astype(np.uint8)
        ).to(torch.bool)

        predictor.add_new_mask(
            inference_state=state,
            frame_idx=0,
            obj_id=oid,
            mask=mask0,
        )

    predictions = {}
    decisions = []

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    generator = predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=cfg["n_frames"],
        reverse=False,
        propagate_preflight=True,
    )

    for out in generator:
        frame_idx = int(out[0])
        ids = [int(x) for x in out[1]]
        video_res = out[3]

        if set(ids) != set(object_ids):
            raise RuntimeError(
                "Tracked object IDs changed at frame {}".format(frame_idx)
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

        should_block = (
            block
            and cfg["block_start_frame"]
            <= frame_idx
            <= cfg["block_end_frame"]
        )

        if should_block:
            global_bank = state["output_dict"]["non_cond_frame_outputs"]

            per_object_banks = [
                obj_output["non_cond_frame_outputs"]
                for obj_output in state["output_dict_per_obj"].values()
            ]

            global_before = int(frame_idx in global_bank)
            per_object_before = sum(
                int(frame_idx in bank)
                for bank in per_object_banks
            )

            if global_before != 1:
                raise RuntimeError(
                    "Frame {} missing from global bank before block".format(
                        frame_idx
                    )
                )

            if per_object_before != len(object_ids):
                raise RuntimeError(
                    "Frame {} missing from per-object bank before block".format(
                        frame_idx
                    )
                )

            global_size_before = len(global_bank)

            global_bank.pop(frame_idx, None)

            for bank in per_object_banks:
                bank.pop(frame_idx, None)

            global_after = int(frame_idx in global_bank)
            per_object_after = sum(
                int(frame_idx in bank)
                for bank in per_object_banks
            )

            if global_after != 0 or per_object_after != 0:
                raise RuntimeError(
                    "Frame {} still present after block".format(frame_idx)
                )

            decisions.append({
                "frame_idx": frame_idx,
                "action": "BLOCK",
                "global_present_before": global_before,
                "global_present_after": global_after,
                "per_object_present_before": per_object_before,
                "per_object_present_after": per_object_after,
                "global_bank_size_before": global_size_before,
                "global_bank_size_after": len(global_bank),
                "frames_already_tracked_retained": int(
                    frame_idx in state["frames_already_tracked"]
                ),
            })

    peak = float(torch.cuda.max_memory_allocated() / 1024 ** 3)
    runtime = float(time.time() - start)

    del state
    torch.cuda.empty_cache()

    return predictions, decisions, peak, runtime


def main():
    cfg = json.loads(CFG_PATH.read_text())

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(OUT)
        )

    OUT.mkdir(parents=True)
    visual_dir = OUT / "visuals"
    visual_dir.mkdir()

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
        raise RuntimeError(
            "Video has only {} frames".format(len(jpgs))
        )

    if [
        Path(x).stem for x in jpgs[:cfg["n_frames"]]
    ] != [
        Path(x).stem for x in pngs[:cfg["n_frames"]]
    ]:
        raise RuntimeError("Frame stems are misaligned")

    gt0 = np.array(Image.open(ann_dir / pngs[0]))
    object_ids = sorted(
        int(x) for x in np.unique(gt0) if int(x) != 0
    )

    if len(object_ids) == 0:
        raise RuntimeError("No frame-0 objects")

    print("video={}".format(cfg["video"]), flush=True)
    print("frames={}".format(cfg["n_frames"]), flush=True)
    print("object_ids={}".format(object_ids), flush=True)

    print("building frozen SAM3...", flush=True)
    model = build_sam3_video_model()
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    print("running B0...", flush=True)
    b0, _, peak_b0, sec_b0 = run_tracker(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg,
        block=False,
    )

    print("running deterministic BLOCK...", flush=True)
    blocked, decisions, peak_block, sec_block = run_tracker(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg,
        block=True,
    )

    if len(b0) != cfg["n_frames"]:
        raise RuntimeError("B0 frame count mismatch")

    if len(blocked) != cfg["n_frames"]:
        raise RuntimeError("BLOCK frame count mismatch")

    expected_blocks = (
        cfg["block_end_frame"]
        - cfg["block_start_frame"]
        + 1
    )

    if len(decisions) != expected_blocks:
        raise RuntimeError(
            "Block count mismatch: {} vs {}".format(
                len(decisions),
                expected_blocks,
            )
        )

    first_block = cfg["block_start_frame"]

    first_block_equal = all(
        np.array_equal(
            b0[first_block][oid],
            blocked[first_block][oid],
        )
        for oid in object_ids
    )

    if not first_block_equal:
        raise RuntimeError(
            "B0 and BLOCK differ before first intervention can affect inference"
        )

    metric_rows = []
    visible_b0 = []
    visible_block = []

    first_changed_frame = None
    total_xor_after_first_block = 0

    for frame_idx in range(cfg["n_frames"]):
        gt = np.array(Image.open(ann_dir / pngs[frame_idx]))

        for oid in object_ids:
            gt_mask = gt == oid
            p0 = b0[frame_idx][oid]
            pb = blocked[frame_idx][oid]

            i0 = gt_iou(p0, gt_mask)
            ib = gt_iou(pb, gt_mask)

            if gt_mask.any():
                visible_b0.append(i0)
                visible_block.append(ib)

            xor_px = int(np.logical_xor(p0, pb).sum())
            between_iou = mask_iou(p0, pb)

            if frame_idx > first_block and xor_px > 0:
                total_xor_after_first_block += xor_px
                if first_changed_frame is None:
                    first_changed_frame = frame_idx

            metric_rows.append({
                "frame_idx": frame_idx,
                "object_id": oid,
                "target_visible": int(gt_mask.any()),
                "gt_area_px": int(gt_mask.sum()),
                "b0_pred_area_px": int(p0.sum()),
                "block_pred_area_px": int(pb.sum()),
                "b0_target_iou": "" if i0 is None else i0,
                "block_target_iou": "" if ib is None else ib,
                "b0_vs_block_mask_iou": between_iou,
                "xor_px": xor_px,
            })

        if frame_idx in cfg["visual_frames"]:
            raw = np.array(
                Image.open(jpg_dir / jpgs[frame_idx]).convert("RGB")
            )

            make_visual(
                visual_dir / "frame_{:04d}.png".format(frame_idx),
                raw,
                gt,
                b0[frame_idx],
                blocked[frame_idx],
                object_ids,
            )

    behavior_changed = first_changed_frame is not None

    if not behavior_changed:
        raise RuntimeError(
            "Memory entries were blocked but no downstream mask changed"
        )

    write_csv(
        OUT / "write_decisions.csv",
        [
            "frame_idx",
            "action",
            "global_present_before",
            "global_present_after",
            "per_object_present_before",
            "per_object_present_after",
            "global_bank_size_before",
            "global_bank_size_after",
            "frames_already_tracked_retained",
        ],
        decisions,
    )

    write_csv(
        OUT / "frame_metrics.csv",
        [
            "frame_idx",
            "object_id",
            "target_visible",
            "gt_area_px",
            "b0_pred_area_px",
            "block_pred_area_px",
            "b0_target_iou",
            "block_target_iou",
            "b0_vs_block_mask_iou",
            "xor_px",
        ],
        metric_rows,
    )

    summary = {
        "status": "MECHANISM_SANITY_COMPLETE_NOT_GATE_RESULT",
        "video": cfg["video"],
        "frames": cfg["n_frames"],
        "object_ids": object_ids,
        "block_start_frame": cfg["block_start_frame"],
        "block_end_frame": cfg["block_end_frame"],
        "block_count": len(decisions),
        "first_block_prediction_equal_between_runs": first_block_equal,
        "every_block_present_before": all(
            x["global_present_before"] == 1
            and x["per_object_present_before"] == len(object_ids)
            for x in decisions
        ),
        "every_block_absent_after": all(
            x["global_present_after"] == 0
            and x["per_object_present_after"] == 0
            for x in decisions
        ),
        "frames_already_tracked_retained": all(
            x["frames_already_tracked_retained"] == 1
            for x in decisions
        ),
        "downstream_behavior_changed": behavior_changed,
        "first_changed_frame": first_changed_frame,
        "total_xor_px_after_first_block": total_xor_after_first_block,
        "b0_mean_target_iou_visible_rows": float(np.mean(visible_b0)),
        "block_mean_target_iou_visible_rows": float(np.mean(visible_block)),
        "descriptive_delta_mean_iou": float(
            np.mean(visible_block) - np.mean(visible_b0)
        ),
        "b0_runtime_sec": sec_b0,
        "block_runtime_sec": sec_block,
        "b0_peak_vram_gb": peak_b0,
        "block_peak_vram_gb": peak_block,
        "visual_frames": cfg["visual_frames"],
        "claim_boundary": cfg["claim_boundary"],
        "dev_touched": 0,
        "test_touched": 0,
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
