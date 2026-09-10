import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model
from identitygate_attention_filter import (
    get_filter_stats,
    install_identitygate_attention_filter,
    mark_output_block_rows,
    reset_filter_stats,
)


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/EXP027-per-object-filter-sanity-v1.json"
DATA = Path("/mnt/d/thesis_data/mosev2/train")
OUT = ROOT / "experiments/EXP027_per_object_filter_sanity"


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / u)


def add_prompts(predictor, state, gt0, object_ids):
    for oid in object_ids:
        mask = torch.from_numpy(
            (gt0 == oid).astype(np.uint8)
        ).to(torch.bool)

        predictor.add_new_mask(
            inference_state=state,
            frame_idx=0,
            obj_id=oid,
            mask=mask,
        )


def run_tracker(
    predictor,
    jpg_dir,
    gt0,
    object_ids,
    cfg,
    block_object_id=None,
):
    state = predictor.init_state(video_path=str(jpg_dir))
    predictor.clear_all_points_in_video(state)
    add_prompts(predictor, state, gt0, object_ids)

    predictions = {}
    decisions = []

    reset_filter_stats()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    generator = predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=cfg["n_frames"] - 1,
        reverse=False,
        propagate_preflight=True,
    )

    for out in generator:
        frame_idx = int(out[0])
        ids = [int(x) for x in out[1]]
        video_res = out[3]

        if ids != object_ids:
            raise RuntimeError(
                "Object order mismatch: {} vs {}".format(
                    ids, object_ids
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

        should_block = (
            block_object_id is not None
            and cfg["block_start_frame"]
            <= frame_idx
            <= cfg["block_end_frame"]
        )

        if should_block:
            bank = state["output_dict"]["non_cond_frame_outputs"]

            if frame_idx not in bank:
                raise RuntimeError(
                    "Current non-conditioning output missing at frame {}".format(
                        frame_idx
                    )
                )

            rows = [
                oid == block_object_id
                for oid in ids
            ]

            mark_output_block_rows(
                bank[frame_idx],
                rows,
            )

            stored = bank[frame_idx].get(
                "_identitygate_block_rows"
            )

            if stored != rows:
                raise RuntimeError(
                    "Stored block-row metadata mismatch"
                )

            decisions.append({
                "frame_idx": frame_idx,
                "block_object_id": block_object_id,
                "block_rows": json.dumps(rows),
                "bank_entry_present": 1,
            })

    runtime = float(time.time() - t0)
    peak = float(
        torch.cuda.max_memory_allocated() / 1024 ** 3
    )
    stats = get_filter_stats()

    if len(predictions) != cfg["n_frames"]:
        raise RuntimeError(
            "Frame count mismatch: {}".format(
                len(predictions)
            )
        )

    del state
    torch.cuda.empty_cache()

    return predictions, decisions, stats, peak, runtime


def compare(name, left, right, object_ids, n_frames):
    rows = []
    exact = 0
    xor_total = 0
    min_iou = 1.0

    for frame_idx in range(n_frames):
        for oid in object_ids:
            a = left[frame_idx][oid]
            b = right[frame_idx][oid]

            same = bool(np.array_equal(a, b))
            xor_px = int(np.logical_xor(a, b).sum())
            miou = mask_iou(a, b)

            exact += int(same)
            xor_total += xor_px
            min_iou = min(min_iou, miou)

            rows.append({
                "comparison": name,
                "frame_idx": frame_idx,
                "object_id": oid,
                "exact_equal": int(same),
                "xor_px": xor_px,
                "mask_iou": miou,
            })

    return rows, exact, xor_total, min_iou


def main():
    cfg = json.loads(CFG.read_text())

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(OUT)
        )

    OUT.mkdir(parents=True)

    jpg_dir = DATA / "JPEGImages" / cfg["video"]
    ann_dir = DATA / "Annotations" / cfg["video"]

    jpgs = sorted(
        [x for x in os.listdir(jpg_dir) if x.endswith(".jpg")],
        key=lambda x: int(Path(x).stem),
    )
    pngs = sorted(
        [x for x in os.listdir(ann_dir) if x.endswith(".png")],
        key=lambda x: int(Path(x).stem),
    )

    if len(jpgs) != len(pngs):
        raise RuntimeError("JPEG/annotation count mismatch")

    if len(jpgs) < cfg["n_frames"]:
        raise RuntimeError("Video shorter than configured range")

    gt0 = np.array(Image.open(ann_dir / pngs[0]))

    object_ids = sorted(
        int(x)
        for x in np.unique(gt0)
        if int(x) != 0
    )

    if cfg["block_object_id"] not in object_ids:
        raise RuntimeError("Configured block object absent")

    admitted_ids = [
        x for x in object_ids
        if x != cfg["block_object_id"]
    ]

    if len(admitted_ids) == 0:
        raise RuntimeError("No admitted comparison object")

    print("video={}".format(cfg["video"]), flush=True)
    print("frames={}".format(cfg["n_frames"]), flush=True)
    print("object_ids={}".format(object_ids), flush=True)

    print("building frozen SAM3...", flush=True)
    model = build_sam3_video_model()
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    print("running vanilla batched B0...", flush=True)
    vanilla, _, vanilla_stats, vanilla_peak, vanilla_sec = run_tracker(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg,
    )

    print("installing IdentityGate attention filter...", flush=True)
    patch_info = install_identitygate_attention_filter(
        predictor
    )

    print("running patched NO-BLOCK...", flush=True)
    no_block, _, no_block_stats, no_block_peak, no_block_sec = run_tracker(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg,
    )

    print("running selective object block...", flush=True)
    selective, decisions, selective_stats, selective_peak, selective_sec = (
        run_tracker(
            predictor,
            jpg_dir,
            gt0,
            object_ids,
            cfg,
            block_object_id=cfg["block_object_id"],
        )
    )

    rows_ab, exact_ab, xor_ab, min_ab = compare(
        "VANILLA_vs_PATCHED_NO_BLOCK",
        vanilla,
        no_block,
        object_ids,
        cfg["n_frames"],
    )

    rows_bc, exact_bc, xor_bc, min_bc = compare(
        "PATCHED_NO_BLOCK_vs_SELECTIVE",
        no_block,
        selective,
        object_ids,
        cfg["n_frames"],
    )

    total_rows = cfg["n_frames"] * len(object_ids)

    baseline_exact = exact_ab == total_rows

    first_block = cfg["block_start_frame"]

    first_block_equal = all(
        np.array_equal(
            no_block[first_block][oid],
            selective[first_block][oid],
        )
        for oid in object_ids
    )

    admitted_exact = all(
        np.array_equal(
            no_block[f][oid],
            selective[f][oid],
        )
        for f in range(cfg["n_frames"])
        for oid in admitted_ids
    )

    changed = [
        f
        for f in range(first_block + 1, cfg["n_frames"])
        if not np.array_equal(
            no_block[f][cfg["block_object_id"]],
            selective[f][cfg["block_object_id"]],
        )
    ]

    blocked_downstream_changed = len(changed) > 0

    expected_decisions = (
        cfg["block_end_frame"]
        - cfg["block_start_frame"]
        + 1
    )

    decision_count_ok = (
        len(decisions) == expected_decisions
    )

    no_block_zero_masked = (
        no_block_stats["spatial_masked_tokens"] == 0
        and no_block_stats["pointer_masked_tokens"] == 0
    )

    selective_masked = (
        selective_stats["spatial_masked_tokens"] > 0
        and selective_stats["pointer_masked_tokens"] > 0
    )

    passed = all([
        baseline_exact,
        first_block_equal,
        admitted_exact,
        blocked_downstream_changed,
        decision_count_ok,
        no_block_zero_masked,
        selective_masked,
        patch_info["external_source_modified"] is False,
    ])

    summary = {
        "status": (
            "PER_OBJECT_FILTER_SANITY_PASS"
            if passed
            else "PER_OBJECT_FILTER_SANITY_FAIL"
        ),
        "video": cfg["video"],
        "frames": cfg["n_frames"],
        "object_ids": object_ids,
        "block_object_id": cfg["block_object_id"],
        "admitted_object_ids": admitted_ids,
        "block_start_frame": cfg["block_start_frame"],
        "block_end_frame": cfg["block_end_frame"],
        "decision_count": len(decisions),
        "expected_decision_count": expected_decisions,
        "decision_count_ok": decision_count_ok,
        "baseline_comparison_rows": total_rows,
        "vanilla_vs_patched_no_block_exact_rows": exact_ab,
        "vanilla_vs_patched_no_block_exact_all": baseline_exact,
        "vanilla_vs_patched_no_block_xor_px": xor_ab,
        "vanilla_vs_patched_no_block_min_mask_iou": min_ab,
        "first_block_prediction_equal": first_block_equal,
        "admitted_objects_exact_all_frames": admitted_exact,
        "blocked_object_downstream_changed": blocked_downstream_changed,
        "first_blocked_object_changed_frame": (
            changed[0] if changed else None
        ),
        "patched_no_block_vs_selective_exact_rows": exact_bc,
        "patched_no_block_vs_selective_xor_px": xor_bc,
        "patched_no_block_vs_selective_min_mask_iou": min_bc,
        "no_block_zero_masked_tokens": no_block_zero_masked,
        "selective_has_spatial_and_pointer_masking": selective_masked,
        "vanilla_filter_stats": vanilla_stats,
        "no_block_filter_stats": no_block_stats,
        "selective_filter_stats": selective_stats,
        "patch_info": patch_info,
        "vanilla_peak_vram_gb": vanilla_peak,
        "patched_no_block_peak_vram_gb": no_block_peak,
        "selective_peak_vram_gb": selective_peak,
        "vanilla_runtime_sec": vanilla_sec,
        "patched_no_block_runtime_sec": no_block_sec,
        "selective_runtime_sec": selective_sec,
        "dev_touched": 0,
        "test_touched": 0,
        "claim_boundary": cfg["claim_boundary"],
    }

    with open(
        OUT / "frame_object_comparison.csv",
        "w",
        newline="",
    ) as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "comparison",
                "frame_idx",
                "object_id",
                "exact_equal",
                "xor_px",
                "mask_iou",
            ],
            lineterminator="\n",
        )
        wr.writeheader()
        wr.writerows(rows_ab + rows_bc)

    with open(
        OUT / "write_decisions.csv",
        "w",
        newline="",
    ) as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "block_object_id",
                "block_rows",
                "bank_entry_present",
            ],
            lineterminator="\n",
        )
        wr.writeheader()
        wr.writerows(decisions)

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
