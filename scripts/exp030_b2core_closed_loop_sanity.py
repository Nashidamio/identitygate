import csv
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_video_model


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP030-b2core-closed-loop-sanity-v1.json"
OUT = ROOT / "experiments/EXP030_b2core_closed_loop_sanity"


class GateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return float("nan")
    return float(inter / union)


def comparison_iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(inter / union)


def target_iou(pred, gt):
    if int(gt.sum()) == 0:
        return None
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(inter / union)


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )
        wr.writeheader()
        wr.writerows(rows)


def load_gate(cfg):
    model_path = ROOT / cfg["model_path"]

    actual_sha = sha256sum(model_path)
    if actual_sha != cfg["expected_model_sha256"]:
        raise RuntimeError(
            "EXP029 model SHA mismatch: {}".format(actual_sha)
        )

    artifact = json.loads(model_path.read_text())

    if artifact["features"] != cfg["features"]:
        raise RuntimeError("EXP029 feature order mismatch")

    mean = np.asarray(
        artifact["normalization"]["mean"],
        dtype=np.float64,
    )
    std = np.asarray(
        artifact["normalization"]["std"],
        dtype=np.float64,
    )

    if mean.shape != (len(cfg["features"]),):
        raise RuntimeError("Normalizer mean shape mismatch")
    if std.shape != mean.shape:
        raise RuntimeError("Normalizer std shape mismatch")
    if not np.all(np.isfinite(mean)):
        raise RuntimeError("Non-finite normalizer mean")
    if not np.all(np.isfinite(std)):
        raise RuntimeError("Non-finite normalizer std")
    if np.any(std <= 0):
        raise RuntimeError("Non-positive normalizer std")

    heads = {}

    for name in ["drift", "theft"]:
        model = GateMLP(len(cfg["features"])).to(dtype=torch.float64)

        state = {
            key: torch.tensor(value, dtype=torch.float64)
            for key, value
            in artifact["heads"][name]["state_dict"].items()
        }

        model.load_state_dict(state, strict=True)
        model.eval()
        heads[name] = model

    return heads, mean, std, actual_sha


def score_object(feature_values, heads, mean, std):
    x = np.asarray(feature_values, dtype=np.float64)

    finite = bool(np.all(np.isfinite(x)))

    if not finite:
        return {
            "feature_finite": 0,
            "p_unsafe_drift": "",
            "p_unsafe_theft": "",
            "p_safe_object": 0.0,
            "missing_policy": "FAIL_CLOSED",
        }

    z = (x - mean) / std
    xt = torch.from_numpy(z).reshape(1, -1)

    with torch.no_grad():
        p_drift = float(torch.sigmoid(heads["drift"](xt))[0].item())
        p_theft = float(torch.sigmoid(heads["theft"](xt))[0].item())

    if not math.isfinite(p_drift) or not math.isfinite(p_theft):
        raise RuntimeError("Non-finite neural gate probability")

    p_safe = min(1.0 - p_drift, 1.0 - p_theft)

    return {
        "feature_finite": 1,
        "p_unsafe_drift": p_drift,
        "p_unsafe_theft": p_theft,
        "p_safe_object": float(p_safe),
        "missing_policy": "",
    }


def add_frame0_prompts(predictor, state, gt0, object_ids):
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


def evict_current_frame(state, frame_idx, n_objects):
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
            "Blocked frame {} absent from global bank".format(frame_idx)
        )

    if per_object_before != n_objects:
        raise RuntimeError(
            "Blocked frame {} absent from per-object bank".format(frame_idx)
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
            "Blocked frame {} survived eviction".format(frame_idx)
        )

    return {
        "global_present_before": global_before,
        "global_present_after": global_after,
        "per_object_present_before": per_object_before,
        "per_object_present_after": per_object_after,
        "global_bank_size_before": global_size_before,
        "global_bank_size_after": len(global_bank),
        "frames_already_tracked_retained": int(
            frame_idx in state["frames_already_tracked"]
        ),
    }


def run_tracker(
    predictor,
    jpg_dir,
    gt0,
    object_ids,
    cfg,
    gated,
    heads=None,
    mean=None,
    std=None,
):
    state = predictor.init_state(video_path=str(jpg_dir))
    predictor.clear_all_points_in_video(state)
    add_frame0_prompts(predictor, state, gt0, object_ids)

    predictions = {}
    decisions = []
    feature_rows = []

    anchor_areas = {
        oid: int((gt0 == oid).sum())
        for oid in object_ids
    }

    if any(x <= 0 for x in anchor_areas.values()):
        raise RuntimeError("Frame-0 anchor area is zero")

    prev_masks = {
        oid: (gt0 == oid).astype(bool)
        for oid in object_ids
    }

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

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
                "Object order changed at frame {}: {}".format(
                    frame_idx,
                    ids,
                )
            )

        current_masks = {}

        for row, oid in enumerate(ids):
            current_masks[oid] = (
                (video_res[row, 0] > 0)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

        predictions[frame_idx] = {
            oid: current_masks[oid].copy()
            for oid in object_ids
        }

        if frame_idx == 0:
            continue

        if not gated:
            for oid in object_ids:
                prev_masks[oid] = current_masks[oid].copy()
            continue

        current_out = state["output_dict"][
            "non_cond_frame_outputs"
        ].get(frame_idx)

        if current_out is None:
            raise RuntimeError(
                "Current non-conditioning output missing at frame {}".format(
                    frame_idx
                )
            )

        iou_scores = current_out["iou_score"]
        object_logits = current_out["object_score_logits"]

        object_safe_scores = []

        for row, oid in enumerate(ids):
            mask = current_masks[oid]
            pred_area = int(mask.sum())
            frame_area = int(mask.size)

            mask_conf = float(
                iou_scores[row].detach().float().reshape(-1)[0].item()
            )

            occ_logit = float(
                object_logits[row].detach().float().reshape(-1)[0].item()
            )

            temporal = binary_iou(
                mask,
                prev_masks[oid],
            )

            features = [
                mask_conf,
                occ_logit,
                pred_area / frame_area,
                pred_area / anchor_areas[oid],
                temporal,
            ]

            scored = score_object(
                features,
                heads,
                mean,
                std,
            )

            object_safe_scores.append(scored["p_safe_object"])

            feature_rows.append({
                "frame_idx": frame_idx,
                "object_id": oid,
                "mask_conf_iou_head": mask_conf,
                "occ_score_logit": occ_logit,
                "pred_area_px": pred_area,
                "area_norm": pred_area / frame_area,
                "frame0_anchor_area_px": anchor_areas[oid],
                "area_ratio_anchor": pred_area / anchor_areas[oid],
                "temporal_iou_prev": (
                    ""
                    if not math.isfinite(temporal)
                    else temporal
                ),
                "feature_finite": scored["feature_finite"],
                "p_unsafe_drift": scored["p_unsafe_drift"],
                "p_unsafe_theft": scored["p_unsafe_theft"],
                "p_safe_object": scored["p_safe_object"],
                "missing_policy": scored["missing_policy"],
            })

        frame_score = float(min(object_safe_scores))
        tau = float(
            cfg["frame_aggregation"]["tau_admit"]
        )

        action = "ADMIT" if frame_score >= tau else "BLOCK"

        decision = {
            "frame_idx": frame_idx,
            "frame_score": frame_score,
            "tau_admit": tau,
            "action": action,
            "n_objects": len(object_ids),
            "n_finite_objects": sum(
                1
                for r in feature_rows[-len(object_ids):]
                if r["feature_finite"] == 1
            ),
            "global_present_before": "",
            "global_present_after": "",
            "per_object_present_before": "",
            "per_object_present_after": "",
            "global_bank_size_before": "",
            "global_bank_size_after": "",
            "frames_already_tracked_retained": "",
        }

        if action == "BLOCK":
            meta = evict_current_frame(
                state,
                frame_idx,
                len(object_ids),
            )
            decision.update(meta)

        decisions.append(decision)

        for oid in object_ids:
            prev_masks[oid] = current_masks[oid].copy()

    peak = float(torch.cuda.max_memory_allocated() / 1024 ** 3)
    runtime = float(time.time() - start)

    if len(predictions) != cfg["n_frames"]:
        raise RuntimeError(
            "Frame count mismatch: {}".format(len(predictions))
        )

    del state
    torch.cuda.empty_cache()

    return predictions, decisions, feature_rows, peak, runtime


def overlay(raw, masks, object_ids):
    arr = np.asarray(raw.convert("RGB")).astype(np.float32).copy()

    colors = [
        np.array([255, 0, 0], dtype=np.float32),
        np.array([0, 255, 0], dtype=np.float32),
        np.array([0, 128, 255], dtype=np.float32),
        np.array([255, 255, 0], dtype=np.float32),
    ]

    for i, oid in enumerate(object_ids):
        mask = masks[oid]
        color = colors[i % len(colors)]
        arr[mask] = 0.45 * arr[mask] + 0.55 * color

    return Image.fromarray(
        np.clip(arr, 0, 255).astype(np.uint8)
    )


def make_visual(
    path,
    raw,
    gt_labels,
    b0_masks,
    gated_masks,
    object_ids,
):
    gt_masks = {
        oid: (gt_labels == oid)
        for oid in object_ids
    }

    panels = [
        raw.convert("RGB"),
        overlay(raw, gt_masks, object_ids),
        overlay(raw, b0_masks, object_ids),
        overlay(raw, gated_masks, object_ids),
    ]

    labels = ["RAW", "GT", "B0", "B2CORE-GATED"]

    w, h = panels[0].size
    header = 26

    canvas = Image.new(
        "RGB",
        (w * len(panels), h + header),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (i * w, header))
        draw.text((i * w + 5, 5), label, fill="black")

    canvas.save(path)


def main():
    cfg = json.loads(CFG_PATH.read_text())

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(OUT)
        )

    sam_root = Path.home() / "thesis/externals/sam3"
    sam_commit = subprocess.check_output(
        ["git", "-C", str(sam_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    if sam_commit != cfg["expected_sam_commit"]:
        raise RuntimeError("SAM3 commit mismatch")

    heads, mean, std, model_sha = load_gate(cfg)

    data_root = Path(cfg["dataset_root"])
    jpg_dir = data_root / "JPEGImages" / cfg["video"]
    ann_dir = data_root / "Annotations" / cfg["video"]

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
        raise RuntimeError("Video shorter than configured scope")

    gt0 = np.array(Image.open(ann_dir / pngs[0]))

    object_ids = sorted(
        int(x)
        for x in np.unique(gt0)
        if int(x) != 0
    )

    if object_ids != cfg["expected_object_ids"]:
        raise RuntimeError(
            "Object IDs mismatch: {}".format(object_ids)
        )

    OUT.mkdir(parents=True)
    (OUT / "visuals").mkdir()

    print("video={}".format(cfg["video"]), flush=True)
    print("frames={}".format(cfg["n_frames"]), flush=True)
    print("object_ids={}".format(object_ids), flush=True)
    print("model_sha256={}".format(model_sha), flush=True)

    print("building frozen SAM3...", flush=True)
    model = build_sam3_video_model()
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    print("running B0...", flush=True)
    b0 = run_tracker(
        predictor=predictor,
        jpg_dir=jpg_dir,
        gt0=gt0,
        object_ids=object_ids,
        cfg=cfg,
        gated=False,
    )

    print("running learned B2-core gate...", flush=True)
    gated = run_tracker(
        predictor=predictor,
        jpg_dir=jpg_dir,
        gt0=gt0,
        object_ids=object_ids,
        cfg=cfg,
        gated=True,
        heads=heads,
        mean=mean,
        std=std,
    )

    decisions = gated[1]
    feature_rows = gated[2]

    block_rows = [
        row for row in decisions
        if row["action"] == "BLOCK"
    ]

    admit_rows = [
        row for row in decisions
        if row["action"] == "ADMIT"
    ]

    first_block = (
        int(block_rows[0]["frame_idx"])
        if block_rows
        else None
    )

    first_block_equal = None
    downstream_changed = False
    first_changed_frame = None
    total_xor_after_first_block = 0

    if first_block is not None:
        first_block_equal = all(
            np.array_equal(
                b0[0][first_block][oid],
                gated[0][first_block][oid],
            )
            for oid in object_ids
        )

        for frame_idx in range(first_block + 1, cfg["n_frames"]):
            frame_xor = 0

            for oid in object_ids:
                frame_xor += int(
                    np.logical_xor(
                        b0[0][frame_idx][oid],
                        gated[0][frame_idx][oid],
                    ).sum()
                )

            if frame_xor > 0:
                downstream_changed = True
                total_xor_after_first_block += frame_xor

                if first_changed_frame is None:
                    first_changed_frame = frame_idx

    frame_metrics = []
    visible_b0 = []
    visible_gated = []

    for frame_idx in range(cfg["n_frames"]):
        gt = np.array(
            Image.open(ann_dir / pngs[frame_idx])
        )

        for oid in object_ids:
            pb0 = b0[0][frame_idx][oid]
            pg = gated[0][frame_idx][oid]
            tgt = gt == oid

            ib0 = target_iou(pb0, tgt)
            ig = target_iou(pg, tgt)

            if ib0 is not None:
                visible_b0.append(ib0)
                visible_gated.append(ig)

            frame_metrics.append({
                "frame_idx": frame_idx,
                "object_id": oid,
                "target_visible": int(tgt.sum() > 0),
                "b0_target_iou": "" if ib0 is None else ib0,
                "gated_target_iou": "" if ig is None else ig,
                "b0_vs_gated_mask_iou": comparison_iou(pb0, pg),
                "xor_px": int(np.logical_xor(pb0, pg).sum()),
                "b0_pred_area_px": int(pb0.sum()),
                "gated_pred_area_px": int(pg.sum()),
            })

    for frame_idx in cfg["visual_frames"]:
        if frame_idx >= cfg["n_frames"]:
            continue

        raw = Image.open(jpg_dir / jpgs[frame_idx])
        gt = np.array(Image.open(ann_dir / pngs[frame_idx]))

        make_visual(
            OUT / "visuals" / "frame_{:05d}.png".format(frame_idx),
            raw,
            gt,
            b0[0][frame_idx],
            gated[0][frame_idx],
            object_ids,
        )

    every_block_present_before = all(
        row["global_present_before"] == 1
        and row["per_object_present_before"] == len(object_ids)
        for row in block_rows
    )

    every_block_absent_after = all(
        row["global_present_after"] == 0
        and row["per_object_present_after"] == 0
        for row in block_rows
    )

    frames_tracked_retained = all(
        row["frames_already_tracked_retained"] == 1
        for row in block_rows
    )

    mechanism_pass = all([
        len(block_rows) > 0,
        every_block_present_before,
        every_block_absent_after,
        frames_tracked_retained,
        first_block_equal is True,
        downstream_changed,
    ])

    nonfinite_object_rows = sum(
        row["feature_finite"] == 0
        for row in feature_rows
    )

    eligible_frames = cfg["n_frames"] - 1

    summary = {
        "status": (
            "LEARNED_GATE_CLOSED_LOOP_SANITY_PASS"
            if mechanism_pass
            else "LEARNED_GATE_CLOSED_LOOP_SANITY_FAIL"
        ),
        "scope": cfg["claim_boundary"],
        "video": cfg["video"],
        "frames": cfg["n_frames"],
        "object_ids": object_ids,
        "sam_commit": sam_commit,
        "model_sha256": model_sha,
        "tau_admit": cfg["frame_aggregation"]["tau_admit"],
        "dual_head_composition": cfg["dual_head_composition"],
        "missingness": cfg["missingness"],
        "eligible_nonconditioning_frames": eligible_frames,
        "block_count": len(block_rows),
        "admit_count": len(admit_rows),
        "descriptive_admit_fraction_nonconditioning": (
            len(admit_rows) / eligible_frames
        ),
        "nonfinite_object_feature_rows": nonfinite_object_rows,
        "first_block_frame": first_block,
        "first_block_prediction_equal_to_b0": first_block_equal,
        "downstream_mask_changed_after_first_block": downstream_changed,
        "first_changed_frame": first_changed_frame,
        "total_xor_px_after_first_block": total_xor_after_first_block,
        "every_block_present_before": every_block_present_before,
        "every_block_absent_after": every_block_absent_after,
        "frames_already_tracked_retained": frames_tracked_retained,
        "b0_mean_target_iou_visible_rows_descriptive": float(
            np.mean(visible_b0)
        ),
        "gated_mean_target_iou_visible_rows_descriptive": float(
            np.mean(visible_gated)
        ),
        "descriptive_mean_iou_delta_gated_minus_b0": float(
            np.mean(visible_gated) - np.mean(visible_b0)
        ),
        "b0_peak_vram_gb": b0[3],
        "gated_peak_vram_gb": gated[3],
        "b0_runtime_sec": b0[4],
        "gated_runtime_sec": gated[4],
        "dev_touched": 0,
        "test_touched": 0,
    }

    write_csv(
        OUT / "write_decisions.csv",
        [
            "frame_idx",
            "frame_score",
            "tau_admit",
            "action",
            "n_objects",
            "n_finite_objects",
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
        OUT / "object_gate_scores.csv",
        [
            "frame_idx",
            "object_id",
            "mask_conf_iou_head",
            "occ_score_logit",
            "pred_area_px",
            "area_norm",
            "frame0_anchor_area_px",
            "area_ratio_anchor",
            "temporal_iou_prev",
            "feature_finite",
            "p_unsafe_drift",
            "p_unsafe_theft",
            "p_safe_object",
            "missing_policy",
        ],
        feature_rows,
    )

    write_csv(
        OUT / "frame_metrics.csv",
        [
            "frame_idx",
            "object_id",
            "target_visible",
            "b0_target_iou",
            "gated_target_iou",
            "b0_vs_gated_mask_iou",
            "xor_px",
            "b0_pred_area_px",
            "gated_pred_area_px",
        ],
        frame_metrics,
    )

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
