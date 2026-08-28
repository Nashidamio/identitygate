from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

from sam3.model_builder import build_sam3_video_model


REPO = Path.home() / "thesis" / "identitygate"
SAM_REPO = Path.home() / "thesis" / "externals" / "sam3"

ROOT = Path("/mnt/d/thesis_data/mosev2/train")
FRAMES = ROOT / "JPEGImages"
ANN = ROOT / "Annotations"

CONFIG_PATH = REPO / "configs/EXP023-primitive-cache-v1.json"
SCOPE_PATH = REPO / "experiments/EXP021_relational_scope.json"

EXP022_ROOT = REPO / "experiments/EXP022_full/per_video"

SANITY_ROOT = REPO / "experiments/EXP023_sanity"


FIELDS = [
    "split",
    "video",
    "frame_idx",
    "obj_idx",
    "object_id",
    "n_objects",
    "frame_height",
    "frame_width",
    "baseline_memory_encoded",

    "mask_conf_iou_head",
    "occ_score_logit",
    "occ_score_prob",
    "pointer_valid",

    "pred_area_px",
    "area_norm",
    "frame0_anchor_area_px",
    "area_ratio_anchor",

    "temporal_iou_prev",
    "centroid_x_px",
    "centroid_y_px",
    "prev_centroid_x_px",
    "prev_centroid_y_px",

    "target_visible",
    "target_gt_area_px",
    "target_iou",
    "max_other_iou",
    "max_other_object_id",

    "ptr_sim_anchor_fp32",
]


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def cosine_matrix_fp32(current, anchors):
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_precision = torch.get_float32_matmul_precision()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    try:
        with torch.autocast(
            device_type=current.device.type,
            enabled=False,
        ):
            a = F.normalize(
                current.float(),
                dim=-1,
            )
            b = F.normalize(
                anchors.float(),
                dim=-1,
            )
            out = a @ b.T
    finally:
        torch.set_float32_matmul_precision(
            prev_precision
        )
        torch.backends.cuda.matmul.allow_tf32 = (
            prev_tf32
        )

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    return out


def binary_iou(a, b):
    union = np.logical_or(a, b).sum()

    if union == 0:
        return None

    return float(
        np.logical_and(a, b).sum()
        / union
    )


def centroid(mask):
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None, None

    return float(xs.mean()), float(ys.mean())


def csv_float(v):
    if v is None:
        return ""
    return v


def load_context():
    cfg = json.load(open(CONFIG_PATH))
    scope = json.load(open(SCOPE_PATH))

    assert sha256_file(SCOPE_PATH) == (
        cfg["scope"]["manifest_sha256"]
    )

    assert git_commit(SAM_REPO) == (
        cfg["substrate"]["sam_source_commit"]
    )

    train = [
        r
        for r in scope["eligible_videos"]
        if r["split"] == "TRAIN"
    ]

    assert len(train) == 18
    assert scope["test_videos_touched"] == 0

    checkpoint = Path(
        hf_hub_download(
            repo_id=cfg["substrate"]["hf_repo_id"],
            filename=cfg["substrate"][
                "checkpoint_filename"
            ],
            revision=cfg["substrate"]["hf_revision"],
            local_files_only=True,
        )
    )

    checkpoint_sha = sha256_file(checkpoint)

    assert checkpoint_sha == (
        cfg["substrate"]["checkpoint_sha256"]
    )

    return cfg, scope, train, checkpoint


def load_exp022_rows(video):
    path = (
        EXP022_ROOT
        / f"TRAIN_{video}"
        / "features.csv"
    )

    rows = {}

    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (
                int(r["frame_idx"]),
                int(r["object_id"]),
            )

            rows[key] = r

    return rows


def run_video(
    predictor,
    row,
    checkpoint,
    outroot,
):
    video = row["video"]
    split = row["split"]

    frame_dir = FRAMES / video
    ann_dir = ANN / video

    jpgs = sorted(frame_dir.glob("*.jpg"))
    pngs = sorted(ann_dir.glob("*.png"))

    n_frames = int(row["n_frames"])
    object_ids = [
        int(x)
        for x in row["frame0_object_ids"]
    ]
    n_objects = len(object_ids)

    assert len(jpgs) == n_frames
    assert len(pngs) == n_frames

    assert [
        p.stem for p in jpgs
    ] == [
        p.stem for p in pngs
    ]

    gt0 = np.array(Image.open(pngs[0]))

    actual_ids = sorted(
        int(x)
        for x in np.unique(gt0)
        if int(x) != 0
    )

    assert actual_ids == object_ids

    frame_h, frame_w = gt0.shape
    frame_area = frame_h * frame_w

    anchor_areas = {
        oid: int((gt0 == oid).sum())
        for oid in object_ids
    }

    for oid in object_ids:
        assert anchor_areas[oid] > 0

    state = predictor.init_state(
        video_path=str(frame_dir)
    )

    predictor.clear_all_points_in_video(
        state
    )

    for oid in object_ids:
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=0,
            obj_id=oid,
            mask=torch.from_numpy(
                gt0 == oid
            ),
        )

    assert list(state["obj_ids"]) == object_ids

    predictor.propagate_in_video_preflight(
        state,
        run_mem_encoder=True,
    )

    frame0_out = (
        state["output_dict"]
        ["cond_frame_outputs"][0]
    )

    anchors = (
        frame0_out["obj_ptr"]
        .detach()
    )

    assert tuple(anchors.shape) == (
        n_objects,
        predictor.hidden_dim,
    )

    anchor_ptrs = (
        anchors.float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    previous_pred = {
        oid: (gt0 == oid)
        for oid in object_ids
    }

    previous_centroid = {
        oid: centroid(previous_pred[oid])
        for oid in object_ids
    }

    rows = []

    pointer_rows = []
    pointer_frames = []
    pointer_objects = []

    exp022 = load_exp022_rows(video)

    xcheck_score_diffs = []
    xcheck_anchor_diffs = []
    xcheck_valid_equal = []

    start = time.time()

    torch.cuda.reset_peak_memory_stats()

    for out in predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=n_frames - 1,
        reverse=False,
        propagate_preflight=False,
    ):
        frame_idx = int(out[0])

        ids = [
            int(x)
            for x in out[1]
        ]

        assert ids == object_ids

        if frame_idx == 0:
            continue

        video_res = out[3]

        gt = np.array(
            Image.open(pngs[frame_idx])
        )

        assert gt.shape == (
            frame_h,
            frame_w,
        )

        gmask = {
            oid: (gt == oid)
            for oid in object_ids
        }

        current_out = (
            state["output_dict"]
            ["non_cond_frame_outputs"]
            [frame_idx]
        )

        assert (
            current_out["maskmem_features"]
            is not None
        )

        assert "iou_score" in current_out
        assert "object_score_logits" in current_out
        assert "obj_ptr" in current_out

        iou_scores = (
            current_out["iou_score"]
            .detach()
            .float()
            .reshape(-1)
        )

        object_logits = (
            current_out["object_score_logits"]
            .detach()
            .float()
            .reshape(-1)
        )

        ptr = (
            current_out["obj_ptr"]
            .detach()
        )

        assert tuple(iou_scores.shape) == (
            n_objects,
        )

        assert tuple(object_logits.shape) == (
            n_objects,
        )

        assert tuple(ptr.shape) == (
            n_objects,
            predictor.hidden_dim,
        )

        anchor_cos = cosine_matrix_fp32(
            ptr,
            anchors,
        )

        self_anchor = torch.diagonal(
            anchor_cos
        )

        for obj_idx, oid in enumerate(object_ids):
            pred = (
                video_res[obj_idx, 0] > 0
            ).detach().cpu().numpy()

            assert pred.shape == gt.shape

            target = gmask[oid]

            pred_area = int(pred.sum())
            gt_area = int(target.sum())

            target_iou = binary_iou(
                pred,
                target,
            )

            best_other = 0.0
            best_other_oid = None

            for o2 in object_ids:
                if o2 == oid:
                    continue

                val = binary_iou(
                    pred,
                    gmask[o2],
                )

                if val is not None and (
                    best_other_oid is None
                    or val > best_other
                ):
                    best_other = val
                    best_other_oid = o2

            temporal_iou = binary_iou(
                pred,
                previous_pred[oid],
            )

            cx, cy = centroid(pred)

            pcx, pcy = previous_centroid[oid]

            score = float(
                iou_scores[obj_idx].item()
            )

            logit = float(
                object_logits[obj_idx].item()
            )

            prob = float(
                torch.sigmoid(
                    object_logits[obj_idx]
                ).item()
            )

            pointer_valid = logit > 0.0

            ptr_anchor = float(
                self_anchor[obj_idx].item()
            )

            anchor_area = anchor_areas[oid]

            row_out = {
                "split": split,
                "video": video,
                "frame_idx": frame_idx,
                "obj_idx": obj_idx,
                "object_id": oid,
                "n_objects": n_objects,
                "frame_height": frame_h,
                "frame_width": frame_w,
                "baseline_memory_encoded": 1,

                "mask_conf_iou_head": score,
                "occ_score_logit": logit,
                "occ_score_prob": prob,
                "pointer_valid": int(
                    pointer_valid
                ),

                "pred_area_px": pred_area,
                "area_norm": (
                    pred_area / frame_area
                ),
                "frame0_anchor_area_px": (
                    anchor_area
                ),
                "area_ratio_anchor": (
                    pred_area / anchor_area
                ),

                "temporal_iou_prev": csv_float(
                    temporal_iou
                ),

                "centroid_x_px": csv_float(cx),
                "centroid_y_px": csv_float(cy),

                "prev_centroid_x_px": (
                    csv_float(pcx)
                ),
                "prev_centroid_y_px": (
                    csv_float(pcy)
                ),

                "target_visible": int(
                    gt_area > 0
                ),
                "target_gt_area_px": gt_area,
                "target_iou": csv_float(
                    target_iou
                ),
                "max_other_iou": best_other,
                "max_other_object_id": (
                    ""
                    if best_other_oid is None
                    else best_other_oid
                ),

                "ptr_sim_anchor_fp32": (
                    ptr_anchor
                    if pointer_valid
                    else ""
                ),
            }

            rows.append(row_out)

            pointer_rows.append(
                ptr[obj_idx]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            pointer_frames.append(frame_idx)
            pointer_objects.append(oid)

            key = (
                frame_idx,
                oid,
            )

            assert key in exp022

            old = exp022[key]

            old_logit = float(
                old["object_score_logit"]
            )

            old_valid = int(
                old["pointer_valid"]
            )

            xcheck_score_diffs.append(
                abs(logit - old_logit)
            )

            xcheck_valid_equal.append(
                int(pointer_valid) == old_valid
            )

            if pointer_valid:
                old_anchor = float(
                    old[
                        "self_anchor_cos_fp32"
                    ]
                )

                xcheck_anchor_diffs.append(
                    abs(
                        ptr_anchor
                        - old_anchor
                    )
                )

            previous_pred[oid] = pred
            previous_centroid[oid] = (
                cx,
                cy,
            )

    expected_rows = (
        (n_frames - 1)
        * n_objects
    )

    assert len(rows) == expected_rows
    assert len(pointer_rows) == expected_rows

    outroot.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = outroot / "primitives.csv"

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    pointers_path = (
        outroot / "pointers.npz"
    )

    np.savez_compressed(
        pointers_path,
        frame_idx=np.asarray(
            pointer_frames,
            dtype=np.int32,
        ),
        object_id=np.asarray(
            pointer_objects,
            dtype=np.int32,
        ),
        ptr_f32=np.stack(
            pointer_rows,
            axis=0,
        ).astype(np.float32),
        anchor_object_id=np.asarray(
            object_ids,
            dtype=np.int32,
        ),
        anchor_ptr_f32=anchor_ptrs,
    )

    torch.cuda.synchronize()

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024 ** 3
    )

    summary = {
        "experiment": "EXP023",
        "status": "SANITY_PASS",
        "identitygate_commit_at_execution": (
            git_commit(REPO)
        ),
        "sam3_commit": git_commit(
            SAM_REPO
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": (
            sha256_file(checkpoint)
        ),
        "split": split,
        "video": video,
        "n_frames": n_frames,
        "n_objects": n_objects,
        "candidate_rows": len(rows),
        "pointer_shape": list(
            np.stack(
                pointer_rows,
                axis=0,
            ).shape
        ),
        "anchor_pointer_shape": list(
            anchor_ptrs.shape
        ),
        "dev_videos_touched": 0,
        "test_videos_touched": 0,
        "mask_rule": "video_res > 0",
        "feature7_reference": (
            "OPEN_NOT_DEFINED_BY_PLAN"
        ),
        "exp022_object_score_max_abs_diff": (
            max(xcheck_score_diffs)
        ),
        "exp022_pointer_valid_all_equal": (
            all(xcheck_valid_equal)
        ),
        "exp022_valid_anchor_cos_max_abs_diff": (
            max(xcheck_anchor_diffs)
            if xcheck_anchor_diffs
            else None
        ),
        "peak_memory_allocated_gb": peak_gb,
        "runtime_sec": time.time() - start,
        "primitives_sha256": (
            sha256_file(csv_path)
        ),
        "pointers_sha256": (
            sha256_file(pointers_path)
        ),
    }

    summary_path = (
        outroot / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("EXP023 SANITY RESULT")
    for k, v in summary.items():
        print(k, "=", v)

    print()
    print("EXP023_SANITY_PASS")

    del state
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["sanity"],
    )

    args = parser.parse_args()

    cfg, scope, train, checkpoint = (
        load_context()
    )

    if args.mode == "sanity":
        video = cfg["scope"][
            "sanity_video"
        ]

        matches = [
            r
            for r in train
            if r["video"] == video
        ]

        assert len(matches) == 1

        print("EXP023 PRIMITIVE CACHE SANITY")
        print(
            "identitygate_head =",
            git_commit(REPO),
        )
        print(
            "sam3_commit =",
            git_commit(SAM_REPO),
        )
        print(
            "video =",
            video,
        )
        print(
            "frames_dir =",
            FRAMES / video,
        )
        print(
            "annotations_dir =",
            ANN / video,
        )
        print("dev_videos_touched = 0")
        print("test_videos_touched = 0")

        model = build_sam3_video_model(
            checkpoint_path=str(
                checkpoint
            ),
            load_from_HF=False,
        )

        predictor = model.tracker
        predictor.backbone = (
            model.detector.backbone
        )

        print(
            "predictor_class =",
            type(predictor).__name__,
        )

        print(
            "use_memory_selection =",
            predictor.use_memory_selection,
        )

        assert predictor.use_memory_selection

        run_video(
            predictor,
            matches[0],
            checkpoint,
            SANITY_ROOT,
        )


if __name__ == "__main__":
    main()
