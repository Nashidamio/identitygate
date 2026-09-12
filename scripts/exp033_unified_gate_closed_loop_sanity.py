import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP033-unified-gate-closed-loop-sanity-v1.json"
OUT = ROOT / "experiments/EXP033_unified_gate_closed_loop_sanity"


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
        raise RuntimeError(
            "Cannot load source module: {}".format(path)
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def verify_dependency_hashes(cfg):
    for name, item in cfg["source_dependencies"].items():
        path = ROOT / item["path"]

        actual = sha256sum(path)

        if actual != item["sha256"]:
            raise RuntimeError(
                "{} SHA mismatch: {}".format(
                    name,
                    actual,
                )
            )

    p = cfg["protocol_artifacts"]

    checks = [
        (
            "A5",
            p["a5_path"],
            p["a5_sha256"],
        ),
        (
            "A6",
            p["a6_path"],
            p["a6_sha256"],
        ),
        (
            "B1 rule",
            p["b1_rule_path"],
            p["b1_rule_sha256"],
        ),
    ]

    for name, rel, expected in checks:
        actual = sha256sum(ROOT / rel)

        if actual != expected:
            raise RuntimeError(
                "{} SHA mismatch: {}".format(
                    name,
                    actual,
                )
            )


def load_dependencies(cfg):
    verify_dependency_hashes(cfg)

    exp023 = load_module(
        "exp023_frozen",
        ROOT / cfg["source_dependencies"][
            "exp023_primitive_source"
        ]["path"],
    )

    exp030 = load_module(
        "exp030_frozen",
        ROOT / cfg["source_dependencies"][
            "exp030_b2_closed_loop_source"
        ]["path"],
    )

    exp032 = load_module(
        "exp032_frozen",
        ROOT / cfg["source_dependencies"][
            "exp032_b1_closed_loop_source"
        ]["path"],
    )

    return {
        "exp023": exp023,
        "exp030": exp030,
        "exp032": exp032,
    }


def load_heads(exp030, head_artifacts, input_dim):
    heads = {}

    for name in ["drift", "theft"]:
        artifact = head_artifacts[name]

        model = exp030.GateMLP(input_dim).double()

        state = {
            key: torch.tensor(
                value,
                dtype=torch.float64,
            )
            for key, value in artifact[
                "state_dict"
            ].items()
        }

        model.load_state_dict(
            state,
            strict=True,
        )

        model.eval()

        actual_params = sum(
            p.numel()
            for p in model.parameters()
        )

        if actual_params != artifact[
            "parameter_count"
        ]:
            raise RuntimeError(
                "{} parameter-count mismatch".format(
                    name
                )
            )

        heads[name] = model

    return heads


def make_pack(
    exp030,
    features,
    normalization,
    heads_artifact,
):
    mean = np.asarray(
        normalization["mean"],
        dtype=np.float64,
    )

    std = np.asarray(
        normalization["std"],
        dtype=np.float64,
    )

    if mean.shape != (len(features),):
        raise RuntimeError(
            "Normalizer mean shape mismatch"
        )

    if std.shape != mean.shape:
        raise RuntimeError(
            "Normalizer std shape mismatch"
        )

    if not np.all(np.isfinite(mean)):
        raise RuntimeError(
            "Non-finite normalizer mean"
        )

    if not np.all(np.isfinite(std)):
        raise RuntimeError(
            "Non-finite normalizer std"
        )

    if np.any(std <= 0):
        raise RuntimeError(
            "Non-positive normalizer std"
        )

    heads = load_heads(
        exp030,
        heads_artifact,
        len(features),
    )

    return {
        "features": list(features),
        "mean": mean,
        "std": std,
        "heads": heads,
    }


def load_model_packs(cfg, deps):
    b2_path = ROOT / cfg["models"]["B2"]["path"]
    b3_path = ROOT / cfg["models"]["B3"]["path"]

    b2_sha = sha256sum(b2_path)
    b3_sha = sha256sum(b3_path)

    if b2_sha != cfg["models"]["B2"]["sha256"]:
        raise RuntimeError(
            "B2 model SHA mismatch"
        )

    if b3_sha != cfg["models"]["B3"]["sha256"]:
        raise RuntimeError(
            "B3 model SHA mismatch"
        )

    b2 = json.loads(b2_path.read_text())
    b3 = json.loads(b3_path.read_text())

    if b2["status"] != (
        "TRAIN_ONLY_B2_CORE_FAILURE_TYPED_WEIGHTS"
    ):
        raise RuntimeError(
            "Unexpected B2 artifact status"
        )

    if b3["status"] != (
        "TRAIN_ONLY_B3_DEVELOPMENT_WEIGHTS"
    ):
        raise RuntimeError(
            "Unexpected B3 artifact status"
        )

    if b3["pointer_valid_is_feature"] is not False:
        raise RuntimeError(
            "pointer_valid became a predictive feature"
        )

    if b2["features"] != cfg["features"]["B2"]:
        raise RuntimeError(
            "B2 feature mismatch"
        )

    for variant in ["B3_S", "B3_R"]:
        if (
            b3["variants"][variant]["features"]
            != cfg["features"][variant]
        ):
            raise RuntimeError(
                "{} feature mismatch".format(
                    variant
                )
            )

    return {
        "B2": make_pack(
            deps["exp030"],
            b2["features"],
            b2["normalization"],
            b2["heads"],
        ),
        "B3_S": make_pack(
            deps["exp030"],
            b3["variants"]["B3_S"]["features"],
            b3["variants"]["B3_S"][
                "normalization"
            ],
            b3["variants"]["B3_S"]["heads"],
        ),
        "B3_R": make_pack(
            deps["exp030"],
            b3["variants"]["B3_R"]["features"],
            b3["variants"]["B3_R"][
                "normalization"
            ],
            b3["variants"]["B3_R"]["heads"],
        ),
        "b2_sha256": b2_sha,
        "b3_sha256": b3_sha,
    }


def score_pack(values, pack):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if not bool(np.all(np.isfinite(x))):
        return {
            "finite": 0,
            "p_unsafe_drift": "",
            "p_unsafe_theft": "",
            "safe_score": 0.0,
            "missing_policy": "FAIL_CLOSED",
        }

    z = (
        x - pack["mean"]
    ) / pack["std"]

    xt = torch.from_numpy(
        z
    ).reshape(1, -1)

    with torch.no_grad():
        p_drift = float(
            torch.sigmoid(
                pack["heads"]["drift"](xt)
            )[0].item()
        )

        p_theft = float(
            torch.sigmoid(
                pack["heads"]["theft"](xt)
            )[0].item()
        )

    if not math.isfinite(p_drift):
        raise RuntimeError(
            "Non-finite drift probability"
        )

    if not math.isfinite(p_theft):
        raise RuntimeError(
            "Non-finite theft probability"
        )

    safe = min(
        1.0 - p_drift,
        1.0 - p_theft,
    )

    return {
        "finite": 1,
        "p_unsafe_drift": p_drift,
        "p_unsafe_theft": p_theft,
        "safe_score": float(safe),
        "missing_policy": "",
    }


def score_learned_variant(
    variant,
    base_features,
    pointer_valid,
    self_identity,
    competitor_identity,
    packs,
):
    if not all(
        math.isfinite(float(x))
        for x in base_features
    ):
        return {
            "base_finite": 0,
            "used_variant": "FAIL_CLOSED_BASE",
            "p_unsafe_drift": "",
            "p_unsafe_theft": "",
            "safe_score": 0.0,
            "missing_policy": "FAIL_CLOSED",
        }

    if variant == "B2":
        used = "B2"
        values = base_features

    elif variant == "B3_S":
        if not pointer_valid:
            used = "B2"
            values = base_features
        else:
            if (
                self_identity is None
                or not math.isfinite(
                    self_identity
                )
            ):
                raise RuntimeError(
                    "Valid pointer has invalid self identity"
                )

            used = "B3_S"
            values = (
                base_features
                + [self_identity]
            )

    elif variant == "B3_R":
        if not pointer_valid:
            used = "B2"
            values = base_features

        else:
            if (
                self_identity is None
                or not math.isfinite(
                    self_identity
                )
            ):
                raise RuntimeError(
                    "Valid pointer has invalid self identity"
                )

            if competitor_identity is None:
                used = "B3_S"
                values = (
                    base_features
                    + [self_identity]
                )

            else:
                if not math.isfinite(
                    competitor_identity
                ):
                    raise RuntimeError(
                        "Available competitor identity is non-finite"
                    )

                used = "B3_R"
                values = (
                    base_features
                    + [
                        self_identity,
                        competitor_identity,
                    ]
                )

    else:
        raise RuntimeError(
            "Unknown learned variant: {}".format(
                variant
            )
        )

    scored = score_pack(
        values,
        packs[used],
    )

    if scored["finite"] != 1:
        raise RuntimeError(
            "Finite routed features became non-finite"
        )

    return {
        "base_finite": 1,
        "used_variant": used,
        "p_unsafe_drift": scored[
            "p_unsafe_drift"
        ],
        "p_unsafe_theft": scored[
            "p_unsafe_theft"
        ],
        "safe_score": scored[
            "safe_score"
        ],
        "missing_policy": scored[
            "missing_policy"
        ],
    }


def prepare_state(
    predictor,
    jpg_dir,
    gt0,
    object_ids,
    deps,
):
    state = predictor.init_state(
        video_path=str(jpg_dir)
    )

    predictor.clear_all_points_in_video(
        state
    )

    deps["exp032"].add_frame0_prompts(
        predictor,
        state,
        gt0,
        object_ids,
    )

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

    if tuple(anchors.shape) != (
        len(object_ids),
        predictor.hidden_dim,
    ):
        raise RuntimeError(
            "Frame-0 anchor shape mismatch"
        )

    if not bool(
        torch.isfinite(
            anchors.float()
        ).all().item()
    ):
        raise RuntimeError(
            "Non-finite frame-0 anchor"
        )

    return state, anchors


def run_tracker(
    predictor,
    jpg_dir,
    gt0,
    object_ids,
    cfg,
    variant,
    packs,
    deps,
):
    state, anchors = prepare_state(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        deps,
    )

    predictions = {
        0: {
            oid: (
                gt0 == oid
            ).astype(bool)
            for oid in object_ids
        }
    }

    decisions = []
    object_rows = []

    anchor_areas = {
        oid: int(
            (gt0 == oid).sum()
        )
        for oid in object_ids
    }

    if any(
        x <= 0
        for x in anchor_areas.values()
    ):
        raise RuntimeError(
            "Frame-0 anchor area is zero"
        )

    prev_masks = {
        oid: (
            gt0 == oid
        ).astype(bool)
        for oid in object_ids
    }

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    generator = predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=(
            cfg["scope"]["n_frames"] - 1
        ),
        reverse=False,
        propagate_preflight=False,
    )

    for out in generator:
        frame_idx = int(out[0])

        ids = [
            int(x)
            for x in out[1]
        ]

        if ids != object_ids:
            raise RuntimeError(
                "Object order changed at frame {}".format(
                    frame_idx
                )
            )

        video_res = out[3]

        current_masks = {}

        for row, oid in enumerate(ids):
            current_masks[oid] = (
                (
                    video_res[row, 0] > 0
                )
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

        if variant == "B0":
            for oid in object_ids:
                prev_masks[oid] = (
                    current_masks[oid].copy()
                )
            continue

        current_out = (
            state["output_dict"]
            ["non_cond_frame_outputs"]
            .get(frame_idx)
        )

        if current_out is None:
            raise RuntimeError(
                "Missing non-conditioning output at frame {}".format(
                    frame_idx
                )
            )

        iou_scores = (
            current_out["iou_score"]
            .detach()
            .float()
            .reshape(-1)
        )

        object_logits = (
            current_out[
                "object_score_logits"
            ]
            .detach()
            .float()
            .reshape(-1)
        )

        if tuple(iou_scores.shape) != (
            len(object_ids),
        ):
            raise RuntimeError(
                "IoU-score shape mismatch"
            )

        if tuple(object_logits.shape) != (
            len(object_ids),
        ):
            raise RuntimeError(
                "Object-logit shape mismatch"
            )

        self_values = None
        competitor_values = None

        if variant in ["B3_S", "B3_R"]:
            ptr = (
                current_out["obj_ptr"]
                .detach()
            )

            if tuple(ptr.shape) != (
                len(object_ids),
                predictor.hidden_dim,
            ):
                raise RuntimeError(
                    "Current obj_ptr shape mismatch"
                )

            cos = deps[
                "exp023"
            ].cosine_matrix_fp32(
                ptr,
                anchors,
            )

            self_values = torch.diagonal(
                cos
            )

            if len(object_ids) > 1:
                eye = torch.eye(
                    len(object_ids),
                    dtype=torch.bool,
                    device=cos.device,
                )

                competitor_values = (
                    cos.masked_fill(
                        eye,
                        float("-inf"),
                    )
                    .max(dim=1)
                    .values
                )

        object_safe_scores = []
        frame_rows = []

        for row, oid in enumerate(ids):
            mask = current_masks[oid]

            pred_area = int(
                mask.sum()
            )

            frame_area = int(
                mask.size
            )

            area_norm = (
                pred_area / frame_area
            )

            area_ratio = (
                pred_area
                / anchor_areas[oid]
            )

            mask_conf = float(
                iou_scores[row].item()
            )

            occ_logit = float(
                object_logits[row].item()
            )

            temporal = deps[
                "exp032"
            ].binary_iou(
                mask,
                prev_masks[oid],
            )

            pointer_valid = (
                occ_logit > 0.0
            )

            self_identity = None
            competitor_identity = None

            if variant in ["B3_S", "B3_R"]:
                if pointer_valid:
                    self_identity = float(
                        self_values[row].item()
                    )

                    if competitor_values is not None:
                        competitor_identity = float(
                            competitor_values[
                                row
                            ].item()
                        )

            if variant == "B1":
                b1 = deps[
                    "exp032"
                ].score_object([
                    mask_conf,
                    occ_logit,
                    area_ratio,
                    temporal,
                ])

                safe_score = float(
                    b1["b1_score"]
                )

                scored = {
                    "base_finite": int(
                        b1[
                            "feature_finite"
                        ]
                    ),
                    "used_variant": "B1",
                    "p_unsafe_drift": "",
                    "p_unsafe_theft": "",
                    "safe_score": safe_score,
                    "missing_policy": b1[
                        "missing_policy"
                    ],
                }

            else:
                scored = score_learned_variant(
                    variant=variant,
                    base_features=[
                        mask_conf,
                        occ_logit,
                        area_norm,
                        area_ratio,
                        temporal,
                    ],
                    pointer_valid=(
                        pointer_valid
                    ),
                    self_identity=(
                        self_identity
                    ),
                    competitor_identity=(
                        competitor_identity
                    ),
                    packs=packs,
                )

                safe_score = float(
                    scored[
                        "safe_score"
                    ]
                )

            object_safe_scores.append(
                safe_score
            )

            object_row = {
                "variant": variant,
                "frame_idx": frame_idx,
                "object_id": oid,
                "mask_conf_iou_head": (
                    mask_conf
                ),
                "occ_score_logit": (
                    occ_logit
                ),
                "area_norm": area_norm,
                "area_ratio_anchor": (
                    area_ratio
                ),
                "temporal_iou_prev": (
                    ""
                    if not math.isfinite(
                        temporal
                    )
                    else temporal
                ),
                "base_finite": scored[
                    "base_finite"
                ],
                "pointer_valid": (
                    ""
                    if variant
                    not in [
                        "B3_S",
                        "B3_R",
                    ]
                    else int(
                        pointer_valid
                    )
                ),
                "ptr_sim_anchor_fp32": (
                    ""
                    if self_identity is None
                    else self_identity
                ),
                "max_comp_anchor_cos_fp32": (
                    ""
                    if competitor_identity is None
                    else competitor_identity
                ),
                "used_variant": scored[
                    "used_variant"
                ],
                "p_unsafe_drift": scored[
                    "p_unsafe_drift"
                ],
                "p_unsafe_theft": scored[
                    "p_unsafe_theft"
                ],
                "object_safe_score": (
                    safe_score
                ),
                "missing_policy": scored[
                    "missing_policy"
                ],
            }

            object_rows.append(
                object_row
            )

            frame_rows.append(
                object_row
            )

        frame_score = float(
            min(
                object_safe_scores
            )
        )

        tau = float(
            cfg["frame_intervention"][
                "tau"
            ]
        )

        action = (
            "ADMIT"
            if frame_score >= tau
            else "BLOCK"
        )

        decision = {
            "frame_idx": frame_idx,
            "frame_score": frame_score,
            "tau": tau,
            "action": action,
            "n_objects": len(
                object_ids
            ),
            "n_finite_objects": sum(
                int(
                    r[
                        "base_finite"
                    ]
                )
                for r in frame_rows
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
            meta = deps[
                "exp032"
            ].evict_current_frame(
                state,
                frame_idx,
                len(object_ids),
            )

            decision.update(meta)

        decisions.append(
            decision
        )

        for oid in object_ids:
            prev_masks[oid] = (
                current_masks[oid].copy()
            )

    if len(decisions) != (
        cfg["scope"]["n_frames"] - 1
    ) and variant != "B0":
        raise RuntimeError(
            "Decision-count mismatch for {}".format(
                variant
            )
        )

    torch.cuda.synchronize()

    peak = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    runtime = (
        time.time() - start
    )

    del state
    torch.cuda.empty_cache()

    return {
        "predictions": predictions,
        "decisions": decisions,
        "object_rows": object_rows,
        "peak_vram_gb": peak,
        "runtime_sec": runtime,
    }


def evaluate_variant(
    variant,
    b0,
    gated,
    ann_dir,
    pngs,
    object_ids,
    cfg,
    deps,
):
    decisions = gated[
        "decisions"
    ]

    object_rows = gated[
        "object_rows"
    ]

    block_rows = [
        r
        for r in decisions
        if r["action"] == "BLOCK"
    ]

    admit_rows = [
        r
        for r in decisions
        if r["action"] == "ADMIT"
    ]

    first_block = (
        int(
            block_rows[0][
                "frame_idx"
            ]
        )
        if block_rows
        else None
    )

    first_block_equal = None
    downstream_changed = False
    first_changed_frame = None
    total_xor = 0

    if first_block is not None:
        first_block_equal = all(
            np.array_equal(
                b0[
                    "predictions"
                ][first_block][oid],
                gated[
                    "predictions"
                ][first_block][oid],
            )
            for oid in object_ids
        )

        for frame_idx in range(
            first_block + 1,
            cfg["scope"]["n_frames"],
        ):
            frame_xor = 0

            for oid in object_ids:
                frame_xor += int(
                    np.logical_xor(
                        b0[
                            "predictions"
                        ][frame_idx][oid],
                        gated[
                            "predictions"
                        ][frame_idx][oid],
                    ).sum()
                )

            if frame_xor > 0:
                downstream_changed = True
                total_xor += (
                    frame_xor
                )

                if (
                    first_changed_frame
                    is None
                ):
                    first_changed_frame = (
                        frame_idx
                    )

    every_present = all(
        int(
            r[
                "global_present_before"
            ]
        ) == 1
        and int(
            r[
                "per_object_present_before"
            ]
        ) == len(object_ids)
        for r in block_rows
    )

    every_absent = all(
        int(
            r[
                "global_present_after"
            ]
        ) == 0
        and int(
            r[
                "per_object_present_after"
            ]
        ) == 0
        for r in block_rows
    )

    tracked_retained = all(
        int(
            r[
                "frames_already_tracked_retained"
            ]
        ) == 1
        for r in block_rows
    )

    b0_visible = []
    gated_visible = []

    for frame_idx in range(
        cfg["scope"]["n_frames"]
    ):
        gt = np.array(
            Image.open(
                ann_dir
                / pngs[frame_idx]
            )
        )

        for oid in object_ids:
            target = (
                gt == oid
            )

            b0_iou = deps[
                "exp032"
            ].target_iou(
                b0[
                    "predictions"
                ][frame_idx][oid],
                target,
            )

            gated_iou = deps[
                "exp032"
            ].target_iou(
                gated[
                    "predictions"
                ][frame_idx][oid],
                target,
            )

            if b0_iou is not None:
                b0_visible.append(
                    b0_iou
                )
                gated_visible.append(
                    gated_iou
                )

    route_counts = {}

    for row in object_rows:
        key = row[
            "used_variant"
        ]

        route_counts[key] = (
            route_counts.get(
                key,
                0,
            )
            + 1
        )

    pointer_rows = [
        r
        for r in object_rows
        if r["pointer_valid"] != ""
    ]

    pointer_valid_rows = sum(
        int(
            r["pointer_valid"]
        )
        for r in pointer_rows
    )

    mechanism_pass = all([
        len(block_rows) > 0,
        every_present,
        every_absent,
        tracked_retained,
        first_block_equal is True,
        downstream_changed,
    ])

    return {
        "status": (
            "{}_CLOSED_LOOP_SANITY_PASS".format(
                variant
            )
            if mechanism_pass
            else "{}_CLOSED_LOOP_SANITY_FAIL".format(
                variant
            )
        ),
        "mechanism_pass": (
            mechanism_pass
        ),
        "eligible_nonconditioning_frames": (
            cfg["scope"]["n_frames"]
            - 1
        ),
        "admit_count": len(
            admit_rows
        ),
        "block_count": len(
            block_rows
        ),
        "admit_fraction": (
            len(admit_rows)
            / (
                cfg["scope"][
                    "n_frames"
                ]
                - 1
            )
        ),
        "first_block_frame": (
            first_block
        ),
        "first_block_prediction_equal_to_b0": (
            first_block_equal
        ),
        "first_changed_frame": (
            first_changed_frame
        ),
        "downstream_mask_changed_after_first_block": (
            downstream_changed
        ),
        "total_xor_px_after_first_block": (
            total_xor
        ),
        "every_block_present_before": (
            every_present
        ),
        "every_block_absent_after": (
            every_absent
        ),
        "frames_already_tracked_retained": (
            tracked_retained
        ),
        "nonfinite_base_rows": sum(
            int(
                r["base_finite"]
            ) == 0
            for r in object_rows
        ),
        "pointer_rows": len(
            pointer_rows
        ),
        "pointer_valid_rows": (
            pointer_valid_rows
        ),
        "routing_counts": (
            route_counts
        ),
        "b0_mean_target_iou_visible_rows_descriptive": (
            float(
                np.mean(
                    b0_visible
                )
            )
        ),
        "gated_mean_target_iou_visible_rows_descriptive": (
            float(
                np.mean(
                    gated_visible
                )
            )
        ),
        "descriptive_mean_iou_delta_gated_minus_b0": (
            float(
                np.mean(
                    gated_visible
                )
                - np.mean(
                    b0_visible
                )
            )
        ),
        "peak_vram_gb": gated[
            "peak_vram_gb"
        ],
        "runtime_sec": gated[
            "runtime_sec"
        ],
    }


def main():
    cfg = json.loads(
        CFG_PATH.read_text()
    )

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(
                OUT
            )
        )

    deps = load_dependencies(
        cfg
    )

    packs = load_model_packs(
        cfg,
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
        raise RuntimeError(
            "SAM3 commit mismatch"
        )

    data_root = Path(
        cfg["scope"][
            "dataset_root"
        ]
    )

    jpg_dir = (
        data_root
        / "JPEGImages"
        / cfg["scope"]["video"]
    )

    ann_dir = (
        data_root
        / "Annotations"
        / cfg["scope"]["video"]
    )

    jpgs = sorted(
        [
            x
            for x in os.listdir(
                jpg_dir
            )
            if x.endswith(".jpg")
        ],
        key=lambda x: int(
            Path(x).stem
        ),
    )

    pngs = sorted(
        [
            x
            for x in os.listdir(
                ann_dir
            )
            if x.endswith(".png")
        ],
        key=lambda x: int(
            Path(x).stem
        ),
    )

    if len(jpgs) != len(pngs):
        raise RuntimeError(
            "JPEG and annotation count mismatch"
        )

    if len(jpgs) < cfg[
        "scope"
    ]["n_frames"]:
        raise RuntimeError(
            "Video shorter than configured scope"
        )

    gt0 = np.array(
        Image.open(
            ann_dir / pngs[0]
        )
    )

    object_ids = sorted(
        int(x)
        for x in np.unique(
            gt0
        )
        if int(x) != 0
    )

    if object_ids != cfg[
        "scope"
    ]["expected_object_ids"]:
        raise RuntimeError(
            "Object IDs mismatch: {}".format(
                object_ids
            )
        )

    OUT.mkdir(
        parents=True
    )

    print(
        "video={}".format(
            cfg["scope"]["video"]
        ),
        flush=True,
    )

    print(
        "frames={}".format(
            cfg["scope"]["n_frames"]
        ),
        flush=True,
    )

    print(
        "object_ids={}".format(
            object_ids
        ),
        flush=True,
    )

    print(
        "b2_model_sha256={}".format(
            packs["b2_sha256"]
        ),
        flush=True,
    )

    print(
        "b3_model_sha256={}".format(
            packs["b3_sha256"]
        ),
        flush=True,
    )

    print(
        "building frozen SAM3...",
        flush=True,
    )

    model = build_sam3_video_model()

    predictor = model.tracker
    predictor.backbone = (
        model.detector.backbone
    )

    print(
        "running B0...",
        flush=True,
    )

    b0 = run_tracker(
        predictor=predictor,
        jpg_dir=jpg_dir,
        gt0=gt0,
        object_ids=object_ids,
        cfg=cfg,
        variant="B0",
        packs=packs,
        deps=deps,
    )

    decision_fields = [
        "frame_idx",
        "frame_score",
        "tau",
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
    ]

    object_fields = [
        "variant",
        "frame_idx",
        "object_id",
        "mask_conf_iou_head",
        "occ_score_logit",
        "area_norm",
        "area_ratio_anchor",
        "temporal_iou_prev",
        "base_finite",
        "pointer_valid",
        "ptr_sim_anchor_fp32",
        "max_comp_anchor_cos_fp32",
        "used_variant",
        "p_unsafe_drift",
        "p_unsafe_theft",
        "object_safe_score",
        "missing_policy",
    ]

    summaries = {}

    for variant in cfg[
        "variants"
    ]:
        print(
            "running {}...".format(
                variant
            ),
            flush=True,
        )

        gated = run_tracker(
            predictor=predictor,
            jpg_dir=jpg_dir,
            gt0=gt0,
            object_ids=object_ids,
            cfg=cfg,
            variant=variant,
            packs=packs,
            deps=deps,
        )

        summary = evaluate_variant(
            variant=variant,
            b0=b0,
            gated=gated,
            ann_dir=ann_dir,
            pngs=pngs,
            object_ids=object_ids,
            cfg=cfg,
            deps=deps,
        )

        summaries[
            variant
        ] = summary

        deps[
            "exp032"
        ].write_csv(
            OUT
            / "{}_write_decisions.csv".format(
                variant
            ),
            decision_fields,
            gated[
                "decisions"
            ],
        )

        deps[
            "exp032"
        ].write_csv(
            OUT
            / "{}_object_scores.csv".format(
                variant
            ),
            object_fields,
            gated[
                "object_rows"
            ],
        )

        print(
            "{} status={} admit={} block={} routes={}".format(
                variant,
                summary[
                    "status"
                ],
                summary[
                    "admit_count"
                ],
                summary[
                    "block_count"
                ],
                summary[
                    "routing_counts"
                ],
            ),
            flush=True,
        )

    overall_pass = all(
        summaries[v][
            "mechanism_pass"
        ]
        for v in cfg[
            "variants"
        ]
    )

    result = {
        "experiment": "EXP033",
        "status": (
            "UNIFIED_GATE_CLOSED_LOOP_SANITY_PASS"
            if overall_pass
            else "UNIFIED_GATE_CLOSED_LOOP_SANITY_FAIL"
        ),
        "video": cfg[
            "scope"
        ]["video"],
        "frames": cfg[
            "scope"
        ]["n_frames"],
        "object_ids": (
            object_ids
        ),
        "sam_commit": (
            sam_commit
        ),
        "b0_peak_vram_gb": (
            b0[
                "peak_vram_gb"
            ]
        ),
        "b0_runtime_sec": (
            b0[
                "runtime_sec"
            ]
        ),
        "b2_model_sha256": (
            packs[
                "b2_sha256"
            ]
        ),
        "b3_model_sha256": (
            packs[
                "b3_sha256"
            ]
        ),
        "tau": cfg[
            "frame_intervention"
        ]["tau"],
        "tau_status": cfg[
            "frame_intervention"
        ]["tau_status"],
        "variants": (
            summaries
        ),
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
        + "\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    if not overall_pass:
        raise RuntimeError(
            "EXP033 mechanism acceptance failed"
        )


if __name__ == "__main__":
    main()
