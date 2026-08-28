from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import os
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

FRAMES = Path("/mnt/d/thesis_data/mosev2/train/JPEGImages")
ANN = Path("/mnt/d/thesis_data/mosev2/train/Annotations")

CONFIG_PATH = REPO / "configs/EXP022-relational-census-v1.json"
SCOPE_PATH = REPO / "experiments/EXP021_relational_scope.json"
SUBSTRATE_PATH = REPO / "configs/SUBSTRATE-v1.json"

OUTDIR = REPO / "experiments/EXP022_full"
PER_VIDEO_DIR = OUTDIR / "per_video"

FEATURES_PATH = OUTDIR / "features.csv"
ANCHORS_PATH = OUTDIR / "anchor_cosine.csv"
SUMMARY_PATH = OUTDIR / "summary.json"

FEATURE_FIELDS = [
    "split",
    "video",
    "frame_idx",
    "obj_idx",
    "object_id",
    "n_objects",
    "baseline_memory_encoded",
    "pointer_valid",
    "pointer_is_no_obj_sentinel",
    "object_score_logit",
    "object_score_prob",
    "self_anchor_cos_fp32",
    "max_tracked_competitor_cos_fp32",
    "competitor_object_id",
    "identity_margin_fp32",
    "distance_to_no_obj_ptr",
    "max_abs_diff_to_no_obj_ptr",
    "obj_ptr_l2_norm",
]

ANCHOR_FIELDS = [
    "split",
    "video",
    "object_id_i",
    "object_id_j",
    "cosine_fp32",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_committed_clean(paths):
    for path in paths:
        rel = str(path.relative_to(REPO))

        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if tracked.returncode != 0:
            raise RuntimeError(
                f"EXP022-FULL run requires committed file: {rel}"
            )

        status = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--", rel],
            text=True,
        ).strip()

        if status:
            raise RuntimeError(
                f"EXP022-FULL run requires clean committed file: {rel}: {status}"
            )


def cosine_matrix_fp32(current, anchors):
    device_type = current.device.type

    prev_tf32 = None
    prev_precision = None

    if device_type == "cuda":
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_precision = torch.get_float32_matmul_precision()

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    try:
        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            current = F.normalize(current.float(), dim=-1)
            anchors = F.normalize(anchors.float(), dim=-1)
            out = current @ anchors.T
    finally:
        if device_type == "cuda":
            torch.set_float32_matmul_precision(prev_precision)
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    return out


def write_csv_atomic(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp, path)


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def finite_stats(values):
    if not values:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
        }

    x = np.asarray(values, dtype=np.float64)

    assert np.isfinite(x).all()

    return {
        "n": int(len(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def pearson_or_none(x, y):
    if len(x) < 2:
        return None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def load_and_validate_metadata():
    config = json.load(open(CONFIG_PATH))
    scope = json.load(open(SCOPE_PATH))
    substrate = json.load(open(SUBSTRATE_PATH))

    config_sha = sha256_file(CONFIG_PATH)
    scope_sha = sha256_file(SCOPE_PATH)
    substrate_sha = sha256_file(SUBSTRATE_PATH)

    assert scope_sha == config["scope"]["sha256"]
    assert substrate_sha == config["substrate"]["sha256"]

    assert scope["experiment"] == "EXP021"
    assert scope["test_videos_touched"] == 0
    assert scope["n_eligible_videos"] == 26

    eligible = scope["eligible_videos"]

    assert len(eligible) == config["scope"]["n_videos"]

    total_frames = sum(
        int(r["n_frames"])
        for r in eligible
    )

    candidate_rows = sum(
        (int(r["n_frames"]) - 1)
        * int(r["n_frame0_objects"])
        for r in eligible
    )

    anchor_rows = sum(
        int(r["n_frame0_objects"]) ** 2
        for r in eligible
    )

    assert total_frames == config["scope"]["total_video_frames"]
    assert candidate_rows == config["scope"]["expected_candidate_rows"]
    assert anchor_rows == config["scope"]["expected_anchor_rows"]

    sam_commit = git_commit(SAM_REPO)

    assert sam_commit == config["substrate"]["sam_source_commit"]

    checkpoint_path = Path(
        hf_hub_download(
            repo_id=config["substrate"]["hf_repo_id"],
            filename=config["substrate"]["checkpoint_filename"],
            revision=config["substrate"]["hf_revision"],
            local_files_only=True,
        )
    )

    checkpoint_sha = sha256_file(checkpoint_path)

    assert checkpoint_sha == (
        config["substrate"]["checkpoint_sha256"]
    ), (
        checkpoint_sha,
        config["substrate"]["checkpoint_sha256"],
    )

    return {
        "config": config,
        "scope": scope,
        "substrate": substrate,
        "eligible": eligible,
        "config_sha": config_sha,
        "scope_sha": scope_sha,
        "substrate_sha": substrate_sha,
        "sam_commit": sam_commit,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha": checkpoint_sha,
        "expected_candidate_rows": candidate_rows,
        "expected_anchor_rows": anchor_rows,
        "total_frames": total_frames,
    }


def print_plan(ctx):
    eligible = ctx["eligible"]

    train = [r for r in eligible if r["split"] == "TRAIN"]
    dev = [r for r in eligible if r["split"] == "DEV"]

    print("EXP022-FULL PLAN")
    print("identitygate_head =", git_commit(REPO))
    print("sam3_commit =", ctx["sam_commit"])
    print("config_sha256 =", ctx["config_sha"])
    print("scope_sha256 =", ctx["scope_sha"])
    print("substrate_sha256 =", ctx["substrate_sha"])
    print("checkpoint_path =", ctx["checkpoint_path"])
    print("checkpoint_sha256 =", ctx["checkpoint_sha"])
    print("test_videos_touched = 0")
    print("videos =", len(eligible))
    print("train_videos =", len(train))
    print("dev_videos =", len(dev))
    print("total_video_frames =", ctx["total_frames"])
    print(
        "expected_candidate_rows =",
        ctx["expected_candidate_rows"],
    )
    print(
        "expected_anchor_rows =",
        ctx["expected_anchor_rows"],
    )

    print()
    print("pointer_valid_rule = object_score_logit > 0")
    print(
        "invalid_relational_rule = "
        "retain baseline row; mask self/competitor/margin"
    )
    print(
        "cosine_precision =",
        ctx["config"]["feature_protocol"]["cosine_precision"],
    )

    print()
    print("=== VIDEO ORDER ===")

    for idx, r in enumerate(eligible):
        print(
            idx,
            r["split"],
            r["video"],
            "frames=",
            r["n_frames"],
            "objects=",
            r["n_frame0_objects"],
            "candidate_rows=",
            (int(r["n_frames"]) - 1)
            * int(r["n_frame0_objects"]),
        )

    print()
    print("EXP022_FULL_PLAN_PASS")


def video_output_paths(row):
    tag = f'{row["split"]}_{row["video"]}'
    root = PER_VIDEO_DIR / tag

    return {
        "root": root,
        "features": root / "features.csv",
        "anchors": root / "anchor_cosine.csv",
        "summary": root / "summary.json",
    }


def completed_video_is_valid(row, ctx, script_sha, run_commit):
    paths = video_output_paths(row)

    if not paths["summary"].exists():
        return False

    summary = json.load(open(paths["summary"]))

    required = [
        paths["features"],
        paths["anchors"],
    ]

    if not all(p.exists() for p in required):
        raise RuntimeError(
            f"Partial completed output detected for {row['video']}"
        )

    checks = [
        summary["status"] == "COMPLETED",
        summary["video"] == row["video"],
        summary["split"] == row["split"],
        summary["script_sha256"] == script_sha,
        summary["identitygate_commit_at_execution"] == run_commit,
        summary["config_sha256"] == ctx["config_sha"],
        summary["scope_sha256"] == ctx["scope_sha"],
        summary["substrate_sha256"] == ctx["substrate_sha"],
        summary["checkpoint_sha256"] == ctx["checkpoint_sha"],
        summary["test_videos_touched"] == 0,
        summary["features_sha256"] == sha256_file(paths["features"]),
        summary["anchors_sha256"] == sha256_file(paths["anchors"]),
    ]

    if not all(checks):
        raise RuntimeError(
            f"Existing output provenance mismatch for {row['video']}"
        )

    expected_features = (
        (int(row["n_frames"]) - 1)
        * int(row["n_frame0_objects"])
    )

    expected_anchors = int(row["n_frame0_objects"]) ** 2

    if summary["n_candidate_rows"] != expected_features:
        raise RuntimeError(
            f"Feature row mismatch for {row['video']}"
        )

    if summary["n_anchor_rows"] != expected_anchors:
        raise RuntimeError(
            f"Anchor row mismatch for {row['video']}"
        )

    return True


def process_video(predictor, row, ctx, script_sha, run_commit):
    split = row["split"]
    vid = row["video"]
    n_frames = int(row["n_frames"])
    object_ids = [int(x) for x in row["frame0_object_ids"]]
    n_objects = len(object_ids)

    paths = video_output_paths(row)
    paths["root"].mkdir(parents=True, exist_ok=True)

    jpgs = sorted((FRAMES / vid).glob("*.jpg"))
    pngs = sorted((ANN / vid).glob("*.png"))

    assert len(jpgs) == n_frames
    assert pngs

    a0 = np.array(Image.open(pngs[0]))

    actual_ids = sorted(
        int(x)
        for x in np.unique(a0)
        if int(x) != 0
    )

    assert actual_ids == object_ids
    assert n_objects >= 2

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.time()

    print()
    print(
        "VIDEO_START",
        split,
        vid,
        "frames=",
        n_frames,
        "objects=",
        n_objects,
    )

    state = predictor.init_state(
        video_path=str(FRAMES / vid)
    )

    try:
        predictor.clear_all_points_in_video(state)

        for oid in object_ids:
            mask = torch.from_numpy(a0 == oid)
            assert mask.any()

            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=oid,
                mask=mask,
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

        anchors = frame0_out["obj_ptr"].detach()

        assert tuple(anchors.shape) == (
            n_objects,
            predictor.hidden_dim,
        )

        anchor_cos = cosine_matrix_fp32(
            anchors,
            anchors,
        )

        diag = torch.diagonal(anchor_cos)

        assert torch.allclose(
            diag,
            torch.ones_like(diag),
            atol=1e-5,
            rtol=1e-5,
        )

        anchor_rows = []

        anchor_cpu = anchor_cos.detach().cpu().numpy()

        for i, oid_i in enumerate(object_ids):
            for j, oid_j in enumerate(object_ids):
                anchor_rows.append({
                    "split": split,
                    "video": vid,
                    "object_id_i": oid_i,
                    "object_id_j": oid_j,
                    "cosine_fp32": float(anchor_cpu[i, j]),
                })

        feature_rows = []
        valid_self = []
        valid_competitor = []
        valid_margin = []

        seen_frames = []

        no_obj = (
            predictor.no_obj_ptr
            .detach()
            .float()
            .to(anchors.device)
            .reshape(-1)
        )

        invalid_noobj_maxdiff = []

        for out in predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=n_frames - 1,
            reverse=False,
            propagate_preflight=False,
        ):
            frame_idx = int(out[0])
            yielded_ids = [int(x) for x in out[1]]

            seen_frames.append(frame_idx)

            assert yielded_ids == object_ids
            assert list(state["obj_ids"]) == object_ids

            if frame_idx == 0:
                continue

            current_out = (
                state["output_dict"]
                ["non_cond_frame_outputs"][frame_idx]
            )

            memory_encoded = (
                current_out.get("maskmem_features")
                is not None
            )

            assert memory_encoded

            ptr = current_out["obj_ptr"].detach()

            logits = (
                current_out["object_score_logits"]
                .detach()
                .float()
                .reshape(-1)
            )

            assert tuple(ptr.shape) == (
                n_objects,
                predictor.hidden_dim,
            )

            assert tuple(logits.shape) == (
                n_objects,
            )

            cos = cosine_matrix_fp32(
                ptr,
                anchors,
            )

            eye = torch.eye(
                n_objects,
                dtype=torch.bool,
                device=cos.device,
            )

            competitor_values, competitor_indices = (
                cos.masked_fill(
                    eye,
                    float("-inf"),
                )
                .max(dim=1)
            )

            self_values = torch.diagonal(cos)
            margins = self_values - competitor_values

            probs = torch.sigmoid(logits)

            ptr_fp32 = ptr.float()

            no_obj_diff = (
                ptr_fp32
                - no_obj.reshape(1, -1)
            )

            no_obj_l2 = torch.linalg.vector_norm(
                no_obj_diff,
                dim=1,
            )

            no_obj_maxabs = no_obj_diff.abs().amax(dim=1)

            ptr_norm = torch.linalg.vector_norm(
                ptr_fp32,
                dim=1,
            )

            for obj_idx, oid in enumerate(object_ids):
                logit = float(logits[obj_idx].item())
                pointer_valid = logit > 0.0

                maxabs = float(
                    no_obj_maxabs[obj_idx].item()
                )

                distance = float(
                    no_obj_l2[obj_idx].item()
                )

                is_sentinel = maxabs <= 1e-6

                if not pointer_valid:
                    invalid_noobj_maxdiff.append(maxabs)

                    assert maxabs <= 1e-5, (
                        vid,
                        frame_idx,
                        oid,
                        logit,
                        maxabs,
                    )

                    self_value = None
                    competitor_value = None
                    competitor_oid = None
                    margin_value = None

                else:
                    self_value = float(
                        self_values[obj_idx].item()
                    )

                    competitor_value = float(
                        competitor_values[obj_idx].item()
                    )

                    competitor_idx = int(
                        competitor_indices[obj_idx].item()
                    )

                    competitor_oid = object_ids[
                        competitor_idx
                    ]

                    margin_value = float(
                        margins[obj_idx].item()
                    )

                    valid_self.append(self_value)
                    valid_competitor.append(
                        competitor_value
                    )
                    valid_margin.append(margin_value)

                feature_rows.append({
                    "split": split,
                    "video": vid,
                    "frame_idx": frame_idx,
                    "obj_idx": obj_idx,
                    "object_id": oid,
                    "n_objects": n_objects,
                    "baseline_memory_encoded": 1,
                    "pointer_valid": int(pointer_valid),
                    "pointer_is_no_obj_sentinel": int(
                        is_sentinel
                    ),
                    "object_score_logit": logit,
                    "object_score_prob": float(
                        probs[obj_idx].item()
                    ),
                    "self_anchor_cos_fp32": self_value,
                    "max_tracked_competitor_cos_fp32": (
                        competitor_value
                    ),
                    "competitor_object_id": competitor_oid,
                    "identity_margin_fp32": margin_value,
                    "distance_to_no_obj_ptr": distance,
                    "max_abs_diff_to_no_obj_ptr": maxabs,
                    "obj_ptr_l2_norm": float(
                        ptr_norm[obj_idx].item()
                    ),
                })

        assert seen_frames == list(range(n_frames)), (
            vid,
            seen_frames[:10],
            seen_frames[-10:],
            n_frames,
        )

        expected_rows = (
            (n_frames - 1)
            * n_objects
        )

        assert len(feature_rows) == expected_rows
        assert len(anchor_rows) == n_objects ** 2

        write_csv_atomic(
            paths["features"],
            FEATURE_FIELDS,
            feature_rows,
        )

        write_csv_atomic(
            paths["anchors"],
            ANCHOR_FIELDS,
            anchor_rows,
        )

        torch.cuda.synchronize()

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        peak_reserved_gb = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

        valid_rows = len(valid_margin)
        invalid_rows = expected_rows - valid_rows

        offdiag = anchor_cos[
            ~torch.eye(
                n_objects,
                dtype=torch.bool,
                device=anchor_cos.device,
            )
        ]

        summary = {
            "experiment": "EXP022-FULL",
            "status": "COMPLETED",
            "identitygate_commit_at_execution": run_commit,
            "sam3_commit": ctx["sam_commit"],
            "script_sha256": script_sha,
            "config_sha256": ctx["config_sha"],
            "scope_sha256": ctx["scope_sha"],
            "substrate_sha256": ctx["substrate_sha"],
            "checkpoint_path": str(
                ctx["checkpoint_path"]
            ),
            "checkpoint_sha256": ctx["checkpoint_sha"],
            "split": split,
            "video": vid,
            "video_frames": n_frames,
            "n_objects": n_objects,
            "object_ids": object_ids,
            "test_videos_touched": 0,
            "conditioning_frame_is_candidate": False,
            "n_candidate_rows": expected_rows,
            "n_anchor_rows": len(anchor_rows),
            "valid_pointer_rows": valid_rows,
            "invalid_pointer_rows": invalid_rows,
            "valid_pointer_fraction": (
                valid_rows / expected_rows
            ),
            "pointer_valid_definition": (
                "object_score_logit > 0"
            ),
            "invalid_relational_values_masked": True,
            "max_invalid_pointer_no_obj_diff": (
                max(invalid_noobj_maxdiff)
                if invalid_noobj_maxdiff
                else None
            ),
            "anchor_offdiag": finite_stats(
                offdiag.detach().cpu().tolist()
            ),
            "valid_self_anchor_cos": finite_stats(
                valid_self
            ),
            "valid_competitor_cos": finite_stats(
                valid_competitor
            ),
            "valid_identity_margin": finite_stats(
                valid_margin
            ),
            "negative_valid_margin_rows": int(
                sum(x < 0 for x in valid_margin)
            ),
            "negative_valid_margin_fraction": (
                float(
                    np.mean(
                        np.asarray(valid_margin) < 0
                    )
                )
                if valid_margin
                else None
            ),
            "peak_memory_allocated_gb": (
                peak_allocated_gb
            ),
            "peak_memory_reserved_gb": (
                peak_reserved_gb
            ),
            "runtime_sec": time.time() - start,
            "features_sha256": sha256_file(
                paths["features"]
            ),
            "anchors_sha256": sha256_file(
                paths["anchors"]
            ),
        }

        write_json_atomic(
            paths["summary"],
            summary,
        )

        print(
            "VIDEO_PASS",
            vid,
            "candidate_rows=",
            expected_rows,
            "valid_rows=",
            valid_rows,
            "valid_fraction=",
            summary["valid_pointer_fraction"],
            "peak_allocated_gb=",
            peak_allocated_gb,
            "runtime_sec=",
            summary["runtime_sec"],
        )

        return summary

    finally:
        del state
        gc.collect()
        torch.cuda.empty_cache()


def merge_outputs(ctx, script_sha, run_commit):
    all_features = []
    all_anchors = []
    video_summaries = []

    for row in ctx["eligible"]:
        paths = video_output_paths(row)

        summary = json.load(open(paths["summary"]))
        video_summaries.append(summary)

        with open(paths["features"], newline="") as f:
            all_features.extend(
                csv.DictReader(f)
            )

        with open(paths["anchors"], newline="") as f:
            all_anchors.extend(
                csv.DictReader(f)
            )

    assert len(all_features) == ctx["expected_candidate_rows"]
    assert len(all_anchors) == ctx["expected_anchor_rows"]

    write_csv_atomic(
        FEATURES_PATH,
        FEATURE_FIELDS,
        all_features,
    )

    write_csv_atomic(
        ANCHORS_PATH,
        ANCHOR_FIELDS,
        all_anchors,
    )

    valid_rows = [
        r
        for r in all_features
        if int(r["pointer_valid"]) == 1
    ]

    invalid_rows = [
        r
        for r in all_features
        if int(r["pointer_valid"]) == 0
    ]

    for r in invalid_rows:
        assert r["self_anchor_cos_fp32"] == ""
        assert r["max_tracked_competitor_cos_fp32"] == ""
        assert r["competitor_object_id"] == ""
        assert r["identity_margin_fp32"] == ""

    valid_self = [
        float(r["self_anchor_cos_fp32"])
        for r in valid_rows
    ]

    valid_comp = [
        float(r["max_tracked_competitor_cos_fp32"])
        for r in valid_rows
    ]

    valid_margin = [
        float(r["identity_margin_fp32"])
        for r in valid_rows
    ]

    anchor_offdiag = [
        float(r["cosine_fp32"])
        for r in all_anchors
        if r["object_id_i"] != r["object_id_j"]
    ]

    eps_summary = {}

    for eps in ctx["config"]["feature_protocol"][
        "descriptive_abs_margin_eps"
    ]:
        if valid_margin:
            frac = float(
                np.mean(
                    np.abs(
                        np.asarray(
                            valid_margin,
                            dtype=np.float64,
                        )
                    )
                    < float(eps)
                )
            )
        else:
            frac = None

        eps_summary[str(eps)] = frac

    by_object_count = {}

    for summary in video_summaries:
        key = str(summary["n_objects"])

        if key not in by_object_count:
            by_object_count[key] = {
                "videos": 0,
                "candidate_rows": 0,
                "valid_rows": 0,
            }

        by_object_count[key]["videos"] += 1
        by_object_count[key]["candidate_rows"] += (
            summary["n_candidate_rows"]
        )
        by_object_count[key]["valid_rows"] += (
            summary["valid_pointer_rows"]
        )

    for value in by_object_count.values():
        value["valid_fraction"] = (
            value["valid_rows"]
            / value["candidate_rows"]
        )

    full_summary = {
        "experiment": "EXP022-FULL",
        "status": "COMPLETED_DEVELOPMENT_SIGNAL_CENSUS",
        "scientific_role": "DEVELOPMENT_ONLY_SIGNAL_FEASIBILITY",
        "identitygate_commit_at_execution": run_commit,
        "sam3_commit": ctx["sam_commit"],
        "script_sha256": script_sha,
        "config_sha256": ctx["config_sha"],
        "scope_sha256": ctx["scope_sha"],
        "substrate_sha256": ctx["substrate_sha"],
        "checkpoint_path": str(ctx["checkpoint_path"]),
        "checkpoint_sha256": ctx["checkpoint_sha"],
        "test_videos_touched": 0,
        "n_videos": len(ctx["eligible"]),
        "total_video_frames": ctx["total_frames"],
        "candidate_rows": len(all_features),
        "anchor_rows": len(all_anchors),
        "valid_pointer_rows": len(valid_rows),
        "invalid_pointer_rows": len(invalid_rows),
        "valid_pointer_fraction": (
            len(valid_rows) / len(all_features)
        ),
        "videos_with_any_valid_pointer": int(
            sum(
                s["valid_pointer_rows"] > 0
                for s in video_summaries
            )
        ),
        "videos_with_zero_valid_pointer": int(
            sum(
                s["valid_pointer_rows"] == 0
                for s in video_summaries
            )
        ),
        "valid_self_anchor_cos": finite_stats(
            valid_self
        ),
        "valid_competitor_cos": finite_stats(
            valid_comp
        ),
        "valid_identity_margin": finite_stats(
            valid_margin
        ),
        "valid_self_vs_competitor_pearson": (
            pearson_or_none(
                valid_self,
                valid_comp,
            )
        ),
        "negative_valid_margin_rows": int(
            sum(x < 0 for x in valid_margin)
        ),
        "negative_valid_margin_fraction": (
            float(
                np.mean(
                    np.asarray(valid_margin) < 0
                )
            )
            if valid_margin
            else None
        ),
        "abs_margin_near_zero_fraction": eps_summary,
        "anchor_offdiag": finite_stats(
            anchor_offdiag
        ),
        "by_object_count": by_object_count,
        "per_video": video_summaries,
        "features_sha256": sha256_file(
            FEATURES_PATH
        ),
        "anchors_sha256": sha256_file(
            ANCHORS_PATH
        ),
        "missing_identity_fallback": (
            "OPEN_NOT_DECIDED_BY_EXP022"
        ),
        "p_values_reported": False,
    }

    write_json_atomic(
        SUMMARY_PATH,
        full_summary,
    )

    return full_summary


def run_full(ctx):
    require_committed_clean([
        Path(__file__).resolve(),
        CONFIG_PATH,
    ])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    run_commit = git_commit(REPO)
    script_sha = sha256_file(Path(__file__).resolve())

    print("EXP022-FULL RUN")
    print("identitygate_commit =", run_commit)
    print("script_sha256 =", script_sha)
    print("config_sha256 =", ctx["config_sha"])
    print("checkpoint_sha256 =", ctx["checkpoint_sha"])
    print("test_videos_touched = 0")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("building frozen SAM 3 VOS model...")

    model = build_sam3_video_model(
        checkpoint_path=str(
            ctx["checkpoint_path"]
        ),
        load_from_HF=False,
    )

    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    print(
        "predictor_class =",
        type(predictor).__name__,
    )
    print("hidden_dim =", predictor.hidden_dim)
    print(
        "gpu =",
        torch.cuda.get_device_name(0),
    )

    completed = 0
    skipped = 0

    for idx, row in enumerate(ctx["eligible"], start=1):
        print()
        print(
            f"=== VIDEO {idx}/{len(ctx['eligible'])} ==="
        )

        if completed_video_is_valid(
            row,
            ctx,
            script_sha,
            run_commit,
        ):
            print("RESUME_SKIP_PASS =", row["video"])
            skipped += 1
            continue

        process_video(
            predictor,
            row,
            ctx,
            script_sha,
            run_commit,
        )

        completed += 1

    print()
    print("=== MERGE ===")

    summary = merge_outputs(
        ctx,
        script_sha,
        run_commit,
    )

    print("videos_completed_this_run =", completed)
    print("videos_resumed =", skipped)
    print(
        "candidate_rows =",
        summary["candidate_rows"],
    )
    print(
        "valid_pointer_rows =",
        summary["valid_pointer_rows"],
    )
    print(
        "invalid_pointer_rows =",
        summary["invalid_pointer_rows"],
    )
    print(
        "valid_pointer_fraction =",
        summary["valid_pointer_fraction"],
    )
    print(
        "videos_with_any_valid_pointer =",
        summary["videos_with_any_valid_pointer"],
    )
    print(
        "videos_with_zero_valid_pointer =",
        summary["videos_with_zero_valid_pointer"],
    )
    print(
        "negative_valid_margin_fraction =",
        summary["negative_valid_margin_fraction"],
    )
    print(
        "abs_margin_near_zero_fraction =",
        summary["abs_margin_near_zero_fraction"],
    )
    print(
        "features_sha256 =",
        summary["features_sha256"],
    )
    print(
        "anchors_sha256 =",
        summary["anchors_sha256"],
    )
    print("summary =", SUMMARY_PATH)
    print()
    print("EXP022_FULL_PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["plan", "run"],
    )

    args = parser.parse_args()

    ctx = load_and_validate_metadata()

    if args.mode == "plan":
        print_plan(ctx)
    else:
        run_full(ctx)


if __name__ == "__main__":
    main()
