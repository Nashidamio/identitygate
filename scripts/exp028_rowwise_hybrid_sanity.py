import csv
import hashlib
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/EXP028-rowwise-hybrid-sanity-v1.json"
DATA = Path("/mnt/d/thesis_data/mosev2/train")
OUT = ROOT / "experiments/EXP028_rowwise_hybrid_sanity"


def tensor_hash(x):
    if x is None:
        return None
    y = x.detach().contiguous().cpu()
    raw = y.view(torch.uint8).numpy().tobytes()
    h = hashlib.sha256()
    h.update(str(tuple(y.shape)).encode())
    h.update(str(y.dtype).encode())
    h.update(raw)
    return h.hexdigest()


def slice_value(value, row, batch_size):
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[row:row + 1]
        return value

    if isinstance(value, list):
        return [
            slice_value(x, row, batch_size)
            for x in value
        ]

    if isinstance(value, tuple):
        return tuple(
            slice_value(x, row, batch_size)
            for x in value
        )

    return value


def make_row_output_dict(
    output_dict,
    row,
    batch_size,
    blocked_frames,
    controller,
):
    result = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    for bank_name in [
        "cond_frame_outputs",
        "non_cond_frame_outputs",
    ]:
        for frame_idx, out in output_dict[bank_name].items():
            if (
                bank_name == "non_cond_frame_outputs"
                and frame_idx in blocked_frames
            ):
                controller["omission_count"] += 1
                controller["omitted_frames"].add(int(frame_idx))
                continue

            sliced = {
                key: slice_value(value, row, batch_size)
                for key, value in out.items()
            }

            if "eff_iou_score" in out:
                original_eff = out["eff_iou_score"]
                sliced_eff = sliced["eff_iou_score"]

                if tensor_hash(original_eff) != tensor_hash(sliced_eff):
                    raise RuntimeError(
                        "Global eff_iou_score changed during row slicing"
                    )

            result[bank_name][frame_idx] = sliced

    return result


def install_rowwise_hybrid(predictor, controller):
    original = predictor._prepare_memory_conditioned_features

    def wrapped(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):
        vanilla = original(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            use_prev_mem_frame=use_prev_mem_frame,
        )

        if (
            not controller["enabled"]
            or is_init_cond_frame
            or not use_prev_mem_frame
        ):
            return vanilla

        batch_size = current_vision_feats[-1].size(1)
        row = controller["replace_row"]

        if row < 0 or row >= batch_size:
            raise RuntimeError(
                "Row {} outside batch size {}".format(
                    row,
                    batch_size,
                )
            )

        row_output_dict = make_row_output_dict(
            output_dict=output_dict,
            row=row,
            batch_size=batch_size,
            blocked_frames=controller["blocked_frames"],
            controller=controller,
        )

        row_feats = [
            x[:, row:row + 1]
            for x in current_vision_feats
        ]
        row_pos = [
            x[:, row:row + 1]
            for x in current_vision_pos_embeds
        ]

        row_result = original(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=row_feats,
            current_vision_pos_embeds=row_pos,
            feat_sizes=feat_sizes,
            output_dict=row_output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            use_prev_mem_frame=use_prev_mem_frame,
        )

        hybrid = vanilla.clone()
        hybrid[row:row + 1] = row_result

        controller["row_recompute_calls"] += 1

        return hybrid

    predictor._prepare_memory_conditioned_features = types.MethodType(
        wrapped,
        predictor,
    )


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


def capture_signature(current_out, row, batch_size):
    result = {}

    for key in [
        "object_score_logits",
        "iou_score",
        "obj_ptr",
        "maskmem_features",
    ]:
        value = current_out.get(key)

        if value is None:
            result[key] = None
        else:
            result[key] = tensor_hash(
                slice_value(value, row, batch_size)
            )

    return result


def run_tracker(
    predictor,
    jpg_dir,
    gt0,
    object_ids,
    cfg,
    controller=None,
):
    state = predictor.init_state(video_path=str(jpg_dir))
    predictor.clear_all_points_in_video(state)
    add_prompts(predictor, state, gt0, object_ids)

    predictions = {}
    signatures = {}
    eff_hashes = {}

    if controller is not None:
        controller["row_recompute_calls"] = 0
        controller["omission_count"] = 0
        controller["omitted_frames"] = set()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for out in predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=cfg["n_frames"] - 1,
        reverse=False,
        propagate_preflight=True,
    ):
        frame_idx = int(out[0])
        ids = [int(x) for x in out[1]]
        video_res = out[3]

        if ids != object_ids:
            raise RuntimeError(
                "Object order mismatch: {} vs {}".format(
                    ids,
                    object_ids,
                )
            )

        predictions[frame_idx] = {}
        signatures[frame_idx] = {}

        for row, oid in enumerate(ids):
            predictions[frame_idx][oid] = (
                (video_res[row, 0] > 0)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

        if frame_idx in state["output_dict"]["cond_frame_outputs"]:
            current_out = state["output_dict"]["cond_frame_outputs"][
                frame_idx
            ]
        else:
            current_out = state["output_dict"]["non_cond_frame_outputs"][
                frame_idx
            ]

        batch_size = len(ids)

        for row, oid in enumerate(ids):
            signatures[frame_idx][oid] = capture_signature(
                current_out,
                row,
                batch_size,
            )

        eff_hashes[frame_idx] = tensor_hash(
            current_out.get("eff_iou_score")
        )

    runtime = float(time.time() - t0)
    peak = float(
        torch.cuda.max_memory_allocated() / 1024 ** 3
    )

    if len(predictions) != cfg["n_frames"]:
        raise RuntimeError(
            "Frame count mismatch: {}".format(
                len(predictions)
            )
        )

    stats = None

    if controller is not None:
        stats = {
            "row_recompute_calls": controller[
                "row_recompute_calls"
            ],
            "omission_count": controller[
                "omission_count"
            ],
            "omitted_frames": sorted(
                controller["omitted_frames"]
            ),
        }

    del state
    torch.cuda.empty_cache()

    return predictions, signatures, eff_hashes, stats, peak, runtime


def compare_runs(
    name,
    left_masks,
    left_sig,
    left_eff,
    right_masks,
    right_sig,
    right_eff,
    object_ids,
    n_frames,
):
    rows = []
    mask_exact_count = 0
    signal_exact_count = 0
    eff_exact_count = 0

    signal_fields = [
        "object_score_logits",
        "iou_score",
        "obj_ptr",
        "maskmem_features",
    ]

    for frame_idx in range(n_frames):
        if left_eff[frame_idx] == right_eff[frame_idx]:
            eff_exact_count += 1

        for oid in object_ids:
            a = left_masks[frame_idx][oid]
            b = right_masks[frame_idx][oid]

            mask_equal = bool(np.array_equal(a, b))
            xor_px = int(np.logical_xor(a, b).sum())

            field_equal = {
                key: (
                    left_sig[frame_idx][oid][key]
                    == right_sig[frame_idx][oid][key]
                )
                for key in signal_fields
            }

            signals_equal = all(field_equal.values())

            mask_exact_count += int(mask_equal)
            signal_exact_count += int(signals_equal)

            rows.append({
                "comparison": name,
                "frame_idx": frame_idx,
                "object_id": oid,
                "mask_exact": int(mask_equal),
                "xor_px": xor_px,
                "object_score_logits_exact": int(
                    field_equal["object_score_logits"]
                ),
                "iou_score_exact": int(
                    field_equal["iou_score"]
                ),
                "obj_ptr_exact": int(
                    field_equal["obj_ptr"]
                ),
                "maskmem_features_exact": int(
                    field_equal["maskmem_features"]
                ),
            })

    total_rows = n_frames * len(object_ids)

    return {
        "rows": rows,
        "total_rows": total_rows,
        "mask_exact_count": mask_exact_count,
        "signal_exact_count": signal_exact_count,
        "eff_exact_count": eff_exact_count,
        "mask_exact_all": mask_exact_count == total_rows,
        "signals_exact_all": signal_exact_count == total_rows,
        "eff_exact_all": eff_exact_count == n_frames,
    }


def write_csv(path, rows):
    fields = [
        "comparison",
        "frame_idx",
        "object_id",
        "mask_exact",
        "xor_px",
        "object_score_logits_exact",
        "iou_score_exact",
        "obj_ptr_exact",
        "maskmem_features_exact",
    ]

    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )
        wr.writeheader()
        wr.writerows(rows)


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

    if cfg["replace_object_id"] not in object_ids:
        raise RuntimeError("Replace object absent")

    replace_row = object_ids.index(
        cfg["replace_object_id"]
    )

    print("video={}".format(cfg["video"]), flush=True)
    print("frames={}".format(cfg["n_frames"]), flush=True)
    print("object_ids={}".format(object_ids), flush=True)
    print("replace_row={}".format(replace_row), flush=True)

    print("building frozen SAM3...", flush=True)
    model = build_sam3_video_model()
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    if predictor.use_memory_selection is not True:
        raise RuntimeError(
            "Pinned runtime no longer has use_memory_selection=True"
        )

    if predictor.non_overlap_masks_for_mem_enc is not False:
        raise RuntimeError(
            "non_overlap_masks_for_mem_enc must remain False"
        )

    print("running VANILLA...", flush=True)

    vanilla = run_tracker(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg,
    )

    controller = {
        "enabled": True,
        "replace_row": replace_row,
        "blocked_frames": set(),
        "row_recompute_calls": 0,
        "omission_count": 0,
        "omitted_frames": set(),
    }

    install_rowwise_hybrid(
        predictor,
        controller,
    )

    print("running ROWWISE FULL-MEMORY CONTROL...", flush=True)

    control = run_tracker(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg,
        controller,
    )

    control_cmp = compare_runs(
        "VANILLA_vs_ROWWISE_FULL_MEMORY",
        vanilla[0],
        vanilla[1],
        vanilla[2],
        control[0],
        control[1],
        control[2],
        object_ids,
        cfg["n_frames"],
    )

    control_pass = all([
        control_cmp["mask_exact_all"],
        control_cmp["signals_exact_all"],
        control_cmp["eff_exact_all"],
        control[3]["omission_count"] == 0,
    ])

    if not control_pass:
        summary = {
            "status": "ROWWISE_CONTROL_FAIL",
            "video": cfg["video"],
            "frames": cfg["n_frames"],
            "object_ids": object_ids,
            "replace_object_id": cfg["replace_object_id"],
            "control": {
                key: value
                for key, value in control_cmp.items()
                if key != "rows"
            },
            "control_runtime_stats": control[3],
            "vanilla_peak_vram_gb": vanilla[4],
            "control_peak_vram_gb": control[4],
            "vanilla_runtime_sec": vanilla[5],
            "control_runtime_sec": control[5],
            "dev_touched": 0,
            "test_touched": 0,
            "claim_boundary": cfg["claim_boundary"],
        }

        write_csv(
            OUT / "comparisons.csv",
            control_cmp["rows"],
        )

        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    controller["blocked_frames"] = set(
        range(
            cfg["block_start_frame"],
            cfg["block_end_frame"] + 1,
        )
    )

    print("running ROWWISE SELECTIVE BLOCK...", flush=True)

    selective = run_tracker(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        cfg,
        controller,
    )

    selective_cmp = compare_runs(
        "ROWWISE_FULL_MEMORY_vs_SELECTIVE",
        control[0],
        control[1],
        control[2],
        selective[0],
        selective[1],
        selective[2],
        object_ids,
        cfg["n_frames"],
    )

    admitted_ids = [
        oid
        for oid in object_ids
        if oid != cfg["replace_object_id"]
    ]

    admitted_exact = all(
        np.array_equal(
            control[0][frame_idx][oid],
            selective[0][frame_idx][oid],
        )
        and (
            control[1][frame_idx][oid]
            == selective[1][frame_idx][oid]
        )
        for frame_idx in range(cfg["n_frames"])
        for oid in admitted_ids
    )

    first_block = cfg["block_start_frame"]

    first_block_exact = all(
        np.array_equal(
            control[0][first_block][oid],
            selective[0][first_block][oid],
        )
        and (
            control[1][first_block][oid]
            == selective[1][first_block][oid]
        )
        for oid in object_ids
    )

    changed_frames = [
        frame_idx
        for frame_idx in range(
            first_block + 1,
            cfg["n_frames"],
        )
        if not np.array_equal(
            control[0][frame_idx][cfg["replace_object_id"]],
            selective[0][frame_idx][cfg["replace_object_id"]],
        )
    ]

    selective_pass = all([
        admitted_exact,
        first_block_exact,
        len(changed_frames) > 0,
        selective[3]["omission_count"] > 0,
        len(selective[3]["omitted_frames"]) > 0,
    ])

    summary = {
        "status": (
            "ROWWISE_HYBRID_SANITY_PASS"
            if selective_pass
            else "ROWWISE_SELECTIVE_FAIL"
        ),
        "video": cfg["video"],
        "frames": cfg["n_frames"],
        "object_ids": object_ids,
        "replace_object_id": cfg["replace_object_id"],
        "block_start_frame": cfg["block_start_frame"],
        "block_end_frame": cfg["block_end_frame"],
        "control": {
            key: value
            for key, value in control_cmp.items()
            if key != "rows"
        },
        "control_runtime_stats": control[3],
        "selective": {
            key: value
            for key, value in selective_cmp.items()
            if key != "rows"
        },
        "selective_runtime_stats": selective[3],
        "admitted_objects_exact_all_frames": admitted_exact,
        "first_block_frame_exact": first_block_exact,
        "blocked_object_downstream_changed": len(
            changed_frames
        ) > 0,
        "first_changed_frame": (
            changed_frames[0]
            if changed_frames
            else None
        ),
        "vanilla_peak_vram_gb": vanilla[4],
        "control_peak_vram_gb": control[4],
        "selective_peak_vram_gb": selective[4],
        "vanilla_runtime_sec": vanilla[5],
        "control_runtime_sec": control[5],
        "selective_runtime_sec": selective[5],
        "dev_touched": 0,
        "test_touched": 0,
        "claim_boundary": cfg["claim_boundary"],
    }

    write_csv(
        OUT / "comparisons.csv",
        control_cmp["rows"] + selective_cmp["rows"],
    )

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
