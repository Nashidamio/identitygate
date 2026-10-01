import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_video_model

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/DEMO001-real-video-b0-b2-v1.json"
EXP033_PATH = ROOT / "scripts/exp033_unified_gate_closed_loop_sanity.py"
EXP033_CFG_PATH = ROOT / "configs/EXP033-unified-gate-closed-loop-sanity-v1.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load module: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def point_mask(predictor, jpg_dir, xy):
    state = predictor.init_state(video_path=str(jpg_dir))
    predictor.clear_all_points_in_video(state)

    pts = torch.tensor([xy], dtype=torch.float32)
    lbl = torch.tensor([1], dtype=torch.int32)

    predictor.add_new_points(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=pts,
        labels=lbl,
        clear_old_points=False,
    )

    generator = predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=1,
        reverse=False,
        propagate_preflight=True,
    )

    out = next(generator)
    mask = (
        (out[3][0, 0] > 0)
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    del state
    torch.cuda.empty_cache()
    return mask


def overlay(img, mask, label):
    arr = np.asarray(img.convert("RGB")).copy()

    if mask.shape != arr.shape[:2]:
        mask = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (arr.shape[1], arr.shape[0])
            )
        ) > 127

    arr[mask] = (
        0.45 * arr[mask]
        + 0.55 * np.array([255, 60, 60])
    ).astype(np.uint8)

    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    draw.rectangle((10, 10, 150, 45), fill=(0, 0, 0))
    draw.text((20, 18), label, fill=(255, 255, 255))
    return out


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    jpg_dir = Path(cfg["video_frames"])
    out_dir = Path(cfg["output_dir"])

    if out_dir.exists():
        raise RuntimeError("Output already exists: {}".format(out_dir))

    frames = sorted(
        jpg_dir.glob("*.jpg"),
        key=lambda p: int(p.stem),
    )

    if len(frames) != int(cfg["n_frames"]):
        raise RuntimeError(
            "Frame count mismatch: {} != {}".format(
                len(frames), cfg["n_frames"]
            )
        )

    exp033 = load_module("demo_exp033", EXP033_PATH)
    exp033_cfg = json.loads(EXP033_CFG_PATH.read_text(encoding="utf-8"))

    deps = exp033.load_dependencies(exp033_cfg)
    packs = exp033.load_model_packs(exp033_cfg, deps)

    sam_root = Path.home() / "thesis/externals/sam3"
    sam_commit = subprocess.check_output(
        ["git", "-C", str(sam_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    expected = exp033_cfg["substrate"]["expected_sam_commit"]
    if sam_commit != expected:
        raise RuntimeError(
            "SAM3 commit mismatch: {} != {}".format(
                sam_commit, expected
            )
        )

    print("[1] building frozen SAM3...", flush=True)
    model = build_sam3_video_model()
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    print("[2] deriving frozen frame-0 prompt mask...", flush=True)
    mask0 = point_mask(
        predictor,
        jpg_dir,
        cfg["prompt_normalized_xy"],
    )

    gt0 = np.zeros(mask0.shape, dtype=np.uint8)
    gt0[mask0] = 1
    object_ids = [1]

    run_cfg = {
        "scope": {
            "n_frames": int(cfg["n_frames"]),
        },
        "frame_intervention": {
            "tau": float(cfg["b2_tau"]),
        },
    }

    print("[3] running B0...", flush=True)
    b0 = exp033.run_tracker(
        predictor=predictor,
        jpg_dir=jpg_dir,
        gt0=gt0,
        object_ids=object_ids,
        cfg=run_cfg,
        variant="B0",
        packs=packs,
        deps=deps,
    )

    print("[4] running frozen B2 tau=0.1...", flush=True)
    b2 = exp033.run_tracker(
        predictor=predictor,
        jpg_dir=jpg_dir,
        gt0=gt0,
        object_ids=object_ids,
        cfg=run_cfg,
        variant="B2",
        packs=packs,
        deps=deps,
    )

    out_dir.mkdir(parents=True)
    vis_dir = out_dir / "side_by_side"
    vis_dir.mkdir()

    print("[5] writing side-by-side overlays...", flush=True)

    for idx, frame_path in enumerate(frames):
        raw = Image.open(frame_path).convert("RGB")
        left = overlay(raw, b0["predictions"][idx][1], "B0 native")
        right = overlay(raw, b2["predictions"][idx][1], "B2 gated")

        canvas = Image.new(
            "RGB",
            (left.width + right.width, left.height),
        )
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        canvas.save(
            vis_dir / "{:05d}.jpg".format(idx + 1),
            quality=90,
        )

    admits = sum(
        r["action"] == "ADMIT"
        for r in b2["decisions"]
    )
    blocks = sum(
        r["action"] == "BLOCK"
        for r in b2["decisions"]
    )

    summary = {
        "demo": cfg["demo"],
        "claim": cfg["purpose"],
        "sam_commit": sam_commit,
        "frames": len(frames),
        "prompt_normalized_xy": cfg["prompt_normalized_xy"],
        "b2_tau": cfg["b2_tau"],
        "b2_admit_count": admits,
        "b2_block_count": blocks,
        "b2_write_rate": admits / (admits + blocks),
        "b0_peak_vram_gb": b0["peak_vram_gb"],
        "b2_peak_vram_gb": b2["peak_vram_gb"],
        "b0_runtime_sec": b0["runtime_sec"],
        "b2_runtime_sec": b2["runtime_sec"],
        "no_ground_truth": True,
        "quantitative_performance_claim_allowed": False,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("OUTPUT=", out_dir)


if __name__ == "__main__":
    main()
