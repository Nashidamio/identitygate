from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image


REPO = Path.home() / "thesis" / "identitygate"
DINO_REPO = Path.home() / "thesis" / "externals" / "dinov2"
WEIGHT = (
    Path.home()
    / "thesis"
    / "checkpoints"
    / "dinov2"
    / "dinov2_vits14_pretrain.pth"
)

JPEG_ROOT = Path(
    "/mnt/d/thesis_data/mosev2/train/JPEGImages"
)

CONFIG = (
    REPO
    / "configs"
    / "EXP047-visual-embeddings-v1.json"
)

VIDEO_MANIFEST = (
    REPO
    / "experiments"
    / "EXP045_di_gt_primitives"
    / "video_gt_primitives.csv"
)

OUTDIR = (
    REPO
    / "experiments"
    / "EXP047_visual_embeddings"
)

EMBEDDINGS_OUT = OUTDIR / "embeddings.npy"
VIDEO_FRAMES_OUT = OUTDIR / "video_frames.csv"
SUMMARY_OUT = OUTDIR / "summary.json"

if str(DINO_REPO) not in sys.path:
    sys.path.insert(
        0,
        str(DINO_REPO),
    )

from dinov2.data.transforms import make_classification_eval_transform
from dinov2.hub.backbones import dinov2_vits14


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git_head(path):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def git_status(path):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
        ],
        text=True,
    ).strip()


def require_clean_committed():
    status = git_status(REPO)

    if status:
        raise RuntimeError(
            "STOP: run requires clean thesis repo: "
            + status
        )

    for path in (
        Path(__file__).resolve(),
        CONFIG,
    ):
        rel = str(
            path.relative_to(REPO)
        )

        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "ls-files",
                "--error-unmatch",
                rel,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "STOP: run requires committed file: "
                + rel
            )


def read_csv(path):
    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv_atomic(
    path,
    fields,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        tmp,
        path,
    )


def write_json_atomic(
    path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def write_npy_atomic(
    path,
    array,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "wb",
    ) as f:
        np.save(
            f,
            array,
            allow_pickle=False,
        )

    os.replace(
        tmp,
        path,
    )


def middle_frame_index(
    n_frames,
):
    n_frames = int(
        n_frames
    )

    assert n_frames > 0

    return n_frames // 2


def frame_paths(video):
    paths = sorted(
        (
            JPEG_ROOT
            / video
        ).glob("*.jpg")
    )

    if not paths:
        raise RuntimeError(
            "STOP: no JPEG frames for "
            + video
        )

    return paths


def selected_frame(video):
    paths = frame_paths(
        video
    )

    idx = middle_frame_index(
        len(paths)
    )

    return (
        paths[idx],
        idx,
        len(paths),
    )


def validate_embedding_tensor(
    value,
    batch_size,
):
    assert isinstance(
        value,
        torch.Tensor,
    )

    assert tuple(
        value.shape
    ) == (
        int(batch_size),
        384,
    )

    assert (
        value.dtype
        == torch.float32
    )

    assert torch.isfinite(
        value
    ).all()

    return True


def load_context():
    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    manifest_sha = sha256_file(
        VIDEO_MANIFEST
    )

    assert (
        manifest_sha
        == cfg["input"][
            "primary_video_manifest_sha256"
        ]
    ), (
        manifest_sha,
        cfg["input"][
            "primary_video_manifest_sha256"
        ],
    )

    assert (
        git_head(DINO_REPO)
        == cfg["backbone"][
            "source_commit"
        ]
    )

    dino_status = git_status(
        DINO_REPO
    )

    assert not dino_status, (
        "STOP: DINOv2 source dirty: "
        + dino_status
    )

    assert WEIGHT.is_file()

    weight_sha = sha256_file(
        WEIGHT
    )

    assert (
        weight_sha
        == cfg["backbone"][
            "weights_sha256"
        ]
    ), (
        weight_sha,
        cfg["backbone"][
            "weights_sha256"
        ],
    )

    videos = read_csv(
        VIDEO_MANIFEST
    )

    expected = int(
        cfg["input"][
            "expected_primary_videos"
        ]
    )

    assert len(videos) == expected

    names = [
        row["video"]
        for row in videos
    ]

    assert len(set(names)) == expected

    videos = sorted(
        videos,
        key=lambda row:
            row["video"],
    )

    return {
        "cfg": cfg,
        "videos": videos,
        "weight_sha": weight_sha,
        "manifest_sha": manifest_sha,
    }


def load_model(
    device,
):
    state = torch.load(
        WEIGHT,
        map_location="cpu",
        weights_only=True,
    )

    assert isinstance(
        state,
        dict,
    )

    assert state

    model = dinov2_vits14(
        pretrained=False,
    )

    result = model.load_state_dict(
        state,
        strict=True,
    )

    assert not result.missing_keys
    assert not result.unexpected_keys

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    assert params == 22056576

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    model.eval()

    model.to(
        device
    )

    assert not any(
        p.requires_grad
        for p in model.parameters()
    )

    transform = (
        make_classification_eval_transform()
    )

    return (
        model,
        transform,
    )


def prepare_image(
    path,
    transform,
    device,
):
    with Image.open(
        path
    ) as image:
        rgb = image.convert(
            "RGB"
        )

        width, height = (
            rgb.size
        )

        tensor = transform(
            rgb
        )

    assert tuple(
        tensor.shape
    ) == (
        3,
        224,
        224,
    )

    assert (
        tensor.dtype
        == torch.float32
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    return (
        tensor,
        width,
        height,
    )


def embed_tensor(
    model,
    tensor,
):
    device_type = (
        tensor.device.type
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            value = model(
                tensor
            )

    validate_embedding_tensor(
        value,
        tensor.shape[0],
    )

    return value


def configure_fp32():
    previous = {
        "precision":
            torch.get_float32_matmul_precision(),
        "tf32":
            torch.backends.cuda.matmul.allow_tf32,
    }

    torch.set_float32_matmul_precision(
        "highest"
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        False
    )

    return previous


def restore_fp32(previous):
    torch.set_float32_matmul_precision(
        previous["precision"]
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        previous["tf32"]
    )


def plan(ctx):
    counts = []
    selected = []

    for row in ctx["videos"]:
        video = row["video"]

        path, idx, n_frames = (
            selected_frame(
                video
            )
        )

        counts.append(
            n_frames
        )

        selected.append(
            (
                video,
                idx,
                path.name,
            )
        )

    counts_array = np.asarray(
        counts,
        dtype=np.int64,
    )

    max_frames = int(
        counts_array.max()
    )

    max_videos = sorted(
        video
        for (
            video,
            _,
            _
        ), count in zip(
            selected,
            counts,
        )
        if count == max_frames
    )

    middle_row = (
        selected[
            len(selected) // 2
        ]
    )

    print("EXP047 PLAN")

    print(
        "identitygate_head =",
        git_head(REPO),
    )

    print(
        "dinov2_commit =",
        git_head(DINO_REPO),
    )

    print(
        "weight_sha256 =",
        ctx["weight_sha"],
    )

    print(
        "manifest_sha256 =",
        ctx["manifest_sha"],
    )

    print(
        "primary_videos =",
        len(ctx["videos"]),
    )

    print(
        "frame_count_min =",
        int(
            counts_array.min()
        ),
    )

    print(
        "frame_count_median =",
        float(
            np.median(
                counts_array
            )
        ),
    )

    print(
        "frame_count_max =",
        max_frames,
    )

    print(
        "max_frame_count_video =",
        max_videos[0],
    )

    print(
        "first_selected =",
        selected[0],
    )

    print(
        "middle_selected =",
        middle_row,
    )

    print(
        "last_selected =",
        selected[-1],
    )

    print(
        "frame_formula = n_frames // 2"
    )

    print(
        "embedding_dimension = 384"
    )

    print(
        "embedding_postprocessing = NONE"
    )

    print(
        "clustering_performed = false"
    )

    print(
        "planned_k =",
        ctx["cfg"][
            "downstream_predeclared"
        ]["planned_k"],
    )

    print(
        "di_v1_frozen = false"
    )

    print(
        "hard_set_selected = false"
    )

    print(
        "final_split_constructed = false"
    )

    print(
        "fresh_dev_evaluated = false"
    )

    print(
        "test_evaluated = false"
    )

    print(
        "EXP047_PLAN_PASS"
    )


def sanity(ctx):
    require_clean_committed()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "STOP: CUDA required"
        )

    if OUTDIR.exists():
        raise RuntimeError(
            "STOP: EXP047 output directory already exists"
        )

    previous = configure_fp32()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = None

    try:
        model, transform = load_model(
            torch.device("cuda")
        )

        videos = [
            row["video"]
            for row in ctx["videos"]
        ]

        selected_videos = [
            videos[0],
            videos[
                len(videos) // 2
            ],
            videos[-1],
        ]

        for position, video in enumerate(
            selected_videos
        ):
            path, idx, n_frames = (
                selected_frame(
                    video
                )
            )

            tensor, width, height = (
                prepare_image(
                    path,
                    transform,
                    torch.device("cuda"),
                )
            )

            value = embed_tensor(
                model,
                tensor,
            )

            if position == 0:
                repeat = embed_tensor(
                    model,
                    tensor,
                )

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        enabled=False,
                    ):
                        feature = (
                            model.forward_features(
                                tensor
                            )[
                                "x_norm_clstoken"
                            ]
                        )

                validate_embedding_tensor(
                    feature,
                    1,
                )

                repeat_diff = float(
                    (
                        value
                        - repeat
                    )
                    .abs()
                    .max()
                    .item()
                )

                direct_feature_diff = float(
                    (
                        value
                        - feature
                    )
                    .abs()
                    .max()
                    .item()
                )

                assert repeat_diff <= 1e-7, (
                    repeat_diff
                )

                assert (
                    direct_feature_diff
                    <= 1e-7
                ), direct_feature_diff

                print(
                    "REPEAT_MAX_ABS_DIFF =",
                    repeat_diff,
                )

                print(
                    "DIRECT_VS_CLS_MAX_ABS_DIFF =",
                    direct_feature_diff,
                )

            norm = float(
                torch.linalg.vector_norm(
                    value.float(),
                    dim=1,
                )[0].item()
            )

            print(
                "SANITY_VIDEO",
                video,
                "frame_idx=",
                idx,
                "n_frames=",
                n_frames,
                "frame=",
                path.name,
                "original_width=",
                width,
                "original_height=",
                height,
                "embedding_norm=",
                norm,
            )

        print(
            "peak_memory_allocated_gb =",
            torch.cuda.max_memory_allocated()
            / (1024 ** 3),
        )

        print(
            "peak_memory_reserved_gb =",
            torch.cuda.max_memory_reserved()
            / (1024 ** 3),
        )

        print(
            "SANITY_ARTIFACT_WRITTEN = false"
        )

        print(
            "fresh_dev_evaluated = false"
        )

        print(
            "test_evaluated = false"
        )

        print(
            "EXP047_SANITY_PASS"
        )

    finally:
        if model is not None:
            del model

        gc.collect()

        torch.cuda.empty_cache()

        restore_fp32(
            previous
        )


def run(ctx):
    require_clean_committed()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "STOP: CUDA required"
        )

    if OUTDIR.exists():
        raise RuntimeError(
            "STOP: EXP047 output directory exists; do not rerun"
        )

    previous = configure_fp32()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = None

    start = time.time()

    try:
        model, transform = load_model(
            torch.device("cuda")
        )

        embeddings = np.empty(
            (
                len(ctx["videos"]),
                384,
            ),
            dtype=np.float32,
        )

        metadata = []

        total = len(
            ctx["videos"]
        )

        for row_idx, row in enumerate(
            ctx["videos"]
        ):
            video = row["video"]

            path, frame_idx, n_frames = (
                selected_frame(
                    video
                )
            )

            tensor, width, height = (
                prepare_image(
                    path,
                    transform,
                    torch.device("cuda"),
                )
            )

            value = embed_tensor(
                model,
                tensor,
            )

            vector = (
                value[0]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            assert vector.shape == (
                384,
            )

            assert np.isfinite(
                vector
            ).all()

            embeddings[
                row_idx
            ] = vector

            metadata.append({
                "row_idx":
                    row_idx,
                "video":
                    video,
                "frame_count":
                    n_frames,
                "frame_idx":
                    frame_idx,
                "frame_name":
                    path.name,
                "original_width":
                    width,
                "original_height":
                    height,
            })

            if (
                (row_idx + 1) % 100 == 0
                or row_idx + 1 == total
            ):
                print(
                    "PROGRESS="
                    + str(row_idx + 1)
                    + "/"
                    + str(total),
                    flush=True,
                )

        assert embeddings.shape == (
            1170,
            384,
        )

        assert (
            embeddings.dtype
            == np.float32
        )

        assert np.isfinite(
            embeddings
        ).all()

        norms = np.linalg.norm(
            embeddings.astype(
                np.float64
            ),
            axis=1,
        )

        assert np.isfinite(
            norms
        ).all()

        write_npy_atomic(
            EMBEDDINGS_OUT,
            embeddings,
        )

        write_csv_atomic(
            VIDEO_FRAMES_OUT,
            [
                "row_idx",
                "video",
                "frame_count",
                "frame_idx",
                "frame_name",
                "original_width",
                "original_height",
            ],
            metadata,
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

        runtime = (
            time.time()
            - start
        )

        summary = {
            "experiment":
                "EXP047",
            "status":
                "VISUAL_EMBEDDINGS_COMPLETE_NOT_CLUSTERED_NOT_DI_FREEZE",
            "identitygate_commit_at_execution":
                git_head(REPO),
            "dinov2_source_commit":
                git_head(DINO_REPO),
            "weight_sha256":
                sha256_file(
                    WEIGHT
                ),
            "config_sha256":
                sha256_file(
                    CONFIG
                ),
            "script_sha256":
                sha256_file(
                    Path(__file__).resolve()
                ),
            "primary_videos":
                len(ctx["videos"]),
            "frames_per_video":
                1,
            "frame_index_formula":
                "n_frames // 2",
            "embedding_dimension":
                384,
            "embedding_dtype":
                "float32",
            "embedding_representation":
                "model(x) = IdentityHead(x_norm_clstoken)",
            "embedding_postprocessing":
                "NONE",
            "l2_normalization_after_model":
                False,
            "preprocessing":{
                "resize_size":
                    256,
                "interpolation":
                    "BICUBIC",
                "center_crop_size":
                    224,
                "normalize_mean":[
                    0.485,
                    0.456,
                    0.406,
                ],
                "normalize_std":[
                    0.229,
                    0.224,
                    0.225,
                ],
            },
            "embedding_norm":{
                "min":
                    float(
                        norms.min()
                    ),
                "median":
                    float(
                        np.median(
                            norms
                        )
                    ),
                "max":
                    float(
                        norms.max()
                    ),
                "mean":
                    float(
                        norms.mean()
                    ),
            },
            "max_peak_memory_allocated_gb":
                peak_allocated,
            "max_peak_memory_reserved_gb":
                peak_reserved,
            "runtime_sec":
                runtime,
            "clustering_performed":
                False,
            "planned_k":
                20,
            "hard_set_selected":
                False,
            "di_v1_frozen":
                False,
            "final_split_constructed":
                False,
            "fresh_dev_evaluated":
                False,
            "test_evaluated":
                False,
            "output_hashes":{
                "embeddings.npy":
                    sha256_file(
                        EMBEDDINGS_OUT
                    ),
                "video_frames.csv":
                    sha256_file(
                        VIDEO_FRAMES_OUT
                    ),
            },
        }

        write_json_atomic(
            SUMMARY_OUT,
            summary,
        )

        print(
            "primary_videos =",
            len(ctx["videos"]),
        )

        print(
            "embedding_shape =",
            tuple(
                embeddings.shape
            ),
        )

        print(
            "embedding_dtype =",
            embeddings.dtype,
        )

        print(
            "embedding_norm_min =",
            summary[
                "embedding_norm"
            ]["min"],
        )

        print(
            "embedding_norm_median =",
            summary[
                "embedding_norm"
            ]["median"],
        )

        print(
            "embedding_norm_max =",
            summary[
                "embedding_norm"
            ]["max"],
        )

        print(
            "peak_memory_allocated_gb =",
            peak_allocated,
        )

        print(
            "peak_memory_reserved_gb =",
            peak_reserved,
        )

        print(
            "runtime_sec =",
            runtime,
        )

        print(
            "embeddings_sha256 =",
            summary[
                "output_hashes"
            ]["embeddings.npy"],
        )

        print(
            "video_frames_sha256 =",
            summary[
                "output_hashes"
            ]["video_frames.csv"],
        )

        print(
            "summary =",
            SUMMARY_OUT,
        )

        print(
            "EXP047_RUN_PASS"
        )

    finally:
        if model is not None:
            del model

        gc.collect()

        torch.cuda.empty_cache()

        restore_fp32(
            previous
        )


def selftest():
    assert (
        middle_frame_index(1)
        == 0
    )

    assert (
        middle_frame_index(2)
        == 1
    )

    assert (
        middle_frame_index(3)
        == 1
    )

    assert (
        middle_frame_index(4)
        == 2
    )

    value = torch.zeros(
        2,
        384,
        dtype=torch.float32,
    )

    assert validate_embedding_tensor(
        value,
        2,
    )

    print(
        "EXP047_SELFTEST=PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "selftest",
            "plan",
            "sanity",
            "run",
        ],
    )

    args = parser.parse_args()

    if args.mode == "selftest":
        selftest()
        return

    ctx = load_context()

    if args.mode == "plan":
        plan(ctx)

    elif args.mode == "sanity":
        sanity(ctx)

    else:
        run(ctx)


if __name__ == "__main__":
    main()
