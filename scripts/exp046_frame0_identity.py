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
from sam3.model_builder import build_sam3_video_model


REPO = Path.home() / "thesis" / "identitygate"
SAM_REPO = Path.home() / "thesis" / "externals" / "sam3"
FRAMES = Path("/mnt/d/thesis_data/mosev2/train/JPEGImages")
ANN = Path("/mnt/d/thesis_data/mosev2/train/Annotations")

CONFIG = REPO / "configs/EXP046-frame0-identity-v1.json"
EVENT_IN = REPO / "experiments/EXP045_di_gt_primitives/event_gt_primitives.csv"
VIDEO_IN = REPO / "experiments/EXP045_di_gt_primitives/video_gt_primitives.csv"
SUMMARY_IN = REPO / "experiments/EXP045_di_gt_primitives/summary.json"
EXP022_ANCHORS = REPO / "experiments/EXP022_full/anchor_cosine.csv"

OUTDIR = REPO / "experiments/EXP046_frame0_identity"
EVENT_OUT = OUTDIR / "event_identity_primitives.csv"
VIDEO_OUT = OUTDIR / "video_identity_primitives.csv"
ANCHOR_OUT = OUTDIR / "anchor_cosine.csv"
SUMMARY_OUT = OUTDIR / "summary.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_clean_committed():
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        text=True,
    ).strip()

    if status:
        raise RuntimeError(
            "STOP: run requires clean repo: " + status
        )

    for path in (
        Path(__file__).resolve(),
        CONFIG,
    ):
        rel = str(path.relative_to(REPO))

        ok = subprocess.run(
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
        ).returncode == 0

        if not ok:
            raise RuntimeError(
                "STOP: run requires committed " + rel
            )


def read_csv(path):
    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path, fields, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

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

    os.replace(tmp, path)


def write_json_atomic(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def cosine_matrix_fp32(x, y):
    device_type = x.device.type

    prev_tf32 = None
    prev_precision = None

    if device_type == "cuda":
        prev_tf32 = (
            torch.backends.cuda.matmul.allow_tf32
        )

        prev_precision = (
            torch.get_float32_matmul_precision()
        )

        torch.backends.cuda.matmul.allow_tf32 = False

        torch.set_float32_matmul_precision(
            "highest"
        )

    try:
        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            x = F.normalize(
                x.float(),
                dim=-1,
            )

            y = F.normalize(
                y.float(),
                dim=-1,
            )

            out = x @ y.T

    finally:
        if device_type == "cuda":
            torch.set_float32_matmul_precision(
                prev_precision
            )

            torch.backends.cuda.matmul.allow_tf32 = (
                prev_tf32
            )

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    return out


def find_checkpoint(expected_sha):
    candidates = []

    direct = (
        Path.home()
        / "thesis"
        / "checkpoints"
        / "sam3"
        / "sam3.pt"
    )

    if direct.is_file():
        candidates.append(direct)

    root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--facebook--sam3"
        / "snapshots"
    )

    if root.is_dir():
        candidates.extend(
            sorted(
                root.glob("*/sam3.pt")
            )
        )

    checked = set()

    for path in candidates:
        real = str(path.resolve())

        if real in checked:
            continue

        checked.add(real)

        if sha256_file(path) == expected_sha:
            return path

    raise RuntimeError(
        "STOP: no local SAM3 checkpoint matches "
        + expected_sha
    )


def frame0_annotation(video):
    pngs = sorted(
        (ANN / video).glob("*.png")
    )

    if not pngs:
        raise RuntimeError(
            "STOP: no annotations for " + video
        )

    a0 = np.array(
        Image.open(pngs[0])
    )

    object_ids = sorted(
        int(x)
        for x in np.unique(a0)
        if int(x) != 0
    )

    if not object_ids:
        raise RuntimeError(
            "STOP: no frame0 objects for " + video
        )

    return a0, object_ids


def load_context():
    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    actual_hashes = {
        "event_gt_primitives.csv":
            sha256_file(EVENT_IN),
        "video_gt_primitives.csv":
            sha256_file(VIDEO_IN),
        "summary.json":
            sha256_file(SUMMARY_IN),
    }

    assert (
        actual_hashes
        == cfg["inputs"]["sha256"]
    ), (
        actual_hashes,
        cfg["inputs"]["sha256"],
    )

    summary45 = json.loads(
        SUMMARY_IN.read_text(
            encoding="utf-8"
        )
    )

    assert summary45["primary_events"] == 2701
    assert summary45["primary_videos"] == 1170
    assert summary45["di_v1_frozen"] is False

    assert (
        summary45["final_split_constructed"]
        is False
    )

    assert (
        summary45["fresh_dev_evaluated"]
        is False
    )

    assert (
        summary45["test_evaluated"]
        is False
    )

    events = read_csv(EVENT_IN)
    videos = read_csv(VIDEO_IN)

    assert len(events) == 2701
    assert len(videos) == 1170

    multi = [
        row
        for row in videos
        if int(row["frame0_object_count"]) >= 2
    ]

    single = [
        row
        for row in videos
        if int(row["frame0_object_count"]) == 1
    ]

    with_comp = [
        row
        for row in events
        if int(row["frame0_competitor_count"]) >= 1
    ]

    zero_comp = [
        row
        for row in events
        if int(row["frame0_competitor_count"]) == 0
    ]

    scope = cfg["scope"]

    assert (
        len(multi)
        == scope["multi_object_videos"]
    )

    assert (
        len(single)
        == scope["single_object_videos"]
    )

    assert (
        len(with_comp)
        == scope["events_with_competitor"]
    )

    assert (
        len(zero_comp)
        == scope["events_without_competitor"]
    )

    sam_commit = git_head(SAM_REPO)

    assert (
        sam_commit
        == cfg["substrate"]["sam_source_commit"]
    )

    checkpoint = find_checkpoint(
        cfg["substrate"]["checkpoint_sha256"]
    )

    return {
        "cfg": cfg,
        "events": events,
        "videos": videos,
        "multi": multi,
        "single": single,
        "checkpoint": checkpoint,
    }


def build_predictor(ctx):
    model = build_sam3_video_model(
        checkpoint_path=str(
            ctx["checkpoint"]
        ),
        load_from_HF=False,
    )

    predictor = model.tracker

    predictor.backbone = (
        model.detector.backbone
    )

    return model, predictor


def extract_frame0(
    predictor,
    video,
):
    a0, object_ids = frame0_annotation(
        video
    )

    state = predictor.init_state(
        video_path=str(
            FRAMES / video
        )
    )

    try:
        predictor.clear_all_points_in_video(
            state
        )

        for oid in object_ids:
            mask = torch.from_numpy(
                a0 == oid
            )

            assert mask.any()

            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=oid,
                mask=mask,
            )

        assert (
            list(state["obj_ids"])
            == object_ids
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

        assert tuple(
            anchors.shape
        ) == (
            len(object_ids),
            predictor.hidden_dim,
        )

        assert torch.isfinite(
            anchors.float()
        ).all()

        cosine = cosine_matrix_fp32(
            anchors,
            anchors,
        )

        diagonal = torch.diagonal(
            cosine
        )

        assert torch.allclose(
            diagonal,
            torch.ones_like(
                diagonal
            ),
            atol=1e-5,
            rtol=1e-5,
        )

        return (
            object_ids,
            cosine.detach().cpu(),
        )

    finally:
        del state

        gc.collect()

        torch.cuda.empty_cache()


def anchor_rows(
    video,
    object_ids,
    cosine,
):
    matrix = cosine.numpy()

    return [
        {
            "video": video,
            "object_id_i": oid_i,
            "object_id_j": oid_j,
            "cosine_fp32":
                float(matrix[i, j]),
        }
        for i, oid_i in enumerate(
            object_ids
        )
        for j, oid_j in enumerate(
            object_ids
        )
    ]


def event_identity_rows(
    events,
    object_ids,
    cosine,
):
    n_objects = len(
        object_ids
    )

    index = {
        oid: i
        for i, oid in enumerate(
            object_ids
        )
    }

    rows = []

    if n_objects == 1:
        for event in events:
            oid = int(
                event["object_id"]
            )

            assert (
                oid
                == object_ids[0]
            )

            assert (
                int(
                    event[
                        "frame0_competitor_count"
                    ]
                )
                == 0
            )

            rows.append({
                "video":
                    event["video"],
                "object_id":
                    oid,
                "disappear_start":
                    int(
                        event[
                            "disappear_start"
                        ]
                    ),
                "reappear_frame":
                    int(
                        event[
                            "reappear_frame"
                        ]
                    ),
                "frame0_object_count":
                    1,
                "frame0_competitor_count":
                    0,
                "identity_pressure_status":
                    "NO_TRACKED_COMPETITOR",
                "max_other_anchor_cos_fp32":
                    None,
                "max_other_object_id":
                    None,
            })

        return rows

    assert cosine is not None

    masked = cosine.masked_fill(
        torch.eye(
            n_objects,
            dtype=torch.bool,
        ),
        float("-inf"),
    )

    values, indices = masked.max(
        dim=1
    )

    for event in events:
        oid = int(
            event["object_id"]
        )

        assert oid in index

        assert (
            int(
                event[
                    "frame0_object_count"
                ]
            )
            == n_objects
        )

        assert (
            int(
                event[
                    "frame0_competitor_count"
                ]
            )
            == n_objects - 1
        )

        i = index[oid]

        j = int(
            indices[i].item()
        )

        competitor_oid = (
            object_ids[j]
        )

        value = float(
            values[i].item()
        )

        assert competitor_oid != oid
        assert np.isfinite(value)

        rows.append({
            "video":
                event["video"],
            "object_id":
                oid,
            "disappear_start":
                int(
                    event[
                        "disappear_start"
                    ]
                ),
            "reappear_frame":
                int(
                    event[
                        "reappear_frame"
                    ]
                ),
            "frame0_object_count":
                n_objects,
            "frame0_competitor_count":
                n_objects - 1,
            "identity_pressure_status":
                "TRACKED_COMPETITOR_AVAILABLE",
            "max_other_anchor_cos_fp32":
                value,
            "max_other_object_id":
                competitor_oid,
        })

    return rows


def plan(ctx):
    print("EXP046 PLAN")

    print(
        "identitygate_head =",
        git_head(REPO),
    )

    print(
        "sam3_commit =",
        git_head(SAM_REPO),
    )

    print(
        "checkpoint_path =",
        ctx["checkpoint"],
    )

    print(
        "checkpoint_sha256 =",
        sha256_file(
            ctx["checkpoint"]
        ),
    )

    print(
        "primary_events =",
        len(ctx["events"]),
    )

    print(
        "primary_videos =",
        len(ctx["videos"]),
    )

    print(
        "multi_object_videos =",
        len(ctx["multi"]),
    )

    print(
        "single_object_videos =",
        len(ctx["single"]),
    )

    max_objects = max(
        int(
            row[
                "frame0_object_count"
            ]
        )
        for row in ctx["multi"]
    )

    max_videos = sorted(
        row["video"]
        for row in ctx["multi"]
        if int(
            row[
                "frame0_object_count"
            ]
        ) == max_objects
    )

    print(
        "max_frame0_objects =",
        max_objects,
    )

    print(
        "deterministic_max_object_video =",
        max_videos[0],
    )

    print(
        "events_with_competitor =",
        ctx["cfg"]["scope"][
            "events_with_competitor"
        ],
    )

    print(
        "events_without_competitor =",
        ctx["cfg"]["scope"][
            "events_without_competitor"
        ],
    )

    print(
        "propagation_after_frame0 = false"
    )

    print(
        "single_object_raw_cosine_fabricated = false"
    )

    print(
        "di_v1_frozen = false"
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

    print("EXP046_PLAN_PASS")


def sanity(ctx):
    require_clean_committed()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "STOP: CUDA required"
        )

    if not EXP022_ANCHORS.is_file():
        raise RuntimeError(
            "STOP: missing "
            + str(EXP022_ANCHORS)
        )

    historical = read_csv(
        EXP022_ANCHORS
    )

    by_video = {}

    for row in historical:
        by_video.setdefault(
            row["video"],
            [],
        ).append(row)

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    model, predictor = build_predictor(
        ctx
    )

    results = []

    try:
        for video in ctx["cfg"]["sanity"][
            "historical_equivalence_videos"
        ]:
            assert video in by_video

            start = time.time()

            object_ids, cosine = (
                extract_frame0(
                    predictor,
                    video,
                )
            )

            new = {
                (
                    int(row["object_id_i"]),
                    int(row["object_id_j"]),
                ):
                float(
                    row["cosine_fp32"]
                )
                for row in anchor_rows(
                    video,
                    object_ids,
                    cosine,
                )
            }

            old = {
                (
                    int(row["object_id_i"]),
                    int(row["object_id_j"]),
                ):
                float(
                    row["cosine_fp32"]
                )
                for row in by_video[
                    video
                ]
            }

            assert set(new) == set(old)

            max_diff = max(
                abs(
                    new[key]
                    - old[key]
                )
                for key in new
            )

            tolerance = float(
                ctx["cfg"]["sanity"][
                    "max_abs_cosine_diff"
                ]
            )

            assert (
                max_diff
                <= tolerance
            ), (
                video,
                max_diff,
                tolerance,
            )

            results.append({
                "video":
                    video,
                "role":
                    "HISTORICAL_EQUIVALENCE",
                "n_objects":
                    len(object_ids),
                "max_abs_cosine_diff_vs_EXP022":
                    max_diff,
                "runtime_sec":
                    time.time() - start,
            })

        max_objects = max(
            int(
                row[
                    "frame0_object_count"
                ]
            )
            for row in ctx["multi"]
        )

        max_primary = sorted(
            row["video"]
            for row in ctx["multi"]
            if int(
                row[
                    "frame0_object_count"
                ]
            ) == max_objects
        )[0]

        start = time.time()

        object_ids, cosine = (
            extract_frame0(
                predictor,
                max_primary,
            )
        )

        assert (
            len(object_ids)
            == max_objects
        )

        assert tuple(
            cosine.shape
        ) == (
            max_objects,
            max_objects,
        )

        results.append({
            "video":
                max_primary,
            "role":
                "MAX_PRIMARY_OBJECT_COUNT_VRAM",
            "n_objects":
                len(object_ids),
            "max_abs_cosine_diff_vs_EXP022":
                None,
            "runtime_sec":
                time.time() - start,
        })

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

        print("EXP046_SANITY_PASS")

        for row in results:
            print(
                "SANITY_VIDEO",
                row["video"],
                "role=",
                row["role"],
                "objects=",
                row["n_objects"],
                "max_abs_diff=",
                row[
                    "max_abs_cosine_diff_vs_EXP022"
                ],
                "runtime_sec=",
                row["runtime_sec"],
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
            "SANITY_ARTIFACT_WRITTEN = false"
        )

        print(
            "fresh_dev_evaluated = false"
        )

        print(
            "test_evaluated = false"
        )

    finally:
        del predictor
        del model

        gc.collect()

        torch.cuda.empty_cache()


def run(ctx):
    require_clean_committed()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "STOP: CUDA required"
        )

    if OUTDIR.exists():
        raise RuntimeError(
            "STOP: EXP046 output directory exists; do not rerun"
        )

    events_by_video = {}

    for event in ctx["events"]:
        events_by_video.setdefault(
            event["video"],
            [],
        ).append(event)

    torch.cuda.empty_cache()

    model, predictor = build_predictor(
        ctx
    )

    all_anchor_rows = []
    event_rows = []
    video_rows = []

    max_peak = 0.0
    total_runtime = 0.0

    try:
        total_multi = len(
            ctx["multi"]
        )

        for idx, video_row in enumerate(
            ctx["multi"],
            start=1,
        ):
            video = (
                video_row["video"]
            )

            expected_n = int(
                video_row[
                    "frame0_object_count"
                ]
            )

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start = time.time()

            object_ids, cosine = (
                extract_frame0(
                    predictor,
                    video,
                )
            )

            elapsed = (
                time.time()
                - start
            )

            peak = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

            assert (
                len(object_ids)
                == expected_n
            )

            arows = anchor_rows(
                video,
                object_ids,
                cosine,
            )

            erows = event_identity_rows(
                events_by_video[
                    video
                ],
                object_ids,
                cosine,
            )

            all_anchor_rows.extend(
                arows
            )

            event_rows.extend(
                erows
            )

            video_rows.append({
                "video":
                    video,
                "primary_event_count":
                    len(erows),
                "frame0_object_count":
                    expected_n,
                "has_frame0_competitor":
                    1,
                "sam_frame0_extraction_performed":
                    1,
                "anchor_pair_rows":
                    len(arows),
                "runtime_sec":
                    elapsed,
                "peak_memory_allocated_gb":
                    peak,
            })

            max_peak = max(
                max_peak,
                peak,
            )

            total_runtime += elapsed

            if (
                idx % 20 == 0
                or idx == total_multi
            ):
                print(
                    f"PROGRESS={idx}/{total_multi}",
                    flush=True,
                )

        for video_row in ctx["single"]:
            video = (
                video_row["video"]
            )

            _, object_ids = (
                frame0_annotation(
                    video
                )
            )

            assert (
                len(object_ids)
                == 1
            )

            erows = event_identity_rows(
                events_by_video[
                    video
                ],
                object_ids,
                None,
            )

            event_rows.extend(
                erows
            )

            video_rows.append({
                "video":
                    video,
                "primary_event_count":
                    len(erows),
                "frame0_object_count":
                    1,
                "has_frame0_competitor":
                    0,
                "sam_frame0_extraction_performed":
                    0,
                "anchor_pair_rows":
                    0,
                "runtime_sec":
                    None,
                "peak_memory_allocated_gb":
                    None,
            })

        assert len(
            event_rows
        ) == 2701

        assert len(
            video_rows
        ) == 1170

        with_comp = [
            row
            for row in event_rows
            if row[
                "identity_pressure_status"
            ]
            == "TRACKED_COMPETITOR_AVAILABLE"
        ]

        without_comp = [
            row
            for row in event_rows
            if row[
                "identity_pressure_status"
            ]
            == "NO_TRACKED_COMPETITOR"
        ]

        assert len(
            with_comp
        ) == 1279

        assert len(
            without_comp
        ) == 1422

        assert all(
            row[
                "max_other_anchor_cos_fp32"
            ]
            is not None
            for row in with_comp
        )

        assert all(
            row[
                "max_other_anchor_cos_fp32"
            ]
            is None
            and row[
                "max_other_object_id"
            ]
            is None
            for row in without_comp
        )

        event_rows.sort(
            key=lambda row: (
                row["video"],
                int(
                    row["object_id"]
                ),
                int(
                    row[
                        "reappear_frame"
                    ]
                ),
            )
        )

        video_rows.sort(
            key=lambda row:
                row["video"]
        )

        all_anchor_rows.sort(
            key=lambda row: (
                row["video"],
                int(
                    row[
                        "object_id_i"
                    ]
                ),
                int(
                    row[
                        "object_id_j"
                    ]
                ),
            )
        )

        event_fields = [
            "video",
            "object_id",
            "disappear_start",
            "reappear_frame",
            "frame0_object_count",
            "frame0_competitor_count",
            "identity_pressure_status",
            "max_other_anchor_cos_fp32",
            "max_other_object_id",
        ]

        video_fields = [
            "video",
            "primary_event_count",
            "frame0_object_count",
            "has_frame0_competitor",
            "sam_frame0_extraction_performed",
            "anchor_pair_rows",
            "runtime_sec",
            "peak_memory_allocated_gb",
        ]

        anchor_fields = [
            "video",
            "object_id_i",
            "object_id_j",
            "cosine_fp32",
        ]

        write_csv_atomic(
            EVENT_OUT,
            event_fields,
            event_rows,
        )

        write_csv_atomic(
            VIDEO_OUT,
            video_fields,
            video_rows,
        )

        write_csv_atomic(
            ANCHOR_OUT,
            anchor_fields,
            all_anchor_rows,
        )

        summary = {
            "experiment":
                "EXP046",
            "status":
                "FRAME0_IDENTITY_PRIMITIVES_COMPLETE_NOT_DI_FREEZE",
            "identitygate_commit_at_execution":
                git_head(REPO),
            "script_sha256":
                sha256_file(
                    Path(__file__).resolve()
                ),
            "config_sha256":
                sha256_file(CONFIG),
            "checkpoint_sha256":
                sha256_file(
                    ctx["checkpoint"]
                ),
            "primary_events":
                2701,
            "primary_videos":
                1170,
            "multi_object_videos":
                len(ctx["multi"]),
            "single_object_videos":
                len(ctx["single"]),
            "events_with_competitor":
                len(with_comp),
            "events_without_competitor":
                len(without_comp),
            "anchor_pair_rows":
                len(
                    all_anchor_rows
                ),
            "max_peak_memory_allocated_gb":
                max_peak,
            "total_multi_video_runtime_sec":
                total_runtime,
            "single_object_raw_cosine_fabricated":
                False,
            "sam_or_gate_recovery_inference_performed":
                False,
            "propagation_after_frame0_performed":
                False,
            "di_v1_frozen":
                False,
            "hard_set_selected":
                False,
            "final_split_constructed":
                False,
            "fresh_dev_evaluated":
                False,
            "test_evaluated":
                False,
            "output_hashes":{
                EVENT_OUT.name:
                    sha256_file(
                        EVENT_OUT
                    ),
                VIDEO_OUT.name:
                    sha256_file(
                        VIDEO_OUT
                    ),
                ANCHOR_OUT.name:
                    sha256_file(
                        ANCHOR_OUT
                    ),
            },
        }

        write_json_atomic(
            SUMMARY_OUT,
            summary,
        )

        print(
            "events =",
            len(event_rows),
        )

        print(
            "videos =",
            len(video_rows),
        )

        print(
            "events_with_competitor =",
            len(with_comp),
        )

        print(
            "events_without_competitor =",
            len(without_comp),
        )

        print(
            "anchor_pair_rows =",
            len(
                all_anchor_rows
            ),
        )

        print(
            "max_peak_memory_allocated_gb =",
            max_peak,
        )

        print(
            "event_sha256 =",
            summary[
                "output_hashes"
            ][EVENT_OUT.name],
        )

        print(
            "video_sha256 =",
            summary[
                "output_hashes"
            ][VIDEO_OUT.name],
        )

        print(
            "anchor_sha256 =",
            summary[
                "output_hashes"
            ][ANCHOR_OUT.name],
        )

        print(
            "summary =",
            SUMMARY_OUT,
        )

        print("EXP046_RUN_PASS")

    finally:
        del predictor
        del model

        gc.collect()

        torch.cuda.empty_cache()


def selftest():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    cosine = cosine_matrix_fp32(
        x,
        x,
    )

    assert torch.allclose(
        cosine,
        torch.eye(2),
        atol=1e-7,
        rtol=1e-7,
    )

    single = [{
        "video":
            "v",
        "object_id":
            "7",
        "disappear_start":
            "10",
        "reappear_frame":
            "15",
        "frame0_object_count":
            "1",
        "frame0_competitor_count":
            "0",
    }]

    rows = event_identity_rows(
        single,
        [7],
        None,
    )

    assert (
        rows[0][
            "identity_pressure_status"
        ]
        == "NO_TRACKED_COMPETITOR"
    )

    assert (
        rows[0][
            "max_other_anchor_cos_fp32"
        ]
        is None
    )

    multi = [{
        "video":
            "v",
        "object_id":
            "7",
        "disappear_start":
            "10",
        "reappear_frame":
            "15",
        "frame0_object_count":
            "3",
        "frame0_competitor_count":
            "2",
    }]

    matrix = torch.tensor(
        [
            [1.0, 0.2, 0.8],
            [0.2, 1.0, 0.3],
            [0.8, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )

    rows = event_identity_rows(
        multi,
        [7, 8, 9],
        matrix,
    )

    assert (
        rows[0][
            "max_other_object_id"
        ]
        == 9
    )

    assert abs(
        rows[0][
            "max_other_anchor_cos_fp32"
        ]
        - 0.8
    ) < 1e-6

    print("EXP046_SELFTEST=PASS")


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
