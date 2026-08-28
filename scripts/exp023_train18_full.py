from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np

import exp023_train18_primitive_cache as base


REPO = Path.home() / "thesis" / "identitygate"

OUTROOT = (
    REPO
    / "experiments"
    / "EXP023_train18_cache"
)

PER_VIDEO_ROOT = OUTROOT / "per_video"

EXTRACTOR_PATH = (
    REPO
    / "scripts"
    / "exp023_train18_primitive_cache.py"
)

EXPECTED_EXTRACTOR_SHA256 = (
    "e3f78478171083eab8fc5955b711be39337aa2838da7a05bb6c67771a76c536b"
)


def compute_scope_totals(train):
    candidate_rows = 0
    anchor_rows = 0

    for row in train:
        n_frames = int(row["n_frames"])

        object_ids = [
            int(x)
            for x in row["frame0_object_ids"]
        ]

        candidate_rows += (
            (n_frames - 1)
            * len(object_ids)
        )

        anchor_rows += len(object_ids)

    return candidate_rows, anchor_rows


def print_scope(train):
    print("=== EXP023 FULL TRAIN18 SCOPE ===")

    for i, row in enumerate(train, 1):
        video = row["video"]
        n_frames = int(row["n_frames"])

        object_ids = [
            int(x)
            for x in row["frame0_object_ids"]
        ]

        n_objects = len(object_ids)

        candidate_rows = (
            (n_frames - 1)
            * n_objects
        )

        print(
            f"[{i:02d}/{len(train)}] "
            f"{video} "
            f"frames={n_frames} "
            f"objects={n_objects} "
            f"candidate_rows={candidate_rows}"
        )

        print(
            "  frames_dir =",
            base.FRAMES / video,
        )

        print(
            "  annotations_dir =",
            base.ANN / video,
        )


def merge_primitives(train):
    merged_path = OUTROOT / "primitives.csv"

    total_rows = 0

    with merged_path.open(
        "w",
        newline="",
    ) as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=base.FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()

        for row in train:
            video = row["video"]

            path = (
                PER_VIDEO_ROOT
                / f"TRAIN_{video}"
                / "primitives.csv"
            )

            with path.open(newline="") as fin:
                reader = csv.DictReader(fin)

                assert reader.fieldnames == (
                    base.FIELDS
                )

                for r in reader:
                    writer.writerow(r)
                    total_rows += 1

    return merged_path, total_rows


def write_index(train):
    path = OUTROOT / "index.csv"

    fields = [
        "split",
        "video",
        "n_frames",
        "n_objects",
        "candidate_rows",
        "pointer_rows",
        "anchor_pointer_rows",
        "peak_memory_allocated_gb",
        "runtime_sec",
        "exp022_object_score_max_abs_diff",
        "exp022_pointer_valid_all_equal",
        "exp022_valid_anchor_cos_max_abs_diff",
        "primitives_sha256",
        "pointers_sha256",
    ]

    rows = []

    for scope_row in train:
        video = scope_row["video"]

        summary_path = (
            PER_VIDEO_ROOT
            / f"TRAIN_{video}"
            / "summary.json"
        )

        summary = json.load(
            open(summary_path)
        )

        assert summary["video"] == video
        assert summary["split"] == "TRAIN"
        assert summary["status"] == (
            "SANITY_PASS"
        )

        assert (
            summary["dev_videos_touched"]
            == 0
        )

        assert (
            summary["test_videos_touched"]
            == 0
        )

        assert (
            summary[
                "exp022_pointer_valid_all_equal"
            ]
            is True
        )

        rows.append({
            "split": "TRAIN",
            "video": video,
            "n_frames": summary["n_frames"],
            "n_objects": summary["n_objects"],
            "candidate_rows": (
                summary["candidate_rows"]
            ),
            "pointer_rows": (
                summary["pointer_shape"][0]
            ),
            "anchor_pointer_rows": (
                summary[
                    "anchor_pointer_shape"
                ][0]
            ),
            "peak_memory_allocated_gb": (
                summary[
                    "peak_memory_allocated_gb"
                ]
            ),
            "runtime_sec": (
                summary["runtime_sec"]
            ),
            "exp022_object_score_max_abs_diff": (
                summary[
                    "exp022_object_score_max_abs_diff"
                ]
            ),
            "exp022_pointer_valid_all_equal": (
                summary[
                    "exp022_pointer_valid_all_equal"
                ]
            ),
            "exp022_valid_anchor_cos_max_abs_diff": (
                summary[
                    "exp022_valid_anchor_cos_max_abs_diff"
                ]
            ),
            "primitives_sha256": (
                summary["primitives_sha256"]
            ),
            "pointers_sha256": (
                summary["pointers_sha256"]
            ),
        })

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    return path, rows


def verify_pointer_files(train):
    pointer_rows = 0
    anchor_rows = 0

    for scope_row in train:
        video = scope_row["video"]

        path = (
            PER_VIDEO_ROOT
            / f"TRAIN_{video}"
            / "pointers.npz"
        )

        with np.load(path) as x:
            frames = x["frame_idx"]
            objects = x["object_id"]
            ptr = x["ptr_f32"]

            anchor_objects = (
                x["anchor_object_id"]
            )

            anchor_ptr = (
                x["anchor_ptr_f32"]
            )

            assert len(frames) == len(objects)
            assert len(frames) == ptr.shape[0]
            assert ptr.shape[1] == 256

            assert len(anchor_objects) == (
                anchor_ptr.shape[0]
            )

            assert anchor_ptr.shape[1] == 256

            pointer_rows += ptr.shape[0]
            anchor_rows += anchor_ptr.shape[0]

    return pointer_rows, anchor_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "preflight",
            "full",
        ],
    )

    args = parser.parse_args()

    cfg, scope, train, checkpoint = (
        base.load_context()
    )

    assert base.sha256_file(
        EXTRACTOR_PATH
    ) == EXPECTED_EXTRACTOR_SHA256

    train = sorted(
        train,
        key=lambda r: r["video"],
    )

    assert len(train) == 18

    assert {
        r["split"]
        for r in train
    } == {"TRAIN"}

    expected_rows, expected_anchors = (
        compute_scope_totals(train)
    )

    print(
        "identitygate_head =",
        base.git_commit(REPO),
    )

    print(
        "extractor_sha256 =",
        base.sha256_file(
            EXTRACTOR_PATH
        ),
    )

    print(
        "config_sha256 =",
        base.sha256_file(
            base.CONFIG_PATH
        ),
    )

    print(
        "scope_sha256 =",
        base.sha256_file(
            base.SCOPE_PATH
        ),
    )

    print(
        "train_videos =",
        len(train),
    )

    print(
        "computed_candidate_rows =",
        expected_rows,
    )

    print(
        "computed_anchor_pointer_rows =",
        expected_anchors,
    )

    print(
        "config_expected_candidate_rows =",
        cfg["scope"][
            "expected_candidate_rows"
        ],
    )

    print(
        "config_expected_anchor_pointer_rows =",
        cfg["scope"][
            "expected_anchor_pointer_rows"
        ],
    )

    assert expected_rows == (
        cfg["scope"][
            "expected_candidate_rows"
        ]
    )

    assert expected_anchors == (
        cfg["scope"][
            "expected_anchor_pointer_rows"
        ]
    )

    assert (
        cfg["scope"]["dev_videos_touched"]
        == 0
    )

    assert (
        cfg["scope"]["test_videos_touched"]
        == 0
    )

    print_scope(train)

    if args.mode == "preflight":
        print()
        print(
            "EXP023_FULL_PREFLIGHT_PASS"
        )
        return

    assert not OUTROOT.exists(), (
        f"Refusing to overwrite existing "
        f"output root: {OUTROOT}"
    )

    PER_VIDEO_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    print()
    print("=== BUILD SAM3 CORE ===")

    model = base.build_sam3_video_model(
        checkpoint_path=str(checkpoint),
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

    start = time.time()

    for i, row in enumerate(train, 1):
        video = row["video"]

        outdir = (
            PER_VIDEO_ROOT
            / f"TRAIN_{video}"
        )

        print()
        print(
            f"=== FULL VIDEO "
            f"{i}/{len(train)}: "
            f"{video} ==="
        )

        base.run_video(
            predictor,
            row,
            checkpoint,
            outdir,
        )

    merged_path, merged_rows = (
        merge_primitives(train)
    )

    index_path, index_rows = (
        write_index(train)
    )

    pointer_rows, anchor_rows = (
        verify_pointer_files(train)
    )

    assert merged_rows == expected_rows
    assert pointer_rows == expected_rows
    assert anchor_rows == expected_anchors

    assert sum(
        int(r["candidate_rows"])
        for r in index_rows
    ) == expected_rows

    score_max = max(
        float(
            r[
                "exp022_object_score_max_abs_diff"
            ]
        )
        for r in index_rows
    )

    anchor_diffs = [
        float(
            r[
                "exp022_valid_anchor_cos_max_abs_diff"
            ]
        )
        for r in index_rows
        if r[
            "exp022_valid_anchor_cos_max_abs_diff"
        ] not in (None, "")
    ]

    valid_anchor_max = (
        max(anchor_diffs)
        if anchor_diffs
        else None
    )

    all_valid_equal = all(
        str(
            r[
                "exp022_pointer_valid_all_equal"
            ]
        ).lower()
        == "true"
        for r in index_rows
    )

    assert score_max == 0.0
    assert all_valid_equal

    if valid_anchor_max is not None:
        assert valid_anchor_max == 0.0

    summary = {
        "experiment": "EXP023",
        "status": "FULL_PASS",
        "scientific_role": (
            "TRAIN-only primitive signal and "
            "GT-outcome cache"
        ),
        "identitygate_commit_at_execution": (
            base.git_commit(REPO)
        ),
        "sam3_commit": (
            base.git_commit(base.SAM_REPO)
        ),
        "extractor_path": str(
            EXTRACTOR_PATH.relative_to(REPO)
        ),
        "extractor_sha256": (
            base.sha256_file(
                EXTRACTOR_PATH
            )
        ),
        "production_script_sha256": (
            base.sha256_file(
                Path(__file__).resolve()
            )
        ),
        "config_sha256": (
            base.sha256_file(
                base.CONFIG_PATH
            )
        ),
        "scope_sha256": (
            base.sha256_file(
                base.SCOPE_PATH
            )
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": (
            base.sha256_file(checkpoint)
        ),
        "split": "TRAIN",
        "n_videos": len(train),
        "candidate_rows": merged_rows,
        "pointer_rows": pointer_rows,
        "anchor_pointer_rows": (
            anchor_rows
        ),
        "dev_videos_touched": 0,
        "test_videos_touched": 0,
        "exp022_object_score_max_abs_diff": (
            score_max
        ),
        "exp022_pointer_valid_all_equal": (
            all_valid_equal
        ),
        "exp022_valid_anchor_cos_max_abs_diff": (
            valid_anchor_max
        ),
        "peak_memory_allocated_gb_max": (
            max(
                float(
                    r[
                        "peak_memory_allocated_gb"
                    ]
                )
                for r in index_rows
            )
        ),
        "runtime_sec": (
            time.time() - start
        ),
        "merged_primitives_sha256": (
            base.sha256_file(
                merged_path
            )
        ),
        "index_sha256": (
            base.sha256_file(
                index_path
            )
        ),
        "per_video": index_rows,
    }

    summary_path = (
        OUTROOT / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=== EXP023 FULL RESULT ===")

    for k, v in summary.items():
        if k != "per_video":
            print(k, "=", v)

    print()
    print("EXP023_FULL_PASS")


if __name__ == "__main__":
    main()
