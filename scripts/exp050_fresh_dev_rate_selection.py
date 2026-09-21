import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP050-fresh-dev-rate-selection-v1.json"
OUT = ROOT / "experiments/EXP050_dev_rate_selection"
OUT_TMP = ROOT / "experiments/EXP050_dev_rate_selection.tmp"


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
            "Cannot load module: {}".format(path)
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def git_head():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_clean_committed_tree():
    required = [
        "configs/EXP050-fresh-dev-rate-selection-v1.json",
        "scripts/exp050_fresh_dev_rate_selection.py",
    ]

    for rel in required:
        subprocess.check_call(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--error-unmatch",
                rel,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
        ],
        text=True,
    ).strip()

    if status:
        raise RuntimeError(
            "EXP050 requires clean committed tree: {}".format(
                status
            )
        )


def verify_dependencies(cfg):
    for name, item in cfg[
        "source_dependencies"
    ].items():
        actual = sha256sum(
            ROOT / item["path"]
        )

        if actual != item["sha256"]:
            raise RuntimeError(
                "{} SHA mismatch: {}".format(
                    name,
                    actual,
                )
            )

    for name, item in cfg[
        "final_models"
    ].items():
        actual = sha256sum(
            ROOT / item["path"]
        )

        if actual != item["sha256"]:
            raise RuntimeError(
                "{} final model SHA mismatch: {}".format(
                    name,
                    actual,
                )
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
    fieldnames,
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
            fieldnames=fieldnames,
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
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def membership_hash(
    label,
    videos,
):
    payload = (
        label
        + "\n"
        + "\n".join(videos)
        + "\n"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def load_dev_rows(cfg):
    path = (
        ROOT
        / cfg["scope"]["fresh_dev_csv"]
    )

    actual = sha256sum(path)

    if (
        actual
        != cfg["scope"][
            "fresh_dev_csv_sha256"
        ]
    ):
        raise RuntimeError(
            "Fresh DEV CSV SHA mismatch"
        )

    rows = read_csv(path)

    expected_n = int(
        cfg["scope"]["expected_videos"]
    )

    if len(rows) != expected_n:
        raise RuntimeError(
            "Fresh DEV video-count mismatch"
        )

    videos = [
        r["video"]
        for r in rows
    ]

    if videos != sorted(videos):
        raise RuntimeError(
            "Fresh DEV rows are not video-sorted"
        )

    if len(set(videos)) != len(videos):
        raise RuntimeError(
            "Duplicate fresh DEV video"
        )

    actual_membership = membership_hash(
        "DEV40",
        videos,
    )

    if (
        actual_membership
        != cfg["scope"][
            "fresh_dev_membership_sha256"
        ]
    ):
        raise RuntimeError(
            "Fresh DEV membership mismatch"
        )

    events = sum(
        int(r["primary_event_count"])
        for r in rows
    )

    if (
        events
        != int(
            cfg["scope"][
                "expected_primary_events"
            ]
        )
    ):
        raise RuntimeError(
            "Fresh DEV event-count mismatch"
        )

    if any(
        r["partition"] != "FRESH_DEV"
        for r in rows
    ):
        raise RuntimeError(
            "Non-DEV row in fresh DEV manifest"
        )

    return rows


def load_runtime(cfg):
    verify_dependencies(cfg)

    dep = cfg["source_dependencies"]

    exp033 = load_module(
        "exp033_for_exp050",
        ROOT
        / dep["exp033_script"]["path"],
    )

    exp037 = load_module(
        "exp037_for_exp050",
        ROOT
        / dep["exp037_script"]["path"],
    )

    exp038 = load_module(
        "exp038_for_exp050",
        ROOT
        / dep["exp038_script"]["path"],
    )

    exp039 = load_module(
        "exp039_for_exp050",
        ROOT
        / dep["exp039_script"]["path"],
    )

    base_cfg = json.loads(
        (
            ROOT
            / dep[
                "exp033_config"
            ]["path"]
        ).read_text(
            encoding="utf-8"
        )
    )

    exp037_cfg = json.loads(
        (
            ROOT
            / dep[
                "exp037_config"
            ]["path"]
        ).read_text(
            encoding="utf-8"
        )
    )

    exp038_cfg = json.loads(
        (
            ROOT
            / dep[
                "exp038_config"
            ]["path"]
        ).read_text(
            encoding="utf-8"
        )
    )

    exp039_cfg = json.loads(
        (
            ROOT
            / dep[
                "exp039_config"
            ]["path"]
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        cfg["a4_selector"]
        != exp038_cfg["a4_selector"]
    ):
        raise RuntimeError(
            "EXP050 selector differs from EXP038"
        )

    if (
        cfg["a4_selector"]
        != exp039_cfg["a4_selector"]
    ):
        raise RuntimeError(
            "EXP050 selector differs from EXP039"
        )

    exp037.verify_dependencies(
        exp037_cfg
    )

    deps = exp033.load_dependencies(
        base_cfg
    )

    packs = exp033.load_model_packs(
        base_cfg,
        deps,
    )

    if (
        packs["b2_sha256"]
        != cfg["final_models"][
            "B2"
        ]["sha256"]
    ):
        raise RuntimeError(
            "A7 final B2 pack mismatch"
        )

    if (
        packs["b3_sha256"]
        != cfg["final_models"][
            "B3"
        ]["sha256"]
    ):
        raise RuntimeError(
            "A7 final B3 pack mismatch"
        )

    exp039.install_b5_scoring(
        exp033
    )

    static_checks = (
        exp039.static_formula_checks()
    )

    if not all(
        static_checks.values()
    ):
        raise RuntimeError(
            "B5 static formula checks failed"
        )

    sam_root = (
        Path.home()
        / "thesis/externals/sam3"
    )

    sam_commit = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(sam_root),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    )

    expected_sam = (
        exp037_cfg[
            "substrate"
        ][
            "expected_sam_commit"
        ]
    )

    if sam_commit != expected_sam:
        raise RuntimeError(
            "SAM3 commit mismatch: {}".format(
                sam_commit
            )
        )

    return {
        "exp033": exp033,
        "exp037": exp037,
        "exp038": exp038,
        "exp039": exp039,
        "base_cfg": base_cfg,
        "exp037_cfg": exp037_cfg,
        "deps": deps,
        "packs": packs,
        "sam_commit": sam_commit,
        "b5_static_checks":
            static_checks,
    }


def video_context(
    exp037,
    data_root,
    video,
):
    jpg_dir = (
        data_root
        / "JPEGImages"
        / video
    )

    ann_dir = (
        data_root
        / "Annotations"
        / video
    )

    jpgs = exp037.sorted_frames(
        jpg_dir,
        ".jpg",
    )

    pngs = exp037.sorted_frames(
        ann_dir,
        ".png",
    )

    if len(jpgs) != len(pngs):
        raise RuntimeError(
            "{} JPEG/annotation count mismatch".format(
                video
            )
        )

    if len(jpgs) < 2:
        raise RuntimeError(
            "{} has fewer than 2 frames".format(
                video
            )
        )

    gt0 = np.array(
        Image.open(
            ann_dir / pngs[0]
        )
    )

    object_ids = sorted(
        int(x)
        for x in np.unique(gt0)
        if int(x) != 0
    )

    if not object_ids:
        raise RuntimeError(
            "{} has no frame-0 objects".format(
                video
            )
        )

    return {
        "video": video,
        "jpg_dir": jpg_dir,
        "gt0": gt0,
        "object_ids": object_ids,
        "n_frames": len(jpgs),
    }


def make_scope_cfg(
    exp037_cfg,
    data_root,
    ctx,
):
    cfg = copy.deepcopy(
        exp037_cfg
    )

    cfg["scope"][
        "dataset_root"
    ] = str(data_root)

    cfg["scope"][
        "video"
    ] = ctx["video"]

    cfg["scope"][
        "n_frames"
    ] = ctx["n_frames"]

    cfg["scope"][
        "expected_object_ids"
    ] = ctx["object_ids"]

    cfg[
        "engineering_tau_status"
    ] = (
        "FINAL_FRESH_DEV_A4_WRITE_RATE_ONLY"
    )

    return cfg


def aggregate_video_points(
    variant,
    tau,
    rows,
    source,
):
    eligible = sum(
        int(
            r[
                "eligible_opportunities"
            ]
        )
        for r in rows
    )

    admit = sum(
        int(r["admit_count"])
        for r in rows
    )

    block = sum(
        int(r["block_count"])
        for r in rows
    )

    if (
        admit + block
        != eligible
    ):
        raise RuntimeError(
            "Pooled action-count mismatch"
        )

    if eligible <= 0:
        raise RuntimeError(
            "No pooled eligible opportunities"
        )

    return {
        "variant": variant,
        "tau": float(tau),
        "write_rate": float(
            admit / eligible
        ),
        "eligible_opportunities":
            int(eligible),
        "admit_count": int(admit),
        "block_count": int(block),
        "videos": len(rows),
        "max_peak_vram_gb": max(
            float(
                r["peak_vram_gb"]
            )
            for r in rows
        ),
        "runtime_sec": sum(
            float(
                r["runtime_sec"]
            )
            for r in rows
        ),
        "first_source": source,
    }


class MultiVideoRunner:
    def __init__(
        self,
        cfg,
        runtime,
        predictor,
        videos,
        scratch_dir,
        scope_label,
    ):
        self.cfg = cfg
        self.runtime = runtime
        self.predictor = predictor
        self.videos = list(videos)
        self.scratch_dir = Path(
            scratch_dir
        )
        self.scope_label = (
            scope_label
        )

        self.repo_commit = (
            git_head()
        )

        self.config_sha = (
            sha256sum(CFG_PATH)
        )

        self.data_root = Path(
            cfg["scope"][
                "dataset_root"
            ]
        )

        self.cache = {}
        self.executed = []
        self.per_video = {}

        self.cache_hits = 0
        self.cache_misses = 0

        self.context_cache = {}

    def get_context(
        self,
        video,
    ):
        if video not in self.context_cache:
            self.context_cache[
                video
            ] = video_context(
                self.runtime[
                    "exp037"
                ],
                self.data_root,
                video,
            )

        return self.context_cache[
            video
        ]

    def cache_path(
        self,
        variant,
        tau,
        video,
    ):
        tau_key = self.runtime[
            "exp038"
        ].tau_key(tau)

        return (
            self.scratch_dir
            / self.scope_label
            / variant
            / tau_key
            / (video + ".json")
        )

    def validate_cache_row(
        self,
        row,
        variant,
        tau,
        video,
    ):
        expected = {
            "experiment":
                "EXP050",
            "repo_commit":
                self.repo_commit,
            "config_sha256":
                self.config_sha,
            "scope_label":
                self.scope_label,
            "variant":
                variant,
            "tau_key":
                self.runtime[
                    "exp038"
                ].tau_key(tau),
            "video":
                video,
        }

        for key, value in (
            expected.items()
        ):
            if row.get(key) != value:
                raise RuntimeError(
                    "Stale EXP050 scratch cache "
                    "at {} field {}".format(
                        video,
                        key,
                    )
                )

        return row

    def run_video(
        self,
        variant,
        tau,
        video,
    ):
        path = self.cache_path(
            variant,
            tau,
            video,
        )

        key = (
            variant,
            self.runtime[
                "exp038"
            ].tau_key(tau),
            video,
        )

        if path.exists():
            row = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.validate_cache_row(
                row,
                variant,
                tau,
                video,
            )

            self.cache_hits += 1
            self.per_video[
                key
            ] = row

            return row

        ctx = self.get_context(
            video
        )

        scope_cfg = (
            make_scope_cfg(
                self.runtime[
                    "exp037_cfg"
                ],
                self.data_root,
                ctx,
            )
        )

        run_cfg = self.runtime[
            "exp037"
        ].configure_scope(
            self.runtime[
                "base_cfg"
            ],
            scope_cfg,
            float(tau),
        )

        gated = self.runtime[
            "exp033"
        ].run_tracker(
            predictor=self.predictor,
            jpg_dir=ctx["jpg_dir"],
            gt0=ctx["gt0"],
            object_ids=ctx[
                "object_ids"
            ],
            cfg=run_cfg,
            variant=variant,
            packs=self.runtime[
                "packs"
            ],
            deps=self.runtime[
                "deps"
            ],
        )

        decisions = gated[
            "decisions"
        ]

        eligible = (
            ctx["n_frames"] - 1
        )

        if len(decisions) != eligible:
            raise RuntimeError(
                "{} {} tau={} "
                "decision-count mismatch".format(
                    video,
                    variant,
                    tau,
                )
            )

        admit = sum(
            row["action"]
            == "ADMIT"
            for row in decisions
        )

        block = sum(
            row["action"]
            == "BLOCK"
            for row in decisions
        )

        if (
            admit + block
            != eligible
        ):
            raise RuntimeError(
                "{} invalid action count".format(
                    video
                )
            )

        integrity = self.runtime[
            "exp037"
        ].block_integrity_pass(
            decisions,
            len(
                ctx["object_ids"]
            ),
        )

        if not integrity:
            raise RuntimeError(
                "{} {} tau={} "
                "block integrity failure".format(
                    video,
                    variant,
                    tau,
                )
            )

        b5_formula = ""
        b5_frame_min = ""
        b5_action_rule = ""

        if variant == "B5":
            checks = self.runtime[
                "exp039"
            ].validate_gated(
                gated,
                float(tau),
                len(
                    ctx["object_ids"]
                ),
            )

            b5_formula = int(
                checks[
                    "formula_pass"
                ]
            )

            b5_frame_min = int(
                checks[
                    "frame_min_pass"
                ]
            )

            b5_action_rule = int(
                checks[
                    "action_rule_pass"
                ]
            )

            if not (
                b5_formula
                and b5_frame_min
                and b5_action_rule
            ):
                raise RuntimeError(
                    "{} B5 contract failure".format(
                        video
                    )
                )

        row = {
            "experiment":
                "EXP050",
            "repo_commit":
                self.repo_commit,
            "config_sha256":
                self.config_sha,
            "scope_label":
                self.scope_label,
            "variant":
                variant,
            "tau":
                float(tau),
            "tau_key":
                self.runtime[
                    "exp038"
                ].tau_key(tau),
            "video":
                video,
            "n_frames":
                int(
                    ctx["n_frames"]
                ),
            "n_objects":
                len(
                    ctx["object_ids"]
                ),
            "eligible_opportunities":
                int(eligible),
            "admit_count":
                int(admit),
            "block_count":
                int(block),
            "write_rate":
                float(
                    admit / eligible
                ),
            "block_integrity_pass":
                1,
            "b5_formula_pass":
                b5_formula,
            "b5_frame_min_pass":
                b5_frame_min,
            "b5_action_rule_pass":
                b5_action_rule,
            "peak_vram_gb":
                float(
                    gated[
                        "peak_vram_gb"
                    ]
                ),
            "runtime_sec":
                float(
                    gated[
                        "runtime_sec"
                    ]
                ),
            "tracking_outcome_metrics_computed":
                False,
            "test_touched":
                False,
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = Path(
            str(path) + ".tmp"
        )

        tmp.write_text(
            json.dumps(
                row,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        os.replace(
            tmp,
            path,
        )

        self.cache_misses += 1

        self.per_video[
            key
        ] = row

        del gated
        gc.collect()

        if (
            torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

        return row

    def run(
        self,
        variant,
        tau,
        source,
    ):
        key = (
            variant,
            self.runtime[
                "exp038"
            ].tau_key(tau),
        )

        if key in self.cache:
            return self.cache[
                key
            ]

        rows = []

        print(
            "POOL START "
            "variant={} tau={} "
            "videos={}".format(
                variant,
                tau,
                len(self.videos),
            ),
            flush=True,
        )

        for index, video in enumerate(
            self.videos,
            1,
        ):
            row = self.run_video(
                variant,
                tau,
                video,
            )

            rows.append(row)

            print(
                "  {}/{} {} "
                "K={}/{} "
                "rate={:.6f}".format(
                    index,
                    len(self.videos),
                    video,
                    row[
                        "admit_count"
                    ],
                    row[
                        "eligible_opportunities"
                    ],
                    row[
                        "write_rate"
                    ],
                ),
                flush=True,
            )

        point = (
            aggregate_video_points(
                variant,
                tau,
                rows,
                source,
            )
        )

        self.cache[
            key
        ] = point

        self.executed.append(
            point
        )

        print(
            "POOL DONE "
            "variant={} tau={} "
            "K={}/{} rate={:.6f}".format(
                variant,
                tau,
                point[
                    "admit_count"
                ],
                point[
                    "eligible_opportunities"
                ],
                point[
                    "write_rate"
                ],
            ),
            flush=True,
        )

        return point


def selection_row(
    target,
    variant,
    result,
):
    selected = result[
        "selected"
    ]

    return {
        "target_rate":
            float(target),
        "variant":
            variant,
        "status":
            result["status"],
        "selected_tau": (
            ""
            if selected is None
            else float(
                selected["tau"]
            )
        ),
        "realized_write_rate": (
            ""
            if selected is None
            else float(
                selected[
                    "write_rate"
                ]
            )
        ),
        "absolute_rate_error": (
            ""
            if selected is None
            else abs(
                float(
                    selected[
                        "write_rate"
                    ]
                )
                - float(target)
            )
        ),
        "refinements":
            int(
                result[
                    "refinements"
                ]
            ),
        "tested_taus":
            json.dumps(
                result[
                    "tested_taus"
                ]
            ),
    }


def selftest():
    got = membership_hash(
        "DEV40",
        ["a", "b"],
    )

    want = hashlib.sha256(
        b"DEV40\na\nb\n"
    ).hexdigest()

    assert got == want

    rows = [
        {
            "eligible_opportunities":
                9,
            "admit_count":
                9,
            "block_count":
                0,
            "peak_vram_gb":
                6.0,
            "runtime_sec":
                1.0,
        },
        {
            "eligible_opportunities":
                99,
            "admit_count":
                0,
            "block_count":
                99,
            "peak_vram_gb":
                6.2,
            "runtime_sec":
                2.0,
        },
    ]

    point = (
        aggregate_video_points(
            "B2",
            0.5,
            rows,
            "SELFTEST",
        )
    )

    assert (
        point[
            "eligible_opportunities"
        ]
        == 108
    )

    assert (
        point[
            "admit_count"
        ]
        == 9
    )

    assert (
        point[
            "block_count"
        ]
        == 99
    )

    assert (
        point[
            "write_rate"
        ]
        == 9 / 108
    )

    assert (
        point[
            "write_rate"
        ]
        != 0.5
    )

    assert (
        point[
            "max_peak_vram_gb"
        ]
        == 6.2
    )

    print(
        "EXP050_SELFTEST=PASS"
    )


def plan():
    cfg = json.loads(
        CFG_PATH.read_text(
            encoding="utf-8"
        )
    )

    verify_dependencies(
        cfg
    )

    rows = load_dev_rows(
        cfg
    )

    print("EXP050 PLAN")
    print(
        "identitygate_head =",
        git_head(),
    )

    print(
        "fresh_dev_videos =",
        len(rows),
    )

    print(
        "fresh_dev_primary_events =",
        sum(
            int(
                r[
                    "primary_event_count"
                ]
            )
            for r in rows
        ),
    )

    print(
        "fresh_dev_csv_sha256 =",
        sha256sum(
            ROOT
            / cfg["scope"][
                "fresh_dev_csv"
            ]
        ),
    )

    print(
        "fresh_dev_membership_sha256 =",
        membership_hash(
            "DEV40",
            [
                r["video"]
                for r in rows
            ],
        ),
    )

    print(
        "variants =",
        ",".join(
            cfg["variants"]
        ),
    )

    print(
        "common_target_order =",
        cfg[
            "a4_selector"
        ][
            "common_target_rate_order"
        ],
    )

    print(
        "curve_targets =",
        cfg[
            "a4_selector"
        ][
            "curve_target_rates"
        ],
    )

    print(
        "absolute_rate_tolerance =",
        cfg[
            "a4_selector"
        ][
            "absolute_rate_tolerance"
        ],
    )

    print(
        "max_midpoint_refinements =",
        cfg[
            "a4_selector"
        ][
            "max_midpoint_refinements"
        ],
    )

    print(
        "b2_model_sha256 =",
        sha256sum(
            ROOT
            / cfg[
                "final_models"
            ]["B2"]["path"]
        ),
    )

    print(
        "b3_model_sha256 =",
        sha256sum(
            ROOT
            / cfg[
                "final_models"
            ]["B3"]["path"]
        ),
    )

    print(
        "scratch_dir =",
        cfg["runtime"][
            "scratch_dir"
        ],
    )

    print(
        "pooled_rate = SUM_K_OVER_SUM_N"
    )

    print(
        "por_computed = false"
    )

    print(
        "itr_computed = false"
    )

    print(
        "jf_computed = false"
    )

    print(
        "uar_computed = false"
    )

    print(
        "contamination_computed = false"
    )

    print(
        "test_touched = false"
    )

    print(
        "EXP050_PLAN_PASS"
    )


def build_predictor(runtime):
    model = runtime[
        "exp033"
    ].build_sam3_video_model()

    predictor = model.tracker

    predictor.backbone = (
        model.detector.backbone
    )

    return model, predictor


def sanity():
    require_clean_committed_tree()

    cfg = json.loads(
        CFG_PATH.read_text(
            encoding="utf-8"
        )
    )

    runtime = load_runtime(
        cfg
    )

    model, predictor = (
        build_predictor(
            runtime
        )
    )

    video = cfg["scope"][
        "train_exposed_sanity_video"
    ]

    runner = MultiVideoRunner(
        cfg=cfg,
        runtime=runtime,
        predictor=predictor,
        videos=[video],
        scratch_dir=cfg[
            "runtime"
        ]["scratch_dir"],
        scope_label=(
            "TRAIN_EXPOSED_SANITY"
        ),
    )

    for variant in cfg[
        "variants"
    ]:
        point = runner.run(
            variant,
            0.5,
            "SANITY",
        )

        print(
            "SANITY {} "
            "rate={:.6f} "
            "K={}/{} "
            "peak_gb={:.6f}".format(
                variant,
                point[
                    "write_rate"
                ],
                point[
                    "admit_count"
                ],
                point[
                    "eligible_opportunities"
                ],
                point[
                    "max_peak_vram_gb"
                ],
            ),
            flush=True,
        )

    max_peak = max(
        x[
            "max_peak_vram_gb"
        ]
        for x in runner.executed
    )

    print(
        "SANITY_MAX_PEAK_VRAM_GB",
        max_peak,
    )

    print(
        "SANITY_CACHE_HITS",
        runner.cache_hits,
    )

    print(
        "SANITY_CACHE_MISSES",
        runner.cache_misses,
    )

    print(
        "FRESH_DEV_TOUCHED=false"
    )

    print(
        "TEST_TOUCHED=false"
    )

    print(
        "EXP050_SANITY_PASS"
    )

    del predictor
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run():
    require_clean_committed_tree()

    cfg = json.loads(
        CFG_PATH.read_text(
            encoding="utf-8"
        )
    )

    runtime = load_runtime(
        cfg
    )

    dev_rows = load_dev_rows(
        cfg
    )

    if OUT.exists():
        raise RuntimeError(
            "Final EXP050 output "
            "directory already exists: "
            + str(OUT)
        )

    if OUT_TMP.exists():
        raise RuntimeError(
            "EXP050 temporary output "
            "directory exists: "
            + str(OUT_TMP)
        )

    model, predictor = (
        build_predictor(
            runtime
        )
    )

    runner = MultiVideoRunner(
        cfg=cfg,
        runtime=runtime,
        predictor=predictor,
        videos=[
            r["video"]
            for r in dev_rows
        ],
        scratch_dir=cfg[
            "runtime"
        ]["scratch_dir"],
        scope_label="FRESH_DEV40",
    )

    selector = cfg[
        "a4_selector"
    ]

    variants = cfg[
        "variants"
    ]

    common_rows = []
    headline = {}
    r_star = None

    for target in selector[
        "common_target_rate_order"
    ]:
        print(
            "COMMON TARGET {}".format(
                target
            ),
            flush=True,
        )

        all_match = True
        current = {}

        for variant in variants:
            result = runtime[
                "exp038"
            ].select_for_target(
                runner,
                variant,
                float(target),
                selector,
            )

            row = selection_row(
                target,
                variant,
                result,
            )

            common_rows.append(
                row
            )

            current[
                variant
            ] = row

            print(
                "{} target={} "
                "status={} "
                "tau={} rate={}".format(
                    variant,
                    target,
                    row[
                        "status"
                    ],
                    row[
                        "selected_tau"
                    ],
                    row[
                        "realized_write_rate"
                    ],
                ),
                flush=True,
            )

            if (
                result["status"]
                != "MATCH"
            ):
                all_match = False

        if all_match:
            r_star = float(
                target
            )

            headline = current
            break

    curve_rows = []
    curve_map = {}

    if r_star is not None:
        for target in selector[
            "curve_target_rates"
        ]:
            curve_map[
                str(target)
            ] = {}

            for variant in variants:
                result = runtime[
                    "exp038"
                ].select_for_target(
                    runner,
                    variant,
                    float(target),
                    selector,
                )

                row = selection_row(
                    target,
                    variant,
                    result,
                )

                curve_rows.append(
                    row
                )

                curve_map[
                    str(target)
                ][
                    variant
                ] = row

    curve_all_match = bool(
        curve_rows
        and all(
            row["status"]
            == "MATCH"
            for row in curve_rows
        )
    )

    pooled_rows = sorted(
        runner.executed,
        key=lambda row: (
            row["variant"],
            float(
                row["tau"]
            ),
        ),
    )

    per_video_rows = sorted(
        runner.per_video.values(),
        key=lambda row: (
            row["variant"],
            float(
                row["tau"]
            ),
            row["video"],
        ),
    )

    OUT_TMP.mkdir(
        parents=True
    )

    write_csv_atomic(
        OUT_TMP
        / "executed_pooled_points.csv",
        [
            "variant",
            "tau",
            "write_rate",
            "eligible_opportunities",
            "admit_count",
            "block_count",
            "videos",
            "max_peak_vram_gb",
            "runtime_sec",
            "first_source",
        ],
        pooled_rows,
    )

    write_csv_atomic(
        OUT_TMP
        / "per_video_write_counts.csv",
        [
            "experiment",
            "repo_commit",
            "config_sha256",
            "scope_label",
            "variant",
            "tau",
            "tau_key",
            "video",
            "n_frames",
            "n_objects",
            "eligible_opportunities",
            "admit_count",
            "block_count",
            "write_rate",
            "block_integrity_pass",
            "b5_formula_pass",
            "b5_frame_min_pass",
            "b5_action_rule_pass",
            "peak_vram_gb",
            "runtime_sec",
            "tracking_outcome_metrics_computed",
            "test_touched",
        ],
        per_video_rows,
    )

    write_csv_atomic(
        OUT_TMP
        / "common_target_selection.csv",
        [
            "target_rate",
            "variant",
            "status",
            "selected_tau",
            "realized_write_rate",
            "absolute_rate_error",
            "refinements",
            "tested_taus",
        ],
        common_rows,
    )

    write_csv_atomic(
        OUT_TMP
        / "curve_selection.csv",
        [
            "target_rate",
            "variant",
            "status",
            "selected_tau",
            "realized_write_rate",
            "absolute_rate_error",
            "refinements",
            "tested_taus",
        ],
        curve_rows,
    )

    operating = {
        "experiment":
            "EXP050",
        "r_star":
            r_star,
        "headline":
            headline,
        "curve_map":
            curve_map,
        "absolute_rate_tolerance":
            selector[
                "absolute_rate_tolerance"
            ],
        "common_target_rate_order":
            selector[
                "common_target_rate_order"
            ],
        "curve_target_rates":
            selector[
                "curve_target_rates"
            ],
        "threshold_selection_information":
            "POOLED_FRESH_DEV_PHYSICAL_WRITE_RATE_ONLY",
        "tracking_outcomes_used":
            False,
        "test_touched":
            False,
    }

    write_json_atomic(
        OUT_TMP
        / "final_operating_points.json",
        operating,
    )

    if r_star is None:
        status = (
            "EXP050_NO_COMMON_R_STAR"
        )

    elif not curve_all_match:
        status = (
            "EXP050_FULL_CURVE_MATCH_FAILURE"
        )

    else:
        status = (
            "EXP050_DEV_RATE_SELECTION_PASS"
        )

    summary = {
        "experiment":
            "EXP050",
        "status":
            status,
        "repo_commit":
            git_head(),
        "sam_commit":
            runtime[
                "sam_commit"
            ],
        "config_sha256":
            sha256sum(
                CFG_PATH
            ),
        "script_sha256":
            sha256sum(
                Path(__file__)
            ),
        "fresh_dev_csv_sha256":
            cfg["scope"][
                "fresh_dev_csv_sha256"
            ],
        "fresh_dev_membership_sha256":
            cfg["scope"][
                "fresh_dev_membership_sha256"
            ],
        "fresh_dev_videos":
            len(dev_rows),
        "fresh_dev_primary_events":
            sum(
                int(
                    row[
                        "primary_event_count"
                    ]
                )
                for row in dev_rows
            ),
        "variants":
            variants,
        "b2_model_sha256":
            runtime[
                "packs"
            ][
                "b2_sha256"
            ],
        "b3_model_sha256":
            runtime[
                "packs"
            ][
                "b3_sha256"
            ],
        "r_star":
            r_star,
        "curve_all_match":
            curve_all_match,
        "executed_pooled_points":
            len(
                runner.executed
            ),
        "executed_video_evaluations":
            len(
                runner.per_video
            ),
        "scratch_cache_hits":
            runner.cache_hits,
        "scratch_cache_misses":
            runner.cache_misses,
        "max_peak_vram_gb": (
            max(
                row[
                    "max_peak_vram_gb"
                ]
                for row in (
                    runner.executed
                )
            )
            if runner.executed
            else None
        ),
        "fresh_dev_touched_by_closed_loop_tracking":
            True,
        "threshold_selection_uses_tracking_outcomes":
            False,
        "por_computed":
            False,
        "itr_computed":
            False,
        "jf_computed":
            False,
        "uar_computed":
            False,
        "contamination_computed":
            False,
        "test_touched":
            False,
    }

    output_names = [
        "executed_pooled_points.csv",
        "per_video_write_counts.csv",
        "common_target_selection.csv",
        "curve_selection.csv",
        "final_operating_points.json",
    ]

    summary[
        "output_hashes"
    ] = {
        name:
            sha256sum(
                OUT_TMP / name
            )
        for name in output_names
    }

    write_json_atomic(
        OUT_TMP
        / "summary.json",
        summary,
    )

    os.replace(
        OUT_TMP,
        OUT,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    del predictor
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if r_star is None:
        raise RuntimeError(
            "EXP050 STOP: "
            "no common A4 r_star"
        )

    if not curve_all_match:
        raise RuntimeError(
            "EXP050 STOP: "
            "mandatory curve target "
            "match failure"
        )

    print(
        "EXP050_RUN_PASS"
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

    if (
        args.mode
        == "selftest"
    ):
        selftest()

    elif (
        args.mode
        == "plan"
    ):
        plan()

    elif (
        args.mode
        == "sanity"
    ):
        sanity()

    else:
        run()


if __name__ == "__main__":
    main()
