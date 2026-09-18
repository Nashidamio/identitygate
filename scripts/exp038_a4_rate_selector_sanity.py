import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP038-a4-rate-selector-sanity-v1.json"
OUT = ROOT / "experiments/EXP038_a4_rate_selector_sanity"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load module: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_clean_committed_tree():
    required = [
        "configs/EXP038-a4-rate-selector-sanity-v1.json",
        "scripts/exp038_a4_rate_selector_sanity.py",
    ]

    for rel in required:
        subprocess.check_call(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        text=True,
    ).strip()

    if status:
        raise RuntimeError("EXP038 requires a clean committed tree")


def verify_dependencies(cfg):
    for name, item in cfg["source_dependencies"].items():
        actual = sha256sum(ROOT / item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(
                "{} SHA mismatch: {}".format(name, actual)
            )


def tau_key(tau):
    return "{:.10f}".format(float(tau))


def within_tolerance(rate, target, tolerance):
    return abs(float(rate) - float(target)) <= float(tolerance)


def brackets(rate_a, rate_b, target):
    da = float(rate_a) - float(target)
    db = float(rate_b) - float(target)
    return da * db <= 0.0


def choose_best_match(points, target, tolerance):
    matched = [
        p
        for p in points
        if within_tolerance(
            p["write_rate"],
            target,
            tolerance,
        )
    ]

    if not matched:
        return None

    return min(
        matched,
        key=lambda p: (
            abs(
                float(p["write_rate"])
                - float(target)
            ),
            float(p["tau"]),
        ),
    )


class Runner:
    def __init__(
        self,
        exp033,
        predictor,
        packs,
        deps,
        base_cfg,
        exp037_cfg,
        jpg_dir,
        gt0,
        object_ids,
        n_frames,
    ):
        self.exp033 = exp033
        self.predictor = predictor
        self.packs = packs
        self.deps = deps
        self.base_cfg = base_cfg
        self.exp037_cfg = exp037_cfg
        self.jpg_dir = jpg_dir
        self.gt0 = gt0
        self.object_ids = object_ids
        self.n_frames = n_frames
        self.cache = {}
        self.executed = []

    def run(self, variant, tau, source):
        key = (variant, tau_key(tau))

        if key in self.cache:
            return self.cache[key]

        run_cfg = self.exp037.configure_scope(
            self.base_cfg,
            self.exp037_cfg,
            float(tau),
        )

        gated = self.exp033.run_tracker(
            predictor=self.predictor,
            jpg_dir=self.jpg_dir,
            gt0=self.gt0,
            object_ids=self.object_ids,
            cfg=run_cfg,
            variant=variant,
            packs=self.packs,
            deps=self.deps,
        )

        decisions = gated["decisions"]

        if len(decisions) != self.n_frames - 1:
            raise RuntimeError(
                "{} tau={} decision count {}".format(
                    variant,
                    tau,
                    len(decisions),
                )
            )

        admit = sum(
            row["action"] == "ADMIT"
            for row in decisions
        )

        block = sum(
            row["action"] == "BLOCK"
            for row in decisions
        )

        if admit + block != self.n_frames - 1:
            raise RuntimeError("Invalid action count")

        integrity = self.exp037.block_integrity_pass(
            decisions,
            len(self.object_ids),
        )

        if not integrity:
            raise RuntimeError(
                "{} tau={} block integrity failure".format(
                    variant,
                    tau,
                )
            )

        point = {
            "variant": variant,
            "tau": float(tau),
            "write_rate": float(
                admit / (self.n_frames - 1)
            ),
            "admit_count": int(admit),
            "block_count": int(block),
            "block_integrity_pass": 1,
            "peak_vram_gb": float(
                gated["peak_vram_gb"]
            ),
            "runtime_sec": float(
                gated["runtime_sec"]
            ),
            "first_source": source,
        }

        self.cache[key] = point
        self.executed.append(point)

        del gated
        gc.collect()

        return point


def select_for_target(
    runner,
    variant,
    target,
    selector_cfg,
):
    tolerance = float(
        selector_cfg["absolute_rate_tolerance"]
    )

    coarse = [
        runner.run(
            variant,
            tau,
            "COARSE",
        )
        for tau in selector_cfg["coarse_tau"]
    ]

    tested = list(coarse)

    best = choose_best_match(
        tested,
        target,
        tolerance,
    )

    if best is not None:
        return {
            "status": "MATCH",
            "selected": best,
            "refinements": 0,
            "tested_taus": [
                float(x["tau"])
                for x in tested
            ],
        }

    ordered = sorted(
        coarse,
        key=lambda p: float(p["tau"]),
    )

    candidate_brackets = []

    for left, right in zip(
        ordered[:-1],
        ordered[1:],
    ):
        if brackets(
            left["write_rate"],
            right["write_rate"],
            target,
        ):
            candidate_brackets.append(
                (left, right)
            )

    if len(candidate_brackets) == 0:
        return {
            "status": "NO_MATCH_NO_BRACKET",
            "selected": None,
            "refinements": 0,
            "tested_taus": [
                float(x["tau"])
                for x in tested
            ],
        }

    if len(candidate_brackets) > 1:
        raise RuntimeError(
            "{} target={} has multiple adjacent coarse brackets".format(
                variant,
                target,
            )
        )

    left, right = candidate_brackets[0]
    refinements = 0

    for refinement in range(
        int(
            selector_cfg[
                "max_midpoint_refinements"
            ]
        )
    ):
        tau_mid = (
            float(left["tau"])
            + float(right["tau"])
        ) / 2.0

        mid = runner.run(
            variant,
            tau_mid,
            "REFINE_TARGET_{}".format(
                target
            ),
        )

        if all(
            tau_key(mid["tau"])
            != tau_key(x["tau"])
            for x in tested
        ):
            tested.append(mid)

        refinements = refinement + 1

        best = choose_best_match(
            tested,
            target,
            tolerance,
        )

        if best is not None:
            return {
                "status": "MATCH",
                "selected": best,
                "refinements": refinements,
                "tested_taus": [
                    float(x["tau"])
                    for x in tested
                ],
            }

        left_brackets = brackets(
            left["write_rate"],
            mid["write_rate"],
            target,
        )

        right_brackets = brackets(
            mid["write_rate"],
            right["write_rate"],
            target,
        )

        if left_brackets and right_brackets:
            raise RuntimeError(
                "{} target={} refinement produced multiple brackets".format(
                    variant,
                    target,
                )
            )

        if left_brackets:
            right = mid

        elif right_brackets:
            left = mid

        else:
            raise RuntimeError(
                "{} target={} lost deterministic bracket".format(
                    variant,
                    target,
                )
            )

    best = choose_best_match(
        tested,
        target,
        tolerance,
    )

    return {
        "status": (
            "MATCH"
            if best is not None
            else "NO_MATCH_AFTER_REFINEMENT"
        ),
        "selected": best,
        "refinements": refinements,
        "tested_taus": [
            float(x["tau"])
            for x in tested
        ],
    }


def main():
    cfg = json.loads(
        CFG_PATH.read_text(encoding="utf-8")
    )

    require_clean_committed_tree()
    verify_dependencies(cfg)

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(
                OUT
            )
        )

    exp037 = load_module(
        "exp037_frozen",
        ROOT
        / cfg["source_dependencies"][
            "exp037_script"
        ]["path"],
    )

    exp037_cfg = json.loads(
        (
            ROOT
            / cfg["source_dependencies"][
                "exp037_config"
            ]["path"]
        ).read_text(encoding="utf-8")
    )

    exp037.verify_dependencies(
        exp037_cfg
    )

    exp033 = exp037.load_module(
        "exp033_frozen_for_exp038",
        ROOT
        / exp037_cfg[
            "source_dependencies"
        ]["exp033_script"]["path"],
    )

    base_cfg = json.loads(
        (
            ROOT
            / exp037_cfg[
                "source_dependencies"
            ]["exp033_config"]["path"]
        ).read_text(encoding="utf-8")
    )

    deps = exp033.load_dependencies(
        base_cfg
    )

    packs = exp033.load_model_packs(
        base_cfg,
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

    if sam_commit != exp037_cfg[
        "substrate"
    ]["expected_sam_commit"]:
        raise RuntimeError("SAM3 commit mismatch")

    data_root = Path(
        cfg["scope"]["dataset_root"]
    )

    video = cfg["scope"]["video"]
    n_frames = int(
        cfg["scope"]["n_frames"]
    )

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
            "JPEG and annotation count mismatch"
        )

    if len(jpgs) < n_frames:
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
        for x in np.unique(gt0)
        if int(x) != 0
    )

    if object_ids != cfg[
        "scope"
    ]["expected_object_ids"]:
        raise RuntimeError(
            "Object ID mismatch: {}".format(
                object_ids
            )
        )

    print(
        "building frozen SAM3...",
        flush=True,
    )

    model = exp033.build_sam3_video_model()

    predictor = model.tracker
    predictor.backbone = (
        model.detector.backbone
    )

    runner = Runner(
        exp033=exp033,
        predictor=predictor,
        packs=packs,
        deps=deps,
        base_cfg=base_cfg,
        exp037_cfg=exp037_cfg,
        jpg_dir=jpg_dir,
        gt0=gt0,
        object_ids=object_ids,
        n_frames=n_frames,
    )

    selector_cfg = cfg["a4_selector"]
    variants = cfg["sanity_variants"]

    target_rows = []
    subset_common_target = None

    for target in selector_cfg[
        "common_target_rate_order"
    ]:
        print(
            "testing target={}...".format(
                target
            ),
            flush=True,
        )

        per_variant = []
        all_match = True

        for variant in variants:
            result = select_for_target(
                runner,
                variant,
                float(target),
                selector_cfg,
            )

            selected = result[
                "selected"
            ]

            row = {
                "target_rate": float(target),
                "variant": variant,
                "status": result["status"],
                "selected_tau": (
                    ""
                    if selected is None
                    else selected["tau"]
                ),
                "realized_write_rate": (
                    ""
                    if selected is None
                    else selected["write_rate"]
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
                "refinements": int(
                    result["refinements"]
                ),
                "tested_taus": json.dumps(
                    result["tested_taus"]
                ),
            }

            target_rows.append(row)
            per_variant.append(row)

            if result["status"] != "MATCH":
                all_match = False

            print(
                "{} target={} status={} tau={} rate={}".format(
                    variant,
                    target,
                    row["status"],
                    row["selected_tau"],
                    row["realized_write_rate"],
                ),
                flush=True,
            )

        if all_match:
            subset_common_target = float(
                target
            )
            break

    OUT.mkdir(parents=True)

    exp037.write_csv(
        OUT / "executed_tau_points.csv",
        [
            "variant",
            "tau",
            "write_rate",
            "admit_count",
            "block_count",
            "block_integrity_pass",
            "peak_vram_gb",
            "runtime_sec",
            "first_source",
        ],
        sorted(
            runner.executed,
            key=lambda r: (
                r["variant"],
                float(r["tau"]),
            ),
        ),
    )

    exp037.write_csv(
        OUT / "target_selection.csv",
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
        target_rows,
    )

    repo_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    status = (
        "EXP038_A4_SELECTOR_SANITY_PASS"
        if subset_common_target is not None
        else "EXP038_A4_SELECTOR_SANITY_NO_COMMON_TARGET"
    )

    result = {
        "experiment": "EXP038",
        "status": status,
        "repo_commit": repo_commit,
        "sam_commit": sam_commit,
        "config_sha256": sha256sum(
            CFG_PATH
        ),
        "script_sha256": sha256sum(
            Path(__file__)
        ),
        "video": video,
        "frames": n_frames,
        "variants": variants,
        "subset_common_target_not_final_r_star": (
            subset_common_target
        ),
        "executed_tau_points": len(
            runner.executed
        ),
        "max_peak_vram_gb": (
            max(
                x["peak_vram_gb"]
                for x in runner.executed
            )
            if runner.executed
            else None
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
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    if subset_common_target is None:
        raise RuntimeError(
            "EXP038 sanity found no common target for B1/B2/B3-S/B3-R"
        )


if __name__ == "__main__":
    main()
