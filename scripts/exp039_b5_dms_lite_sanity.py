import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP039-b5-dms-lite-sanity-v1.json"
OUT = ROOT / "experiments/EXP039_b5_dms_lite_sanity"


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
        "configs/EXP039-b5-dms-lite-sanity-v1.json",
        "scripts/exp039_b5_dms_lite_sanity.py",
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
        raise RuntimeError("EXP039 requires a clean committed tree")


def verify_dependencies(cfg):
    for name, item in cfg["source_dependencies"].items():
        actual = sha256sum(ROOT / item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(
                "{} SHA mismatch: {}".format(name, actual)
            )


def b5_score(mask_conf, occ_logit):
    mask_conf = float(mask_conf)
    occ_logit = float(occ_logit)

    if not math.isfinite(mask_conf) or not math.isfinite(occ_logit):
        return {
            "base_finite": 0,
            "used_variant": "B5",
            "p_unsafe_drift": "",
            "p_unsafe_theft": "",
            "safe_score": 0.0,
            "missing_policy": "FAIL_CLOSED",
        }

    if occ_logit <= 0.0:
        presence = 0.0
    else:
        presence = 2.0 * (1.0 / (1.0 + math.exp(-occ_logit))) - 1.0

    score = presence * mask_conf

    if not math.isfinite(score):
        raise RuntimeError("Finite B5 inputs produced non-finite score")

    return {
        "base_finite": 1,
        "used_variant": "B5",
        "p_unsafe_drift": "",
        "p_unsafe_theft": "",
        "safe_score": float(score),
        "missing_policy": "",
    }


def static_formula_checks():
    positive = b5_score(0.8, math.log(3.0))
    absent = b5_score(0.8, -1.0)
    nan_mask = b5_score(float("nan"), 1.0)
    inf_occ = b5_score(0.8, float("inf"))

    checks = {
        "positive_formula_pass": int(
            math.isclose(
                positive["safe_score"],
                0.4,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "nonpositive_occ_zero_pass": int(
            absent["safe_score"] == 0.0
        ),
        "nan_fail_closed_pass": int(
            nan_mask["safe_score"] == 0.0
            and nan_mask["base_finite"] == 0
            and nan_mask["missing_policy"] == "FAIL_CLOSED"
        ),
        "inf_fail_closed_pass": int(
            inf_occ["safe_score"] == 0.0
            and inf_occ["base_finite"] == 0
            and inf_occ["missing_policy"] == "FAIL_CLOSED"
        ),
    }

    if not all(checks.values()):
        raise RuntimeError("B5 static formula contract failure")

    return checks


def install_b5_scoring(exp033):
    original = exp033.score_learned_variant

    def routed(
        variant,
        base_features,
        pointer_valid,
        self_identity,
        competitor_identity,
        packs,
    ):
        if variant == "B5":
            if len(base_features) < 2:
                raise RuntimeError("B5 missing required base features")
            return b5_score(
                base_features[0],
                base_features[1],
            )

        return original(
            variant=variant,
            base_features=base_features,
            pointer_valid=pointer_valid,
            self_identity=self_identity,
            competitor_identity=competitor_identity,
            packs=packs,
        )

    exp033.score_learned_variant = routed


def validate_gated(gated, tau, n_objects):
    rows = gated["object_rows"]
    decisions = gated["decisions"]

    if len(rows) != len(decisions) * n_objects:
        raise RuntimeError("B5 object-row count mismatch")

    rows_by_frame = {}
    formula_rows = 0
    fail_closed_rows = 0

    for row in rows:
        if row["variant"] != "B5":
            raise RuntimeError("Non-B5 row in EXP039")

        expected = b5_score(
            row["mask_conf_iou_head"],
            row["occ_score_logit"],
        )

        actual_score = float(row["object_safe_score"])

        if not math.isclose(
            actual_score,
            float(expected["safe_score"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "B5 object score mismatch at frame {} object {}".format(
                    row["frame_idx"],
                    row["object_id"],
                )
            )

        if int(row["base_finite"]) != int(expected["base_finite"]):
            raise RuntimeError("B5 base_finite mismatch")

        if row["used_variant"] != "B5":
            raise RuntimeError("B5 routing label mismatch")

        if row["missing_policy"] != expected["missing_policy"]:
            raise RuntimeError("B5 missing-policy mismatch")

        formula_rows += 1
        fail_closed_rows += int(expected["base_finite"] == 0)

        rows_by_frame.setdefault(
            int(row["frame_idx"]),
            [],
        ).append(row)

    decision_by_frame = {
        int(row["frame_idx"]): row
        for row in decisions
    }

    if set(rows_by_frame) != set(decision_by_frame):
        raise RuntimeError("B5 frame coverage mismatch")

    for frame_idx, frame_rows in rows_by_frame.items():
        if len(frame_rows) != n_objects:
            raise RuntimeError("B5 per-frame object count mismatch")

        expected_min = min(
            float(row["object_safe_score"])
            for row in frame_rows
        )

        decision = decision_by_frame[frame_idx]

        if not math.isclose(
            float(decision["frame_score"]),
            expected_min,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "A3 frame-min mismatch at frame {}".format(frame_idx)
            )

        expected_action = (
            "ADMIT"
            if expected_min >= float(tau)
            else "BLOCK"
        )

        if decision["action"] != expected_action:
            raise RuntimeError(
                "B5 action-rule mismatch at frame {}".format(frame_idx)
            )

    return {
        "formula_rows_checked": formula_rows,
        "actual_fail_closed_rows": fail_closed_rows,
        "formula_pass": 1,
        "frame_min_pass": 1,
        "action_rule_pass": 1,
    }


class Runner:
    def __init__(
        self,
        exp033,
        exp037,
        exp038,
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
        self.exp037 = exp037
        self.exp038 = exp038
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
        if variant != "B5":
            raise RuntimeError("EXP039 Runner only accepts B5")

        key = (
            variant,
            self.exp038.tau_key(tau),
        )

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
            variant="B5",
            packs=self.packs,
            deps=self.deps,
        )

        decisions = gated["decisions"]

        if len(decisions) != self.n_frames - 1:
            raise RuntimeError(
                "B5 tau={} decision count {}".format(
                    tau,
                    len(decisions),
                )
            )

        checks = validate_gated(
            gated,
            float(tau),
            len(self.object_ids),
        )

        integrity = self.exp037.block_integrity_pass(
            decisions,
            len(self.object_ids),
        )

        if not integrity:
            raise RuntimeError(
                "B5 tau={} block integrity failure".format(tau)
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
            raise RuntimeError("Invalid B5 action count")

        point = {
            "variant": "B5",
            "tau": float(tau),
            "write_rate": float(admit / (self.n_frames - 1)),
            "admit_count": int(admit),
            "block_count": int(block),
            "block_integrity_pass": 1,
            "formula_rows_checked": int(checks["formula_rows_checked"]),
            "actual_fail_closed_rows": int(checks["actual_fail_closed_rows"]),
            "formula_pass": int(checks["formula_pass"]),
            "frame_min_pass": int(checks["frame_min_pass"]),
            "action_rule_pass": int(checks["action_rule_pass"]),
            "peak_vram_gb": float(gated["peak_vram_gb"]),
            "runtime_sec": float(gated["runtime_sec"]),
            "first_source": source,
        }

        self.cache[key] = point
        self.executed.append(point)

        del gated
        gc.collect()

        return point


def main():
    cfg = json.loads(
        CFG_PATH.read_text(encoding="utf-8")
    )

    require_clean_committed_tree()
    verify_dependencies(cfg)

    if OUT.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(OUT)
        )

    exp037 = load_module(
        "exp037_frozen_for_exp039",
        ROOT / cfg["source_dependencies"]["exp037_script"]["path"],
    )

    exp038 = load_module(
        "exp038_frozen_for_exp039",
        ROOT / cfg["source_dependencies"]["exp038_script"]["path"],
    )

    exp037_cfg = json.loads(
        (
            ROOT
            / cfg["source_dependencies"]["exp037_config"]["path"]
        ).read_text(encoding="utf-8")
    )

    exp038_cfg = json.loads(
        (
            ROOT
            / cfg["source_dependencies"]["exp038_config"]["path"]
        ).read_text(encoding="utf-8")
    )

    exp037.verify_dependencies(exp037_cfg)

    if cfg["a4_selector"] != exp038_cfg["a4_selector"]:
        raise RuntimeError("EXP039 A4 selector differs from frozen EXP038 selector")

    exp033 = exp037.load_module(
        "exp033_frozen_for_exp039",
        ROOT
        / exp037_cfg["source_dependencies"]["exp033_script"]["path"],
    )

    base_cfg = json.loads(
        (
            ROOT
            / exp037_cfg["source_dependencies"]["exp033_config"]["path"]
        ).read_text(encoding="utf-8")
    )

    deps = exp033.load_dependencies(
        base_cfg
    )

    packs = exp033.load_model_packs(
        base_cfg,
        deps,
    )

    install_b5_scoring(exp033)
    synthetic_checks = static_formula_checks()

    patched_nan = exp033.score_learned_variant(
        variant="B5",
        base_features=[
            float("nan"),
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        pointer_valid=False,
        self_identity=None,
        competitor_identity=None,
        packs=packs,
    )

    if not (
        patched_nan["safe_score"] == 0.0
        and patched_nan["base_finite"] == 0
        and patched_nan["missing_policy"] == "FAIL_CLOSED"
    ):
        raise RuntimeError("Patched EXP033 B5 FAIL_CLOSED contract failure")

    sam_root = Path.home() / "thesis/externals/sam3"

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

    if sam_commit != exp037_cfg["substrate"]["expected_sam_commit"]:
        raise RuntimeError("SAM3 commit mismatch")

    data_root = Path(
        cfg["scope"]["dataset_root"]
    )
    video = cfg["scope"]["video"]
    n_frames = int(cfg["scope"]["n_frames"])

    jpg_dir = data_root / "JPEGImages" / video
    ann_dir = data_root / "Annotations" / video

    jpgs = exp037.sorted_frames(
        jpg_dir,
        ".jpg",
    )
    pngs = exp037.sorted_frames(
        ann_dir,
        ".png",
    )

    if len(jpgs) != len(pngs):
        raise RuntimeError("JPEG and annotation count mismatch")

    if len(jpgs) < n_frames:
        raise RuntimeError("Video shorter than configured scope")

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

    if object_ids != cfg["scope"]["expected_object_ids"]:
        raise RuntimeError(
            "Object ID mismatch: {}".format(object_ids)
        )

    print("building frozen SAM3...", flush=True)

    model = exp033.build_sam3_video_model()

    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    runner = Runner(
        exp033=exp033,
        exp037=exp037,
        exp038=exp038,
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
    target_rows = []
    selected_target = None
    selected_point = None

    for target in selector_cfg["common_target_rate_order"]:
        print(
            "testing B5 target={}...".format(target),
            flush=True,
        )

        result = exp038.select_for_target(
            runner,
            "B5",
            float(target),
            selector_cfg,
        )

        selected = result["selected"]

        row = {
            "target_rate": float(target),
            "variant": "B5",
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
                    float(selected["write_rate"])
                    - float(target)
                )
            ),
            "refinements": int(result["refinements"]),
            "tested_taus": json.dumps(result["tested_taus"]),
        }

        target_rows.append(row)

        print(
            "B5 target={} status={} tau={} rate={}".format(
                target,
                row["status"],
                row["selected_tau"],
                row["realized_write_rate"],
            ),
            flush=True,
        )

        if result["status"] == "MATCH":
            selected_target = float(target)
            selected_point = selected
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
            "formula_rows_checked",
            "actual_fail_closed_rows",
            "formula_pass",
            "frame_min_pass",
            "action_rule_pass",
            "peak_vram_gb",
            "runtime_sec",
            "first_source",
        ],
        sorted(
            runner.executed,
            key=lambda r: float(r["tau"]),
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

    contract = {
        "a8_sha256": sha256sum(
            ROOT / cfg["source_dependencies"]["a8"]["path"]
        ),
        "synthetic_checks": synthetic_checks,
        "patched_exp033_nan_fail_closed_pass": 1,
        "executed_points": len(runner.executed),
        "formula_rows_checked_total": sum(
            x["formula_rows_checked"]
            for x in runner.executed
        ),
        "actual_fail_closed_rows_total": sum(
            x["actual_fail_closed_rows"]
            for x in runner.executed
        ),
        "all_formula_pass": int(
            all(x["formula_pass"] == 1 for x in runner.executed)
        ),
        "all_frame_min_pass": int(
            all(x["frame_min_pass"] == 1 for x in runner.executed)
        ),
        "all_action_rule_pass": int(
            all(x["action_rule_pass"] == 1 for x in runner.executed)
        ),
        "all_block_integrity_pass": int(
            all(x["block_integrity_pass"] == 1 for x in runner.executed)
        ),
    }

    (
        OUT / "contract_checks.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
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
        "EXP039_B5_DMS_LITE_SANITY_PASS"
        if selected_target is not None
        else "EXP039_B5_DMS_LITE_SANITY_NO_MATCH"
    )

    result = {
        "experiment": "EXP039",
        "status": status,
        "repo_commit": repo_commit,
        "sam_commit": sam_commit,
        "config_sha256": sha256sum(CFG_PATH),
        "script_sha256": sha256sum(Path(__file__)),
        "video": video,
        "frames": n_frames,
        "variant": "B5",
        "b5_first_match_target_not_final_r_star": selected_target,
        "selected_tau": (
            None
            if selected_point is None
            else float(selected_point["tau"])
        ),
        "selected_realized_write_rate": (
            None
            if selected_point is None
            else float(selected_point["write_rate"])
        ),
        "executed_tau_points": len(runner.executed),
        "max_peak_vram_gb": (
            max(
                x["peak_vram_gb"]
                for x in runner.executed
            )
            if runner.executed
            else None
        ),
        "contract_checks_pass": int(
            contract["all_formula_pass"] == 1
            and contract["all_frame_min_pass"] == 1
            and contract["all_action_rule_pass"] == 1
            and contract["all_block_integrity_pass"] == 1
            and all(synthetic_checks.values())
        ),
        "claim_boundary": cfg["claim_boundary"],
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

    if selected_target is None:
        raise RuntimeError(
            "EXP039 B5 sanity found no A4 target-rate match"
        )


if __name__ == "__main__":
    main()
