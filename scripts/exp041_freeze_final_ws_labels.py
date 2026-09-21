#!/usr/bin/env python3

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP041-ws-final-labels-v1.json"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_clean_committed_tree():
    required = [
        "configs/EXP041-ws-final-labels-v1.json",
        "scripts/exp041_freeze_final_ws_labels.py",
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
        raise RuntimeError("EXP041 requires a clean committed tree")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value):
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return ROOT / p


def verify_input(entry):
    path = resolve_path(entry["path"])

    if not path.is_file():
        raise RuntimeError("Missing input: {}".format(path))

    actual = sha256sum(path)

    if actual != entry["sha256"]:
        raise RuntimeError(
            "SHA mismatch for {}: {} != {}".format(
                path,
                actual,
                entry["sha256"],
            )
        )

    return path


def main():
    require_clean_committed_tree()

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

    if cfg["experiment"] != "EXP041":
        raise RuntimeError("Unexpected experiment ID")

    inputs = cfg["inputs"]

    population_path = verify_input(inputs["scene_population"])
    auto_path = verify_input(inputs["auto_confirmed"])
    required_path = verify_input(inputs["manual_required"])
    labels_path = verify_input(inputs["manual_labels"])
    audit_path = verify_input(inputs["blinded_audit_result"])

    population = read_csv(population_path)
    auto = read_csv(auto_path)
    required = read_csv(required_path)
    labels = read_csv(labels_path)

    if len(population) != int(inputs["scene_population"]["expected_rows"]):
        raise RuntimeError("Population row-count mismatch")

    if len(auto) != int(inputs["auto_confirmed"]["expected_rows"]):
        raise RuntimeError("Auto-confirmed row-count mismatch")

    if len(required) != int(inputs["manual_required"]["expected_rows"]):
        raise RuntimeError("Manual-required row-count mismatch")

    if len(labels) != int(inputs["manual_labels"]["expected_rows"]):
        raise RuntimeError("Manual-label row-count mismatch")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if audit["status"] != "FINAL_BLINDED_WS_AUDIT_COMPLETE":
        raise RuntimeError("EXP040 audit is not complete")

    if audit["thresholds_changed_after_audit"] is not False:
        raise RuntimeError("Audit threshold-change contract violated")

    population_by_id = {
        r["scene_event_id"]: r
        for r in population
    }
    auto_by_id = {
        r["scene_event_id"]: r
        for r in auto
    }
    required_by_id = {
        r["scene_event_id"]: r
        for r in required
    }
    labels_by_id = {
        r["scene_event_id"]: r
        for r in labels
    }

    if len(population_by_id) != 2179:
        raise RuntimeError("Duplicate population scene IDs")

    if len(auto_by_id) != 361:
        raise RuntimeError("Duplicate auto-confirmed scene IDs")

    if len(required_by_id) != 1818:
        raise RuntimeError("Duplicate manual-required scene IDs")

    if len(labels_by_id) != 1818:
        raise RuntimeError("Duplicate manual-label scene IDs")

    auto_ids = set(auto_by_id)
    required_ids = set(required_by_id)
    population_ids = set(population_by_id)
    label_ids = set(labels_by_id)

    if auto_ids & required_ids:
        raise RuntimeError("Auto/manual populations overlap")

    if required_ids != label_ids:
        raise RuntimeError("Manual label IDs do not match disagreement IDs")

    if auto_ids | required_ids != population_ids:
        raise RuntimeError("Final-label populations do not cover EXP019 population")

    allowed = set(cfg["label_rule"]["allowed_final_labels"])

    output_rows = []

    for sid in sorted(population_ids):
        base = population_by_id[sid]

        if sid in auto_ids:
            if int(auto_by_id[sid]["pixel_confirmation_positive"]) != 1:
                raise RuntimeError(
                    "Auto-confirmed row is not pixel-positive: {}".format(sid)
                )

            final_label = "WHOLE_SCENE"
            label_source = "EXP034_AUTO_CONFIRMED"
            manual_reason = ""

        else:
            if int(required_by_id[sid]["pixel_confirmation_positive"]) != 0:
                raise RuntimeError(
                    "Manual-required row is not pixel-negative: {}".format(sid)
                )

            manual = labels_by_id[sid]
            final_label = (manual.get("manual_label") or "").strip()
            manual_reason = (manual.get("reason") or "").strip()

            if final_label not in allowed:
                raise RuntimeError(
                    "Invalid manual final label for {}: {}".format(
                        sid,
                        final_label,
                    )
                )

            if not manual_reason:
                raise RuntimeError(
                    "Missing manual reason for {}".format(sid)
                )

            label_source = "EXP036_MANUAL_ADJUDICATION"

        if final_label not in allowed:
            raise RuntimeError("Invalid final label")

        output_rows.append({
            "scene_event_id": sid,
            "video": base["video"],
            "n_object_events": base["n_object_events"],
            "n_unique_objects": base["n_unique_objects"],
            "disappear_min": base["disappear_min"],
            "disappear_max": base["disappear_max"],
            "reappear_min": base["reappear_min"],
            "reappear_max": base["reappear_max"],
            "final_ws_label": final_label,
            "label_source": label_source,
            "manual_reason": manual_reason,
        })

    counts = Counter(
        r["final_ws_label"]
        for r in output_rows
    )

    expected = cfg["expected_output"]

    if len(output_rows) != int(expected["rows"]):
        raise RuntimeError("Final output row-count mismatch")

    if counts["WHOLE_SCENE"] != int(expected["whole_scene"]):
        raise RuntimeError("Final WHOLE_SCENE count mismatch")

    if counts["NORMAL_OCCLUSION"] != int(expected["normal_occlusion"]):
        raise RuntimeError("Final NORMAL_OCCLUSION count mismatch")

    out_dir = ROOT / cfg["output_dir"]

    if out_dir.exists():
        raise RuntimeError("Output directory already exists")

    out_dir.mkdir(parents=True)

    labels_out = out_dir / "final_ws_labels.csv"

    write_csv(
        labels_out,
        output_rows,
        [
            "scene_event_id",
            "video",
            "n_object_events",
            "n_unique_objects",
            "disappear_min",
            "disappear_max",
            "reappear_min",
            "reappear_max",
            "final_ws_label",
            "label_source",
            "manual_reason",
        ],
    )

    whole_scene_ids = [
        r["scene_event_id"]
        for r in output_rows
        if r["final_ws_label"] == "WHOLE_SCENE"
    ]

    normal_ids = [
        r["scene_event_id"]
        for r in output_rows
        if r["final_ws_label"] == "NORMAL_OCCLUSION"
    ]

    (out_dir / "whole_scene_scene_ids.json").write_text(
        json.dumps(whole_scene_ids, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment": "EXP041",
        "status": "FINAL_WHOLE_SCENE_LABELS_FROZEN",
        "repo_commit_at_execution": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "population_rows": len(output_rows),
        "whole_scene_count": counts["WHOLE_SCENE"],
        "normal_occlusion_count": counts["NORMAL_OCCLUSION"],
        "auto_confirmed_count": len(auto_ids),
        "manual_adjudicated_count": len(required_ids),
        "manual_whole_scene_count": sum(
            1
            for sid in required_ids
            if labels_by_id[sid]["manual_label"] == "WHOLE_SCENE"
        ),
        "manual_normal_occlusion_count": sum(
            1
            for sid in required_ids
            if labels_by_id[sid]["manual_label"] == "NORMAL_OCCLUSION"
        ),
        "final_ws_labels_sha256": sha256sum(labels_out),
        "whole_scene_scene_ids_sha256": sha256sum(
            out_dir / "whole_scene_scene_ids.json"
        ),
        "input_hashes": {
            "scene_population": sha256sum(population_path),
            "auto_confirmed": sha256sum(auto_path),
            "manual_required": sha256sum(required_path),
            "manual_labels": sha256sum(labels_path),
            "blinded_audit_result": sha256sum(audit_path),
        },
        "fresh_final_dev_touched": False,
        "test_touched": False,
        "thresholds_changed": False,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("EXP041_FINAL_WS_FREEZE=PASS")
    print("TOTAL="+str(len(output_rows)))
    print("WHOLE_SCENE="+str(counts["WHOLE_SCENE"]))
    print("NORMAL_OCCLUSION="+str(counts["NORMAL_OCCLUSION"]))
    print("AUTO_CONFIRMED="+str(len(auto_ids)))
    print("MANUAL_ADJUDICATED="+str(len(required_ids)))
    print("FINAL_WS_LABELS_SHA256="+sha256sum(labels_out))
    print(
        "WHOLE_SCENE_IDS_SHA256="
        + sha256sum(out_dir / "whole_scene_scene_ids.json")
    )
    print("SUMMARY_SHA256="+sha256sum(summary_path))
    print("FRESH_FINAL_DEV_TOUCHED=0")
    print("TEST_TOUCHED=0")


if __name__ == "__main__":
    main()
