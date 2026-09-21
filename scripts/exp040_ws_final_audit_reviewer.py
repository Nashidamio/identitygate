#!/usr/bin/env python3

import csv
import hashlib
import importlib.util
import json
import subprocess
from http.server import ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP040-ws-final-audit-review-v1.json"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_clean_committed_tree():
    required = [
        "configs/EXP040-ws-final-audit-review-v1.json",
        "scripts/exp040_ws_final_audit_reviewer.py",
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
        raise RuntimeError("EXP040 reviewer requires a clean committed tree")


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "exp036_reviewer",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load EXP036 reviewer module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    require_clean_committed_tree()

    cfg = json.loads(
        CFG_PATH.read_text(encoding="utf-8")
    )

    if cfg["experiment"] != "EXP040":
        raise RuntimeError("Unexpected experiment ID")

    if cfg["stage"] != "FINAL_BLINDED_AUDIT30_MANUAL_REVIEW":
        raise RuntimeError("Unexpected review stage")

    source = ROOT / cfg["source_csv"]

    if sha256sum(source) != cfg["source_csv_sha256"]:
        raise RuntimeError("EXP040 final audit source SHA mismatch")

    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    expected_fields = [
        "scene_event_id",
        "video",
        "n_object_events",
        "n_unique_objects",
        "disappear_min",
        "disappear_max",
        "reappear_min",
        "reappear_max",
    ]

    if fields != expected_fields:
        raise RuntimeError("Blinding failure: unexpected source schema")

    if len(rows) != 30:
        raise RuntimeError("Expected exactly 30 audit scenes")

    if len({r["scene_event_id"] for r in rows}) != 30:
        raise RuntimeError("Duplicate audit scene ID")

    banned = {
        "automatic_status",
        "pixel_confirmation_positive",
        "camera_cut_positive",
        "global_mad_positive",
        "transnet_disappear_max_prob",
        "transnet_reappear_max_prob",
        "event_max_mad",
    }

    if banned.intersection(fields):
        raise RuntimeError("Blinding failure: outcome field present")

    blind = cfg["blinding_contract"]

    if not (
        blind["automatic_status_exposed"] is False
        and blind["pixel_scores_exposed"] is False
        and blind["component_decisions_exposed"] is False
        and blind["sam_or_gate_outcomes_exposed"] is False
        and blind["source_is_annotation_only"] is True
    ):
        raise RuntimeError("Blinding contract failure")

    exp036_path = ROOT / cfg["exp036_reviewer_script"]

    if sha256sum(exp036_path) != cfg["exp036_reviewer_script_sha256"]:
        raise RuntimeError("EXP036 reviewer script SHA mismatch")

    exp036 = load_module(exp036_path)

    state = exp036.ReviewState(cfg)

    if len(state.rows) != 30:
        raise RuntimeError("Reviewer state scene-count mismatch")

    host = cfg["host"]
    port = int(cfg["port"])

    print("EXP040_FINAL_AUDIT_REVIEW_READY")
    print("URL=http://{}:{}".format(host, port))
    print("TOTAL={}".format(len(state.rows)))
    print("FILLED={}".format(state.filled()))
    print("LABELS={}".format(state.labels_path))
    print("BLINDING_CONTRACT=PASS")
    print("Press Ctrl+C only when you want to stop the server.")

    server = ThreadingHTTPServer(
        (host, port),
        exp036.make_handler(state),
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
