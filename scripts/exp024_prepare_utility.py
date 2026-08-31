import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/EXP024-utility-probe-v1.json"
OUT_DIR = ROOT / "experiments/EXP024_utility_prepare"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def as_bool(value):
    return str(value).strip().lower() in {"1", "true"}


def drift_label(value):
    value = parse_float(value)
    if value is None:
        return ""
    if value < 0.3:
        return 1
    if value > 0.7:
        return 0
    return ""


def theft_label(value):
    value = parse_float(value)
    if value is None:
        return ""
    if value > 0.5:
        return 1
    if value < 0.3:
        return 0
    return ""


def main():
    cfg = json.loads(CONFIG_PATH.read_text())

    input_csv = ROOT / cfg["input_csv"]
    per_video_root = ROOT / cfg["per_video_root"]

    assert sha256_file(input_csv) == cfg["expected_input_sha256"]

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == cfg["expected_rows"]
    assert {r["split"] for r in rows} == {cfg["allowed_split"]}

    by_video = defaultdict(list)
    for idx, row in enumerate(rows):
        by_video[row["video"]].append(idx)

    assert len(by_video) == cfg["expected_videos"]

    max_self_diff = 0.0
    valid_rows = 0
    enriched = [None] * len(rows)

    for video in sorted(by_video):
        indices = by_video[video]
        npz_path = per_video_root / ("TRAIN_" + video) / "pointers.npz"

        with np.load(npz_path, allow_pickle=False) as z:
            frame_idx = z["frame_idx"]
            object_id = z["object_id"]
            ptr = z["ptr_f32"]
            anchor_ids = z["anchor_object_id"]
            anchors = z["anchor_ptr_f32"]

        assert ptr.dtype == np.float32
        assert anchors.dtype == np.float32
        assert len(indices) == ptr.shape[0]
        assert ptr.shape[1] == 256
        assert anchors.shape[1] == 256
        assert len(anchor_ids) >= 2

        anchor_id_to_idx = {
            int(oid): i for i, oid in enumerate(anchor_ids.tolist())
        }
        assert len(anchor_id_to_idx) == len(anchor_ids)

        ptr_t = torch.from_numpy(ptr).to(dtype=torch.float32)
        anc_t = torch.from_numpy(anchors).to(dtype=torch.float32)

        ptr_norm = torch.linalg.vector_norm(ptr_t, dim=1, keepdim=True)
        anc_norm = torch.linalg.vector_norm(anc_t, dim=1, keepdim=True)

        assert torch.all(ptr_norm > 0)
        assert torch.all(anc_norm > 0)

        ptr_u = ptr_t / ptr_norm
        anc_u = anc_t / anc_norm

        with torch.no_grad():
            sims = ptr_u @ anc_u.T

        for local_i, global_i in enumerate(indices):
            row = dict(rows[global_i])

            assert int(row["frame_idx"]) == int(frame_idx[local_i])
            assert int(row["object_id"]) == int(object_id[local_i])

            oid = int(row["object_id"])
            own_idx = anchor_id_to_idx[oid]

            comp_cos = ""
            comp_oid = ""
            margin = ""

            if as_bool(row["pointer_valid"]):
                valid_rows += 1

                stored_self = parse_float(row["ptr_sim_anchor_fp32"])
                assert stored_self is not None

                recomputed_self = float(sims[local_i, own_idx].item())
                diff = abs(stored_self - recomputed_self)
                max_self_diff = max(max_self_diff, diff)

                candidate = sims[local_i].clone()
                candidate[own_idx] = -torch.inf

                best_idx = int(torch.argmax(candidate).item())
                best_cos = float(candidate[best_idx].item())
                best_oid = int(anchor_ids[best_idx])

                comp_cos = format(best_cos, ".9g")
                comp_oid = str(best_oid)
                margin = format(stored_self - best_cos, ".9g")

            row["max_comp_anchor_cos_fp32"] = comp_cos
            row["max_comp_anchor_object_id"] = comp_oid
            row["identity_margin_fp32_diagnostic"] = margin
            row["drift_label_s2_probe"] = drift_label(row["target_iou"])
            row["theft_label_s2_probe"] = theft_label(row["max_other_iou"])

            enriched[global_i] = row

    assert all(r is not None for r in enriched)
    assert max_self_diff <= 1e-5

    output_fields = list(rows[0].keys()) + [
        "max_comp_anchor_cos_fp32",
        "max_comp_anchor_object_id",
        "identity_margin_fp32_diagnostic",
        "drift_label_s2_probe",
        "theft_label_s2_probe",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "train18_utility.csv"

    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=output_fields)
        wr.writeheader()
        wr.writerows(enriched)

    overall = Counter()
    positive_videos = {
        "drift": set(),
        "theft": set(),
    }

    per_video = {}

    for video in sorted(by_video):
        c = Counter()

        for i in by_video[video]:
            r = enriched[i]
            valid = as_bool(r["pointer_valid"])

            c["rows"] += 1
            if valid:
                c["pointer_valid"] += 1

            for outcome, col in [
                ("drift", "drift_label_s2_probe"),
                ("theft", "theft_label_s2_probe"),
            ]:
                label = r[col]

                if label == "":
                    c[outcome + "_excluded"] += 1
                    continue

                c[outcome + "_labeled"] += 1

                if int(label) == 1:
                    c[outcome + "_positive"] += 1
                    positive_videos[outcome].add(video)
                else:
                    c[outcome + "_negative"] += 1

                if valid:
                    c["pv_" + outcome + "_labeled"] += 1
                    if int(label) == 1:
                        c["pv_" + outcome + "_positive"] += 1
                    else:
                        c["pv_" + outcome + "_negative"] += 1

        per_video[video] = dict(c)
        overall.update(c)

    summary = {
        "experiment": "EXP024",
        "status": "PREPARATION_COMPLETE_NOT_MODEL_UTILITY_RESULT",
        "input_sha256": sha256_file(input_csv),
        "output_csv_sha256": sha256_file(out_csv),
        "rows": len(enriched),
        "videos": len(by_video),
        "pointer_valid_rows": valid_rows,
        "pointer_valid_fraction": valid_rows / len(enriched),
        "max_self_anchor_recompute_abs_diff": max_self_diff,
        "positive_video_counts": {
            k: len(v) for k, v in positive_videos.items()
        },
        "positive_videos": {
            k: sorted(v) for k, v in positive_videos.items()
        },
        "overall_counts": dict(overall),
        "per_video": per_video,
        "dev_touched": 0,
        "test_touched": 0,
        "notes": [
            "All B2/B3-S/B3-R comparisons use the same pointer-valid conditional population.",
            "pointer_valid is a validity mask and is not a predictive feature.",
            "identity margin is diagnostic only.",
            "Features 4, 8, 9, and 11 remain deferred.",
            "Feature 7 remains open and is excluded.",
            "S2 labels are development utility probe labels, not final frozen safety labels."
        ]
    }

    out_summary = OUT_DIR / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "rows": summary["rows"],
        "videos": summary["videos"],
        "pointer_valid_rows": summary["pointer_valid_rows"],
        "pointer_valid_fraction": summary["pointer_valid_fraction"],
        "max_self_anchor_recompute_abs_diff": summary[
            "max_self_anchor_recompute_abs_diff"
        ],
        "positive_video_counts": summary["positive_video_counts"],
        "overall_counts": summary["overall_counts"],
        "dev_touched": 0,
        "test_touched": 0
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
