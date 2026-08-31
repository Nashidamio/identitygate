import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def roc_auc(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ss = score[order]
    ranks = np.empty(len(y), dtype=np.float64)
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def average_precision(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    ss = score[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and ss[j] == ss[i]:
            j += 1
        group = ys[i:j]
        gp = int(group.sum())
        tp += gp
        fp += int(len(group) - gp)
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def fit_probe(x_train, y_train, x_test, l2, max_iter):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0

    xtr = torch.from_numpy((x_train - mean) / std)
    xte = torch.from_numpy((x_test - mean) / std)
    ytr = torch.from_numpy(y_train.astype(np.float64))

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("Training fold has one class")

    w = torch.zeros(xtr.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float64)

    opt = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = xtr @ w + b
        loss = F.binary_cross_entropy_with_logits(
            logits,
            ytr,
            pos_weight=pos_weight,
        )
        loss = loss + 0.5 * l2 * torch.sum(w * w)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        train_logits = xtr @ w + b
        loss = F.binary_cross_entropy_with_logits(
            train_logits,
            ytr,
            pos_weight=pos_weight,
        ) + 0.5 * l2 * torch.sum(w * w)
        score = torch.sigmoid(xte @ w + b).numpy()

    return score, w.detach().numpy(), float(b.detach()), float(loss.detach())


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/EXP024-model-utility-v1.json")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(24024)

    cfg = json.loads((ROOT / args.config).read_text())
    input_path = ROOT / cfg["input_csv"]
    actual_sha = sha256sum(input_path)
    if actual_sha != cfg["expected_input_sha256"]:
        raise RuntimeError("Input SHA mismatch")

    union = cfg["feature_families"]["B3_R"]
    common = []
    with open(input_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["pointer_valid"] != "1":
                continue
            if all(finite_float(r[x]) for x in union):
                common.append(r)

    if len(common) != cfg["expected_common_rows"]:
        raise RuntimeError("Common population mismatch")

    out_dir = ROOT / (
        "experiments/EXP024_model_utility_sanity"
        if args.sanity
        else "experiments/EXP024_model_utility"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_rows = []
    video_rows = []
    coef_rows = []
    results = {}

    for label_name, label_col in cfg["labels"].items():
        data = [r for r in common if r[label_col] in {"0", "1"}]
        y = np.array([int(r[label_col]) for r in data], dtype=np.int64)
        videos = np.array([r["video"] for r in data], dtype=object)
        expected = cfg["expected_label_counts"][label_name]

        if len(data) != expected["labeled"] or int(y.sum()) != expected["positive"]:
            raise RuntimeError("{} label count mismatch".format(label_name))

        heldouts = sorted(set(videos.tolist()))
        if args.sanity:
            heldouts = [
                v for v in cfg["sanity_holdout_videos"]
                if v in heldouts
            ]

        scores = {
            name: np.full(len(data), np.nan, dtype=np.float64)
            for name in cfg["feature_families"]
        }

        for heldout in heldouts:
            train_idx = np.flatnonzero(videos != heldout)
            test_idx = np.flatnonzero(videos == heldout)

            for model_name, features in cfg["feature_families"].items():
                x = np.array(
                    [[float(r[f]) for f in features] for r in data],
                    dtype=np.float64,
                )
                pred, coef, intercept, loss = fit_probe(
                    x[train_idx],
                    y[train_idx],
                    x[test_idx],
                    cfg["model"]["l2"],
                    cfg["model"]["max_iter"],
                )
                scores[model_name][test_idx] = pred

                for feature, value in zip(features, coef):
                    coef_rows.append({
                        "label": label_name,
                        "heldout_video": heldout,
                        "model": model_name,
                        "feature": feature,
                        "coef_standardized": float(value),
                        "intercept_standardized": intercept,
                        "train_loss": loss,
                        "train_positive": int(y[train_idx].sum()),
                        "train_negative": int(len(train_idx) - y[train_idx].sum()),
                    })

        mask = np.isfinite(scores["B2_core"])

        for model_name in scores:
            if not np.array_equal(np.isfinite(scores[model_name]), mask):
                raise RuntimeError("Model populations differ")

        if args.sanity:
            yy = y[mask]
            results[label_name] = {}

            for model_name in scores:
                ss = scores[model_name][mask]
                results[label_name][model_name] = {
                    "n": int(len(yy)),
                    "positive": int(yy.sum()),
                    "score_min": float(ss.min()),
                    "score_max": float(ss.max()),
                    "average_precision": average_precision(yy, ss),
                    "roc_auc": roc_auc(yy, ss),
                }
            continue

        if not np.all(mask):
            raise RuntimeError("{} missing OOF predictions".format(label_name))

        results[label_name] = {
            "n": int(len(y)),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "positive_videos": sorted(set(videos[y == 1].tolist())),
            "models": {},
        }

        for model_name in scores:
            results[label_name]["models"][model_name] = {
                "average_precision": average_precision(y, scores[model_name]),
                "roc_auc": roc_auc(y, scores[model_name]),
            }

        for i, r in enumerate(data):
            oof_rows.append({
                "label": label_name,
                "video": r["video"],
                "frame_idx": r["frame_idx"],
                "obj_idx": r["obj_idx"],
                "object_id": r["object_id"],
                "y_unsafe": int(y[i]),
                "B2_core_p_unsafe": float(scores["B2_core"][i]),
                "B3_S_p_unsafe": float(scores["B3_S"][i]),
                "B3_R_p_unsafe": float(scores["B3_R"][i]),
            })

        for video in sorted(set(videos.tolist())):
            idx = np.flatnonzero(videos == video)
            yy = y[idx]

            for model_name in scores:
                ss = scores[model_name][idx]
                video_rows.append({
                    "label": label_name,
                    "video": video,
                    "model": model_name,
                    "n": int(len(idx)),
                    "positive": int(yy.sum()),
                    "negative": int(len(yy) - yy.sum()),
                    "average_precision": (
                        "" if yy.sum() == 0 else average_precision(yy, ss)
                    ),
                    "roc_auc": (
                        ""
                        if yy.sum() == 0 or yy.sum() == len(yy)
                        else roc_auc(yy, ss)
                    ),
                })

    if args.sanity:
        summary = {
            "status": "SANITY_PASS_NOT_RESULT",
            "input_sha256": actual_sha,
            "common_rows": len(common),
            "heldout_videos": cfg["sanity_holdout_videos"],
            "results": results,
            "dev_touched": 0,
            "test_touched": 0,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    write_csv(
        out_dir / "oof_predictions.csv",
        [
            "label",
            "video",
            "frame_idx",
            "obj_idx",
            "object_id",
            "y_unsafe",
            "B2_core_p_unsafe",
            "B3_S_p_unsafe",
            "B3_R_p_unsafe",
        ],
        oof_rows,
    )

    write_csv(
        out_dir / "per_video_metrics.csv",
        [
            "label",
            "video",
            "model",
            "n",
            "positive",
            "negative",
            "average_precision",
            "roc_auc",
        ],
        video_rows,
    )

    write_csv(
        out_dir / "fold_coefficients.csv",
        [
            "label",
            "heldout_video",
            "model",
            "feature",
            "coef_standardized",
            "intercept_standardized",
            "train_loss",
            "train_positive",
            "train_negative",
        ],
        coef_rows,
    )

    result = {
        "status": "TRAIN_ONLY_OOF_DISCRIMINATION_COMPLETE_NOT_INFERENCE",
        "scope": "Utility probe only. Not closed-loop tracking improvement.",
        "input_sha256": actual_sha,
        "common_complete_case_rows": len(common),
        "feature_families": cfg["feature_families"],
        "model": cfg["model"],
        "validation": cfg["validation"],
        "results": results,
        "dev_touched": 0,
        "test_touched": 0,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
