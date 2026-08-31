import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp024_model_utility import average_precision, roc_auc


ROOT = Path(__file__).resolve().parents[1]


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def metric_value(metric, y, score):
    if metric == "average_precision":
        if int(y.sum()) == 0:
            return None
        return average_precision(y, score)

    if metric == "roc_auc":
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None
        return roc_auc(y, score)

    raise ValueError(metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/EXP024-cluster-bootstrap-v1.json",
    )
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    input_path = ROOT / cfg["input_csv"]

    actual_sha = sha256sum(input_path)
    if actual_sha != cfg["expected_input_sha256"]:
        raise RuntimeError("OOF input SHA mismatch")

    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != cfg["expected_rows"]:
        raise RuntimeError("OOF row-count mismatch")

    all_videos = sorted({r["video"] for r in rows})
    if len(all_videos) != cfg["expected_videos"]:
        raise RuntimeError("Video-count mismatch")

    models = ["B2_core", "B3_S", "B3_R"]
    metrics = ["average_precision", "roc_auc"]

    for endpoint, expected in cfg["expected_endpoints"].items():
        erows = [r for r in rows if r["label"] == endpoint]
        y = np.asarray(
            [int(r["y_unsafe"]) for r in erows],
            dtype=np.int64,
        )

        if len(erows) != expected["n"]:
            raise RuntimeError("{} n mismatch".format(endpoint))

        if int(y.sum()) != expected["positive"]:
            raise RuntimeError("{} positive-count mismatch".format(endpoint))

        positive_videos = {
            r["video"]
            for r in erows
            if r["y_unsafe"] == "1"
        }

        if len(positive_videos) != expected["positive_videos"]:
            raise RuntimeError(
                "{} positive-video mismatch".format(endpoint)
            )

        for model in models:
            score = np.asarray(
                [
                    float(r[cfg["score_columns"][model]])
                    for r in erows
                ],
                dtype=np.float64,
            )

            for metric in metrics:
                observed = metric_value(metric, y, score)
                target = expected["point_metrics"][model][metric]

                if observed is None or abs(observed - target) > 1e-12:
                    raise RuntimeError(
                        "{} {} {} point-metric mismatch: {} vs {}".format(
                            endpoint,
                            model,
                            metric,
                            observed,
                            target,
                        )
                    )

    n_boot = (
        cfg["sanity_replicates"]
        if args.sanity
        else cfg["replicates"]
    )
    rng = np.random.default_rng(cfg["seed"])

    draw_rows = []
    summary = {
        "status": (
            "SANITY_PASS_NOT_INFERENCE"
            if args.sanity
            else "TRAIN_ONLY_CLUSTER_BOOTSTRAP_COMPLETE"
        ),
        "input_sha256": actual_sha,
        "cluster_unit": cfg["cluster_unit"],
        "method": cfg["method"],
        "replicates_requested": n_boot,
        "seed": cfg["seed"],
        "confidence": cfg["confidence"],
        "endpoints": {},
        "dev_touched": 0,
        "test_touched": 0,
    }

    for endpoint, expected in cfg["expected_endpoints"].items():
        erows = [r for r in rows if r["label"] == endpoint]
        y = np.asarray(
            [int(r["y_unsafe"]) for r in erows],
            dtype=np.int64,
        )
        videos = np.asarray(
            [r["video"] for r in erows],
            dtype=object,
        )

        endpoint_videos = sorted(set(videos.tolist()))
        by_video = {
            v: np.flatnonzero(videos == v)
            for v in endpoint_videos
        }

        positive_video_set = {
            v
            for v in endpoint_videos
            if int(y[by_video[v]].sum()) > 0
        }

        scores = {
            model: np.asarray(
                [
                    float(r[cfg["score_columns"][model]])
                    for r in erows
                ],
                dtype=np.float64,
            )
            for model in models
        }

        observed_metrics = {
            model: {
                metric: metric_value(metric, y, scores[model])
                for metric in metrics
            }
            for model in models
        }

        samples = {
            "{}_minus_{}__{}".format(a, b, metric): []
            for a, b in cfg["comparisons"]
            for metric in metrics
        }

        invalid = {key: 0 for key in samples}

        for rep in range(n_boot):
            sampled = rng.choice(
                endpoint_videos,
                size=len(endpoint_videos),
                replace=True,
            )

            idx = np.concatenate([by_video[v] for v in sampled])
            yy = y[idx]

            sampled_positive_draws = sum(
                1 for v in sampled if v in positive_video_set
            )
            sampled_positive_distinct = len(
                set(sampled.tolist()) & positive_video_set
            )

            model_metrics = {}
            for model in models:
                model_metrics[model] = {}
                ss = scores[model][idx]

                for metric in metrics:
                    model_metrics[model][metric] = metric_value(
                        metric,
                        yy,
                        ss,
                    )

            out = {
                "endpoint": endpoint,
                "replicate": rep,
                "n_rows": int(len(idx)),
                "positive_rows": int(yy.sum()),
                "negative_rows": int(len(yy) - yy.sum()),
                "positive_video_draws": sampled_positive_draws,
                "distinct_positive_videos": sampled_positive_distinct,
            }

            for a, b in cfg["comparisons"]:
                for metric in metrics:
                    key = "{}_minus_{}__{}".format(a, b, metric)
                    va = model_metrics[a][metric]
                    vb = model_metrics[b][metric]

                    if va is None or vb is None:
                        invalid[key] += 1
                        out[key] = ""
                    else:
                        delta = float(va - vb)
                        samples[key].append(delta)
                        out[key] = delta

            draw_rows.append(out)

        alpha = (1.0 - cfg["confidence"]) / 2.0
        endpoint_summary = {
            "n": expected["n"],
            "positive": expected["positive"],
            "negative": expected["n"] - expected["positive"],
            "positive_videos": expected["positive_videos"],
            "observed_metrics": observed_metrics,
            "comparisons": {},
        }

        for a, b in cfg["comparisons"]:
            cname = "{}_minus_{}".format(a, b)
            endpoint_summary["comparisons"][cname] = {}

            for metric in metrics:
                key = "{}__{}".format(cname, metric)
                vals = np.asarray(samples[key], dtype=np.float64)

                observed_delta = (
                    observed_metrics[a][metric]
                    - observed_metrics[b][metric]
                )

                if len(vals) == 0:
                    raise RuntimeError(
                        "{} {} has zero valid bootstrap replicates".format(
                            endpoint,
                            key,
                        )
                    )

                endpoint_summary["comparisons"][cname][metric] = {
                    "observed_delta": float(observed_delta),
                    "valid_replicates": int(len(vals)),
                    "invalid_replicates": int(invalid[key]),
                    "bootstrap_median": float(np.median(vals)),
                    "ci_lower": float(np.quantile(vals, alpha)),
                    "ci_upper": float(np.quantile(vals, 1.0 - alpha)),
                    "fraction_delta_gt_zero": float(np.mean(vals > 0.0)),
                    "fraction_delta_lt_zero": float(np.mean(vals < 0.0)),
                }

        summary["endpoints"][endpoint] = endpoint_summary

    out_dir = ROOT / (
        "experiments/EXP024_cluster_bootstrap_sanity"
        if args.sanity
        else "experiments/EXP024_cluster_bootstrap"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "endpoint",
        "replicate",
        "n_rows",
        "positive_rows",
        "negative_rows",
        "positive_video_draws",
        "distinct_positive_videos",
    ]

    for a, b in cfg["comparisons"]:
        for metric in metrics:
            fields.append(
                "{}_minus_{}__{}".format(a, b, metric)
            )

    write_csv(out_dir / "bootstrap_draws.csv", fields, draw_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
