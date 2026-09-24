import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
KEY_FIELDS = ["video", "object_id", "reappear_frame", "disappear_start"]


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def repo_state():
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    ).strip()
    return head, dirty


def event_key(row):
    return (
        row["video"],
        int(row["object_id"]),
        int(row["reappear_frame"]),
        int(row["disappear_start"]),
    )


def load_and_validate(cfg):
    input_path = ROOT / cfg["input_csv"]
    actual_sha = sha256sum(input_path)
    if actual_sha != cfg["expected_input_sha256"]:
        raise RuntimeError("EXP051 input SHA mismatch")

    rows = read_csv(input_path)
    if len(rows) != cfg["expected_rows"]:
        raise RuntimeError("EXP051 row-count mismatch")

    labels = sorted({r["run_label"] for r in rows})
    expected_labels = sorted(cfg["run_labels"])
    if labels != expected_labels:
        raise RuntimeError("Run-label set mismatch")

    by_label = {}
    reference_keys = None
    reference_videos = None

    for label in cfg["run_labels"]:
        selected = [r for r in rows if r["run_label"] == label]
        if len(selected) != cfg["expected_events_per_label"]:
            raise RuntimeError("{} event-count mismatch".format(label))

        mapping = {}
        for row in selected:
            key = event_key(row)
            if key in mapping:
                raise RuntimeError("Duplicate event key for {}".format(label))

            rec = int(row["recovered_w30"])
            theft = int(row["theft_event_w30"])
            if rec not in (0, 1) or theft not in (0, 1):
                raise RuntimeError("Non-binary endpoint value")

            mapping[key] = {
                "POR30": rec,
                "ITR30": theft,
            }

        keys = set(mapping)
        videos = {k[0] for k in keys}

        if reference_keys is None:
            reference_keys = keys
            reference_videos = videos
        else:
            if keys != reference_keys:
                raise RuntimeError("Paired event-key mismatch for {}".format(label))
            if videos != reference_videos:
                raise RuntimeError("Video-set mismatch for {}".format(label))

        by_label[label] = mapping

    if len(reference_videos) != cfg["expected_videos"]:
        raise RuntimeError("Video-count mismatch")

    return input_path, by_label, sorted(reference_keys), sorted(reference_videos)


def build_cluster_arrays(by_label, event_keys, videos, run_labels):
    video_index = {v: i for i, v in enumerate(videos)}
    counts = np.zeros(len(videos), dtype=np.int64)
    success = {
        endpoint: {
            label: np.zeros(len(videos), dtype=np.int64)
            for label in run_labels
        }
        for endpoint in ("POR30", "ITR30")
    }

    for key in event_keys:
        i = video_index[key[0]]
        counts[i] += 1
        for label in run_labels:
            for endpoint in ("POR30", "ITR30"):
                success[endpoint][label][i] += by_label[label][key][endpoint]

    if np.any(counts <= 0):
        raise RuntimeError("Every statistical cluster must contain at least one event")

    return counts, success


def pooled_rate(counts, successes):
    return float(np.sum(successes) / np.sum(counts))


def pooled_delta(counts, successes_a, successes_b):
    return pooled_rate(counts, successes_a) - pooled_rate(counts, successes_b)


def bootstrap_delta(draw_index, counts, successes_a, successes_b):
    denom = counts[draw_index].sum(axis=1, dtype=np.int64)
    num_a = successes_a[draw_index].sum(axis=1, dtype=np.int64)
    num_b = successes_b[draw_index].sum(axis=1, dtype=np.int64)
    return num_a / denom - num_b / denom


def jackknife_delta(counts, successes_a, successes_b):
    total_n = int(counts.sum())
    total_a = int(successes_a.sum())
    total_b = int(successes_b.sum())

    out = np.empty(len(counts), dtype=np.float64)
    for i in range(len(counts)):
        denom = total_n - int(counts[i])
        if denom <= 0:
            raise RuntimeError("Invalid delete-one-cluster denominator")
        out[i] = (
            (total_a - int(successes_a[i])) / denom
            - (total_b - int(successes_b[i])) / denom
        )
    return out


def bca_interval(observed, bootstrap, jackknife, confidence):
    bootstrap = np.asarray(bootstrap, dtype=np.float64)
    jackknife = np.asarray(jackknife, dtype=np.float64)

    if bootstrap.ndim != 1 or len(bootstrap) < 2:
        raise ValueError("Bootstrap sample must be one-dimensional")
    if jackknife.ndim != 1 or len(jackknife) < 3:
        raise ValueError("Jackknife sample must contain at least three clusters")
    if not np.all(np.isfinite(bootstrap)) or not np.all(np.isfinite(jackknife)):
        raise ValueError("BCa inputs must be finite")

    b = len(bootstrap)
    less = int(np.sum(bootstrap < observed))
    equal = int(np.sum(bootstrap == observed))
    prop = (less + 0.5 * equal) / b
    prop = float(np.clip(prop, 0.5 / b, 1.0 - 0.5 / b))
    z0 = float(norm.ppf(prop))

    jack_mean = float(np.mean(jackknife))
    centered = jack_mean - jackknife
    numerator = float(np.sum(centered ** 3))
    denominator = float(6.0 * (np.sum(centered ** 2) ** 1.5))
    acceleration = 0.0 if denominator == 0.0 else numerator / denominator

    alpha = (1.0 - confidence) / 2.0

    def adjusted(prob):
        z = float(norm.ppf(prob))
        term = z0 + z
        denom = 1.0 - acceleration * term
        if denom == 0.0:
            value = 0.0 if term < 0.0 else 1.0
        else:
            value = float(norm.cdf(z0 + term / denom))
        return float(np.clip(value, 0.0, 1.0))

    q_lower = adjusted(alpha)
    q_upper = adjusted(1.0 - alpha)
    if q_lower > q_upper:
        q_lower, q_upper = q_upper, q_lower

    ci_lower = float(np.quantile(bootstrap, q_lower))
    ci_upper = float(np.quantile(bootstrap, q_upper))

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bias_correction_z0": z0,
        "acceleration": float(acceleration),
        "adjusted_alpha_lower": q_lower,
        "adjusted_alpha_upper": q_upper,
        "bootstrap_less_than_observed": less,
        "bootstrap_equal_to_observed": equal,
    }


def interpret_primary(ci_lower, ci_upper, practical):
    if ci_upper < 0.0:
        return "SUPPORTED_HARM"
    if ci_lower >= practical:
        return "SUPPORTED_AT_OR_ABOVE_PRACTICAL_THRESHOLD"
    if ci_lower > 0.0 and ci_upper < practical:
        return "SUPPORTED_POSITIVE_BELOW_PRACTICAL_THRESHOLD"
    if ci_lower > 0.0 and ci_upper >= practical:
        return "SUPPORTED_POSITIVE_PRACTICAL_THRESHOLD_UNRESOLVED"
    if ci_lower <= 0.0 and ci_upper < practical:
        return "NO_SUPPORTED_POSITIVE_AND_PRACTICAL_THRESHOLD_RULED_OUT"
    return "INCONCLUSIVE_BETWEEN_ZERO_AND_PRACTICAL_THRESHOLD"


def run_selftest():
    observed = 0.25
    bootstrap = np.full(200, observed, dtype=np.float64)
    jackknife = np.full(8, observed, dtype=np.float64)
    out = bca_interval(observed, bootstrap, jackknife, 0.95)
    assert out["ci_lower"] == observed
    assert out["ci_upper"] == observed
    assert abs(out["bias_correction_z0"]) < 1e-12
    assert out["acceleration"] == 0.0

    counts = np.asarray([2, 3, 1, 4], dtype=np.int64)
    a = np.asarray([2, 1, 1, 2], dtype=np.int64)
    b = np.asarray([1, 1, 0, 2], dtype=np.int64)
    expected = 0.2
    assert abs(pooled_delta(counts, a, b) - expected) < 1e-12
    jack = jackknife_delta(counts, a, b)
    assert len(jack) == 4
    assert np.all(np.isfinite(jack))
    print("EXP052_SELFTEST=PASS")


def run(cfg, config_path):
    head, dirty = repo_state()
    if cfg["require_clean_git"] and dirty:
        raise RuntimeError("STOP: working tree is not clean: " + dirty)

    input_path, by_label, event_keys, videos = load_and_validate(cfg)
    counts, success = build_cluster_arrays(
        by_label,
        event_keys,
        videos,
        cfg["run_labels"],
    )

    rng = np.random.default_rng(cfg["seed"])
    draw_index = rng.integers(
        0,
        len(videos),
        size=(cfg["bootstrap_replicates"], len(videos)),
        dtype=np.int64,
    )

    point_rates = {}
    for endpoint in ("POR30", "ITR30"):
        point_rates[endpoint] = {
            label: pooled_rate(counts, success[endpoint][label])
            for label in cfg["run_labels"]
        }

    comparison_rows = []
    bootstrap_cache = {}

    for endpoint in ("POR30", "ITR30"):
        for spec in cfg["comparisons"]:
            a = spec["a"]
            b = spec["b"]
            observed = pooled_delta(
                counts,
                success[endpoint][a],
                success[endpoint][b],
            )
            boot = bootstrap_delta(
                draw_index,
                counts,
                success[endpoint][a],
                success[endpoint][b],
            )
            jack = jackknife_delta(
                counts,
                success[endpoint][a],
                success[endpoint][b],
            )
            bca = bca_interval(
                observed,
                boot,
                jack,
                cfg["confidence"],
            )
            key = (endpoint, spec["name"])
            bootstrap_cache[key] = boot

            comparison_rows.append({
                "group": spec["group"],
                "endpoint": endpoint,
                "comparison": spec["name"],
                "a": a,
                "b": b,
                "observed_delta": observed,
                "ci_lower": bca["ci_lower"],
                "ci_upper": bca["ci_upper"],
                "bias_correction_z0": bca["bias_correction_z0"],
                "acceleration": bca["acceleration"],
                "adjusted_alpha_lower": bca["adjusted_alpha_lower"],
                "adjusted_alpha_upper": bca["adjusted_alpha_upper"],
                "bootstrap_less_than_observed": bca["bootstrap_less_than_observed"],
                "bootstrap_equal_to_observed": bca["bootstrap_equal_to_observed"],
                "bootstrap_replicates": cfg["bootstrap_replicates"],
                "seed": cfg["seed"],
            })

    primary_name = cfg["primary_comparison"]
    primary_row = next(
        row
        for row in comparison_rows
        if row["endpoint"] == "POR30"
        and row["comparison"] == primary_name
    )
    primary_interpretation = interpret_primary(
        primary_row["ci_lower"],
        primary_row["ci_upper"],
        cfg["minimum_practically_important_benefit"],
    )

    out_dir = ROOT / cfg["output_dir"]
    if out_dir.exists():
        raise RuntimeError("STOP: output directory already exists")
    out_dir.mkdir(parents=True)

    fields = [
        "group",
        "endpoint",
        "comparison",
        "a",
        "b",
        "observed_delta",
        "ci_lower",
        "ci_upper",
        "bias_correction_z0",
        "acceleration",
        "adjusted_alpha_lower",
        "adjusted_alpha_upper",
        "bootstrap_less_than_observed",
        "bootstrap_equal_to_observed",
        "bootstrap_replicates",
        "seed",
    ]
    write_csv(out_dir / "comparison_summary.csv", fields, comparison_rows)

    primary_boot = bootstrap_cache[("POR30", primary_name)]
    write_csv(
        out_dir / "primary_bootstrap_draws.csv",
        ["replicate", "delta_por30"],
        [
            {
                "replicate": i,
                "delta_por30": float(value),
            }
            for i, value in enumerate(primary_boot)
        ],
    )

    summary = {
        "status": "FRESH_DEV_BCA_INFERENCE_COMPLETE",
        "repo_commit": head,
        "git_dirty_before_run": False,
        "input_csv": cfg["input_csv"],
        "input_sha256": sha256sum(input_path),
        "script_sha256": sha256sum(Path(__file__)),
        "config_sha256": sha256sum(config_path),
        "cluster_unit": "video",
        "paired": True,
        "method": "paired nonparametric video-cluster bootstrap with BCa 95 percent CI",
        "bca_bias_correction_ties": "midrank",
        "bca_acceleration": "delete-one-video jackknife",
        "confidence": cfg["confidence"],
        "bootstrap_replicates": cfg["bootstrap_replicates"],
        "seed": cfg["seed"],
        "videos": len(videos),
        "events_per_label": len(event_keys),
        "test_touched": False,
        "point_rates": point_rates,
        "primary": {
            **primary_row,
            "minimum_practically_important_benefit": cfg[
                "minimum_practically_important_benefit"
            ],
            "interpretation": primary_interpretation,
        },
        "comparisons": comparison_rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["primary"], indent=2, sort_keys=True))
    print("EXP052_RUN=PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["selftest", "run"],
    )
    parser.add_argument(
        "--config",
        default="configs/EXP052-fresh-dev-bca-inference-v1.json",
    )
    args = parser.parse_args()

    if args.mode == "selftest":
        run_selftest()
        return

    config_path = ROOT / args.config
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run(cfg, config_path)


if __name__ == "__main__":
    main()
