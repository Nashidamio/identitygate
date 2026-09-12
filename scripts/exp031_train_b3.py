import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class GateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def json_state_dict(model):
    return {
        name: tensor.detach().cpu().numpy().tolist()
        for name, tensor in model.state_dict().items()
    }


def train_head(
    rows,
    features,
    label_name,
    label_cfg,
    mean,
    std,
    cfg,
    seed,
    epochs,
):
    col = label_cfg["column"]
    data = [r for r in rows if r[col] in {"0", "1"}]

    y_np = np.asarray([int(r[col]) for r in data], dtype=np.float64)
    x_np = np.asarray(
        [[float(r[f]) for f in features] for r in data],
        dtype=np.float64,
    )

    pos = int(y_np.sum())
    neg = int(len(y_np) - pos)
    pos_videos = len({
        r["video"]
        for r in data
        if r[col] == "1"
    })

    assert len(data) == label_cfg["expected_labeled"]
    assert pos == label_cfg["expected_positive"]
    assert neg == label_cfg["expected_negative"]
    assert pos_videos == label_cfg["expected_positive_videos"]

    x = torch.from_numpy((x_np - mean) / std)
    y = torch.from_numpy(y_np)

    torch.manual_seed(seed)
    model = GateMLP(len(features)).to(dtype=torch.float64)

    pos_weight = torch.tensor(neg / pos, dtype=torch.float64)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=tuple(cfg["training"]["betas"]),
        eps=cfg["training"]["eps"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    with torch.no_grad():
        initial_loss = float(
            F.binary_cross_entropy_with_logits(
                model(x),
                y,
                pos_weight=pos_weight,
            ).item()
        )

    curve = []

    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(x)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=pos_weight,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "{} non-finite loss at epoch {}".format(
                    label_name,
                    epoch,
                )
            )

        loss.backward()
        opt.step()

        if (
            epoch == 1
            or epoch == epochs
            or epoch % cfg["training"]["log_every"] == 0
        ):
            curve.append({
                "epoch": epoch,
                "loss": float(loss.detach().item()),
            })

    with torch.no_grad():
        logits = model(x)
        scores = torch.sigmoid(logits)
        final_loss = float(
            F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=pos_weight,
            ).item()
        )

    if not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError("Non-finite scores")

    artifact = {
        "label_column": col,
        "labeled_rows": len(data),
        "positive": pos,
        "negative": neg,
        "positive_videos": pos_videos,
        "positive_weight": float(pos_weight.item()),
        "seed": seed,
        "initial_weighted_bce": initial_loss,
        "final_weighted_bce": final_loss,
        "score_min_train_descriptive": float(scores.min().item()),
        "score_max_train_descriptive": float(scores.max().item()),
        "parameter_count": parameter_count(model),
        "state_dict": json_state_dict(model),
    }

    return artifact, curve


def write_curve(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=["variant", "head", "epoch", "loss"],
            lineterminator="\n",
        )
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/EXP031-b3-development-train-v1.json",
    )
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    cfg = json.loads((ROOT / args.config).read_text())
    input_path = ROOT / cfg["input_csv"]

    actual_sha = sha256sum(input_path)
    if actual_sha != cfg["expected_input_sha256"]:
        raise RuntimeError("Input SHA mismatch")

    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    splits = sorted(set(r["split"] for r in rows))
    if splits != [cfg["expected_split"]]:
        raise RuntimeError("Unexpected split population")

    if len(set(r["video"] for r in rows)) != cfg["expected_videos"]:
        raise RuntimeError("Video count mismatch")

    pv = [r for r in rows if r["pointer_valid"] == "1"]

    if len(pv) != cfg["population"]["expected_pointer_valid_rows"]:
        raise RuntimeError("Pointer-valid population mismatch")

    union = cfg["variants"]["B3_R"]["features"]

    complete = [
        r for r in pv
        if all(finite_float(r[f]) for f in union)
    ]

    if len(complete) != cfg["population"]["expected_complete_rows"]:
        raise RuntimeError("Complete population mismatch")

    epochs = (
        cfg["training"]["sanity_epochs"]
        if args.sanity
        else cfg["training"]["epochs"]
    )

    out_dir = ROOT / (
        "experiments/EXP031_b3_train_sanity"
        if args.sanity
        else "experiments/EXP031_b3_train"
    )

    if out_dir.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(out_dir)
        )

    out_dir.mkdir(parents=True)

    summary_variants = {}
    curve_rows = []
    model_artifact = {
        "experiment": cfg["experiment"],
        "status": (
            "SANITY_MODEL_NOT_RESULT"
            if args.sanity
            else "TRAIN_ONLY_B3_DEVELOPMENT_WEIGHTS"
        ),
        "input_sha256": actual_sha,
        "pointer_valid_is_feature": False,
        "population_rows": len(complete),
        "variants": {},
        "training": cfg["training"],
        "claim_boundary": cfg["claim_boundary"],
        "dev_touched": 0,
        "test_touched": 0,
    }

    for variant_index, variant in enumerate(["B3_S", "B3_R"]):
        vcfg = cfg["variants"][variant]
        features = vcfg["features"]

        x_all = np.asarray(
            [[float(r[f]) for f in features] for r in complete],
            dtype=np.float64,
        )

        mean = x_all.mean(axis=0)
        std = x_all.std(axis=0)
        std[std < 1e-12] = cfg["normalization"][
            "zero_std_replacement"
        ]

        heads = {}

        for head_index, head in enumerate(["drift", "theft"]):
            seed = (
                cfg["training"]["base_seed"]
                + variant_index * 100
                + head_index
            )

            artifact, curve = train_head(
                rows=complete,
                features=features,
                label_name=head,
                label_cfg=cfg["labels"][head],
                mean=mean,
                std=std,
                cfg=cfg,
                seed=seed,
                epochs=epochs,
            )

            expected_per_head = vcfg[
                "parameter_count_expected_per_head"
            ]

            if artifact["parameter_count"] != expected_per_head:
                raise RuntimeError(
                    "{} {} parameter mismatch".format(
                        variant,
                        head,
                    )
                )

            heads[head] = artifact

            for r in curve:
                curve_rows.append({
                    "variant": variant,
                    "head": head,
                    "epoch": r["epoch"],
                    "loss": r["loss"],
                })

        total = sum(
            heads[h]["parameter_count"]
            for h in heads
        )

        if total != vcfg["parameter_count_expected_total"]:
            raise RuntimeError(
                "{} total parameter mismatch".format(variant)
            )

        model_artifact["variants"][variant] = {
            "features": features,
            "normalization": {
                "population": "pointer-valid common TRAIN rows",
                "n_rows": len(complete),
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            "heads": heads,
            "total_parameters": total,
        }

        summary_variants[variant] = {
            "features": features,
            "total_parameters": total,
            "heads": {
                h: {
                    k: v
                    for k, v in heads[h].items()
                    if k != "state_dict"
                }
                for h in heads
            },
        }

    model_path = out_dir / "model.json"
    model_path.write_text(
        json.dumps(model_artifact, indent=2, sort_keys=True) + "\n"
    )

    write_curve(out_dir / "training_curve.csv", curve_rows)

    summary = {
        "status": (
            "SANITY_PASS_NOT_SCIENTIFIC_RESULT"
            if args.sanity
            else "TRAIN_ONLY_B3_DEVELOPMENT_TRAINING_COMPLETE"
        ),
        "input_sha256": actual_sha,
        "pointer_valid_rows": len(pv),
        "complete_rows": len(complete),
        "videos": len(set(r["video"] for r in complete)),
        "epochs_executed": epochs,
        "variants": summary_variants,
        "model_sha256": sha256sum(model_path),
        "dev_touched": 0,
        "test_touched": 0,
        "claim_boundary": cfg["claim_boundary"],
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
