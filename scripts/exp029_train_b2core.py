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


def finite_float(value):
    try:
        return math.isfinite(float(value))
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


def train_head(name, rows, label_cfg, features, mean, std, cfg, epochs, head_index):
    column = label_cfg["column"]
    data = [r for r in rows if r[column] in {"0", "1"}]

    y_np = np.asarray(
        [int(r[column]) for r in data],
        dtype=np.float64,
    )
    x_np = np.asarray(
        [[float(r[f]) for f in features] for r in data],
        dtype=np.float64,
    )

    positive = int(y_np.sum())
    negative = int(len(y_np) - positive)
    positive_videos = len({
        r["video"]
        for r in data
        if r[column] == "1"
    })

    if len(data) != label_cfg["expected_labeled"]:
        raise RuntimeError("{} labeled count mismatch".format(name))
    if positive != label_cfg["expected_positive"]:
        raise RuntimeError("{} positive count mismatch".format(name))
    if negative != label_cfg["expected_negative"]:
        raise RuntimeError("{} negative count mismatch".format(name))
    if positive_videos != label_cfg["expected_positive_videos"]:
        raise RuntimeError("{} positive-video count mismatch".format(name))

    x = torch.from_numpy((x_np - mean) / std)
    y = torch.from_numpy(y_np)

    torch.manual_seed(cfg["training"]["seed"] + head_index)
    model = GateMLP(len(features)).to(dtype=torch.float64)

    if parameter_count(model) != cfg["architecture"]["parameter_count_expected_per_head"]:
        raise RuntimeError("{} parameter count mismatch".format(name))

    pos_weight = torch.tensor(
        negative / positive,
        dtype=torch.float64,
    )

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
                "{} non-finite loss at epoch {}".format(name, epoch)
            )

        loss.backward()
        opt.step()

        if (
            epoch == 1
            or epoch == epochs
            or epoch % cfg["training"]["log_every"] == 0
        ):
            curve.append({
                "head": name,
                "epoch": epoch,
                "loss": float(loss.detach().item()),
            })

    with torch.no_grad():
        logits = model(x)
        final_loss = float(
            F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=pos_weight,
            ).item()
        )
        scores = torch.sigmoid(logits)

    if not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError("{} produced non-finite scores".format(name))

    artifact = {
        "label_column": column,
        "labeled_rows": len(data),
        "positive": positive,
        "negative": negative,
        "positive_videos": positive_videos,
        "positive_weight": float(pos_weight.item()),
        "seed": cfg["training"]["seed"] + head_index,
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
            fieldnames=["head", "epoch", "loss"],
            lineterminator="\n",
        )
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/EXP029-b2core-train-v1.json",
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
        raise RuntimeError("Input SHA256 mismatch")

    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    splits = sorted(set(r["split"] for r in rows))
    if splits != [cfg["expected_split"]]:
        raise RuntimeError("Unexpected split population: {}".format(splits))

    videos = sorted(set(r["video"] for r in rows))
    if len(videos) != cfg["expected_videos"]:
        raise RuntimeError("Video count mismatch")

    features = cfg["features"]

    finite_rows = [
        r
        for r in rows
        if all(finite_float(r[f]) for f in features)
    ]

    if len(finite_rows) != cfg["expected_finite_rows"]:
        raise RuntimeError(
            "Finite B2-core population mismatch: {}".format(
                len(finite_rows)
            )
        )

    x_all = np.asarray(
        [[float(r[f]) for f in features] for r in finite_rows],
        dtype=np.float64,
    )

    mean = x_all.mean(axis=0)
    std = x_all.std(axis=0)
    std[std < 1e-12] = cfg["normalization"]["zero_std_replacement"]

    epochs = (
        cfg["training"]["sanity_epochs"]
        if args.sanity
        else cfg["training"]["epochs"]
    )

    out_dir = ROOT / (
        "experiments/EXP029_b2core_train_sanity"
        if args.sanity
        else "experiments/EXP029_b2core_train"
    )

    if out_dir.exists():
        raise RuntimeError(
            "Output directory already exists: {}".format(out_dir)
        )

    out_dir.mkdir(parents=True)

    heads = {}
    curve = []

    for head_index, name in enumerate(["drift", "theft"]):
        artifact, rows_curve = train_head(
            name=name,
            rows=finite_rows,
            label_cfg=cfg["labels"][name],
            features=features,
            mean=mean,
            std=std,
            cfg=cfg,
            epochs=epochs,
            head_index=head_index,
        )
        heads[name] = artifact
        curve.extend(rows_curve)

    total_parameters = sum(
        heads[name]["parameter_count"]
        for name in heads
    )

    if total_parameters != cfg["architecture"]["parameter_count_expected_total"]:
        raise RuntimeError("Combined parameter count mismatch")

    model_artifact = {
        "experiment": cfg["experiment"],
        "status": (
            "SANITY_MODEL_NOT_RESULT"
            if args.sanity
            else "TRAIN_ONLY_B2_CORE_FAILURE_TYPED_WEIGHTS"
        ),
        "input_sha256": actual_sha,
        "features": features,
        "normalization": {
            "population": "all finite B2-core TRAIN rows",
            "n_rows": len(finite_rows),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "architecture": cfg["architecture"],
        "training": cfg["training"],
        "epochs_executed": epochs,
        "heads": heads,
        "total_parameters": total_parameters,
        "dev_touched": 0,
        "test_touched": 0,
        "claim_boundary": cfg["claim_boundary"],
    }

    model_path = out_dir / "model.json"
    model_path.write_text(
        json.dumps(model_artifact, indent=2, sort_keys=True) + "\n"
    )

    write_curve(out_dir / "training_curve.csv", curve)

    summary = {
        "status": (
            "SANITY_PASS_NOT_SCIENTIFIC_RESULT"
            if args.sanity
            else "TRAIN_ONLY_B2_CORE_TRAINING_COMPLETE_NOT_FINAL_B2"
        ),
        "input_sha256": actual_sha,
        "videos": len(videos),
        "finite_b2_rows": len(finite_rows),
        "features": features,
        "epochs_executed": epochs,
        "total_parameters": total_parameters,
        "heads": {
            name: {
                key: value
                for key, value in heads[name].items()
                if key != "state_dict"
            }
            for name in heads
        },
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
