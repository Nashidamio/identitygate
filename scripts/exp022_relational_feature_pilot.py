from pathlib import Path
import csv
import hashlib
import json
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam3.model_builder import build_sam3_video_model


REPO = Path.home() / "thesis" / "identitygate"
SAM_REPO = Path.home() / "thesis" / "externals" / "sam3"

FRAMES = Path("/mnt/d/thesis_data/mosev2/train/JPEGImages")
ANN = Path("/mnt/d/thesis_data/mosev2/train/Annotations")

SCOPE_PATH = REPO / "experiments/EXP021_relational_scope.json"
SUBSTRATE_PATH = REPO / "configs/SUBSTRATE-v1.json"

OUTDIR = REPO / "experiments/EXP022_pilot"
FEATURES_PATH = OUTDIR / "features.csv"
ANCHOR_PATH = OUTDIR / "anchor_cosine.csv"
SUMMARY_PATH = OUTDIR / "summary.json"

EXPECTED_SCOPE_SHA256 = (
    "e528ab0b422f57fee371425199f56193c2b54ca103836ae6cf3f364cd8f5e2b2"
)

EXPECTED_SUBSTRATE_SHA256 = (
    "cf48aff216a544409823421993f54b42c0f2b0e9bfd8c6772b21901ba32ddb31"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def cosine_matrix_fp32(current, anchors):
    device_type = current.device.type

    prev_tf32 = None
    prev_precision = None

    if device_type == "cuda":
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_precision = torch.get_float32_matmul_precision()

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    try:
        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            current = F.normalize(current.float(), dim=-1)
            anchors = F.normalize(anchors.float(), dim=-1)
            out = current @ anchors.T
    finally:
        if device_type == "cuda":
            torch.set_float32_matmul_precision(prev_precision)
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    return out


start_time = time.time()

scope_sha = sha256_file(SCOPE_PATH)
substrate_sha = sha256_file(SUBSTRATE_PATH)

assert scope_sha == EXPECTED_SCOPE_SHA256, (
    scope_sha,
    EXPECTED_SCOPE_SHA256,
)

assert substrate_sha == EXPECTED_SUBSTRATE_SHA256, (
    substrate_sha,
    EXPECTED_SUBSTRATE_SHA256,
)

scope = json.load(open(SCOPE_PATH))
substrate = json.load(open(SUBSTRATE_PATH))

assert scope["test_videos_touched"] == 0
assert scope["protocol"] == "DEVELOPMENT_EXPOSED_ONLY_NOT_FINAL_SPLIT"

assert substrate["core"]["model_family"] == "SAM 3"
assert substrate["core"]["task_mode"] == "VOS/PVS"
assert substrate["core"]["checkpoint_filename"] == "sam3.pt"

eligible = scope["eligible_videos"]
assert len(eligible) == 26

# Deterministic stress selection:
# highest frame-0 object count, then longest video, then video ID.
pilot = max(
    eligible,
    key=lambda r: (
        int(r["n_frame0_objects"]),
        int(r["n_frames"]),
        str(r["video"]),
    ),
)

split = pilot["split"]
vid = pilot["video"]
object_ids = [int(x) for x in pilot["frame0_object_ids"]]
n_frames = int(pilot["n_frames"])

assert split in {"TRAIN", "DEV"}
assert len(object_ids) == int(pilot["n_frame0_objects"])
assert len(object_ids) >= 2

jpgs = sorted((FRAMES / vid).glob("*.jpg"))
pngs = sorted((ANN / vid).glob("*.png"))

assert len(jpgs) == n_frames
assert pngs

a0 = np.array(Image.open(pngs[0]))

actual_ids = sorted(
    int(x)
    for x in np.unique(a0)
    if int(x) != 0
)

assert actual_ids == object_ids, (
    actual_ids,
    object_ids,
)

print("EXP022 RELATIONAL FEATURE PILOT")
print("identitygate_commit =", git_commit(REPO))
print("sam3_commit =", git_commit(SAM_REPO))
print("scope_sha256 =", scope_sha)
print("substrate_sha256 =", substrate_sha)
print("split =", split)
print("video =", vid)
print("video_frames =", n_frames)
print("n_objects =", len(object_ids))
print("object_ids =", object_ids)
print("selection_rule = max frame0 objects, then max frames, then video ID")
print("test_videos_touched = 0")
print()

print("building SAM 3 VOS model...")

model = build_sam3_video_model()

predictor = model.tracker
predictor.backbone = model.detector.backbone

print("predictor_class =", type(predictor).__name__)
print("hidden_dim =", predictor.hidden_dim)
print("checkpoint_identity = facebook/sam3/sam3.pt")
print()

assert torch.cuda.is_available()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

memory_after_model_gb = (
    torch.cuda.memory_allocated() / (1024 ** 3)
)

state = predictor.init_state(
    video_path=str(FRAMES / vid)
)

predictor.clear_all_points_in_video(state)

for oid in object_ids:
    mask = torch.from_numpy(a0 == oid)
    assert mask.any()

    predictor.add_new_mask(
        inference_state=state,
        frame_idx=0,
        obj_id=oid,
        mask=mask,
    )

assert list(state["obj_ids"]) == object_ids

# Freeze the frame-0 packed conditioning outputs.
predictor.propagate_in_video_preflight(
    state,
    run_mem_encoder=True,
)

assert 0 in state["output_dict"]["cond_frame_outputs"]

frame0_out = state["output_dict"]["cond_frame_outputs"][0]

anchors = frame0_out["obj_ptr"].detach()

assert tuple(anchors.shape) == (
    len(object_ids),
    predictor.hidden_dim,
)

anchor_cos = cosine_matrix_fp32(
    anchors,
    anchors,
)

assert torch.allclose(
    torch.diagonal(anchor_cos),
    torch.ones(
        len(object_ids),
        device=anchor_cos.device,
        dtype=torch.float32,
    ),
    atol=1e-5,
    rtol=1e-5,
)

offdiag_mask = ~torch.eye(
    len(object_ids),
    dtype=torch.bool,
    device=anchor_cos.device,
)

offdiag = anchor_cos[offdiag_mask]

print("anchor_shape =", tuple(anchors.shape))
print(
    "anchor_offdiag_cos_range =",
    float(offdiag.min()),
    float(offdiag.max()),
)
print(
    "anchor_offdiag_cos_mean =",
    float(offdiag.mean()),
)
print()

feature_rows = []
anchor_rows = []

anchor_cos_cpu = anchor_cos.detach().cpu().numpy()

for i, oid_i in enumerate(object_ids):
    for j, oid_j in enumerate(object_ids):
        anchor_rows.append({
            "video": vid,
            "split": split,
            "object_id_i": oid_i,
            "object_id_j": oid_j,
            "cosine_fp32": float(anchor_cos_cpu[i, j]),
        })

print("=== PROPAGATION ===")

yield_count = 0
noncond_frames_recorded = 0

for out in predictor.propagate_in_video(
    state,
    start_frame_idx=0,
    max_frame_num_to_track=3,
    reverse=False,
    propagate_preflight=False,
):
    frame_idx = int(out[0])
    yielded_ids = [int(x) for x in out[1]]

    assert yielded_ids == object_ids
    assert list(state["obj_ids"]) == object_ids

    print(
        "yield",
        yield_count,
        "frame_idx=",
        frame_idx,
    )

    yield_count += 1

    # Frame 0 is the clean anchor frame.
    if frame_idx == 0:
        continue

    assert frame_idx in state["output_dict"]["non_cond_frame_outputs"]

    current_out = (
        state["output_dict"]["non_cond_frame_outputs"][frame_idx]
    )

    ptr = current_out["obj_ptr"].detach()
    logits = (
        current_out["object_score_logits"]
        .detach()
        .float()
        .reshape(-1)
    )

    assert tuple(ptr.shape) == (
        len(object_ids),
        predictor.hidden_dim,
    )

    assert tuple(logits.shape) == (
        len(object_ids),
    )

    cos = cosine_matrix_fp32(
        ptr,
        anchors,
    )

    eye = torch.eye(
        len(object_ids),
        dtype=torch.bool,
        device=cos.device,
    )

    competitor_values, competitor_indices = (
        cos.masked_fill(
            eye,
            float("-inf"),
        )
        .max(dim=1)
    )

    self_values = torch.diagonal(cos)
    margins = self_values - competitor_values

    probs = torch.sigmoid(logits)

    no_obj = (
        predictor.no_obj_ptr
        .detach()
        .float()
        .to(ptr.device)
    )

    no_obj_l2 = torch.linalg.vector_norm(
        ptr.float() - no_obj,
        dim=1,
    )

    ptr_norm = torch.linalg.vector_norm(
        ptr.float(),
        dim=1,
    )

    tensors_to_check = [
        self_values,
        competitor_values,
        margins,
        logits,
        probs,
        no_obj_l2,
        ptr_norm,
    ]

    assert all(
        torch.isfinite(x).all()
        for x in tensors_to_check
    )

    for obj_idx, oid in enumerate(object_ids):
        competitor_idx = int(
            competitor_indices[obj_idx].item()
        )

        competitor_oid = object_ids[competitor_idx]

        feature_rows.append({
            "split": split,
            "video": vid,
            "frame_idx": frame_idx,
            "obj_idx": obj_idx,
            "object_id": oid,
            "n_objects": len(object_ids),
            "self_anchor_cos_fp32": float(
                self_values[obj_idx].item()
            ),
            "max_tracked_competitor_cos_fp32": float(
                competitor_values[obj_idx].item()
            ),
            "competitor_object_id": competitor_oid,
            "identity_margin_fp32": float(
                margins[obj_idx].item()
            ),
            "object_score_logit": float(
                logits[obj_idx].item()
            ),
            "object_score_prob": float(
                probs[obj_idx].item()
            ),
            "distance_to_no_obj_ptr": float(
                no_obj_l2[obj_idx].item()
            ),
            "obj_ptr_l2_norm": float(
                ptr_norm[obj_idx].item()
            ),
        })

    print(
        "  self_range =",
        (
            float(self_values.min()),
            float(self_values.max()),
        ),
    )
    print(
        "  competitor_range =",
        (
            float(competitor_values.min()),
            float(competitor_values.max()),
        ),
    )
    print(
        "  margin_range =",
        (
            float(margins.min()),
            float(margins.max()),
        ),
    )

    noncond_frames_recorded += 1

assert noncond_frames_recorded == 3
assert len(feature_rows) == len(object_ids) * 3

torch.cuda.synchronize()

peak_allocated_gb = (
    torch.cuda.max_memory_allocated() / (1024 ** 3)
)

peak_reserved_gb = (
    torch.cuda.max_memory_reserved() / (1024 ** 3)
)

total_gpu_gb = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

margins_np = np.array(
    [r["identity_margin_fp32"] for r in feature_rows],
    dtype=np.float64,
)

self_np = np.array(
    [r["self_anchor_cos_fp32"] for r in feature_rows],
    dtype=np.float64,
)

competitor_np = np.array(
    [
        r["max_tracked_competitor_cos_fp32"]
        for r in feature_rows
    ],
    dtype=np.float64,
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

feature_fields = list(feature_rows[0].keys())

with open(
    FEATURES_PATH,
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=feature_fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(feature_rows)

anchor_fields = list(anchor_rows[0].keys())

with open(
    ANCHOR_PATH,
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=anchor_fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(anchor_rows)

summary = {
    "experiment": "EXP022-PILOT",
    "status": "COMPLETED_PILOT_NOT_SCIENTIFIC_RESULT",
    "purpose": (
        "Stress-test production relational feature extraction "
        "before the frozen 26-video development census"
    ),
    "identitygate_commit_at_execution": git_commit(REPO),
    "sam3_commit": git_commit(SAM_REPO),
    "script_sha256": sha256_file(Path(__file__)),
    "scope_manifest": str(SCOPE_PATH.relative_to(REPO)),
    "scope_sha256": scope_sha,
    "substrate_config": str(SUBSTRATE_PATH.relative_to(REPO)),
    "substrate_sha256": substrate_sha,
    "model_family": "SAM 3",
    "task_mode": "VOS/PVS",
    "checkpoint_identity": "facebook/sam3/sam3.pt",
    "split": split,
    "video": vid,
    "video_frames": n_frames,
    "n_objects": len(object_ids),
    "object_ids": object_ids,
    "selection_rule": (
        "max frame0 object count, then max frames, then video ID "
        "within frozen EXP021 development-exposed scope"
    ),
    "test_videos_touched": 0,
    "propagated_nonconditioning_frames": noncond_frames_recorded,
    "n_feature_rows": len(feature_rows),
    "cosine_precision": "FP32_AUTOCAST_DISABLED_TF32_DISABLED_HIGHEST_MATMUL",
    "anchor_offdiag_cos_min": float(offdiag.min()),
    "anchor_offdiag_cos_max": float(offdiag.max()),
    "anchor_offdiag_cos_mean": float(offdiag.mean()),
    "self_anchor_cos_min": float(self_np.min()),
    "self_anchor_cos_max": float(self_np.max()),
    "competitor_cos_min": float(competitor_np.min()),
    "competitor_cos_max": float(competitor_np.max()),
    "identity_margin_min": float(margins_np.min()),
    "identity_margin_max": float(margins_np.max()),
    "identity_margin_mean": float(margins_np.mean()),
    "negative_margin_rows": int(
        np.sum(margins_np < 0)
    ),
    "negative_margin_fraction": float(
        np.mean(margins_np < 0)
    ),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_memory_gb": total_gpu_gb,
    "memory_allocated_after_model_gb": memory_after_model_gb,
    "peak_memory_allocated_gb": peak_allocated_gb,
    "peak_memory_reserved_gb": peak_reserved_gb,
    "runtime_sec": time.time() - start_time,
    "seed": "N/A_MODEL_INFERENCE",
    "scientific_claim": "NONE_FROM_PILOT",
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)

print()
print("=== EXP022 PILOT SUMMARY ===")

for key, value in summary.items():
    print(key, "=", value)

print()
print("features ->", FEATURES_PATH)
print("anchors ->", ANCHOR_PATH)
print("summary ->", SUMMARY_PATH)

print("EXP022_PILOT_PASS")
