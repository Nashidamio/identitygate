from pathlib import Path
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam3.model_builder import build_sam3_video_model


REPO = Path.home() / "thesis" / "identitygate"
FRAMES = Path("/mnt/d/thesis_data/mosev2/train/JPEGImages")
ANN = Path("/mnt/d/thesis_data/mosev2/train/Annotations")

manifest = json.load(
    open(REPO / "experiments/EXP017_split_manifest.json")
)

train = set(manifest["splits"]["TRAIN"]["videos"])
dev = set(manifest["splits"]["DEV"]["videos"])
test = set(manifest["splits"]["TEST"]["videos"])

assert train.isdisjoint(dev)
assert train.isdisjoint(test)

# Match EXP018 selection: shortest TRAIN video with >=2 frame-0 objects.
candidates = []

for vid in sorted(train):
    jpgs = sorted((FRAMES / vid).glob("*.jpg"))
    pngs = sorted((ANN / vid).glob("*.png"))

    if not jpgs or not pngs:
        continue

    a0 = np.array(Image.open(pngs[0]))
    ids = sorted(int(x) for x in np.unique(a0) if int(x) != 0)

    if len(ids) >= 2:
        candidates.append((len(jpgs), vid, ids, pngs[0]))

assert candidates

n_frames, vid, object_ids, ann0_path = min(candidates)
object_ids = object_ids[:4]

assert vid in train
assert vid not in dev
assert vid not in test

print("EXP020 RELATIONAL POINTER FEASIBILITY")
print(
    "identitygate_commit =",
    subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
)
print(
    "sam3_commit =",
    subprocess.check_output(
        [
            "git",
            "-C",
            str(Path.home() / "thesis/externals/sam3"),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip(),
)
print("split = TRAIN")
print("video =", vid)
print("video_frames =", n_frames)
print("object_ids =", object_ids)

a0 = np.array(Image.open(ann0_path))

print()
print("building model...")

m = build_sam3_video_model()
predictor = m.tracker
predictor.backbone = m.detector.backbone

print("predictor_class =", type(predictor).__name__)
print("hidden_dim =", predictor.hidden_dim)
print("no_obj_ptr_shape =", tuple(predictor.no_obj_ptr.shape))

orig_track_step = predictor.track_step

track_count = [0]
prompt_anchors = []
propagation_pointer_calls = [0]

torch.set_printoptions(precision=4, sci_mode=False)


def cosine_matrix(current, anchors):
    with torch.autocast(device_type=current.device.type, enabled=False):
        current = F.normalize(current.float(), dim=-1)
        anchors = F.normalize(anchors.float(), dim=-1)
        out = current @ anchors.T

    assert out.dtype == torch.float32
    print("cosine_compute_dtype =", out.dtype)
    return out


def track_step_probe(*args, **kwargs):
    out = orig_track_step(*args, **kwargs)

    call = track_count[0]
    ptr = out["obj_ptr"].detach()
    frame_idx = kwargs.get("frame_idx", "UNKNOWN")

    print()
    print("=== TRACK_STEP {} ===".format(call))
    print("frame_idx =", frame_idx)
    print("obj_ptr_shape =", tuple(ptr.shape))

    scores = out["object_score_logits"].detach().float().reshape(-1)
    print("object_score_logits =", scores.cpu().tolist())
    print("object_score_probs =", torch.sigmoid(scores).cpu().tolist())

    if call < len(object_ids):
        assert tuple(ptr.shape) == (1, predictor.hidden_dim)
        oid = object_ids[call]

        prompt_anchors.append(
            ptr[0].float().clone()
        )

        print("prompt_anchor_for_obj_id =", oid)

    else:
        assert len(prompt_anchors) == len(object_ids)
        assert ptr.shape[0] == len(object_ids)
        assert ptr.shape[1] == predictor.hidden_dim

        anchors = torch.stack(prompt_anchors).to(ptr.device)
        cos = cosine_matrix(ptr, anchors)

        eye = torch.eye(
            len(object_ids),
            dtype=torch.bool,
            device=cos.device,
        )

        self_cos = cos.diagonal()
        competitor_cos = cos.masked_fill(
            eye, float("-inf")
        ).max(dim=1).values
        margin = self_cos - competitor_cos

        no_obj = predictor.no_obj_ptr.detach().float().to(ptr.device)
        no_obj_l2 = torch.linalg.vector_norm(
            ptr.float() - no_obj,
            dim=1,
        )

        print("packed_obj_ids =", object_ids)
        print("current_to_anchor_cosine =")
        print(cos.cpu())
        print("self_anchor_cos =", self_cos.cpu().tolist())
        print(
            "max_competitor_anchor_cos =",
            competitor_cos.cpu().tolist(),
        )
        print("identity_margin =", margin.cpu().tolist())
        print("distance_to_no_obj_ptr =", no_obj_l2.cpu().tolist())

        propagation_pointer_calls[0] += 1

    track_count[0] += 1
    return out


predictor.track_step = track_step_probe

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

assert len(prompt_anchors) == len(object_ids)

anchors = torch.stack(prompt_anchors)

print()
print("=== FRAME-0 ANCHOR MATRIX ===")
print("anchor_shape =", tuple(anchors.shape))
print(cosine_matrix(anchors, anchors).cpu())

print()
print("=== PROPAGATION + STATE MAPPING ===")

yield_count = 0

for out in predictor.propagate_in_video(
    state,
    start_frame_idx=0,
    max_frame_num_to_track=3,
    reverse=False,
    propagate_preflight=True,
):
    frame_idx = out[0]
    yielded_ids = list(out[1])

    print()
    print("--- YIELD {} ---".format(yield_count))
    print("frame_idx =", frame_idx)
    print("yielded_obj_ids =", yielded_ids)
    print("state_obj_ids =", list(state["obj_ids"]))

    assert yielded_ids == object_ids
    assert list(state["obj_ids"]) == object_ids

    if frame_idx in state["output_dict"]["cond_frame_outputs"]:
        storage_key = "cond_frame_outputs"
    else:
        assert frame_idx in state["output_dict"]["non_cond_frame_outputs"]
        storage_key = "non_cond_frame_outputs"

    packed = state["output_dict"][storage_key][frame_idx]["obj_ptr"]

    print("storage_key =", storage_key)
    print("packed_state_obj_ptr_shape =", tuple(packed.shape))

    assert tuple(packed.shape) == (
        len(object_ids),
        predictor.hidden_dim,
    )

    max_diff = 0.0

    for obj_idx, oid in enumerate(object_ids):
        per_obj = (
            state["output_dict_per_obj"][obj_idx]
            [storage_key][frame_idx]["obj_ptr"]
        )

        diff = (
            packed[obj_idx : obj_idx + 1].float()
            - per_obj.float()
        ).abs().max().item()

        max_diff = max(max_diff, diff)

        print(
            "mapping obj_idx={} obj_id={} "
            "per_obj_shape={} max_abs_diff={:.9f}".format(
                obj_idx,
                oid,
                tuple(per_obj.shape),
                diff,
            )
        )

        assert torch.equal(
            packed[obj_idx : obj_idx + 1],
            per_obj,
        )

    print(
        "packed_vs_per_object_max_abs_diff =",
        max_diff,
    )

    if frame_idx == 0:
        frame0_diff = (
            packed.float()
            - anchors.to(packed.device)
        ).abs().max().item()

        print(
            "frame0_state_vs_prompt_anchor_max_abs_diff =",
            frame0_diff,
        )

    yield_count += 1

print()
print("=== FINAL ===")
print("track_step_calls =", track_count[0])
print(
    "propagation_pointer_calls =",
    propagation_pointer_calls[0],
)
print("yield_count =", yield_count)

assert propagation_pointer_calls[0] >= 1
assert yield_count >= 1

print("EXP020_RELATIONAL_POINTER_FEASIBILITY_PASS")
