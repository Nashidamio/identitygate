import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP051-fresh-dev-headline-outcomes-v1.json"
OUT = ROOT / "experiments/EXP051_fresh_dev_headline_outcomes"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load module: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv_atomic(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def require_frozen_clean_tree():
    required = [
        "configs/EXP051-fresh-dev-headline-outcomes-v1.json",
        "scripts/exp051_fresh_dev_headline_outcomes.py",
        "tests/test_exp051_fresh_dev_headline_outcomes.py",
    ]
    for rel in required:
        subprocess.check_call(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError("EXP051 requires a clean committed tree: " + status)


def verify_dependencies(cfg):
    for name, item in cfg["source_dependencies"].items():
        actual = sha256sum(ROOT / item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(
                "{} SHA mismatch: {} != {}".format(
                    name, actual, item["sha256"]
                )
            )


def load_cfg():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    verify_dependencies(cfg)
    return cfg


def load_runtime(cfg):
    exp050 = load_module(
        "exp050_for_exp051",
        ROOT / cfg["source_dependencies"]["exp050_script"]["path"],
    )
    exp050_cfg = json.loads(
        (
            ROOT
            / cfg["source_dependencies"]["exp050_config"]["path"]
        ).read_text(encoding="utf-8")
    )
    runtime = exp050.load_runtime(exp050_cfg)
    return exp050, runtime


def load_dev_and_primary(cfg, exp050):
    dev_rows = exp050.load_dev_rows(cfg)
    videos = [r["video"] for r in dev_rows]

    if len(videos) != int(cfg["scope"]["expected_videos"]):
        raise RuntimeError("Fresh DEV video count mismatch")

    event_rows = read_csv(
        ROOT / cfg["scope"]["event_pool_csv"]
    )
    primary = [
        r for r in event_rows
        if r["video"] in set(videos)
        and int(r["primary_pool_eligible"]) == 1
    ]

    if len(primary) != int(cfg["scope"]["expected_primary_events"]):
        raise RuntimeError(
            "Fresh DEV primary-event count mismatch: {}".format(len(primary))
        )

    by_video_expected = {
        r["video"]: int(r["primary_event_count"])
        for r in dev_rows
    }
    by_video_actual = defaultdict(int)
    keys = set()
    for r in primary:
        key = (
            r["video"],
            int(r["object_id"]),
            int(r["reappear_frame"]),
        )
        if key in keys:
            raise RuntimeError("Duplicate primary event key: {}".format(key))
        keys.add(key)
        by_video_actual[r["video"]] += 1

    if dict(by_video_actual) != by_video_expected:
        raise RuntimeError(
            "Per-video primary event counts do not match EXP049 fresh DEV"
        )

    primary_by_video = defaultdict(list)
    for r in primary:
        primary_by_video[r["video"]].append(r)

    for video in primary_by_video:
        primary_by_video[video].sort(
            key=lambda r: (
                int(r["object_id"]),
                int(r["reappear_frame"]),
            )
        )

    return dev_rows, videos, primary_by_video, keys


def neutral_admit_frames(n_eligible, k_admit):
    n = int(n_eligible)
    k = int(k_admit)
    if not 0 <= k <= n:
        raise ValueError("Invalid neutral budget")
    if k == 0:
        return []
    q = [
        int(math.floor((j + 0.5) * n / k))
        for j in range(k)
    ]
    if len(set(q)) != k:
        raise RuntimeError("Neutral schedule contains duplicate opportunities")
    if min(q) < 0 or max(q) >= n:
        raise RuntimeError("Neutral schedule outside eligible range")
    return [x + 1 for x in q]


def iou_bool(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(inter / union)


def event_interval(visibility, reappear_frame, min_gap, window):
    vis = list(bool(x) for x in visibility)
    reappear = []
    gap_starts = []
    run = 0
    seen = False

    for frame_idx, visible in enumerate(vis):
        if visible:
            if run >= min_gap and seen:
                reappear.append(frame_idx)
            run = 0
            seen = True
        elif seen:
            run += 1
            if run == min_gap:
                gap_starts.append(frame_idx - min_gap + 1)

    if int(reappear_frame) not in reappear:
        raise RuntimeError(
            "Frozen event reappearance not reproduced: {}".format(
                reappear_frame
            )
        )

    stop = next(
        (g for g in gap_starts if g > int(reappear_frame)),
        len(vis),
    )

    frames = []
    evaluable = 0
    for frame_idx in range(int(reappear_frame), stop):
        frames.append(frame_idx)
        if vis[frame_idx]:
            evaluable += 1
            if evaluable >= int(window):
                break

    return frames, evaluable, stop


def theft_episode_from_flags(frame_indices, flags, min_len):
    best = 0
    streak = 0
    first_start = None
    prev = None

    for frame_idx, flag in zip(frame_indices, flags):
        if flag:
            if prev is not None and frame_idx == prev + 1:
                streak += 1
            else:
                streak = 1
            if streak >= min_len and first_start is None:
                first_start = frame_idx - min_len + 1
            best = max(best, streak)
        else:
            streak = 0
        prev = frame_idx

    return int(best >= min_len), first_start, best


def score_primary_events(
    run_label,
    tau,
    predictions,
    gt_cache,
    visibility,
    object_ids,
    primary_rows,
    endpoint,
):
    out = []
    object_ids = [int(x) for x in object_ids]

    for frozen in primary_rows:
        oid = int(frozen["object_id"])
        reappear_frame = int(frozen["reappear_frame"])

        if oid not in object_ids:
            raise RuntimeError(
                "Frozen event object {} absent from tracker IDs".format(oid)
            )

        frames, evaluable, _ = event_interval(
            visibility[oid],
            reappear_frame,
            int(endpoint["qualifying_gap_frames"]),
            int(endpoint["por_window"]),
        )

        recovered = 0
        first_recovery = ""
        theft_flags = []

        for frame_idx in frames:
            if frame_idx not in predictions:
                raise RuntimeError(
                    "Missing prediction frame {}".format(frame_idx)
                )

            pred = predictions[frame_idx][oid]
            target = gt_cache[frame_idx] == oid
            target_iou = iou_bool(pred, target)

            if (
                visibility[oid][frame_idx]
                and not recovered
                and target_iou
                > float(endpoint["recovery_iou_strictly_greater_than"])
            ):
                recovered = 1
                first_recovery = frame_idx

            max_other = 0.0
            for other in object_ids:
                if other == oid:
                    continue
                other_gt = gt_cache[frame_idx] == other
                max_other = max(
                    max_other,
                    iou_bool(pred, other_gt),
                )

            theft_flags.append(
                target_iou
                < float(endpoint["theft_target_iou_strictly_less_than"])
                and max_other
                > float(endpoint["theft_other_iou_strictly_greater_than"])
            )

        theft_event, first_theft, max_streak = theft_episode_from_flags(
            frames,
            theft_flags,
            int(endpoint["theft_consecutive_frames"]),
        )

        out.append(
            {
                "run_label": run_label,
                "tau": "" if tau is None else float(tau),
                "video": frozen["video"],
                "object_id": oid,
                "reappear_frame": reappear_frame,
                "disappear_start": int(frozen["disappear_start"]),
                "recovered_w30": recovered,
                "first_recovery_frame": first_recovery,
                "evaluable_visible_frames": evaluable,
                "theft_event_w30": theft_event,
                "first_theft_episode_start": (
                    "" if first_theft is None else first_theft
                ),
                "max_consecutive_theft_frames": max_streak,
            }
        )

    return out


def context_gt(runtime, exp050, cfg, video):
    data_root = Path(cfg["scope"]["dataset_root"])
    ctx = exp050.video_context(
        runtime["exp037"],
        data_root,
        video,
    )

    ann_dir = data_root / "Annotations" / video
    pngs = runtime["exp037"].sorted_frames(
        ann_dir,
        ".png",
    )
    if len(pngs) != int(ctx["n_frames"]):
        raise RuntimeError("{} annotation frame-count mismatch".format(video))

    gt_cache, visibility = runtime["exp037"].build_gt_cache(
        ann_dir,
        pngs,
        ctx["object_ids"],
        int(ctx["n_frames"]),
    )
    return ctx, gt_cache, visibility


def make_run_cfg(runtime, exp050, cfg, ctx, tau):
    scope_cfg = exp050.make_scope_cfg(
        runtime["exp037_cfg"],
        Path(cfg["scope"]["dataset_root"]),
        ctx,
    )
    return runtime["exp037"].configure_scope(
        runtime["base_cfg"],
        scope_cfg,
        float(tau),
    )


def run_neutral(
    predictor,
    jpg_dir,
    gt0,
    object_ids,
    run_cfg,
    admit_frames,
    runtime,
):
    exp033 = runtime["exp033"]
    deps = runtime["deps"]

    state, _ = exp033.prepare_state(
        predictor,
        jpg_dir,
        gt0,
        object_ids,
        deps,
    )

    predictions = {
        0: {
            oid: (gt0 == oid).astype(bool)
            for oid in object_ids
        }
    }
    decisions = []
    admit_set = set(int(x) for x in admit_frames)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    generator = predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=run_cfg["scope"]["n_frames"] - 1,
        reverse=False,
        propagate_preflight=False,
    )

    for out in generator:
        frame_idx = int(out[0])
        ids = [int(x) for x in out[1]]

        if ids != object_ids:
            raise RuntimeError(
                "Object order changed at frame {}".format(frame_idx)
            )

        video_res = out[3]
        predictions[frame_idx] = {
            oid: (
                (video_res[row, 0] > 0)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            for row, oid in enumerate(ids)
        }

        if frame_idx == 0:
            continue

        action = "ADMIT" if frame_idx in admit_set else "BLOCK"
        row = {
            "frame_idx": frame_idx,
            "action": action,
        }
        if action == "BLOCK":
            meta = deps["exp032"].evict_current_frame(
                state,
                frame_idx,
                len(object_ids),
            )
            row.update(meta)
        decisions.append(row)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    runtime_sec = time.time() - start

    del state
    torch.cuda.empty_cache()

    if len(decisions) != run_cfg["scope"]["n_frames"] - 1:
        raise RuntimeError("Neutral decision-count mismatch")

    actual_admits = [
        int(r["frame_idx"])
        for r in decisions
        if r["action"] == "ADMIT"
    ]
    if actual_admits != sorted(admit_set):
        raise RuntimeError("Neutral exact-K schedule mismatch")

    return {
        "predictions": predictions,
        "decisions": decisions,
        "peak_vram_gb": peak,
        "runtime_sec": runtime_sec,
    }


def cache_path(cfg, scope_label, run_label, video):
    return (
        Path(cfg["runtime"]["scratch_dir"])
        / scope_label
        / run_label
        / (video + ".json")
    )


def validate_cached(row, expected):
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(
                "Stale EXP051 cache field {}: {} != {}".format(
                    key, row.get(key), value
                )
            )
    return row


def run_or_load(
    cfg,
    exp050,
    runtime,
    predictor,
    scope_label,
    video,
    primary_rows,
    run_label,
    variant,
    tau,
    neutral_k=None,
):
    config_sha = sha256sum(CFG_PATH)
    expected = {
        "experiment": "EXP051",
        "repo_commit": git_head(),
        "config_sha256": config_sha,
        "scope_label": scope_label,
        "run_label": run_label,
        "video": video,
        "neutral_source_k": neutral_k,
    }

    path = cache_path(
        cfg,
        scope_label,
        run_label,
        video,
    )

    if path.exists():
        row = json.loads(path.read_text(encoding="utf-8"))
        return validate_cached(row, expected), True

    ctx, gt_cache, visibility = context_gt(
        runtime,
        exp050,
        cfg,
        video,
    )

    effective_tau = (
        float(tau)
        if tau is not None
        else float(cfg["headline"]["taus"]["B2"])
    )
    run_cfg = make_run_cfg(
        runtime,
        exp050,
        cfg,
        ctx,
        effective_tau,
    )

    if neutral_k is None:
        tracked = runtime["exp033"].run_tracker(
            predictor=predictor,
            jpg_dir=ctx["jpg_dir"],
            gt0=ctx["gt0"],
            object_ids=ctx["object_ids"],
            cfg=run_cfg,
            variant=variant,
            packs=runtime["packs"],
            deps=runtime["deps"],
        )
        if variant == "B0":
            eligible = int(ctx["n_frames"]) - 1
            admit_count = eligible
            block_count = 0
        else:
            eligible = int(ctx["n_frames"]) - 1
            decisions = tracked["decisions"]
            if len(decisions) != eligible:
                raise RuntimeError("Signal decision-count mismatch")
            admit_count = sum(
                r["action"] == "ADMIT"
                for r in decisions
            )
            block_count = sum(
                r["action"] == "BLOCK"
                for r in decisions
            )
            if admit_count + block_count != eligible:
                raise RuntimeError("Signal action-count mismatch")
            if not runtime["exp037"].block_integrity_pass(
                decisions,
                len(ctx["object_ids"]),
            ):
                raise RuntimeError("Signal block-integrity failure")
    else:
        eligible = int(ctx["n_frames"]) - 1
        admit_frames = neutral_admit_frames(
            eligible,
            int(neutral_k),
        )
        tracked = run_neutral(
            predictor=predictor,
            jpg_dir=ctx["jpg_dir"],
            gt0=ctx["gt0"],
            object_ids=ctx["object_ids"],
            run_cfg=run_cfg,
            admit_frames=admit_frames,
            runtime=runtime,
        )
        admit_count = int(neutral_k)
        block_count = eligible - admit_count

    event_rows = score_primary_events(
        run_label=run_label,
        tau=tau,
        predictions=tracked["predictions"],
        gt_cache=gt_cache,
        visibility=visibility,
        object_ids=ctx["object_ids"],
        primary_rows=primary_rows,
        endpoint=cfg["endpoint"],
    )

    if len(event_rows) != len(primary_rows):
        raise RuntimeError("Primary event denominator mismatch")

    result = {
        **expected,
        "variant": variant,
        "tau": tau,
        "n_frames": int(ctx["n_frames"]),
        "eligible_nonconditioning_frames": eligible,
        "admit_count": int(admit_count),
        "block_count": int(block_count),
        "write_rate": float(admit_count / eligible),
        "event_count": len(event_rows),
        "por30": float(
            sum(r["recovered_w30"] for r in event_rows)
            / len(event_rows)
        ),
        "itr30": float(
            sum(r["theft_event_w30"] for r in event_rows)
            / len(event_rows)
        ),
        "peak_vram_gb": float(tracked["peak_vram_gb"]),
        "runtime_sec": float(tracked["runtime_sec"]),
        "event_rows": event_rows,
        "test_touched": False,
    }

    write_json_atomic(path, result)

    del tracked
    gc.collect()
    torch.cuda.empty_cache()

    return result, False


def aggregate(run_rows, cfg):
    labels = ["B0"]
    for variant in cfg["variants"]:
        labels.extend(
            [variant, variant + "_NEUTRAL"]
        )

    by_label = {}
    for label in labels:
        rows = [
            r for r in run_rows
            if r["run_label"] == label
        ]
        events = [
            e
            for r in rows
            for e in r["event_rows"]
        ]
        expected_events = int(
            cfg["scope"]["expected_primary_events"]
        )
        if len(events) != expected_events:
            raise RuntimeError(
                "{} event count {} != {}".format(
                    label, len(events), expected_events
                )
            )
        sum_k = sum(r["admit_count"] for r in rows)
        sum_n = sum(
            r["eligible_nonconditioning_frames"]
            for r in rows
        )
        by_label[label] = {
            "run_label": label,
            "videos": len(rows),
            "events": len(events),
            "por30": float(
                sum(e["recovered_w30"] for e in events)
                / len(events)
            ),
            "itr30": float(
                sum(e["theft_event_w30"] for e in events)
                / len(events)
            ),
            "sum_admit_count": int(sum_k),
            "sum_eligible_count": int(sum_n),
            "pooled_write_rate": float(sum_k / sum_n),
            "max_peak_vram_gb": float(
                max(r["peak_vram_gb"] for r in rows)
            ),
            "runtime_sec": float(
                sum(r["runtime_sec"] for r in rows)
            ),
        }

    for variant in cfg["variants"]:
        signal = by_label[variant]
        neutral = by_label[variant + "_NEUTRAL"]
        if (
            signal["sum_admit_count"]
            != neutral["sum_admit_count"]
            or signal["sum_eligible_count"]
            != neutral["sum_eligible_count"]
        ):
            raise RuntimeError(
                "{} neutral pooled budget mismatch".format(variant)
            )

        expected_rate = float(
            cfg["headline"]["expected_fresh_dev_rates"][variant]
        )
        if abs(
            signal["pooled_write_rate"] - expected_rate
        ) > float(cfg["reproducibility"]["write_rate_abs_tolerance"]):
            raise RuntimeError(
                "{} rerun write rate {} != frozen {}".format(
                    variant,
                    signal["pooled_write_rate"],
                    expected_rate,
                )
            )

    return by_label


def selftest():
    cfg = load_cfg()
    exp050, runtime = load_runtime(cfg)
    _, videos, primary_by_video, keys = load_dev_and_primary(
        cfg,
        exp050,
    )

    op = json.loads(
        (
            ROOT
            / cfg["source_dependencies"]["final_operating_points"]["path"]
        ).read_text(encoding="utf-8")
    )
    if float(op["r_star"]) != float(cfg["headline"]["r_star"]):
        raise RuntimeError("r_star contract mismatch")

    for variant in cfg["variants"]:
        observed = op["headline"][variant]
        if observed["status"] != "MATCH":
            raise RuntimeError("{} headline is not MATCH".format(variant))
        if float(observed["selected_tau"]) != float(
            cfg["headline"]["taus"][variant]
        ):
            raise RuntimeError("{} tau mismatch".format(variant))

    if len(videos) != 40 or len(keys) != 233:
        raise RuntimeError("Frozen DEV contract mismatch")
    if sum(len(x) for x in primary_by_video.values()) != 233:
        raise RuntimeError("Primary event partition mismatch")

    assert neutral_admit_frames(10, 0) == []
    assert neutral_admit_frames(10, 10) == list(range(1, 11))
    assert neutral_admit_frames(10, 3) == [2, 6, 9]

    event, first, best = theft_episode_from_flags(
        [7, 8, 9, 10, 11, 12],
        [False, True, True, True, True, True],
        5,
    )
    assert event == 1 and first == 8 and best == 5

    event, first, best = theft_episode_from_flags(
        [7, 8, 10, 11, 12, 13, 14],
        [True, True, True, True, True, True, True],
        5,
    )
    assert event == 1 and first == 10 and best == 5

    print("EXP051_SELFTEST=PASS")
    print("fresh_dev_videos=40")
    print("primary_event_keys=233")
    print("test_touched=false")


def plan():
    cfg = load_cfg()
    exp050, _ = load_runtime(cfg)
    _, videos, _, keys = load_dev_and_primary(cfg, exp050)
    trajectory_count = len(videos) * (
        1 + 2 * len(cfg["variants"])
    )
    print("EXP051_PLAN=PASS")
    print("videos={}".format(len(videos)))
    print("primary_events={}".format(len(keys)))
    print("headline_trajectories={}".format(trajectory_count))
    print("r_star={}".format(cfg["headline"]["r_star"]))
    print("test_touched=false")


def sanity():
    require_frozen_clean_tree()
    cfg = load_cfg()
    exp050, runtime = load_runtime(cfg)

    video = cfg["scope"]["train_exposed_sanity_video"]
    event_pool = read_csv(
        ROOT / cfg["scope"]["event_pool_csv"]
    )
    rows = [
        r for r in event_pool
        if r["video"] == video
    ]
    if not rows:
        raise RuntimeError(
            "Sanity video has no frozen event-pool event"
        )
    rows = [rows[0]]

    predictor = exp050.build_predictor(runtime)

    variant = "B2"
    tau = float(cfg["headline"]["taus"][variant])

    signal, signal_hit = run_or_load(
        cfg,
        exp050,
        runtime,
        predictor,
        "TRAIN_EXPOSED_SANITY",
        video,
        rows,
        variant,
        variant,
        tau,
    )
    neutral, neutral_hit = run_or_load(
        cfg,
        exp050,
        runtime,
        predictor,
        "TRAIN_EXPOSED_SANITY",
        video,
        rows,
        variant + "_NEUTRAL",
        "B0",
        None,
        neutral_k=signal["admit_count"],
    )

    if signal["admit_count"] != neutral["admit_count"]:
        raise RuntimeError("Sanity neutral exact-K mismatch")

    print("EXP051_SANITY=PASS")
    print("video={}".format(video))
    print("signal_K={}".format(signal["admit_count"]))
    print("neutral_K={}".format(neutral["admit_count"]))
    print("signal_cache_hit={}".format(int(signal_hit)))
    print("neutral_cache_hit={}".format(int(neutral_hit)))
    print(
        "peak_vram_gb={}".format(
            max(
                signal["peak_vram_gb"],
                neutral["peak_vram_gb"],
            )
        )
    )
    print("test_touched=false")


def run():
    require_frozen_clean_tree()
    cfg = load_cfg()
    exp050, runtime = load_runtime(cfg)
    _, videos, primary_by_video, keys = load_dev_and_primary(
        cfg,
        exp050,
    )

    predictor = exp050.build_predictor(runtime)
    run_rows = []
    cache_hits = 0
    cache_misses = 0

    for idx, video in enumerate(videos, start=1):
        primary_rows = primary_by_video[video]
        print(
            "EXP051 video {}/{} {} events={}".format(
                idx, len(videos), video, len(primary_rows)
            ),
            flush=True,
        )

        b0, hit = run_or_load(
            cfg,
            exp050,
            runtime,
            predictor,
            "FRESH_DEV40",
            video,
            primary_rows,
            "B0",
            "B0",
            None,
        )
        run_rows.append(b0)
        cache_hits += int(hit)
        cache_misses += int(not hit)

        for variant in cfg["variants"]:
            tau = float(
                cfg["headline"]["taus"][variant]
            )
            signal, hit = run_or_load(
                cfg,
                exp050,
                runtime,
                predictor,
                "FRESH_DEV40",
                video,
                primary_rows,
                variant,
                variant,
                tau,
            )
            run_rows.append(signal)
            cache_hits += int(hit)
            cache_misses += int(not hit)

            neutral, hit = run_or_load(
                cfg,
                exp050,
                runtime,
                predictor,
                "FRESH_DEV40",
                video,
                primary_rows,
                variant + "_NEUTRAL",
                "B0",
                None,
                neutral_k=signal["admit_count"],
            )
            run_rows.append(neutral)
            cache_hits += int(hit)
            cache_misses += int(not hit)

            print(
                "  {} K={}/{} POR30={:.6f} ITR30={:.6f}".format(
                    variant,
                    signal["admit_count"],
                    signal["eligible_nonconditioning_frames"],
                    signal["por30"],
                    signal["itr30"],
                ),
                flush=True,
            )

    if len(keys) != int(cfg["scope"]["expected_primary_events"]):
        raise RuntimeError("Primary key count changed during run")

    summary_by_label = aggregate(run_rows, cfg)

    event_rows = [
        event
        for row in run_rows
        for event in row["event_rows"]
    ]

    trajectory_rows = []
    for row in run_rows:
        trajectory_rows.append(
            {
                "run_label": row["run_label"],
                "variant": row["variant"],
                "tau": "" if row["tau"] is None else row["tau"],
                "video": row["video"],
                "eligible_nonconditioning_frames":
                    row["eligible_nonconditioning_frames"],
                "admit_count": row["admit_count"],
                "block_count": row["block_count"],
                "write_rate": row["write_rate"],
                "event_count": row["event_count"],
                "por30": row["por30"],
                "itr30": row["itr30"],
                "peak_vram_gb": row["peak_vram_gb"],
                "runtime_sec": row["runtime_sec"],
            }
        )

    summary_rows = [
        summary_by_label[label]
        for label in summary_by_label
    ]

    write_csv_atomic(
        OUT / "event_outcomes.csv",
        [
            "run_label",
            "tau",
            "video",
            "object_id",
            "reappear_frame",
            "disappear_start",
            "recovered_w30",
            "first_recovery_frame",
            "evaluable_visible_frames",
            "theft_event_w30",
            "first_theft_episode_start",
            "max_consecutive_theft_frames",
        ],
        event_rows,
    )

    write_csv_atomic(
        OUT / "trajectory_summary.csv",
        [
            "run_label",
            "variant",
            "tau",
            "video",
            "eligible_nonconditioning_frames",
            "admit_count",
            "block_count",
            "write_rate",
            "event_count",
            "por30",
            "itr30",
            "peak_vram_gb",
            "runtime_sec",
        ],
        trajectory_rows,
    )

    write_csv_atomic(
        OUT / "headline_summary.csv",
        [
            "run_label",
            "videos",
            "events",
            "por30",
            "itr30",
            "sum_admit_count",
            "sum_eligible_count",
            "pooled_write_rate",
            "max_peak_vram_gb",
            "runtime_sec",
        ],
        summary_rows,
    )

    summary = {
        "experiment": "EXP051",
        "status": "FRESH_DEV_HEADLINE_OUTCOMES_COMPLETE",
        "repo_commit": git_head(),
        "config_sha256": sha256sum(CFG_PATH),
        "fresh_dev_videos": len(videos),
        "primary_event_keys": len(keys),
        "r_star": cfg["headline"]["r_star"],
        "headline": summary_by_label,
        "trajectory_count": len(run_rows),
        "scratch_cache_hits": cache_hits,
        "scratch_cache_misses": cache_misses,
        "max_peak_vram_gb": max(
            r["peak_vram_gb"] for r in run_rows
        ),
        "fresh_dev_tracking_outcomes_inspected": True,
        "test_touched": False,
        "claim_boundary": cfg["claim_boundary"],
    }
    write_json_atomic(
        OUT / "summary.json",
        summary,
    )

    print(
        json.dumps(summary, indent=2, sort_keys=True),
        flush=True,
    )
    print("EXP051_RUN=PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["selftest", "plan", "sanity", "run"],
    )
    args = parser.parse_args()

    if args.mode == "selftest":
        selftest()
    elif args.mode == "plan":
        plan()
    elif args.mode == "sanity":
        sanity()
    else:
        run()


if __name__ == "__main__":
    main()
