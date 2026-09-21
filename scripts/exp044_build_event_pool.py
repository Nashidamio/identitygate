#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/EXP044-event-pool-v1.json"
SCRIPT = "scripts/exp044_build_event_pool.py"
EVENT_FIELDS = [
    "video", "object_id", "disappear_start", "reappear_frame", "gap_len",
    "lookback_complete", "reference_objects", "vanished_objects",
    "returned_objects", "vanish_share", "return_share", "ws_gt_candidate",
    "ws_scene_event_id",
]
SCENE_FIELDS = [
    "scene_event_id", "video", "n_object_events", "n_unique_objects",
    "disappear_min", "disappear_max", "reappear_min", "reappear_max",
    "final_ws_label", "label_source", "manual_reason",
]
EXTRA_FIELDS = [
    "resolved_ws_status", "ws_label_source", "is_final_whole_scene",
    "development_exposed", "primary_pool_eligible", "ws_control_pool_eligible",
]
VIDEO_FIELDS = [
    "video", "development_exposed", "total_events", "whole_scene_events",
    "non_whole_scene_events", "primary_pool_events", "ws_control_pool_events",
    "has_both_ws_and_non_ws_events",
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def parse_flag(value):
    token = str(value).strip().lower()
    require(token in {"0", "1", "false", "true"}, "Invalid flag: " + str(value))
    return token in {"1", "true"}


def read_csv_bytes(data, expected_fields):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    require(reader.fieldnames == expected_fields, "CSV header mismatch")
    rows = list(reader)
    require(all(set(r) == set(expected_fields) for r in rows), "Malformed CSV row")
    require(all(v is not None for r in rows for v in r.values()), "Missing CSV cell")
    return rows


def build_tables(events, scenes, excluded):
    scene_map = {r["scene_event_id"]: r for r in scenes}
    require(len(scene_map) == len(scenes), "Duplicate scene ID")
    require(all(scene_map), "Empty scene ID")
    for r in scenes:
        require(r["final_ws_label"] in {"WHOLE_SCENE", "NORMAL_OCCLUSION"},
                "Invalid frozen scene label")
        require(r["label_source"] in {
            "EXP034_AUTO_CONFIRMED", "EXP036_MANUAL_ADJUDICATION"
        }, "Invalid scene-label provenance")
        require(r["label_source"] != "EXP034_AUTO_CONFIRMED"
                or r["final_ws_label"] == "WHOLE_SCENE", "Invalid automatic label")

    keys, groups, output = set(), defaultdict(list), []
    for r in events:
        video = r["video"]
        oid, ds, rf, gap = (int(r[k]) for k in
                            ("object_id", "disappear_start", "reappear_frame", "gap_len"))
        require(video and video == video.strip(), "Invalid video ID")
        require(oid > 0 and ds >= 0 and rf > ds and gap > 0, "Invalid event coordinates")
        key = (video, oid, rf)
        require(key not in keys, "Duplicate object event: " + repr(key))
        keys.add(key)
        sid = r["ws_scene_event_id"]
        candidate = parse_flag(r["ws_gt_candidate"])
        require(bool(sid) == candidate, "Candidate/membership mismatch: " + repr(key))
        if candidate:
            require(sid in scene_map, "Unknown scene membership: " + sid)
            scene = scene_map[sid]
            require(scene["video"] == video, "Cross-video scene membership: " + sid)
            resolved = scene["final_ws_label"]
            label_source = scene["label_source"]
            groups[sid].append((oid, ds, rf))
        else:
            resolved = "NOT_ANNOTATION_CANDIDATE"
            label_source = "EXP019_NOT_ANNOTATION_CANDIDATE"

        is_ws = resolved == "WHOLE_SCENE"
        exposed = video in excluded
        row = {k: r[k] for k in EVENT_FIELDS}
        row.update({
            "resolved_ws_status": resolved,
            "ws_label_source": label_source,
            "is_final_whole_scene": int(is_ws),
            "development_exposed": int(exposed),
            "primary_pool_eligible": int(not exposed and not is_ws),
            "ws_control_pool_eligible": int(not exposed and is_ws),
        })
        output.append(row)

    require(set(groups) == set(scene_map), "Scene membership coverage mismatch")
    for sid, members in groups.items():
        expected = {
            "n_object_events": len(members),
            "n_unique_objects": len({x[0] for x in members}),
            "disappear_min": min(x[1] for x in members),
            "disappear_max": max(x[1] for x in members),
            "reappear_min": min(x[2] for x in members),
            "reappear_max": max(x[2] for x in members),
        }
        for field, value in expected.items():
            require(int(scene_map[sid][field]) == value,
                    "Scene membership metadata mismatch: " + sid + "/" + field)

    output.sort(key=lambda r: (r["video"], int(r["object_id"]), int(r["reappear_frame"])))
    totals = defaultdict(Counter)
    for r in output:
        c = totals[r["video"]]
        c["total_events"] += 1
        c["whole_scene_events"] += r["is_final_whole_scene"]
        c["non_whole_scene_events"] += 1 - r["is_final_whole_scene"]
        c["primary_pool_events"] += r["primary_pool_eligible"]
        c["ws_control_pool_events"] += r["ws_control_pool_eligible"]
    videos = []
    for video, c in sorted(totals.items()):
        videos.append({
            "video": video, "development_exposed": int(video in excluded),
            **dict(c),
            "has_both_ws_and_non_ws_events": int(
                c["whole_scene_events"] > 0 and c["non_whole_scene_events"] > 0
            ),
        })
    require(len(output) == len(events), "Event loss or multiplication")
    return output, videos


def self_test():
    def event(video, oid, ds, rf, candidate, sid):
        r = {k: "" for k in EVENT_FIELDS}
        r.update(video=video, object_id=str(oid), disappear_start=str(ds),
                 reappear_frame=str(rf), gap_len=str(rf-ds),
                 ws_gt_candidate=str(candidate), ws_scene_event_id=sid)
        return r

    events = [
        event("mix", 1, 10, 16, 1, "mix:WS001"),
        event("mix", 2, 11, 17, 1, "mix:WS001"),
        event("mix", 1, 30, 36, 0, ""),
        event("normal", 1, 20, 27, 1, "normal:WS001"),
        event("exposed", 1, 40, 46, 0, ""),
    ]
    scenes = [
        dict(zip(SCENE_FIELDS, ["mix:WS001", "mix", "2", "2", "10", "11", "16", "17",
                               "WHOLE_SCENE", "EXP034_AUTO_CONFIRMED", ""])),
        dict(zip(SCENE_FIELDS, ["normal:WS001", "normal", "1", "1", "20", "20", "27", "27",
                               "NORMAL_OCCLUSION", "EXP036_MANUAL_ADJUDICATION", "fixture"])),
    ]
    rows, videos = build_tables(events, scenes, {"exposed"})
    require(len(rows) == 5 and len(videos) == 3, "Synthetic count failure")
    require(sum(r["primary_pool_eligible"] for r in rows) == 2, "Synthetic primary failure")
    require(sum(r["ws_control_pool_eligible"] for r in rows) == 2, "Synthetic WS failure")
    mixed = next(v for v in videos if v["video"] == "mix")
    require(mixed["primary_pool_events"] == 1, "Mixed-video non-WS event incorrectly removed")
    require(parse_flag("True") and not parse_flag("False"), "Boolean parsing failure")
    bad_events = [dict(r) for r in events]
    bad_events[0]["ws_scene_event_id"] = "unknown:WS001"
    bad_scenes = [dict(r) for r in scenes]
    bad_scenes[0]["n_object_events"] = "3"
    failures = 0
    for ev, sc in [(events + [events[0]], scenes), (bad_events, scenes), (events, bad_scenes)]:
        try:
            build_tables(ev, sc, {"exposed"})
        except RuntimeError:
            failures += 1
    require(failures == 3, "Synthetic invalid inputs were not rejected")
    print("EXP044_SYNTHETIC_TESTS=PASS", flush=True)


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    require(not git("status", "--porcelain"), "STOP: clean committed tree required")
    cfg = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    require(cfg["experiment"] == "EXP044", "Wrong configuration")
    tracked = set(git("ls-files").splitlines())
    require({CONFIG, SCRIPT} <= tracked, "STOP: freeze both implementation files first")
    subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                           cfg["source_commit"], "HEAD"])
    out = ROOT / cfg["output_dir"]
    require(not out.exists(), "STOP: output already exists; do not overwrite")
    data = {}
    for name, item in cfg["inputs"].items():
        require(item["path"] in tracked, "Untracked source: " + item["path"])
        raw = (ROOT / item["path"]).read_bytes()
        require(digest(raw) == item["sha256"], "Input hash mismatch: " + item["path"])
        data[name] = raw

    events = read_csv_bytes(data["events"], EVENT_FIELDS)
    scenes = read_csv_bytes(data["scene_labels"], SCENE_FIELDS)
    lock = json.loads(data["exposure_lock"])
    ws_cfg = json.loads(data["ws_rule"])
    require(lock["development_exposure_lock_complete"] is True, "Exposure lock incomplete")
    require(ws_cfg["primary_endpoint_excludes_final_whole_scene_stratum"] is True,
            "WS primary-endpoint rule changed")
    require(ws_cfg["sensitivity_analysis_retains_whole_scene_stratum"] is True,
            "WS sensitivity rule changed")
    excluded = set(lock["video_ids"])
    e = cfg["expected"]
    require(len(excluded) == len(lock["video_ids"]) == e["excluded_videos"], "Exposure count mismatch")
    require(len(events) == e["events"], "Event count mismatch")
    require(len(scenes) == e["candidate_scenes"], "Scene count mismatch")
    require(sum(r["final_ws_label"] == "WHOLE_SCENE" for r in scenes) == e["whole_scene_scenes"],
            "Frozen WHOLE_SCENE scene count mismatch")
    rows, videos = build_tables(events, scenes, excluded)
    require(len(videos) == e["event_bearing_videos"], "Event-bearing video count mismatch")
    ids = {r["video"] for r in videos}
    partitions = Counter()
    for r in rows:
        key = ("exposed" if r["development_exposed"] else "unexposed")
        key += "_ws" if r["is_final_whole_scene"] else "_non_ws"
        partitions[key] += 1
    require(sum(partitions.values()) == len(events), "Event partition mismatch")
    primary_videos = [r["video"] for r in videos if r["primary_pool_events"] > 0]
    require(not set(primary_videos) & excluded, "Exposed video entered primary pool")
    summary = {
        "experiment": "EXP044", "status": "EVENT_POOL_JOIN_VERIFIED_NOT_A_SPLIT",
        "repo_commit_at_execution": git("rev-parse", "HEAD"),
        "config_sha256": digest((ROOT / CONFIG).read_bytes()),
        "script_sha256": digest((ROOT / SCRIPT).read_bytes()),
        "python_version": sys.version.split()[0],
        "input_hashes": {k: digest(v) for k, v in data.items()},
        "all_events_retained": len(rows), "event_bearing_videos": len(videos),
        "candidate_scenes_mapped": len(scenes),
        "known_excluded_videos": len(excluded),
        "excluded_videos_in_event_corpus": len(ids & excluded),
        "excluded_videos_outside_event_corpus": len(excluded - ids),
        "event_partition": dict(sorted(partitions.items())),
        "resolved_ws_status_counts": dict(sorted(Counter(r["resolved_ws_status"] for r in rows).items())),
        "primary_pool_events": sum(r["primary_pool_eligible"] for r in rows),
        "primary_pool_videos": len(primary_videos),
        "ws_control_pool_events": sum(r["ws_control_pool_eligible"] for r in rows),
        "ws_control_pool_videos": sum(r["ws_control_pool_events"] > 0 for r in videos),
        "mixed_ws_non_ws_videos": sum(r["has_both_ws_and_non_ws_events"] for r in videos),
        "p2_status": lock["historical_p2_reference"]["status"],
        "di_v1_frozen": False, "final_split_constructed": False,
        "sam_or_gate_inference_performed": False,
        "fresh_dev_evaluation_performed": False, "test_evaluation_performed": False,
        "claim_boundary": cfg["claim_boundary"],
    }
    out.mkdir(parents=True)
    write_csv(out / "event_pool.csv", rows, EVENT_FIELDS + EXTRA_FIELDS)
    write_csv(out / "per_video_pool.csv", videos, VIDEO_FIELDS)
    summary["output_hashes"] = {name: digest((out / name).read_bytes())
                                for name in ("event_pool.csv", "per_video_pool.csv")}
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("EXP044_EVENT_POOL_JOIN=PASS")
    for name in ("event_pool.csv", "per_video_pool.csv", "summary.json"):
        print("SHA256=" + digest((out / name).read_bytes()) + " " + str(out / name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
