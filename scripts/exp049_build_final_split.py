from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter

import numpy as np


REPO=Path.home()/"thesis"/"identitygate"
CONFIG=REPO/"configs"/"EXP049-final-split-v1.json"

OUTDIR=REPO/"experiments"/"EXP049_final_split"
HARD_OUT=OUTDIR/"hard_pool.csv"
DEV_OUT=OUTDIR/"fresh_dev.csv"
TEST_OUT=OUTDIR/"hard_test.csv"
REP_OUT=OUTDIR/"representative_test.csv"
MANIFEST_OUT=OUTDIR/"manifest.json"


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    return subprocess.check_output(
        ["git","-C",str(REPO),"rev-parse","HEAD"],
        text=True,
    ).strip()


def git_status():
    return subprocess.check_output(
        ["git","-C",str(REPO),"status","--porcelain"],
        text=True,
    ).strip()


def require_clean_committed():
    status=git_status()

    if status:
        raise RuntimeError(
            "STOP: run requires clean thesis repo: "+status
        )

    for path in (
        Path(__file__).resolve(),
        CONFIG,
    ):
        rel=str(path.relative_to(REPO))

        result=subprocess.run(
            [
                "git","-C",str(REPO),
                "ls-files","--error-unmatch",rel
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "STOP: run requires committed file: "+rel
            )


def read_csv(path):
    with open(path,newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path,fieldnames,rows):
    path.parent.mkdir(parents=True,exist_ok=True)

    tmp=Path(str(path)+".tmp")

    with open(tmp,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp,path)


def write_json_atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)

    tmp=Path(str(path)+".tmp")

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )+"\n",
        encoding="utf-8",
    )

    os.replace(tmp,path)


def membership_hash(label,videos):
    payload=(
        label+"\n"
        +"\n".join(videos)
        +"\n"
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def select_hard_pool(rows,hard_n,max_fraction):
    scores=np.asarray(
        [float(r["di_v1_score"]) for r in rows],
        dtype=np.float64,
    )

    clusters=np.asarray(
        [int(r["visual_cluster"]) for r in rows],
        dtype=np.int64,
    )

    videos=[r["video"] for r in rows]

    cap=int(np.floor(max_fraction*hard_n))

    assert cap>=1

    order=sorted(
        range(len(rows)),
        key=lambda i:(
            -float(scores[i]),
            videos[i],
        ),
    )

    selected=[]
    used=Counter()

    for i in order:
        cluster=int(clusters[i])

        if used[cluster] >= cap:
            continue

        selected.append(i)
        used[cluster]+=1

        if len(selected)==hard_n:
            break

    assert len(selected)==hard_n
    assert max(used.values())/hard_n <= max_fraction+1e-12

    return selected,cap,used


def allocate_dev_quotas(cluster_counts,dev_n,hard_n):
    exact={
        c:cluster_counts[c]*dev_n/hard_n
        for c in sorted(cluster_counts)
    }

    quota={
        c:int(np.floor(exact[c]))
        for c in exact
    }

    remaining=dev_n-sum(quota.values())

    assert remaining>=0

    priority=sorted(
        exact,
        key=lambda c:(
            -(exact[c]-quota[c]),
            c,
        ),
    )

    for c in priority[:remaining]:
        quota[c]+=1

    assert sum(quota.values())==dev_n

    return quota


def partition_hard_pool(
    rows,
    hard,
    dev_n,
    seed,
):
    videos=[r["video"] for r in rows]
    clusters=np.asarray(
        [int(r["visual_cluster"]) for r in rows],
        dtype=np.int64,
    )

    hard_cluster_counts=Counter(
        int(clusters[i])
        for i in hard
    )

    quota=allocate_dev_quotas(
        hard_cluster_counts,
        dev_n,
        len(hard),
    )

    rng=np.random.default_rng(seed)

    dev=[]

    for cluster in sorted(hard_cluster_counts):
        members=sorted(
            [
                i
                for i in hard
                if int(clusters[i])==cluster
            ],
            key=lambda i:videos[i],
        )

        perm=rng.permutation(len(members))

        dev.extend(
            members[int(j)]
            for j in perm[:quota[cluster]]
        )

    dev=sorted(
        dev,
        key=lambda i:videos[i],
    )

    dev_set=set(dev)

    test=sorted(
        [
            i
            for i in hard
            if i not in dev_set
        ],
        key=lambda i:videos[i],
    )

    assert len(dev)==dev_n
    assert len(dev_set)==dev_n
    assert len(dev_set & set(test))==0
    assert dev_set | set(test)==set(hard)

    return dev,test,quota


def select_representative(
    rows,
    hard,
    representative_n,
    seed,
):
    videos=[r["video"] for r in rows]

    hard_set=set(hard)

    remaining=sorted(
        [
            i
            for i in range(len(rows))
            if i not in hard_set
        ],
        key=lambda i:videos[i],
    )

    rng=np.random.default_rng(seed)

    positions=rng.choice(
        len(remaining),
        size=representative_n,
        replace=False,
    )

    representative=sorted(
        [
            remaining[int(j)]
            for j in positions
        ],
        key=lambda i:videos[i],
    )

    assert len(set(representative))==representative_n
    assert not (set(representative) & hard_set)

    return representative


def build(cfg,rows):
    videos=[r["video"] for r in rows]

    assert len(rows)==cfg["population"]["videos"]
    assert videos==sorted(videos)
    assert len(set(videos))==len(videos)

    hard,cap,hard_cluster_counts=select_hard_pool(
        rows,
        cfg["hard_pool"]["videos"],
        cfg["hard_pool"]["visual_cluster_max_fraction"],
    )

    dev,test,quota=partition_hard_pool(
        rows,
        hard,
        cfg["hard_partition"]["fresh_dev_videos"],
        cfg["hard_partition"]["rng_seed"],
    )

    representative=select_representative(
        rows,
        hard,
        cfg["representative_test"]["videos"],
        cfg["representative_test"]["rng_seed"],
    )

    assert len(test)==cfg["hard_partition"]["hard_test_videos"]

    events=np.asarray(
        [int(r["primary_event_count"]) for r in rows],
        dtype=np.int64,
    )

    event_counts={
        "hard_pool":int(events[hard].sum()),
        "fresh_dev":int(events[dev].sum()),
        "hard_test":int(events[test].sum()),
        "representative_test":
            int(events[representative].sum()),
    }

    assert (
        event_counts["hard_test"]
        >= cfg["acceptance"]["hard_test_min_primary_events"]
    )

    hashes={
        "HARD120":
            membership_hash(
                "HARD120",
                [videos[i] for i in hard],
            ),
        "DEV40":
            membership_hash(
                "DEV40",
                [videos[i] for i in dev],
            ),
        "TEST80":
            membership_hash(
                "TEST80",
                [videos[i] for i in test],
            ),
        "REPRESENTATIVE40":
            membership_hash(
                "REPRESENTATIVE40",
                [videos[i] for i in representative],
            ),
    }

    expected=cfg["expected_diagnostic"]

    assert (
        event_counts["hard_pool"]
        == expected["hard_pool_primary_events"]
    )

    assert (
        event_counts["fresh_dev"]
        == expected["fresh_dev_primary_events"]
    )

    assert (
        event_counts["hard_test"]
        == expected["hard_test_primary_events"]
    )

    assert (
        event_counts["representative_test"]
        == expected["representative_primary_events"]
    )

    assert hashes==expected["membership_sha256"], (
        hashes,
        expected["membership_sha256"],
    )

    return {
        "hard":hard,
        "dev":dev,
        "test":test,
        "representative":representative,
        "cap":cap,
        "hard_cluster_counts":hard_cluster_counts,
        "quota":quota,
        "event_counts":event_counts,
        "hashes":hashes,
    }


def output_rows(rows,indices,cohort,partition):
    result=[]

    for i in indices:
        row=rows[i]

        result.append({
            "video":row["video"],
            "cohort":cohort,
            "partition":partition,
            "primary_event_count":
                int(row["primary_event_count"]),
            "di_v1_score":row["di_v1_score"],
            "visual_cluster":
                int(row["visual_cluster"]),
        })

    return result


def load_context():
    cfg=json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    source=REPO/cfg["input"]["path"]

    got=sha256_file(source)
    want=cfg["input"]["sha256"]

    assert got==want,(got,want)

    rows=read_csv(source)

    assert len(rows)==cfg["input"]["expected_videos"]

    return cfg,rows


def selftest():
    synthetic=[
        {
            "video":"a",
            "di_v1_score":"0.9",
            "visual_cluster":"0",
            "primary_event_count":"1",
        },
        {
            "video":"b",
            "di_v1_score":"0.8",
            "visual_cluster":"0",
            "primary_event_count":"1",
        },
        {
            "video":"c",
            "di_v1_score":"0.7",
            "visual_cluster":"1",
            "primary_event_count":"1",
        },
        {
            "video":"d",
            "di_v1_score":"0.6",
            "visual_cluster":"1",
            "primary_event_count":"1",
        },
        {
            "video":"e",
            "di_v1_score":"0.5",
            "visual_cluster":"2",
            "primary_event_count":"1",
        },
    ]

    selected,cap,_=select_hard_pool(
        synthetic,
        4,
        0.5,
    )

    assert cap==2
    assert selected==[0,1,2,3]

    quota=allocate_dev_quotas(
        Counter({0:2,1:2}),
        2,
        4,
    )

    assert quota=={0:1,1:1}

    got=membership_hash(
        "X",
        ["a","b"],
    )

    want=hashlib.sha256(
        b"X\na\nb\n"
    ).hexdigest()

    assert got==want

    print("EXP049_SELFTEST=PASS")


def plan():
    cfg,rows=load_context()
    result=build(cfg,rows)

    print("EXP049 PLAN")
    print("identitygate_head =",git_head())
    print("population_videos =",len(rows))
    print("hard_pool_videos =",len(result["hard"]))
    print("fresh_dev_videos =",len(result["dev"]))
    print("hard_test_videos =",len(result["test"]))
    print(
        "representative_test_videos =",
        len(result["representative"]),
    )

    print(
        "hard_pool_events =",
        result["event_counts"]["hard_pool"],
    )
    print(
        "fresh_dev_events =",
        result["event_counts"]["fresh_dev"],
    )
    print(
        "hard_test_events =",
        result["event_counts"]["hard_test"],
    )
    print(
        "representative_test_events =",
        result["event_counts"]["representative_test"],
    )

    for label,value in result["hashes"].items():
        print(
            label+"_VIDEO_SHA256 =",
            value,
        )

    print("model_outcomes_used = false")
    print("fresh_dev_evaluated = false")
    print("test_evaluated = false")
    print("split_manifest_written = false")
    print("EXP049_PLAN_PASS")


def run():
    require_clean_committed()

    if OUTDIR.exists():
        raise RuntimeError(
            "STOP: EXP049 output directory exists; do not rerun"
        )

    cfg,rows=load_context()
    result=build(cfg,rows)

    hard_set=set(result["hard"])
    dev_set=set(result["dev"])
    test_set=set(result["test"])
    rep_set=set(result["representative"])

    assert dev_set | test_set == hard_set
    assert not (dev_set & test_set)
    assert not (rep_set & hard_set)

    hard_rows=[]

    for i in result["hard"]:
        partition=(
            "FRESH_DEV"
            if i in dev_set
            else "HARD_TEST"
        )

        hard_rows.extend(
            output_rows(
                rows,
                [i],
                "HARD_POOL",
                partition,
            )
        )

    dev_rows=output_rows(
        rows,
        result["dev"],
        "HARD_POOL",
        "FRESH_DEV",
    )

    test_rows=output_rows(
        rows,
        result["test"],
        "HARD_POOL",
        "HARD_TEST",
    )

    rep_rows=output_rows(
        rows,
        result["representative"],
        "REPRESENTATIVE",
        "REPRESENTATIVE_TEST",
    )

    fields=[
        "video",
        "cohort",
        "partition",
        "primary_event_count",
        "di_v1_score",
        "visual_cluster",
    ]

    write_csv_atomic(
        HARD_OUT,
        fields,
        hard_rows,
    )

    write_csv_atomic(
        DEV_OUT,
        fields,
        dev_rows,
    )

    write_csv_atomic(
        TEST_OUT,
        fields,
        test_rows,
    )

    write_csv_atomic(
        REP_OUT,
        fields,
        rep_rows,
    )

    manifest={
        "experiment":"EXP049",
        "status":"FINAL_SPLIT_FROZEN",
        "identitygate_commit_at_execution":
            git_head(),
        "config_sha256":
            sha256_file(CONFIG),
        "script_sha256":
            sha256_file(Path(__file__).resolve()),

        "population_videos":1170,

        "hard_pool":{
            "videos":120,
            "primary_events":
                result["event_counts"]["hard_pool"],
            "membership_sha256":
                result["hashes"]["HARD120"],
            "visual_cluster_cap":0.15,
            "visual_cluster_counts":{
                str(c):int(
                    result["hard_cluster_counts"].get(c,0)
                )
                for c in range(20)
            },
        },

        "fresh_dev":{
            "videos":40,
            "primary_events":
                result["event_counts"]["fresh_dev"],
            "membership_sha256":
                result["hashes"]["DEV40"],
        },

        "hard_test":{
            "videos":80,
            "primary_events":
                result["event_counts"]["hard_test"],
            "membership_sha256":
                result["hashes"]["TEST80"],
            "minimum_required_primary_events":200,
        },

        "representative_test":{
            "videos":40,
            "primary_events":
                result["event_counts"]["representative_test"],
            "membership_sha256":
                result["hashes"]["REPRESENTATIVE40"],
            "disjoint_from_entire_hard_pool":True,
        },

        "hard_dev_allocation":{
            "method":
                "visual-cluster-stratified largest-remainder quota",
            "rng_seed":42,
            "dev_quota_by_cluster":{
                str(c):int(result["quota"].get(c,0))
                for c in range(20)
            },
        },

        "representative_sampling":{
            "method":
                "uniform without replacement from primary-eligible non-hard videos",
            "rng_seed":42,
        },

        "boundary":{
            "model_outcomes_used":False,
            "gate_inference_performed":False,
            "fresh_dev_evaluated":False,
            "test_evaluated":False,
            "test_membership_metadata_only":True,
        },
    }

    manifest["output_hashes"]={
        "hard_pool.csv":
            sha256_file(HARD_OUT),
        "fresh_dev.csv":
            sha256_file(DEV_OUT),
        "hard_test.csv":
            sha256_file(TEST_OUT),
        "representative_test.csv":
            sha256_file(REP_OUT),
    }

    write_json_atomic(
        MANIFEST_OUT,
        manifest,
    )

    print("hard_pool_videos = 120")
    print(
        "hard_pool_events =",
        result["event_counts"]["hard_pool"],
    )
    print("fresh_dev_videos = 40")
    print(
        "fresh_dev_events =",
        result["event_counts"]["fresh_dev"],
    )
    print("hard_test_videos = 80")
    print(
        "hard_test_events =",
        result["event_counts"]["hard_test"],
    )
    print("representative_test_videos = 40")
    print(
        "representative_test_events =",
        result["event_counts"]["representative_test"],
    )

    for label,value in result["hashes"].items():
        print(label+"_VIDEO_SHA256 =",value)

    print("model_outcomes_used = false")
    print("fresh_dev_evaluated = false")
    print("test_evaluated = false")
    print("EXP049_RUN_PASS")


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "selftest",
            "plan",
            "run",
        ],
    )

    args=parser.parse_args()

    if args.mode=="selftest":
        selftest()
    elif args.mode=="plan":
        plan()
    else:
        run()


if __name__=="__main__":
    main()
