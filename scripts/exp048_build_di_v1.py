from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess

import numpy as np
import scipy
from scipy.cluster.vq import kmeans2
from scipy.stats import rankdata


REPO=Path.home()/"thesis"/"identitygate"

CONFIG=REPO/"configs"/"EXP048-di-v1-components-v1.json"

OUTDIR=REPO/"experiments"/"EXP048_di_v1"

COMPONENTS_OUT=OUTDIR/"di_video_components.csv"
CLUSTERS_OUT=OUTDIR/"visual_clusters.csv"
CORR_OUT=OUTDIR/"component_correlations.csv"
STABILITY_OUT=OUTDIR/"stability_grid.csv"
SUMMARY_OUT=OUTDIR/"summary.json"


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

    for path in (Path(__file__).resolve(),CONFIG):
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
        json.dumps(data,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    os.replace(tmp,path)


def percentile_rank(values):
    x=np.asarray(values,dtype=np.float64)

    assert x.ndim==1
    assert x.size>0
    assert np.isfinite(x).all()

    return rankdata(
        x,
        method="average",
    )/len(x)


def build_identity_raw(videos,identity_rows):
    from collections import defaultdict

    target_values=defaultdict(list)

    for row in identity_rows:
        status=row["identity_pressure_status"]

        if status=="TRACKED_COMPETITOR_AVAILABLE":
            key=(row["video"],int(row["object_id"]))
            target_values[key].append(
                float(row["max_other_anchor_cos_fp32"])
            )
        else:
            assert status=="NO_TRACKED_COMPETITOR"

    unique={}

    for key,values in target_values.items():
        assert max(values)-min(values) <= 1e-12
        unique[key]=values[0]

    by_video=defaultdict(list)

    for (video,_),value in unique.items():
        by_video[video].append(value)

    raw=np.empty(len(videos),dtype=np.float64)
    defined=np.zeros(len(videos),dtype=bool)

    defined_values=[
        max(by_video[v])
        for v in videos
        if by_video.get(v)
    ]

    assert defined_values

    floor=float(min(defined_values))-1.0

    for i,video in enumerate(videos):
        values=by_video.get(video)

        if values:
            raw[i]=float(max(values))
            defined[i]=True
        else:
            raw[i]=floor

    return raw,defined,unique


def select_with_cap(scores,labels,videos,hard_n,max_fraction=0.15):
    from collections import Counter

    hard_n=int(hard_n)
    cap=int(np.floor(max_fraction*hard_n))

    assert cap>=1

    order=sorted(
        range(len(videos)),
        key=lambda i:(-float(scores[i]),videos[i]),
    )

    selected=[]
    used=Counter()

    for i in order:
        cluster=int(labels[i])

        if used[cluster] >= cap:
            continue

        selected.append(i)
        used[cluster]+=1

        if len(selected)==hard_n:
            break

    if len(selected) != hard_n:
        return None,cap,used

    assert max(used.values())/hard_n <= max_fraction+1e-12

    return selected,cap,used


def load_context():
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))

    gt_path=REPO/cfg["inputs"]["gt_event_primitives"]["path"]
    id_path=REPO/cfg["inputs"]["identity_event_primitives"]["path"]
    emb_path=REPO/cfg["inputs"]["visual_embeddings"]["path"]
    visual_path=REPO/cfg["inputs"]["visual_video_rows"]["path"]

    checks=[
        (gt_path,cfg["inputs"]["gt_event_primitives"]["sha256"]),
        (id_path,cfg["inputs"]["identity_event_primitives"]["sha256"]),
        (emb_path,cfg["inputs"]["visual_embeddings"]["sha256"]),
        (visual_path,cfg["inputs"]["visual_video_rows"]["sha256"]),
    ]

    for path,want in checks:
        got=sha256_file(path)
        assert got==want,(str(path),got,want)

    gt=read_csv(gt_path)
    identity=read_csv(id_path)
    visual=read_csv(visual_path)
    emb=np.load(emb_path,allow_pickle=False)

    assert len(gt)==2701
    assert len(identity)==2701
    assert len(visual)==1170
    assert emb.shape==(1170,384)
    assert emb.dtype==np.float32
    assert np.isfinite(emb).all()

    videos=sorted({row["video"] for row in gt})

    assert len(videos)==1170
    assert sorted({row["video"] for row in identity})==videos
    assert [row["video"] for row in visual]==videos

    return cfg,gt,identity,visual,emb,videos


def compute(cfg,gt,identity,emb,videos):
    from collections import defaultdict,Counter

    by_video=defaultdict(list)

    for row in gt:
        by_video[row["video"]].append(row)

    identity_raw,identity_defined,unique_identity=build_identity_raw(
        videos,
        identity,
    )

    gap=[]
    size=[]
    crowding=[]
    displacement=[]
    event_counts=[]

    for video in videos:
        rows=by_video[video]

        gap.append(max(float(r["gap_len"]) for r in rows))

        size.append(
            min(
                float(r["pre10_min_visible_area_fraction"])
                for r in rows
            )
        )

        crowding.append(
            max(
                float(r["crowding_mean_visible_objects"])
                for r in rows
            )
        )

        displacement.append(
            max(
                float(r["reappearance_displacement_diag_norm"])
                for r in rows
            )
        )

        event_counts.append(len(rows))

    gap=np.asarray(gap,dtype=np.float64)
    size=np.asarray(size,dtype=np.float64)
    crowding=np.asarray(crowding,dtype=np.float64)
    displacement=np.asarray(displacement,dtype=np.float64)
    event_counts=np.asarray(event_counts,dtype=np.int64)

    r_identity=percentile_rank(identity_raw)
    r_gap=percentile_rank(gap)
    r_size=percentile_rank(-size)
    r_crowding=percentile_rank(crowding)
    r_displacement=percentile_rank(displacement)

    components=np.column_stack([
        r_identity,
        r_gap,
        r_size,
        r_crowding,
        r_displacement,
    ])

    di=components.mean(axis=1)

    assert components.shape==(1170,5)
    assert np.isfinite(components).all()
    assert np.isfinite(di).all()

    # Preserve literal EXP047 representation; no extra normalization.
    x=emb.astype(np.float64,copy=False)

    centroids,labels=kmeans2(
        x,
        cfg["visual_clustering"]["k"],
        iter=cfg["visual_clustering"]["iter"],
        minit=cfg["visual_clustering"]["minit"],
        missing=cfg["visual_clustering"]["missing"],
        check_finite=True,
        rng=np.random.default_rng(
            cfg["visual_clustering"]["rng_seed"]
        ),
    )

    assert labels.shape==(1170,)
    assert np.isfinite(centroids).all()
    assert labels.min()>=0
    assert labels.max()<20
    assert len(set(labels.tolist()))==20

    ranked=np.column_stack([
        rankdata(components[:,i],method="average")
        for i in range(5)
    ])

    corr=np.corrcoef(ranked,rowvar=False)

    names=[
        "identity",
        "gap",
        "size",
        "crowding",
        "displacement",
    ]

    full_score=di
    stability=[]

    for hard_n in cfg["stability"]["diagnostic_hard_n"]:
        selected,cap,used=select_with_cap(
            full_score,
            labels,
            videos,
            hard_n,
            cfg["visual_clustering"][
                "hard_set_max_fraction_per_cluster"
            ],
        )

        assert selected is not None

        full_set=set(selected)
        overlaps={}

        for omit,name in enumerate(names):
            cols=[i for i in range(5) if i != omit]
            loo=components[:,cols].mean(axis=1)

            other,other_cap,_=select_with_cap(
                loo,
                labels,
                videos,
                hard_n,
                cfg["visual_clustering"][
                    "hard_set_max_fraction_per_cluster"
                ],
            )

            assert other is not None
            assert other_cap==cap

            overlaps[name]=len(
                full_set & set(other)
            )/hard_n

        min_overlap=min(overlaps.values())

        stability.append({
            "hard_n":hard_n,
            "event_count":
                int(event_counts[selected].sum()),
            "cap_count":cap,
            "max_selected_cluster_count":
                int(max(used.values())),
            "max_selected_cluster_fraction":
                float(max(used.values())/hard_n),
            "overlap_omit_identity":
                float(overlaps["identity"]),
            "overlap_omit_gap":
                float(overlaps["gap"]),
            "overlap_omit_size":
                float(overlaps["size"]),
            "overlap_omit_crowding":
                float(overlaps["crowding"]),
            "overlap_omit_displacement":
                float(overlaps["displacement"]),
            "min_loo_overlap":
                float(min_overlap),
            "stability_pass":
                bool(
                    min_overlap
                    >= cfg["stability"][
                        "required_min_leave_one_component_out_overlap"
                    ]
                ),
        })

    assert all(row["stability_pass"] for row in stability)

    cluster_counts=Counter(labels.tolist())

    return {
        "identity_raw":identity_raw,
        "identity_defined":identity_defined,
        "unique_identity":unique_identity,
        "gap":gap,
        "size":size,
        "crowding":crowding,
        "displacement":displacement,
        "event_counts":event_counts,
        "components":components,
        "di":di,
        "labels":labels,
        "corr":corr,
        "names":names,
        "stability":stability,
        "cluster_counts":cluster_counts,
    }


def selftest():
    ranks=percentile_rank([1,1,3,4])

    expected=np.asarray([
        1.5/4,
        1.5/4,
        3/4,
        4/4,
    ])

    assert np.allclose(ranks,expected)

    scores=np.asarray([5,4,3,2,1],dtype=np.float64)
    labels=np.asarray([0,0,1,1,2])
    videos=["a","b","c","d","e"]

    selected,cap,_=select_with_cap(
        scores,
        labels,
        videos,
        4,
        0.5,
    )

    assert cap==2
    assert selected==[0,1,2,3]

    print("EXP048_SELFTEST=PASS")


def plan():
    cfg,gt,identity,visual,emb,videos=load_context()

    print("EXP048 PLAN")
    print("identitygate_head =",git_head())
    print("primary_events =",len(gt))
    print("primary_videos =",len(videos))
    print("visual_shape =",tuple(emb.shape))
    print("components =",",".join(cfg["di_v1"]["components"]))
    print("weights =",cfg["di_v1"]["weights"])
    print("rank_method = average_rank_divided_by_N")
    print("identity_no_competitor = lowest_tied_group")
    print("visual_k =",cfg["visual_clustering"]["k"])
    print("visual_l2_normalize =",cfg["visual_clustering"]["l2_normalize"])
    print("kmeans_seed =",cfg["visual_clustering"]["rng_seed"])
    print("hard_set_selected = false")
    print("hard_set_size_frozen = false")
    print("final_split_constructed = false")
    print("fresh_dev_evaluated = false")
    print("test_evaluated = false")
    print("EXP048_PLAN_PASS")


def run():
    require_clean_committed()

    if OUTDIR.exists():
        raise RuntimeError(
            "STOP: EXP048 output directory exists; do not rerun"
        )

    cfg,gt,identity,visual,emb,videos=load_context()

    result=compute(
        cfg,gt,identity,emb,videos
    )

    component_rows=[]

    for i,video in enumerate(videos):
        component_rows.append({
            "video":video,
            "primary_event_count":
                int(result["event_counts"][i]),
            "identity_competitor_defined":
                int(result["identity_defined"][i]),
            "identity_pressure_value":
                (
                    format(float(result["identity_raw"][i]),".17g")
                    if result["identity_defined"][i]
                    else ""
                ),
            "identity_pressure_rank":
                format(float(result["components"][i,0]),".17g"),
            "gap_duration_value":
                format(float(result["gap"][i]),".17g"),
            "gap_duration_rank":
                format(float(result["components"][i,1]),".17g"),
            "target_size_value":
                format(float(result["size"][i]),".17g"),
            "target_size_rank":
                format(float(result["components"][i,2]),".17g"),
            "crowding_value":
                format(float(result["crowding"][i]),".17g"),
            "crowding_rank":
                format(float(result["components"][i,3]),".17g"),
            "reappearance_displacement_value":
                format(float(result["displacement"][i]),".17g"),
            "reappearance_displacement_rank":
                format(float(result["components"][i,4]),".17g"),
            "di_v1_score":
                format(float(result["di"][i]),".17g"),
            "visual_cluster":
                int(result["labels"][i]),
        })

    write_csv_atomic(
        COMPONENTS_OUT,
        list(component_rows[0].keys()),
        component_rows,
    )

    cluster_rows=[
        {
            "video":video,
            "visual_cluster":int(result["labels"][i]),
        }
        for i,video in enumerate(videos)
    ]

    write_csv_atomic(
        CLUSTERS_OUT,
        ["video","visual_cluster"],
        cluster_rows,
    )

    corr_rows=[]

    for i,name in enumerate(result["names"]):
        row={"component":name}

        for j,other in enumerate(result["names"]):
            row[other]=format(
                float(result["corr"][i,j]),
                ".17g",
            )

        corr_rows.append(row)

    write_csv_atomic(
        CORR_OUT,
        ["component"]+result["names"],
        corr_rows,
    )

    write_csv_atomic(
        STABILITY_OUT,
        list(result["stability"][0].keys()),
        result["stability"],
    )

    di=result["di"]

    summary={
        "experiment":"EXP048",
        "status":
            "DI_V1_COMPONENT_TABLE_COMPLETE_HARD_SET_NOT_SELECTED",
        "identitygate_commit_at_execution":git_head(),
        "config_sha256":sha256_file(CONFIG),
        "script_sha256":sha256_file(Path(__file__).resolve()),
        "scipy_version":scipy.__version__,
        "primary_events":2701,
        "primary_videos":1170,
        "identity_competitor_defined_videos":
            int(result["identity_defined"].sum()),
        "identity_no_competitor_videos":
            int((~result["identity_defined"]).sum()),
        "unique_identity_video_object_measurements":
            len(result["unique_identity"]),
        "di_components":cfg["di_v1"]["components"],
        "di_weights":cfg["di_v1"]["weights"],
        "percentile_rank":cfg["di_v1"]["percentile_rank"],
        "visual_clustering":cfg["visual_clustering"],
        "visual_cluster_counts":{
            str(i):int(result["cluster_counts"][i])
            for i in range(20)
        },
        "di_score":{
            "min":float(di.min()),
            "median":float(np.median(di)),
            "max":float(di.max()),
            "mean":float(di.mean()),
        },
        "stability_grid":result["stability"],
        "stability_all_pass":True,
        "di_v1_materialized":True,
        "hard_set_selected":False,
        "hard_set_size_frozen":False,
        "final_split_constructed":False,
        "fresh_dev_evaluated":False,
        "test_evaluated":False,
    }

    summary["output_hashes"]={
        "di_video_components.csv":
            sha256_file(COMPONENTS_OUT),
        "visual_clusters.csv":
            sha256_file(CLUSTERS_OUT),
        "component_correlations.csv":
            sha256_file(CORR_OUT),
        "stability_grid.csv":
            sha256_file(STABILITY_OUT),
    }

    write_json_atomic(
        SUMMARY_OUT,
        summary,
    )

    print("primary_videos =",1170)
    print(
        "identity_defined_videos =",
        summary["identity_competitor_defined_videos"],
    )
    print(
        "identity_no_competitor_videos =",
        summary["identity_no_competitor_videos"],
    )
    print(
        "unique_identity_video_object_measurements =",
        summary["unique_identity_video_object_measurements"],
    )
    print("visual_cluster_counts =",summary["visual_cluster_counts"])
    print("di_min =",summary["di_score"]["min"])
    print("di_median =",summary["di_score"]["median"])
    print("di_max =",summary["di_score"]["max"])
    print("stability_all_pass = true")
    print("hard_set_selected = false")
    print("hard_set_size_frozen = false")
    print("fresh_dev_evaluated = false")
    print("test_evaluated = false")
    print("EXP048_RUN_PASS")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["selftest","plan","run"],
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
