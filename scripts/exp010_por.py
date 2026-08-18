"""
EXP010 - vanilla SAM 3.1 (B0) POR/ITR on the DEV hard stratum.
Plan section 17 headroom checkpoint. Sole primary endpoint per section 14.

Protocol:
  - prompt every frame-0 GT object (anchor policy (a), verified in EXP009)
  - propagate once; record per-object per-frame IoU, pred_px, gt_px, cross-IoU
  - derive events with the SAME rule as count_events.py (gap >= 5 after first appearance)
  - POR: recovered if IoU(pred,target_GT) > 0.5 within first W *evaluable* frames
    after reappearance. Evaluable = GT-visible, truncated at next qualifying gap.
  - windows 15 / 30 / 60 (30 = primary)
  - ITR: theft frame = IoU(pred,target)<0.3 AND max IoU(pred,other)>0.5;
    theft event = >= 5 consecutive theft frames. Denominator NOT pre-registered:
    all candidates reported.
Usage: python scripts/exp010_por.py sanity | python scripts/exp010_por.py full
"""
import os, sys, json, time
import numpy as np, torch
from PIL import Image

REPO = os.path.expanduser("~/thesis/identitygate")
ROOT = "/mnt/d/thesis_data/mosev2/train"
MODE = sys.argv[1] if len(sys.argv) > 1 else "sanity"
N_EVENT, WINDOWS, IOU_REC = 5, (15, 30, 60), 0.5
THEFT_SELF, THEFT_OTHER, THEFT_RUN = 0.3, 0.5, 5
OUT = f"{REPO}/experiments/EXP010_{MODE}"
os.makedirs(OUT, exist_ok=True)

man = json.load(open(f"{REPO}/experiments/EXP009_dev_manifest.json"))
dev = man["dev_videos"]
lens = {v: len([f for f in os.listdir(f"{ROOT}/Annotations/{v}") if f.endswith(".png")])
        for v in dev}
srt = sorted(dev, key=lambda v: lens[v])
print(f"DEV frame counts: min={lens[srt[0]]} med={lens[srt[len(srt)//2]]} max={lens[srt[-1]]}")
vids = [srt[0], srt[len(srt)//2], srt[-1]] if MODE == "sanity" else srt
print(f"MODE={MODE}  running {len(vids)} videos: {[(v, lens[v]) for v in vids]}\n", flush=True)

from sam3.model_builder import build_sam3_video_model
print("building model...", flush=True)
m = build_sam3_video_model(); pred_or = m.tracker
pred_or.backbone = m.detector.backbone


def events_for_track(vis):
    """IDENTICAL rule to scripts/count_events.py. Returns list of reappearance frame idx."""
    reap, run, seen = [], 0, False
    for t, v in enumerate(vis):
        if v:
            if run >= N_EVENT and seen:
                reap.append(t)
            run = 0; seen = True
        else:
            if seen: run += 1
    return reap


def score_track(vis, ious, reap):
    """POR per window + theft events. vis/ious indexed by frame."""
    n = len(vis)
    gap_starts = []          # start frame of every qualifying gap
    run, seen = 0, False
    for t, v in enumerate(vis):
        if v:
            run = 0; seen = True
        else:
            if seen:
                run += 1
                if run == N_EVENT: gap_starts.append(t - N_EVENT + 1)
    res = {}
    for W in WINDOWS:
        rec = []
        for r in reap:
            stop = next((g for g in gap_starts if g > r), n)   # truncate at next qualifying gap
            ev, hit = 0, 0
            for t in range(r, stop):
                if not vis[t]:      continue                    # non-evaluable
                ev += 1
                if ious[t] is not None and ious[t] > IOU_REC: hit = 1; break
                if ev >= W: break
            rec.append(hit)
        res[f"por_w{W}"] = rec
    return res


rows, per_video, t_all = [], [], time.time()
for vi, vid in enumerate(vids):
    JPG, ANN = f"{ROOT}/JPEGImages/{vid}", f"{ROOT}/Annotations/{vid}"
    pngs = sorted(f for f in os.listdir(ANN) if f.endswith(".png"))
    jpgs = sorted(f for f in os.listdir(JPG) if f.endswith(".jpg"))
    assert len(pngs) == len(jpgs), f"{vid}: {len(jpgs)} jpg vs {len(pngs)} png"
    assert [os.path.splitext(a)[0] for a in pngs] == [os.path.splitext(b)[0] for b in jpgs], \
        f"{vid}: frame/mask stems misaligned"
    nf = len(pngs)

    gt0 = np.array(Image.open(os.path.join(ANN, pngs[0])))
    oids = sorted(int(x) for x in np.unique(gt0) if x != 0)

    st = pred_or.init_state(video_path=JPG)
    pred_or.clear_all_points_in_video(st)
    for oid in oids:
        pred_or.add_new_mask(inference_state=st, frame_idx=0, obj_id=oid,
                             mask=torch.from_numpy((gt0 == oid).astype(np.uint8)).to(torch.bool))

    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    vis  = {o: [False]*nf for o in oids}
    iou  = {o: [None]*nf for o in oids}
    xiou = {o: [0.0]*nf for o in oids}
    for out in pred_or.propagate_in_video(st, start_frame_idx=0, max_frame_num_to_track=nf,
                                          reverse=False, propagate_preflight=True):
        fi, ids, _, vres = int(out[0]), out[1], out[2], out[3]
        gt = np.array(Image.open(os.path.join(ANN, pngs[fi])))
        gmask = {o: (gt == o) for o in oids}
        for i, oid in enumerate(ids):
            oid = int(oid)
            if oid not in gmask: continue
            p = (vres[i, 0] > 0).detach().cpu().numpy()
            assert p.shape == gt.shape, f"{vid} f{fi}: pred {p.shape} vs gt {gt.shape}"
            g = gmask[oid]; vis[oid][fi] = bool(g.sum())
            u = np.logical_or(p, g).sum()
            iou[oid][fi] = float(np.logical_and(p, g).sum() / u) if u else None
            best = 0.0
            for o2 in oids:
                if o2 == oid: continue
                g2 = gmask[o2]; u2 = np.logical_or(p, g2).sum()
                if u2: best = max(best, float(np.logical_and(p, g2).sum() / u2))
            xiou[oid][fi] = best
    dt, pk = time.time()-t0, torch.cuda.max_memory_allocated()/1024**3

    nev = 0
    for oid in oids:
        reap = events_for_track(vis[oid])
        if not reap: continue
        sc = score_track(vis[oid], iou[oid], reap)
        theft = [1 if (iou[oid][t] is not None and iou[oid][t] < THEFT_SELF
                       and xiou[oid][t] > THEFT_OTHER) else 0 for t in range(nf)]
        run = ev = 0
        for x in theft:
            run = run+1 if x else 0
            if run == THEFT_RUN: ev += 1
        nev += len(reap)
        rows.append({"video": vid, "object_id": oid, "n_frames": nf,
                     "n_events": len(reap), "reappear_frames": reap,
                     **{k: v for k, v in sc.items()},
                     "theft_events": ev,
                     "theft_frames": int(sum(theft)),
                     "visible_frames": int(sum(vis[oid]))})
    per_video.append({"video": vid, "n_frames": nf, "n_objects": len(oids),
                      "events": nev, "sec": round(dt,1), "peak_vram_gb": round(pk,2)})
    print(f"[{vi+1}/{len(vids)}] {vid}  {nf}f {len(oids)}obj  {nev}ev  "
          f"{dt:.0f}s  {pk:.2f}GB", flush=True)
    del st; torch.cuda.empty_cache()

tot = {f"por_w{W}": [h for r in rows for h in r[f"por_w{W}"]] for W in WINDOWS}
summary = {"mode": MODE, "videos": len(vids), "tracks": len(rows),
           "events": sum(r["n_events"] for r in rows),
           "theft_events": sum(r["theft_events"] for r in rows),
           "tracks_with_theft": sum(1 for r in rows if r["theft_events"] > 0),
           "runtime_min": round((time.time()-t_all)/60, 1),
           "max_peak_vram_gb": max(v["peak_vram_gb"] for v in per_video)}
for W in WINDOWS:
    h = tot[f"por_w{W}"]
    summary[f"POR_w{W}"] = round(sum(h)/len(h), 4) if h else None
    summary[f"n_events_w{W}"] = len(h)

json.dump({"summary": summary, "per_video": per_video, "per_track": rows},
          open(f"{OUT}/results.json", "w"), indent=2)
print("\n=== SUMMARY ===")
for k, v in summary.items(): print(f"  {k:22s}: {v}")
print(f"\n-> {OUT}/results.json")
