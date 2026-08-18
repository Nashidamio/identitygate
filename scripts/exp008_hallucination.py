"""EXP008 - Render RAW | GT | PREDICTION for hallucination frames."""
import os, json, torch, numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

REPO=os.path.expanduser("~/thesis/identitygate")
ROOT="/mnt/d/thesis_data/mosev2/train"; VID="6042d64a"
JPG=f"{ROOT}/JPEGImages/{VID}"; ANN=f"{ROOT}/Annotations/{VID}"
OUT=f"{REPO}/experiments/EXP008_{VID}_hallucinations"; os.makedirs(OUT, exist_ok=True)
COLS={1:(255,40,40), 2:(40,150,255)}

# find hallucination frames from EXP007
d=json.load(open(f"{REPO}/experiments/EXP007_{VID}.json"))
targets=[]
for r in d["per_frame"]:
    for oid in (1,2):
        g,p = r.get(f"obj{oid}_gt_px"), r.get(f"obj{oid}_pred_px")
        if g==0 and p and p>0:
            targets.append((r["frame"], oid, p))
print("HALLUCINATION FRAMES (GT absent, prediction present):")
for f,o,p in targets: print(f"   frame {f:3d}  obj {o}  predicted {p:,} px  (GT = 0)")
want=sorted({f for f,_,_ in targets})

from sam3.model_builder import build_sam3_video_model
print("\nbuilding model + re-running to capture predictions...", flush=True)
m=build_sam3_video_model(); pr=m.tracker; pr.backbone=m.detector.backbone
st=pr.init_state(video_path=JPG); pr.clear_all_points_in_video(st)
gt0=np.array(Image.open(f"{ANN}/00000.png"))
for oid in [int(o) for o in np.unique(gt0) if o]:
    pr.add_new_mask(inference_state=st, frame_idx=0, obj_id=oid,
                    mask=torch.from_numpy((gt0==oid).astype(np.uint8)).to(torch.bool))

def paint(img, mask, col):
    e = binary_dilation(mask, iterations=5) & ~mask
    o=img.copy(); o[e]=col; o[mask]=(0.5*o[mask]+0.5*np.array(col)).astype(np.uint8); return o

nf=len(sorted(os.listdir(JPG)))
for out in pr.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=nf,
                                 reverse=False,propagate_preflight=True):
    fi,ids,_,vres = out[0],out[1],out[2],out[3]
    if fi not in want: continue
    img=np.array(Image.open(f"{JPG}/{fi:05d}.jpg").convert("RGB"))
    gt=np.array(Image.open(f"{ANN}/{fi:05d}.png")); H,W=img.shape[:2]
    gtv=img.copy()
    for oid in [int(o) for o in np.unique(gt) if o]: gtv=paint(gtv,gt==oid,COLS.get(oid,(0,255,0)))
    pv=img.copy()
    for i,oid in enumerate(ids):
        pm=(vres[i,0]>0).detach().cpu().numpy()
        if pm.shape!=(H,W):
            pm=np.array(Image.fromarray((pm*255).astype(np.uint8)).resize((W,H)))>127
        if pm.sum(): pv=paint(pv,pm,COLS.get(int(oid),(0,255,0)))
    c=Image.new("RGB",(W*3+40,H),(15,15,15))
    for k,arr in enumerate([img,gtv,pv]): c.paste(Image.fromarray(arr),(k*(W+20),0))
    dr=ImageDraw.Draw(c)
    gi=[int(o) for o in np.unique(gt) if o]
    dr.text((10,10),f"RAW frame {fi}",fill=(255,255,0))
    dr.text((W+30,10),f"GROUND TRUTH  objects present={gi}",fill=(0,255,0))
    dr.text((2*W+50,10),f"SAM 3.1 PREDICTION  <-- hallucination if GT empty",fill=(255,80,80))
    c.save(f"{OUT}/hallu_{fi:03d}.png")
    print(f"  rendered frame {fi}: GT={gi}", flush=True)
print("\nsaved ->", OUT)
