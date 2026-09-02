#!/usr/bin/env python3
"""Build a private Model #0001 Foundation-v3 LR-pilot package.

The package contains DATA only (v3 train/v3 validation/v1 validation) plus a
deterministic pilot protocol. Canonical weights remain in the existing local
.atb bundle. No test split is read or packaged.
"""
from __future__ import annotations
import argparse, hashlib, json, random, zipfile
from pathlib import Path

MODEL_SHA="047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
V3_TRAIN_SHA="19c7d23661aee08d11ac243347d4b943661084f1dc6fa740a222de01ae970975"
V3_VAL_SHA="90dde5eeb7d7934c08213d73bd2d79abd69639c9d00bbfc2bc497137f59096c1"
SEQ=256
SEED=20260903
TRAIN_STEPS=96
WARMUP_STEPS=3
EVAL_WINDOWS=24
LR_CANDIDATES=[5e-5,1e-4,2e-4]

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def u16_tokens(p:Path)->int:
    n=p.stat().st_size
    if n%2: raise SystemExit(f"STOP: odd uint16 byte count: {p}")
    return n//2

def full_windows(p:Path)->int:
    return max(0,(u16_tokens(p)-1)//SEQ)

def parse_sums(path:Path)->dict[str,str]:
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts=line.strip().split(None,1)
        if len(parts)==2: out[parts[1].strip().lstrip("*")]=parts[0].lower()
    return out

def sample_indices(count:int,n:int,seed:int)->list[int]:
    if count<=0: raise SystemExit("STOP: zero windows")
    n=min(n,count)
    r=random.Random(seed)
    return sorted(r.sample(range(count),n))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-v3-lr-pilot.atpilot")
    args=ap.parse_args()
    project=Path(args.project).resolve()

    v3=project/"artifacts"/"model0001_dataset_v3"
    report=v3/"DATASET_V3_REPORT.json"
    train=v3/"train.bin"
    val=v3/"validation.bin"
    for p in (report,train,val):
        if not p.is_file(): raise SystemExit(f"STOP: missing {p}")
    r=json.loads(report.read_text(encoding="utf-8"))
    if r.get("status")!="PASS": raise SystemExit("STOP: Dataset-v3 report not PASS")
    if sha256(train)!=V3_TRAIN_SHA or sha256(val)!=V3_VAL_SHA:
        raise SystemExit("STOP: Dataset-v3 identity mismatch")
    if r["hard_guards"].get("test_split_used") is not False:
        raise SystemExit("STOP: Dataset-v3 test guard failed")

    v1=project/"artifacts"/"model0001_dataset_v1"
    v1_val=v1/"validation.bin"
    sums=v1/"SHA256SUMS.txt"
    if not v1_val.is_file() or not sums.is_file():
        raise SystemExit("STOP: frozen Dataset-v1 validation evidence missing")
    listed=parse_sums(sums)
    expected=listed.get("validation.bin")
    if not expected:
        raise SystemExit("STOP: v1 SHA256SUMS has no validation.bin")
    got=sha256(v1_val)
    if got!=expected:
        raise SystemExit("STOP: v1 validation.bin SHA mismatch")

    tw=full_windows(train)
    vv=full_windows(val)
    v1w=full_windows(v1_val)
    if tw<TRAIN_STEPS:
        raise SystemExit("STOP: v3 train too small for pilot")

    manifest={
      "schema":"model0001_v3_lr_pilot_v1",
      "status":"LOCKED",
      "source_model_state_sha256":MODEL_SHA,
      "stage_objective":"friend_foundation_v3_cpt",
      "target_tokens_per_update":SEQ,
      "optimizer_init":"fresh_zero_moments_each_candidate",
      "optimizer":{"name":"AdamW","betas":[0.9,0.95],"eps":1e-8,"grad_clip":1.0},
      "protocol":{
        "seed":SEED,
        "lr_candidates":LR_CANDIDATES,
        "train_steps_per_candidate":TRAIN_STEPS,
        "warmup_steps":WARMUP_STEPS,
        "schedule":"linear_warmup_then_constant_for_pilot_only",
        "v3_eval_windows":EVAL_WINDOWS,
        "v1_eval_windows":EVAL_WINDOWS,
        "same_train_windows_for_all_candidates":True,
        "production_lr_locked":False
      },
      "indices":{
        "train":sample_indices(tw,TRAIN_STEPS,SEED+1),
        "v3_validation":sample_indices(vv,EVAL_WINDOWS,SEED+2),
        "v1_validation":sample_indices(v1w,EVAL_WINDOWS,SEED+3)
      },
      "data":{
        "v3_train":{"path":"data/v3_train.bin","sha256":sha256(train),"uint16_tokens":u16_tokens(train),"full_windows":tw},
        "v3_validation":{"path":"data/v3_validation.bin","sha256":sha256(val),"uint16_tokens":u16_tokens(val),"full_windows":vv},
        "v1_validation":{"path":"data/v1_validation.bin","sha256":got,"uint16_tokens":u16_tokens(v1_val),"full_windows":v1w}
      },
      "hard_guards":{
        "test_split_packaged":False,
        "dataset_v2_train_bin_packaged":False,
        "gate_lr_is_production_lr":False,
        "candidate_runs_reset_source_weights":True,
        "candidate_runs_reset_adam_moments":True
      }
    }

    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr("manifest.json",json.dumps(manifest,sort_keys=True,separators=(",",":")))
        z.write(train,"data/v3_train.bin")
        z.write(val,"data/v3_validation.bin")
        z.write(v1_val,"data/v1_validation.bin")
    result={"status":"PASS","output":str(out),"sha256":sha256(out),"manifest":manifest}
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__": main()
