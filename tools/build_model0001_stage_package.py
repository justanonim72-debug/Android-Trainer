#!/usr/bin/env python3
"""Build the private recipe-driven Model #0001 Foundation-v3 stage package.

The package is created only after a physical LR pilot has selected the
production recipe. It packages Dataset-v3 train/validation plus frozen
Dataset-v1 validation for retention diagnostics. No test split is read.
"""
from __future__ import annotations
import argparse, hashlib, json, random, zipfile
from pathlib import Path

SOURCE_MODEL_SHA="047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
START_LIFETIME_TOKENS=5_535_744
V3_TRAIN_SHA="19c7d23661aee08d11ac243347d4b943661084f1dc6fa740a222de01ae970975"
V3_VAL_SHA="90dde5eeb7d7934c08213d73bd2d79abd69639c9d00bbfc2bc497137f59096c1"
SEQ=256

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def need(obj,key,typ):
    if key not in obj or not isinstance(obj[key],typ):
        raise SystemExit(f"STOP: recipe missing/invalid {key}")
    return obj[key]

def tokens(path:Path)->int:
    if path.stat().st_size%2: raise SystemExit(f"STOP: odd uint16 file {path}")
    return path.stat().st_size//2

def windows(path:Path)->int:
    return max(0,(tokens(path)-1)//SEQ)

def parse_sums(path:Path)->dict[str,str]:
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        p=line.strip().split(None,1)
        if len(p)==2: out[p[1].strip().lstrip("*")]=p[0].lower()
    return out

def resolve_sum(sums:dict[str,str],basename:str)->str:
    if basename in sums: return sums[basename]
    for name,value in sums.items():
        if Path(name).name==basename: return value
    raise SystemExit(f"STOP: SHA256SUMS missing {basename}")

def sample_indices(count:int,n:int,seed:int)->list[int]:
    n=min(count,n)
    if n<=0: raise SystemExit("STOP: validation has no full windows")
    return sorted(random.Random(seed).sample(range(count),n))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit",required=True)
    ap.add_argument("--dataset-report",required=True)
    ap.add_argument("--recipe",required=True)
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-foundation-v3.atstage")
    args=ap.parse_args()

    audit_path=Path(args.audit).resolve()
    dataset_report_path=Path(args.dataset_report).resolve()
    recipe_path=Path(args.recipe).resolve()
    project=Path(args.project).resolve()
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    dataset=json.loads(dataset_report_path.read_text(encoding="utf-8"))
    recipe=json.loads(recipe_path.read_text(encoding="utf-8"))

    if audit.get("schema")!="model0001_native_transition_audit_v1":
        raise SystemExit("STOP: wrong transition audit")
    source=audit["known"]["source_weights"]
    if source["model_state_sha256"]!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: wrong source model")
    if dataset.get("schema")!="model0001_dataset_v3_report_v1" or dataset.get("status")!="PASS":
        raise SystemExit("STOP: Dataset-v3 report not PASS")
    dg=dataset.get("hard_guards",{})
    if dg.get("test_split_used") is not False or dg.get("training_started") is not False:
        raise SystemExit("STOP: Dataset-v3 hard guard failed")

    train=Path(dataset["train"]["path"]).resolve()
    val=Path(dataset["validation"]["path"]).resolve()
    if not train.is_file() or sha256(train)!=V3_TRAIN_SHA:
        raise SystemExit("STOP: Dataset-v3 train identity mismatch")
    if not val.is_file() or sha256(val)!=V3_VAL_SHA:
        raise SystemExit("STOP: Dataset-v3 validation identity mismatch")

    v1=project/"artifacts"/"model0001_dataset_v1"
    v1val=v1/"validation.bin"
    sums=v1/"SHA256SUMS.txt"
    if not v1val.is_file() or not sums.is_file():
        raise SystemExit("STOP: frozen v1 validation missing")
    v1sha=resolve_sum(parse_sums(sums),"validation.bin")
    if sha256(v1val)!=v1sha:
        raise SystemExit("STOP: frozen v1 validation SHA mismatch")

    if recipe.get("schema")!="model0001_native_stage_recipe_v1":
        raise SystemExit("STOP: wrong recipe schema")
    if recipe.get("source_model_state_sha256")!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: recipe source SHA mismatch")
    if recipe.get("train_bin_sha256")!=V3_TRAIN_SHA:
        raise SystemExit("STOP: recipe train SHA mismatch")
    if recipe.get("validation_bin_sha256")!=V3_VAL_SHA:
        raise SystemExit("STOP: recipe validation SHA mismatch")
    if recipe.get("optimizer_init")!="fresh_zero_moments":
        raise SystemExit("STOP: optimizer_init must be fresh_zero_moments")
    if recipe.get("test_split_used") is not False:
        raise SystemExit("STOP: test split must remain untouched")

    stage=need(recipe,"stage_name",str)
    total=need(recipe,"total_updates",int)
    if stage!="friend_foundation_v3_cpt":
        raise SystemExit("STOP: stage name drift")
    if total<=0 or total>windows(train):
        raise SystemExit("STOP: total_updates must be within one Dataset-v3 pass")
    if need(recipe,"target_tokens_per_update",int)!=SEQ:
        raise SystemExit("STOP: target_tokens_per_update must be 256")
    if need(recipe,"start_lifetime_tokens",int)!=START_LIFETIME_TOKENS:
        raise SystemExit("STOP: lifetime boundary mismatch")

    opt=need(recipe,"optimizer",dict)
    if opt.get("name")!="AdamW" or [float(x) for x in need(opt,"betas",list)]!=[0.9,0.95]:
        raise SystemExit("STOP: AdamW/betas drift")
    if float(opt.get("eps",0))!=1e-8 or float(opt.get("grad_clip",0))!=1.0:
        raise SystemExit("STOP: optimizer epsilon/clip drift")

    sched=need(recipe,"lr_schedule",dict)
    typ=sched.get("type")
    if typ=="constant":
        peak=float(sched.get("lr",0))
        if not (0<peak<=2e-4): raise SystemExit("STOP: constant LR invalid")
    elif typ=="linear_warmup_cosine":
        peak=float(sched.get("peak_lr",0)); minimum=float(sched.get("min_lr",0))
        warm=int(sched.get("warmup_steps",-1))
        if not (0<minimum<=peak<=2e-4 and 0<warm<total):
            raise SystemExit("STOP: warmup/cosine schedule invalid")
    else:
        raise SystemExit("STOP: unsupported schedule")

    order=need(recipe,"sample_order",dict)
    if order.get("type")!="sequential_packed_windows":
        raise SystemExit("STOP: production sample order must be sequential_packed_windows")
    if int(order.get("seed",-1))!=20260903:
        raise SystemExit("STOP: packing/sample-order seed drift")
    for key in ("checkpoint_every","eval_every","log_every"):
        if need(recipe,key,int)<=0: raise SystemExit(f"STOP: {key} must be >0")

    eval_n=int(recipe.get("eval_windows_per_set",64))
    if eval_n<=0 or eval_n>128: raise SystemExit("STOP: eval_windows_per_set invalid")
    eval_seed=int(recipe.get("eval_seed",20260903))

    recipe_bytes=json.dumps(recipe,sort_keys=True,separators=(",",":")).encode()
    recipe_sha=hashlib.sha256(recipe_bytes).hexdigest()
    manifest={
      "schema":"model0001_native_stage_package_v2",
      "stage_name":stage,
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "source_checkpoint_sha256":source["checkpoint_sha256"],
      "recipe":recipe,
      "recipe_sha256":recipe_sha,
      "audit_sha256":sha256(audit_path),
      "dataset_report_sha256":sha256(dataset_report_path),
      "data":{
        "train":{"path":"data/train.bin","sha256":sha256(train),"uint16_tokens":tokens(train),"full_windows":windows(train)},
        "v3_validation":{"path":"data/v3_validation.bin","sha256":sha256(val),"uint16_tokens":tokens(val),"full_windows":windows(val)},
        "v1_validation":{"path":"data/v1_validation.bin","sha256":v1sha,"uint16_tokens":tokens(v1val),"full_windows":windows(v1val)}
      },
      "eval_indices":{
        "v3_validation":sample_indices(windows(val),eval_n,eval_seed+1),
        "v1_validation":sample_indices(windows(v1val),eval_n,eval_seed+2)
      },
      "hard_guards":{"test_split_packaged":False,"dataset_v2_train_bin_packaged":False}
    }

    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr("manifest.json",json.dumps(manifest,sort_keys=True,separators=(",",":")))
        z.write(train,"data/train.bin")
        z.write(val,"data/v3_validation.bin")
        z.write(v1val,"data/v1_validation.bin")
    print(json.dumps({"status":"PASS","output":str(out),"sha256":sha256(out),"recipe_sha256":recipe_sha,"stage_name":stage,"total_updates":total},indent=2))

if __name__=="__main__": main()
