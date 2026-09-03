#!/usr/bin/env python3
"""Build a private recipe-driven Friend-Core F2 SFT stage package.

The package is intentionally recipe-agnostic until the physical masked-SFT LR
pilot locks production hyperparameters. It packages the audited masked F2
train/validation windows plus frozen V3/V1 retention validation. No project or
external test split is read.
"""
from __future__ import annotations

import argparse, hashlib, json, random, zipfile
from pathlib import Path

SOURCE_MODEL_SHA="10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
TOKENIZER_SHA="3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
SOURCE_FOUNDATION_LIFETIME=6_563_072
SEQ=256
WINDOW=257
SEED=20260903

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

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
    if count<=0: raise SystemExit("STOP: zero validation windows")
    n=min(count,n)
    return sorted(random.Random(seed).sample(range(count),n))

def validate_masked(spec:dict):
    tokens=Path(spec["tokens_path"])
    mask=Path(spec["mask_path"])
    windows=int(spec["windows"])
    if not tokens.is_file() or not mask.is_file():
        raise SystemExit("STOP: masked F2 file missing")
    if tokens.stat().st_size != windows*WINDOW*2:
        raise SystemExit("STOP: masked F2 token file size mismatch")
    if mask.stat().st_size != windows*SEQ:
        raise SystemExit("STOP: masked F2 mask file size mismatch")
    if sha256(tokens)!=spec["tokens_sha256"] or sha256(mask)!=spec["mask_sha256"]:
        raise SystemExit("STOP: masked F2 file SHA mismatch")
    return tokens,mask,windows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--source-audit",default="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json")
    ap.add_argument("--recipe",required=True)
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-f2-sft.atsftstage")
    args=ap.parse_args()

    project=Path(args.project).resolve()
    source_audit_path=Path(args.source_audit).resolve()
    recipe_path=Path(args.recipe).resolve()
    if not source_audit_path.is_file() or not recipe_path.is_file():
        raise SystemExit("STOP: F2 source audit or recipe missing")

    source_audit=json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status")!="PASS" or source_audit.get("model_state_sha256")!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: wrong promoted Foundation-v3 source")
    if source_audit.get("optimizer_moments_carried_forward") is not False:
        raise SystemExit("STOP: F2 source must not carry old optimizer moments")

    pack_path=project/"artifacts"/"model0001_f2_sft"/"F2_SFT_PACK_REPORT.json"
    if not pack_path.is_file():
        raise SystemExit("STOP: audited F2 pack report missing")
    pack=json.loads(pack_path.read_text(encoding="utf-8"))
    if pack.get("status")!="PASS" or pack.get("objective")!="assistant_content_only_cross_entropy":
        raise SystemExit("STOP: F2 pack is not assistant-only PASS")
    if pack.get("tokenizer_sha256")!=TOKENIZER_SHA:
        raise SystemExit("STOP: tokenizer identity drift")
    guards=pack.get("hard_guards",{})
    if guards.get("assistant_only_loss") is not True or guards.get("test_split_used") is not False:
        raise SystemExit("STOP: F2 pack hard guard failed")
    protocol_fraction=float(pack.get("train_deterministic_protocol_scored_fraction",-1))
    if not (0.08<=protocol_fraction<=0.15):
        raise SystemExit("STOP: F2 protocol scored fraction outside lock")

    train_tokens,train_mask,train_windows=validate_masked(pack["train"])
    val_tokens,val_mask,val_windows=validate_masked(pack["validation"])

    v3=project/"artifacts"/"model0001_dataset_v3"/"validation.bin"
    v3_report_path=project/"artifacts"/"model0001_dataset_v3"/"DATASET_V3_REPORT.json"
    if not v3.is_file() or not v3_report_path.is_file():
        raise SystemExit("STOP: frozen V3 validation missing")
    v3_report=json.loads(v3_report_path.read_text(encoding="utf-8"))
    if sha256(v3)!=v3_report["validation"]["sha256"]:
        raise SystemExit("STOP: V3 validation SHA mismatch")
    v3_tokens=v3.stat().st_size//2
    v3_windows=max(0,(v3_tokens-1)//SEQ)

    v1root=project/"artifacts"/"model0001_dataset_v1"
    v1=v1root/"validation.bin"; sums=v1root/"SHA256SUMS.txt"
    if not v1.is_file() or not sums.is_file():
        raise SystemExit("STOP: frozen V1 validation missing")
    v1sha=resolve_sum(parse_sums(sums),"validation.bin")
    if sha256(v1)!=v1sha:
        raise SystemExit("STOP: V1 validation SHA mismatch")
    v1_tokens=v1.stat().st_size//2
    v1_windows=max(0,(v1_tokens-1)//SEQ)

    recipe=json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("schema")!="model0001_f2_sft_stage_recipe_v1":
        raise SystemExit("STOP: wrong F2 recipe schema")
    if recipe.get("source_model_state_sha256")!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: F2 recipe source SHA mismatch")
    if recipe.get("stage_name")!="friend_f2_sft":
        raise SystemExit("STOP: F2 stage-name drift")
    if recipe.get("objective")!="assistant_content_only_cross_entropy":
        raise SystemExit("STOP: F2 recipe objective drift")
    if recipe.get("optimizer_init")!="fresh_zero_moments":
        raise SystemExit("STOP: F2 optimizer init must be fresh-zero")
    if recipe.get("source_foundation_lifetime_tokens")!=SOURCE_FOUNDATION_LIFETIME:
        raise SystemExit("STOP: F2 source lifetime boundary mismatch")
    if recipe.get("test_split_used") is not False:
        raise SystemExit("STOP: F2 recipe must not touch test split")

    total=int(recipe.get("total_updates",0))
    max_epochs=int(recipe.get("max_epochs",0))
    if total<=0 or max_epochs<1 or max_epochs>3 or total>train_windows*max_epochs:
        raise SystemExit("STOP: F2 total_updates/max_epochs invalid")
    order=recipe.get("sample_order",{})
    if order.get("type")!="sequential_masked_windows" or int(order.get("seed",-1))!=SEED:
        raise SystemExit("STOP: F2 sample-order contract drift")
    opt=recipe.get("optimizer",{})
    if opt.get("name")!="AdamW" or [float(x) for x in opt.get("betas",[])]!=[0.9,0.95]:
        raise SystemExit("STOP: F2 AdamW/betas drift")
    if float(opt.get("eps",0))!=1e-8 or float(opt.get("grad_clip",0))!=1.0:
        raise SystemExit("STOP: F2 AdamW eps/clip drift")
    for key in ("checkpoint_every","eval_every","log_every"):
        if int(recipe.get(key,0))<=0:
            raise SystemExit(f"STOP: F2 {key} must be >0")

    sched=recipe.get("lr_schedule",{})
    typ=sched.get("type")
    if typ=="constant":
        lr=float(sched.get("lr",0))
        if not (1e-6<=lr<=5e-5): raise SystemExit("STOP: F2 constant LR invalid")
    elif typ=="linear_warmup_cosine":
        peak=float(sched.get("peak_lr",0)); minimum=float(sched.get("min_lr",0))
        warm=int(sched.get("warmup_steps",-1))
        if not (1e-6<=minimum<=peak<=5e-5 and 0<warm<total):
            raise SystemExit("STOP: F2 warmup/cosine schedule invalid")
    else:
        raise SystemExit("STOP: unsupported F2 LR schedule")

    eval_n=int(recipe.get("eval_windows_per_set",64))
    if eval_n<=0 or eval_n>128:
        raise SystemExit("STOP: F2 eval_windows_per_set invalid")
    eval_seed=int(recipe.get("eval_seed",SEED))

    recipe_bytes=json.dumps(recipe,sort_keys=True,separators=(",",":")).encode()
    recipe_sha=hashlib.sha256(recipe_bytes).hexdigest()
    manifest={
      "schema":"model0001_f2_sft_stage_package_v1",
      "stage_name":"friend_f2_sft",
      "objective":"assistant_content_only_cross_entropy",
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "recipe":recipe,
      "recipe_sha256":recipe_sha,
      "source_bundle_audit_sha256":sha256(source_audit_path),
      "f2_pack_report_sha256":sha256(pack_path),
      "data":{
        "sft_train":{
          "tokens_path":"data/f2_train.tokens.u16",
          "tokens_sha256":sha256(train_tokens),
          "mask_path":"data/f2_train.lossmask.u8",
          "mask_sha256":sha256(train_mask),
          "windows":train_windows,
          "tokens_per_window":WINDOW,
          "mask_targets_per_window":SEQ
        },
        "sft_validation":{
          "tokens_path":"data/f2_validation.tokens.u16",
          "tokens_sha256":sha256(val_tokens),
          "mask_path":"data/f2_validation.lossmask.u8",
          "mask_sha256":sha256(val_mask),
          "windows":val_windows,
          "tokens_per_window":WINDOW,
          "mask_targets_per_window":SEQ
        },
        "v3_validation":{
          "path":"data/v3_validation.bin","sha256":sha256(v3),
          "uint16_tokens":v3_tokens,"full_windows":v3_windows
        },
        "v1_validation":{
          "path":"data/v1_validation.bin","sha256":v1sha,
          "uint16_tokens":v1_tokens,"full_windows":v1_windows
        }
      },
      "eval_indices":{
        "sft_validation":sample_indices(val_windows,eval_n,eval_seed+21),
        "v3_validation":sample_indices(v3_windows,eval_n,eval_seed+22),
        "v1_validation":sample_indices(v1_windows,eval_n,eval_seed+23)
      },
      "hard_guards":{
        "assistant_only_loss":True,
        "test_split_packaged":False,
        "dataset_v2_train_bin_packaged":False,
        "foundation_v3_train_bin_packaged":False
      }
    }

    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr("manifest.json",json.dumps(manifest,sort_keys=True,separators=(",",":")))
        z.write(train_tokens,"data/f2_train.tokens.u16")
        z.write(train_mask,"data/f2_train.lossmask.u8")
        z.write(val_tokens,"data/f2_validation.tokens.u16")
        z.write(val_mask,"data/f2_validation.lossmask.u8")
        z.write(v3,"data/v3_validation.bin")
        z.write(v1,"data/v1_validation.bin")

    print(json.dumps({
      "status":"PASS",
      "schema":"model0001_f2_sft_stage_package_build_v1",
      "output":str(out),
      "sha256":sha256(out),
      "recipe_sha256":recipe_sha,
      "total_updates":total,
      "max_epochs":max_epochs,
      "train_windows":train_windows,
      "test_split_used":False
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
