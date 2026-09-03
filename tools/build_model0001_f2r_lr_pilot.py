#!/usr/bin/env python3
"""Build physical-phone LR pilot for Model #0001 F2R replacement SFT.

Starts every candidate from the promoted Foundation-v3 source with fresh-zero
AdamW moments. The old collapsed F2 checkpoint is never loaded.

The repair pack is record-isolated. Pilot train and SFT-validation indices are
stratified by measured active assistant targets so the behavior-core share is
close to the locked 27% training objective rather than accidentally vanishing
from a small random sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

SOURCE_MODEL_SHA="10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
SEQ=256
WINDOW=257
SEED=20260903
TRAIN_STEPS=96
WARMUP_STEPS=3
SFT_EVAL_WINDOWS=24
RETENTION_EVAL_WINDOWS=24
LR_CANDIDATES=[5e-6,1e-5,2e-5]
TARGET_CORE_FRACTION=0.27

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def parse_sums(path:Path)->dict[str,str]:
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts=line.strip().split(None,1)
        if len(parts)==2:
            out[parts[1].strip().lstrip("*")]=parts[0].lower()
    return out

def resolve_sum(sums:dict[str,str],basename:str)->str:
    if basename in sums: return sums[basename]
    for name,value in sums.items():
        if Path(name).name==basename: return value
    raise SystemExit(f"STOP: SHA256SUMS missing {basename}")

def sample_indices(count:int,n:int,seed:int)->list[int]:
    if count<=0: raise SystemExit("STOP: zero candidate windows")
    n=min(count,n)
    return sorted(random.Random(seed).sample(range(count),n))

def active_counts(mask:Path,windows:int)->list[int]:
    raw=mask.read_bytes()
    if len(raw)!=windows*SEQ:
        raise SystemExit(f"STOP: mask size mismatch {mask}")
    out=[]
    for i in range(windows):
        c=sum(raw[i*SEQ:(i+1)*SEQ])
        if c<=0 or c>SEQ:
            raise SystemExit("STOP: invalid active-target count")
        out.append(c)
    return out

def stratified_indices(rows:list[dict],counts:list[int],n:int,seed:int):
    if len(rows)!=len(counts):
        raise SystemExit("STOP: F2R record/window mapping mismatch")
    core=[i for i,r in enumerate(rows)
          if r.get("source_family")=="repair_behavior_core"]
    human=[i for i,r in enumerate(rows)
           if r.get("source_family") in
              ("repair_human_natural","repair_human_task")]
    if not core or not human:
        raise SystemExit("STOP: F2R stratification families missing")
    rng=random.Random(seed)
    rng.shuffle(core); rng.shuffle(human)

    best=None
    for nc in range(max(1,n-len(human)),min(n-1,len(core))+1):
        nh=n-nc
        if nh>len(human): continue
        ci=core[:nc]; hi=human[:nh]
        core_tokens=sum(counts[i] for i in ci)
        total=core_tokens+sum(counts[i] for i in hi)
        frac=core_tokens/max(1,total)
        score=abs(frac-TARGET_CORE_FRACTION)
        candidate=(score,sorted(ci+hi),frac,nc,nh,total)
        if best is None or candidate[0]<best[0]:
            best=candidate
    if best is None:
        raise SystemExit("STOP: cannot stratify F2R pilot indices")
    _,indices,frac,nc,nh,total=best
    return indices,{
      "behavior_core_windows":nc,
      "human_windows":nh,
      "scored_assistant_tokens":total,
      "behavior_core_scored_fraction":frac,
    }

def sft_spec(tokens:Path,mask:Path,windows:int,tokens_name:str,mask_name:str):
    if tokens.stat().st_size!=windows*WINDOW*2:
        raise SystemExit("STOP: F2R token-window file size mismatch")
    if mask.stat().st_size!=windows*SEQ:
        raise SystemExit("STOP: F2R mask-window file size mismatch")
    return {
      "tokens_path":tokens_name,
      "tokens_sha256":sha256(tokens),
      "mask_path":mask_name,
      "mask_sha256":sha256(mask),
      "windows":windows,
      "tokens_per_window":WINDOW,
      "mask_targets_per_window":SEQ,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--source-audit",default="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-f2r-lr-pilot.atsftpilot")
    ap.add_argument("--report-output",default="/storage/emulated/0/Download/model0001-f2r-lr-pilot-package-report.json")
    args=ap.parse_args()

    project=Path(args.project).resolve()
    source_audit_path=Path(args.source_audit).resolve()
    if not source_audit_path.is_file():
        raise SystemExit("STOP: promoted Foundation-v3 source audit missing")
    source_audit=json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status")!="PASS" or source_audit.get("model_state_sha256")!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: promoted Foundation-v3 source identity mismatch")
    if source_audit.get("optimizer_moments_carried_forward") is not False:
        raise SystemExit("STOP: F2R source must start fresh-zero")

    root=project/"artifacts"/"model0001_f2r_repair"
    pack_path=root/"F2R_PACK_REPORT.json"
    source_path=project/"data"/"f2r_repair"/"friend_f2r_repair_source.jsonl"
    source_report_path=project/"data"/"f2r_repair"/"F2R_SOURCE_REPORT.json"
    for p in (pack_path,source_path,source_report_path):
        if not p.is_file(): raise SystemExit(f"STOP: F2R input missing {p}")
    pack=json.loads(pack_path.read_text(encoding="utf-8"))
    if pack.get("schema")!="model0001_f2r_repair_pack_report_v1" or pack.get("status")!="PASS":
        raise SystemExit("STOP: F2R pack report not PASS")
    guards=pack.get("hard_guards",{})
    if guards.get("assistant_only_loss") is not True:
        raise SystemExit("STOP: F2R assistant-only loss guard missing")
    if guards.get("record_isolated_packing") is not True or int(guards.get("cross_record_windows",-1))!=0:
        raise SystemExit("STOP: F2R record-isolated packing guard failed")
    if guards.get("behavior_eval_prompts_used_for_train") is not False:
        raise SystemExit("STOP: F2R behavior-eval prompts leaked into train")
    if guards.get("test_split_used") is not False:
        raise SystemExit("STOP: F2R test split was used")

    source_rows=[]
    with source_path.open("r",encoding="utf-8") as f:
        for raw in f:
            if raw.strip(): source_rows.append(json.loads(raw))
    train_rows=[r for r in source_rows if r["split"]=="train"]
    val_rows=[r for r in source_rows if r["split"]=="validation"]

    train_tokens=Path(pack["train"]["tokens_path"])
    train_mask=Path(pack["train"]["mask_path"])
    val_tokens=Path(pack["validation"]["tokens_path"])
    val_mask=Path(pack["validation"]["mask_path"])
    train_windows=int(pack["train"]["windows"])
    val_windows=int(pack["validation"]["windows"])
    if pack["train"].get("multiwindow_record_windows")!=0 or pack["validation"].get("multiwindow_record_windows")!=0:
        raise SystemExit("STOP: F2R pilot mapping requires one window per record")
    if len(train_rows)!=train_windows or len(val_rows)!=val_windows:
        raise SystemExit("STOP: F2R row/window one-to-one contract drift")
    if train_windows<TRAIN_STEPS or val_windows<SFT_EVAL_WINDOWS:
        raise SystemExit("STOP: F2R pack too small for pilot")

    train_counts=active_counts(train_mask,train_windows)
    val_counts=active_counts(val_mask,val_windows)
    train_indices,train_mix=stratified_indices(
        train_rows,train_counts,TRAIN_STEPS,SEED+31)
    val_indices,val_mix=stratified_indices(
        val_rows,val_counts,SFT_EVAL_WINDOWS,SEED+32)

    v3=project/"artifacts"/"model0001_dataset_v3"/"validation.bin"
    v3_report=project/"artifacts"/"model0001_dataset_v3"/"DATASET_V3_REPORT.json"
    if not v3.is_file() or not v3_report.is_file():
        raise SystemExit("STOP: frozen V3 validation missing")
    v3r=json.loads(v3_report.read_text(encoding="utf-8"))
    if sha256(v3)!=v3r["validation"]["sha256"]:
        raise SystemExit("STOP: V3 validation identity mismatch")
    v3_windows=int(v3r["validation"]["full_256_target_windows"])

    v1root=project/"artifacts"/"model0001_dataset_v1"
    v1=v1root/"validation.bin"; sums=v1root/"SHA256SUMS.txt"
    if not v1.is_file() or not sums.is_file():
        raise SystemExit("STOP: frozen V1 validation missing")
    v1sha=resolve_sum(parse_sums(sums),"validation.bin")
    if sha256(v1)!=v1sha:
        raise SystemExit("STOP: V1 validation identity mismatch")
    v1_tokens=v1.stat().st_size//2
    v1_windows=max(0,(v1_tokens-1)//SEQ)

    manifest={
      "schema":"model0001_f2r_lr_pilot_v1",
      "status":"LOCKED",
      "stage_objective":"friend_f2r_repair_sft",
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "objective":"assistant_content_only_cross_entropy",
      "source_policy":"RETRAIN_FROM_FOUNDATION_V3_NOT_F2_FINAL",
      "optimizer_init":"fresh_zero_moments_each_candidate",
      "optimizer":{"name":"AdamW","betas":[0.9,0.95],"eps":1e-8,"grad_clip":1.0},
      "protocol":{
        "seed":SEED,
        "lr_candidates":LR_CANDIDATES,
        "train_steps_per_candidate":TRAIN_STEPS,
        "warmup_steps":WARMUP_STEPS,
        "schedule":"linear_warmup_then_constant_for_pilot_only",
        "sft_eval_windows":SFT_EVAL_WINDOWS,
        "v3_eval_windows":RETENTION_EVAL_WINDOWS,
        "v1_eval_windows":RETENTION_EVAL_WINDOWS,
        "same_train_windows_for_all_candidates":True,
        "train_mix":train_mix,
        "sft_validation_mix":val_mix,
        "production_lr_locked":False
      },
      "indices":{
        "train":train_indices,
        "sft_validation":val_indices,
        "v3_validation":sample_indices(v3_windows,RETENTION_EVAL_WINDOWS,SEED+33),
        "v1_validation":sample_indices(v1_windows,RETENTION_EVAL_WINDOWS,SEED+34)
      },
      "data":{
        "sft_train":sft_spec(
            train_tokens,train_mask,train_windows,
            "data/f2r_train.tokens.u16","data/f2r_train.lossmask.u8"),
        "sft_validation":sft_spec(
            val_tokens,val_mask,val_windows,
            "data/f2r_validation.tokens.u16","data/f2r_validation.lossmask.u8"),
        "v3_validation":{
          "path":"data/v3_validation.bin","sha256":sha256(v3),
          "uint16_tokens":v3.stat().st_size//2,"full_windows":v3_windows
        },
        "v1_validation":{
          "path":"data/v1_validation.bin","sha256":v1sha,
          "uint16_tokens":v1_tokens,"full_windows":v1_windows
        }
      },
      "evidence":{
        "source_bundle_audit_sha256":sha256(source_audit_path),
        "f2r_source_report_sha256":sha256(source_report_path),
        "f2r_pack_report_sha256":sha256(pack_path),
      },
      "hard_guards":{
        "assistant_only_loss":True,
        "record_isolated_packing":True,
        "cross_record_windows":0,
        "behavior_eval_prompts_used_for_train":False,
        "old_f2_validation_used_for_train":False,
        "test_split_packaged":False,
        "dataset_v2_train_bin_packaged":False,
        "foundation_v3_train_bin_packaged":False,
        "collapsed_f2_checkpoint_packaged":False,
        "candidate_runs_reset_source_weights":True,
        "candidate_runs_reset_adam_moments":True
      }
    }

    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr("manifest.json",json.dumps(manifest,sort_keys=True,separators=(",",":")))
        z.write(train_tokens,"data/f2r_train.tokens.u16")
        z.write(train_mask,"data/f2r_train.lossmask.u8")
        z.write(val_tokens,"data/f2r_validation.tokens.u16")
        z.write(val_mask,"data/f2r_validation.lossmask.u8")
        z.write(v3,"data/v3_validation.bin")
        z.write(v1,"data/v1_validation.bin")

    build_report={
      "status":"PASS",
      "schema":"model0001_f2r_lr_pilot_package_build_v1",
      "output":str(out),
      "sha256":sha256(out),
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "stage_objective":"friend_f2r_repair_sft",
      "train_windows_total":train_windows,
      "validation_windows_total":val_windows,
      "train_steps_per_candidate":TRAIN_STEPS,
      "train_mix":train_mix,
      "validation_mix":val_mix,
      "lr_candidates":LR_CANDIDATES,
      "fresh_zero_moments_each_candidate":True,
      "record_isolated_packing":True,
      "cross_record_windows":0,
      "collapsed_f2_checkpoint_used":False,
      "test_split_used":False
    }
    rp=Path(args.report_output).resolve()
    rp.write_text(json.dumps(build_report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(build_report,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
