#!/usr/bin/env python3
"""Build the private physical-phone F2 SFT LR-pilot package.

The promoted Foundation-v3 source .atb stays separate. This package contains
only masked SFT train/validation windows plus frozen V3/V1 validation for
retention checks. No test split is read or packaged.
"""
from __future__ import annotations

import argparse, hashlib, json, random, zipfile
from pathlib import Path

SOURCE_MODEL_SHA="10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
SEQ=256
WINDOW=257
SEED=20260903
TRAIN_STEPS=96
WARMUP_STEPS=3
SFT_EVAL_WINDOWS=24
RETENTION_EVAL_WINDOWS=24
LR_CANDIDATES=[1e-5,2e-5,5e-5]

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
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
    if count<=0: raise SystemExit("STOP: zero candidate windows")
    n=min(count,n)
    return sorted(random.Random(seed).sample(range(count),n))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--source-audit",default="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-f2-sft-lr-pilot.atsftpilot")
    args=ap.parse_args()
    project=Path(args.project).resolve()

    source_audit_path=Path(args.source_audit).resolve()
    if not source_audit_path.is_file():
        raise SystemExit("STOP: promoted Foundation-v3 source audit missing")
    source_audit=json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status")!="PASS" or source_audit.get("model_state_sha256")!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: promoted Foundation-v3 source identity mismatch")
    if source_audit.get("optimizer_moments_carried_forward") is not False:
        raise SystemExit("STOP: F2 source must not carry Foundation-v3 moments")

    f2=project/"artifacts"/"model0001_f2_sft"
    pack_report_path=f2/"F2_SFT_PACK_REPORT.json"
    if not pack_report_path.is_file():
        raise SystemExit("STOP: F2 SFT pack report missing")
    pack=json.loads(pack_report_path.read_text(encoding="utf-8"))
    if pack.get("status")!="PASS" or pack.get("objective")!="assistant_content_only_cross_entropy":
        raise SystemExit("STOP: F2 pack not assistant-only PASS")
    guards=pack.get("hard_guards",{})
    if guards.get("assistant_only_loss") is not True or guards.get("test_split_used") is not False:
        raise SystemExit("STOP: F2 pack hard guard failed")

    train_tokens=Path(pack["train"]["tokens_path"])
    train_mask=Path(pack["train"]["mask_path"])
    val_tokens=Path(pack["validation"]["tokens_path"])
    val_mask=Path(pack["validation"]["mask_path"])
    for p in (train_tokens,train_mask,val_tokens,val_mask):
        if not p.is_file(): raise SystemExit(f"STOP: missing F2 pack file {p}")

    train_windows=int(pack["train"]["windows"])
    val_windows=int(pack["validation"]["windows"])
    if train_windows<TRAIN_STEPS:
        raise SystemExit("STOP: F2 train has too few windows for pilot")

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

    def sft_spec(tokens:Path,mask:Path,windows:int):
        if tokens.stat().st_size != windows*WINDOW*2:
            raise SystemExit(f"STOP: SFT token-window file size mismatch {tokens}")
        if mask.stat().st_size != windows*SEQ:
            raise SystemExit(f"STOP: SFT mask-window file size mismatch {mask}")
        return {
          "tokens_path":None,
          "tokens_sha256":sha256(tokens),
          "mask_path":None,
          "mask_sha256":sha256(mask),
          "windows":windows,
          "tokens_per_window":WINDOW,
          "mask_targets_per_window":SEQ,
        }

    train_spec=sft_spec(train_tokens,train_mask,train_windows)
    val_spec=sft_spec(val_tokens,val_mask,val_windows)
    train_spec["tokens_path"]="data/f2_train.tokens.u16"
    train_spec["mask_path"]="data/f2_train.lossmask.u8"
    val_spec["tokens_path"]="data/f2_validation.tokens.u16"
    val_spec["mask_path"]="data/f2_validation.lossmask.u8"

    manifest={
      "schema":"model0001_f2_sft_lr_pilot_v1",
      "status":"LOCKED",
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "stage_objective":"friend_f2_sft",
      "objective":"assistant_content_only_cross_entropy",
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
        "production_lr_locked":False
      },
      "indices":{
        "train":sample_indices(train_windows,TRAIN_STEPS,SEED+11),
        "sft_validation":sample_indices(val_windows,SFT_EVAL_WINDOWS,SEED+12),
        "v3_validation":sample_indices(v3_windows,RETENTION_EVAL_WINDOWS,SEED+13),
        "v1_validation":sample_indices(v1_windows,RETENTION_EVAL_WINDOWS,SEED+14)
      },
      "data":{
        "sft_train":train_spec,
        "sft_validation":val_spec,
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
        "f2_pack_report_sha256":sha256(pack_report_path)
      },
      "hard_guards":{
        "assistant_only_loss":True,
        "test_split_packaged":False,
        "dataset_v2_train_bin_packaged":False,
        "foundation_v3_train_bin_packaged":False,
        "candidate_runs_reset_source_weights":True,
        "candidate_runs_reset_adam_moments":True
      }
    }

    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
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
      "schema":"model0001_f2_sft_lr_pilot_package_build_v1",
      "output":str(out),
      "sha256":sha256(out),
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "train_windows":train_windows,
      "validation_windows":val_windows,
      "lr_candidates":LR_CANDIDATES,
      "test_split_used":False
    },indent=2,sort_keys=True))

if __name__=="__main__": main()
