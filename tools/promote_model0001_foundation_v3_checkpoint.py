#!/usr/bin/env python3
"""Promote the completed Foundation-v3 native checkpoint into an immutable F2 source .atb.

The native .atnckpt contains parameters + Adam moments. This transition keeps
ONLY parameter tensors and deliberately discards optimizer moments so F2 starts
with fresh-zero AdamW state.

The old canonical .atb is used only as a strict tensor-layout/config template.
Its tensor payloads are replaced by the completed Foundation-v3 parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path

import numpy as np

EXPECTED_FINAL_CKPT_SHA = "773d685b81a736de795e8b3d93cf1833dc01a1f6a7e0fd6edfd9edefd7a36a67"
EXPECTED_V3_REPORT_SCHEMA = "model0001_native_stage_report_v1"
EXPECTED_V3_COMMIT = "660638e350f3190a9578e7de9a0c2c26fd8a6cf9"
EXPECTED_OLD_SOURCE_MODEL_SHA = "047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
EXPECTED_OLD_SOURCE_CKPT_SHA = "cbc6dec84e51d2a19e50ea38607e64cef78e62c627e0aa44f3dbe838d100ddf9"
EXPECTED_V3_TRAIN_SHA = "19c7d23661aee08d11ac243347d4b943661084f1dc6fa740a222de01ae970975"
EXPECTED_GEOMETRY = [256, 14000, 384, 8, 6, 2, 64, 1152]
EXPECTED_PARAMS = 19_145_088
EXPECTED_SLOTS = 74
EXPECTED_PROMOTED_MODEL_SHA = "10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):
            h.update(c)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_exact(f, n:int) -> bytes:
    b=f.read(n)
    if len(b)!=n:
        raise SystemExit("STOP: truncated native checkpoint")
    return b

def u32(f)->int: return struct.unpack("<I",read_exact(f,4))[0]
def u64(f)->int: return struct.unpack("<Q",read_exact(f,8))[0]
def f64(f)->float: return struct.unpack("<d",read_exact(f,8))[0]

def read_string(f)->str:
    n=u32(f)
    if n>4096:
        raise SystemExit("STOP: native checkpoint string too long")
    return read_exact(f,n).decode("utf-8")

def tensor_state_hash(slots:dict[str,tuple[list[int],bytes]])->str:
    """Stable normalized-state hash for the promoted native source bundle."""
    h=hashlib.sha256()
    for name in sorted(slots):
        shape,data=slots[name]
        h.update(name.encode("utf-8"))
        h.update(b"float32")
        h.update(struct.pack("<I",len(shape)))
        for d in shape:
            h.update(struct.pack("<Q",int(d)))
        h.update(data)
    return h.hexdigest()

def parse_checkpoint(path:Path):
    with path.open("rb") as f:
        if read_exact(f,8)!=b"ATNCL01\x00":
            raise SystemExit("STOP: native checkpoint magic mismatch")
        version=u32(f); endian=u32(f); count=u32(f); reserved=u32(f)
        step=u64(f); params=u64(f)
        geometry=[u32(f) for _ in range(8)]
        beta1=f64(f); beta2=f64(f); eps=f64(f); lr=f64(f)
        commit=read_string(f)
        source_ckpt=read_string(f)
        source_model=read_string(f)

        if (version,endian,count,reserved)!=(1,0x01020304,EXPECTED_SLOTS,0):
            raise SystemExit("STOP: native checkpoint header drift")
        if step!=4013 or params!=EXPECTED_PARAMS or geometry!=EXPECTED_GEOMETRY:
            raise SystemExit("STOP: Foundation-v3 completion/geometry mismatch")
        if not (math.isclose(beta1,0.9,abs_tol=1e-12) and
                math.isclose(beta2,0.95,abs_tol=1e-12) and
                math.isclose(eps,1e-8,abs_tol=1e-16) and
                math.isclose(lr,2e-5,rel_tol=0,abs_tol=1e-10)):
            raise SystemExit("STOP: Foundation-v3 optimizer/footer contract mismatch")
        if commit!=EXPECTED_V3_COMMIT:
            raise SystemExit("STOP: Foundation-v3 checkpoint build commit mismatch")
        if source_ckpt!=EXPECTED_OLD_SOURCE_CKPT_SHA or source_model!=EXPECTED_OLD_SOURCE_MODEL_SHA:
            raise SystemExit("STOP: Foundation-v3 checkpoint source lineage mismatch")

        slots={}
        decay={}
        total=0
        for _ in range(count):
            name=read_string(f)
            rank=u32(f)
            if rank<=0 or rank>8:
                raise SystemExit(f"STOP: bad rank for {name}")
            shape=[u32(f) for _ in range(rank)]
            elements=u64(f)
            wd=f64(f)
            expected=1
            for d in shape: expected*=d
            if expected!=elements:
                raise SystemExit(f"STOP: shape/elements mismatch for {name}")
            nbytes=elements*4
            param=read_exact(f,nbytes)
            arr=np.frombuffer(param,dtype="<f4")
            if not np.isfinite(arr).all():
                raise SystemExit(f"STOP: nonfinite parameter in {name}")
            # Deliberately discard optimizer moments.
            read_exact(f,nbytes)  # moment1
            read_exact(f,nbytes)  # moment2
            if name in slots:
                raise SystemExit(f"STOP: duplicate checkpoint slot {name}")
            slots[name]=(shape,param)
            decay[name]=wd
            total+=elements

        if total!=EXPECTED_PARAMS:
            raise SystemExit("STOP: checkpoint parameter total mismatch")
        if f.read(1):
            raise SystemExit("STOP: trailing bytes after native checkpoint")
    return slots,decay,{
        "optimizer_step":step,
        "learning_rate":lr,
        "commit":commit,
        "source_checkpoint_sha256":source_ckpt,
        "source_model_state_sha256":source_model,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--template-bundle",default="/storage/emulated/0/Download/model0001-gpu-gate.atb")
    ap.add_argument("--checkpoint",default="/storage/emulated/0/Download/model0001-foundation-v3-final.atnckpt")
    ap.add_argument("--stage-report",default="/storage/emulated/0/Download/model0001-foundation-v3-stage-report.json")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-foundation-v3-source.atb")
    ap.add_argument("--audit-output",default="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json")
    args=ap.parse_args()

    template=Path(args.template_bundle).resolve()
    checkpoint=Path(args.checkpoint).resolve()
    stage_report=Path(args.stage_report).resolve()
    for p in (template,checkpoint,stage_report):
        if not p.is_file():
            raise SystemExit(f"STOP: missing transition input {p}")

    ckpt_sha=sha256_file(checkpoint)
    if ckpt_sha!=EXPECTED_FINAL_CKPT_SHA:
        raise SystemExit(f"STOP: Foundation-v3 checkpoint SHA mismatch: {ckpt_sha}")

    report=json.loads(stage_report.read_text(encoding="utf-8"))
    if report.get("schema")!=EXPECTED_V3_REPORT_SCHEMA or report.get("status")!="PASS" or report.get("pass") is not True:
        raise SystemExit("STOP: Foundation-v3 stage report not PASS")
    if report.get("commit")!=EXPECTED_V3_COMMIT or report.get("ending_optimizer_step")!=4013:
        raise SystemExit("STOP: Foundation-v3 report completion/commit mismatch")
    if report.get("test_split_used") is not False:
        raise SystemExit("STOP: Foundation-v3 report says test was used")

    slots,decay,checkpoint_meta=parse_checkpoint(checkpoint)

    with zipfile.ZipFile(template,"r") as zin:
        manifest=json.loads(zin.read("manifest.json"))
        if manifest.get("schema")!="android_trainer_bundle_v2":
            raise SystemExit("STOP: template bundle schema mismatch")
        if manifest.get("model_state_sha256")!=EXPECTED_OLD_SOURCE_MODEL_SHA:
            raise SystemExit("STOP: template is not canonical CPT-v2 source bundle")
        tensors=manifest.get("tensors",{})
        if set(tensors)!=set(slots):
            missing=sorted(set(tensors)-set(slots))
            extra=sorted(set(slots)-set(tensors))
            raise SystemExit(f"STOP: slot-set mismatch missing={missing} extra={extra}")
        for name,spec in tensors.items():
            shape,param=slots[name]
            if list(spec["shape"])!=shape:
                raise SystemExit(f"STOP: slot shape drift {name}")
            expected_wd=float(manifest["optimizer"]["slot_weight_decay"][name])
            if not math.isclose(float(decay[name]),expected_wd,rel_tol=0,abs_tol=1e-7):
                raise SystemExit(f"STOP: slot weight-decay drift {name}")

        new_model_sha=tensor_state_hash(slots)
        if new_model_sha != EXPECTED_PROMOTED_MODEL_SHA:
            raise SystemExit(
                "STOP: promoted Foundation-v3 model-state SHA mismatch: "
                f"{new_model_sha} != {EXPECTED_PROMOTED_MODEL_SHA}"
            )
        manifest["checkpoint_sha256"]=ckpt_sha
        manifest["model_state_sha256"]=new_model_sha
        manifest["train_bin_sha256"]=EXPECTED_V3_TRAIN_SHA
        manifest["source_stage"]="friend_foundation_v3_cpt"
        manifest["source_stage_optimizer_step"]=4013
        manifest["source_stage_lifetime_tokens"]=6_563_072
        manifest["promotion"]={
            "schema":"model0001_native_checkpoint_promotion_v1",
            "source_checkpoint_file_sha256":ckpt_sha,
            "source_stage_report_sha256":sha256_file(stage_report),
            "parent_model_state_sha256":EXPECTED_OLD_SOURCE_MODEL_SHA,
            "optimizer_moments_carried_forward":False,
            "f2_optimizer_init":"fresh_zero_moments",
        }
        for name,spec in tensors.items():
            param=slots[name][1]
            spec["sha256"]=sha256_bytes(param)
            spec["nbytes"]=len(param)

        out=Path(args.output).resolve()
        out.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as zout:
            # Preserve non-tensor template files (sample/reference metadata etc).
            tensor_paths={spec["path"] for spec in tensors.values()}
            for info in zin.infolist():
                if info.filename=="manifest.json" or info.filename in tensor_paths:
                    continue
                zout.writestr(info,zin.read(info.filename))
            for name,spec in tensors.items():
                zout.writestr(spec["path"],slots[name][1])
            zout.writestr("manifest.json",json.dumps(manifest,sort_keys=True,separators=(",",":")))

    audit={
        "status":"PASS",
        "schema":"model0001_foundation_v3_source_bundle_audit_v1",
        "output":str(out),
        "output_sha256":sha256_file(out),
        "foundation_v3_checkpoint_sha256":ckpt_sha,
        "foundation_v3_stage_report_sha256":sha256_file(stage_report),
        "model_state_sha256":new_model_sha,
        "parameter_count":EXPECTED_PARAMS,
        "tensor_count":EXPECTED_SLOTS,
        "source_stage_optimizer_step":4013,
        "source_stage_lifetime_tokens":6_563_072,
        "optimizer_moments_carried_forward":False,
        "next_stage_optimizer_init":"fresh_zero_moments",
        "test_split_used":False,
    }
    audit_path=Path(args.audit_output).resolve()
    audit_path.write_text(json.dumps(audit,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(audit,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
