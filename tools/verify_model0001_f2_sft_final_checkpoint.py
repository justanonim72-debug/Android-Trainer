#!/usr/bin/env python3
"""Fail-closed verifier for the completed Model #0001 F2 SFT native checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, math, struct
from pathlib import Path
import numpy as np

EXPECTED_FILE_SHA="ed6556dbe293e9bb78af82f5ce410e3f37ad8f529b5b1cd9b84b4883f078d9d6"
EXPECTED_MODEL_SHA="d09d31d9759790c12ba62b4ae101c53807ffc2a95fe452c2706abe4a09ea0e11"
EXPECTED_PARENT_CKPT="773d685b81a736de795e8b3d93cf1833dc01a1f6a7e0fd6edfd9edefd7a36a67"
EXPECTED_PARENT_MODEL="10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
EXPECTED_COMMIT="51761fcf7aa9dc7c589cabc855a1798366378716"
EXPECTED_GEOMETRY=[256,14000,384,8,6,2,64,1152]
EXPECTED_PARAMS=19_145_088
EXPECTED_SLOTS=74
EXPECTED_STEP=2786

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def read_exact(f,n:int)->bytes:
    b=f.read(n)
    if len(b)!=n: raise SystemExit("STOP: truncated checkpoint")
    return b

def u32(f): return struct.unpack("<I",read_exact(f,4))[0]
def u64(f): return struct.unpack("<Q",read_exact(f,8))[0]
def f64(f): return struct.unpack("<d",read_exact(f,8))[0]
def read_string(f):
    n=u32(f)
    if n>4096: raise SystemExit("STOP: checkpoint string too long")
    return read_exact(f,n).decode("utf-8")

def normalized_model_sha(slots):
    h=hashlib.sha256()
    for name in sorted(slots):
        shape,data=slots[name]
        h.update(name.encode("utf-8"))
        h.update(b"float32")
        h.update(struct.pack("<I",len(shape)))
        for d in shape: h.update(struct.pack("<Q",int(d)))
        h.update(data)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="/storage/emulated/0/Download/model0001-f2-sft-final.atnckpt")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-f2-sft-final-audit.json")
    args=ap.parse_args()

    path=Path(args.checkpoint).resolve()
    if not path.is_file(): raise SystemExit(f"STOP: checkpoint missing: {path}")
    file_sha=sha256_file(path)
    if file_sha!=EXPECTED_FILE_SHA:
        raise SystemExit(f"STOP: F2 checkpoint SHA mismatch: {file_sha}")

    with path.open("rb") as f:
        if read_exact(f,8)!=b"ATNCL01\x00": raise SystemExit("STOP: bad magic")
        version=u32(f); endian=u32(f); count=u32(f); reserved=u32(f)
        step=u64(f); params=u64(f)
        geometry=[u32(f) for _ in range(8)]
        beta1=f64(f); beta2=f64(f); eps=f64(f); lr=f64(f)
        commit=read_string(f); parent_ckpt=read_string(f); parent_model=read_string(f)

        if (version,endian,count,reserved)!=(1,0x01020304,EXPECTED_SLOTS,0):
            raise SystemExit("STOP: checkpoint header drift")
        if step!=EXPECTED_STEP or params!=EXPECTED_PARAMS or geometry!=EXPECTED_GEOMETRY:
            raise SystemExit("STOP: F2 completion/geometry mismatch")
        if not (
            math.isclose(beta1,0.9,abs_tol=1e-12) and
            math.isclose(beta2,0.95,abs_tol=1e-12) and
            math.isclose(eps,1e-8,abs_tol=1e-16) and
            math.isclose(lr,1e-6,rel_tol=0,abs_tol=1e-12)
        ):
            raise SystemExit("STOP: F2 optimizer/footer mismatch")
        if commit!=EXPECTED_COMMIT or parent_ckpt!=EXPECTED_PARENT_CKPT or parent_model!=EXPECTED_PARENT_MODEL:
            raise SystemExit("STOP: F2 lineage mismatch")

        slots={}
        total=0
        nonfinite_param=nonfinite_m1=nonfinite_m2=0
        nonzero_m1=nonzero_m2=0
        for _ in range(count):
            name=read_string(f)
            rank=u32(f)
            shape=[u32(f) for _ in range(rank)]
            elements=u64(f)
            _wd=f64(f)
            expected=1
            for d in shape: expected*=d
            if expected!=elements: raise SystemExit(f"STOP: shape mismatch {name}")
            nbytes=elements*4
            pb=read_exact(f,nbytes); m1b=read_exact(f,nbytes); m2b=read_exact(f,nbytes)
            p=np.frombuffer(pb,dtype="<f4")
            m1=np.frombuffer(m1b,dtype="<f4")
            m2=np.frombuffer(m2b,dtype="<f4")
            nonfinite_param += int((~np.isfinite(p)).sum())
            nonfinite_m1 += int((~np.isfinite(m1)).sum())
            nonfinite_m2 += int((~np.isfinite(m2)).sum())
            nonzero_m1 += int(np.count_nonzero(m1))
            nonzero_m2 += int(np.count_nonzero(m2))
            slots[name]=(shape,pb); total+=elements

        if f.read(1): raise SystemExit("STOP: trailing checkpoint bytes")
        if total!=EXPECTED_PARAMS: raise SystemExit("STOP: parameter total mismatch")
        if nonfinite_param or nonfinite_m1 or nonfinite_m2:
            raise SystemExit("STOP: nonfinite tensor/moment values")

    model_sha=normalized_model_sha(slots)
    if model_sha!=EXPECTED_MODEL_SHA:
        raise SystemExit(f"STOP: F2 model-state SHA mismatch: {model_sha}")

    audit={
      "status":"PASS",
      "schema":"model0001_f2_sft_final_checkpoint_audit_v1",
      "checkpoint":str(path),
      "checkpoint_sha256":file_sha,
      "model_state_sha256":model_sha,
      "optimizer_step":step,
      "parameter_count":params,
      "tensor_count":count,
      "geometry":geometry,
      "adamw":{"beta1":beta1,"beta2":beta2,"eps":eps,"stored_final_lr":lr},
      "training_commit":commit,
      "parent_foundation_checkpoint_sha256":parent_ckpt,
      "parent_foundation_model_state_sha256":parent_model,
      "optimizer_moment1_nonzero_values":nonzero_m1,
      "optimizer_moment2_nonzero_values":nonzero_m2,
      "nonfinite_parameter_values":0,
      "nonfinite_moment1_values":0,
      "nonfinite_moment2_values":0,
      "test_split_used":False
    }
    out=Path(args.output).resolve()
    out.write_text(json.dumps(audit,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(audit,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
