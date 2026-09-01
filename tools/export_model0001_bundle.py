#!/usr/bin/env python3
"""
Export the exact Friend-Core Model #0001 checkpoint into an Android-Trainer gate bundle.

This script is intentionally strict:
- imports the real scripts/17_pretrain_model0001.py implementation;
- obtains the real model config from the checkpoint/contract/run config;
- maps state tensors by shape + semantic key names and refuses ambiguity;
- computes PyTorch FP32 reference loss, logit probes, gradient probes/global norm;
- computes one fresh-AdamW reference update using optimizer hyperparameters read
  from the checkpoint when available (or explicit CLI values);
- never uploads or modifies the checkpoint or dataset.

Output is a ZIP-compatible .atb file. The public GitHub repository never needs it.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, math, os, re, struct, sys, tempfile, zipfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

EXPECTED = {
    "vocab_size": 14000, "seq_len": 256, "d_model": 384, "n_layers": 8,
    "n_heads": 6, "n_kv_heads": 2, "head_dim": 64, "d_ff": 1152,
    "params": 19145088,
}
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)(?:\.|$)", re.I)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def import_engine(project: Path):
    p=project/"scripts"/"17_pretrain_model0001.py"
    if not p.exists(): raise SystemExit(f"missing {p}")
    spec=importlib.util.spec_from_file_location("friend_model0001_engine17", p)
    if spec is None or spec.loader is None: raise SystemExit("cannot import script 17")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def extract_cfg(ck: dict, run_config: dict | None, eng) -> dict:
    candidates=[]
    for x in [
        ck.get("model_config"),
        (ck.get("contract") or {}).get("model") if isinstance(ck.get("contract"),dict) else None,
        (ck.get("run_config") or {}).get("model") if isinstance(ck.get("run_config"),dict) else None,
        (run_config or {}).get("model"),
        getattr(eng,"DEFAULT_MODEL",None),
    ]:
        if isinstance(x,dict): candidates.append(dict(x))
    if not candidates: raise SystemExit("cannot find model config")
    cfg=candidates[0]
    # normalize only aliases; never invent architecture values.
    aliases={
      "vocab_size":["vocab_size","vocab"], "seq_len":["seq_len","max_seq_len","context_length"],
      "d_model":["d_model","hidden_size"], "n_layers":["n_layers","num_layers"],
      "n_heads":["n_heads","num_heads"], "n_kv_heads":["n_kv_heads","num_kv_heads"],
      "d_ff":["d_ff","hidden_dim","intermediate_size"], "rope_theta":["rope_theta"],
      "rms_norm_eps":["rms_norm_eps","norm_eps"],
    }
    out={}
    for dst,names in aliases.items():
        vals=[cfg[n] for n in names if n in cfg]
        if vals: out[dst]=vals[0]
    required=["vocab_size","seq_len","d_model","n_layers","n_heads","n_kv_heads","d_ff","rope_theta","rms_norm_eps"]
    miss=[x for x in required if x not in out]
    if miss: raise SystemExit(f"config missing exact fields {miss}; refusing to guess")
    out["head_dim"]=out["d_model"]//out["n_heads"]
    for k in ["vocab_size","seq_len","d_model","n_layers","n_heads","n_kv_heads","d_ff","head_dim"]:
        if int(out[k]) != EXPECTED[k]: raise SystemExit(f"architecture drift {k}: {out[k]} != {EXPECTED[k]}")
    return out

def semantic_kind(name:str, shape:tuple[int,...]) -> str | None:
    n=name.lower()
    if shape==(14000,384): return "tok_embeddings"
    if shape==(384,384):
        if re.search(r"(^|[._])(q|query)([._]|$)",n): return "q_proj"
        if re.search(r"(^|[._])(o|out|output)([._]|$)",n): return "o_proj"
    if shape==(128,384):
        if re.search(r"(^|[._])(k|key)([._]|$)",n): return "k_proj"
        if re.search(r"(^|[._])(v|value)([._]|$)",n): return "v_proj"
    if shape==(1152,384):
        if "gate" in n or re.search(r"(^|[._])w1([._]|$)",n): return "gate_proj"
        if re.search(r"(^|[._])up([._]|$)",n) or re.search(r"(^|[._])w3([._]|$)",n): return "up_proj"
    if shape==(384,1152): return "down_proj"
    if shape==(384,):
        if any(x in n for x in ["attn_norm","attention_norm","input_layernorm","ln1","norm1"]): return "attn_norm"
        if any(x in n for x in ["ffn_norm","post_attention","post_attn","ln2","norm2"]): return "ffn_norm"
    return None

def normalize_state(state:dict[str,torch.Tensor]) -> tuple[dict[str,torch.Tensor],dict[str,str]]:
    tensors={k:v.detach().cpu().float().contiguous() for k,v in state.items() if torch.is_tensor(v)}
    if any(v.ndim==1 and v.numel()!=384 for v in tensors.values()):
        # biases would alter frozen parameter geometry; list them rather than silently ignore.
        extra=[(k,tuple(v.shape)) for k,v in tensors.items() if v.ndim==1 and v.numel()!=384]
        if extra: raise SystemExit(f"unexpected 1-D trainable/state tensors: {extra[:12]}")
    out={}; origin={}
    # embedding must be unique
    emb=[(k,v) for k,v in tensors.items() if tuple(v.shape)==(14000,384)]
    if len(emb)!=1: raise SystemExit(f"expected one tied embedding tensor, got {[k for k,_ in emb]}")
    out["tok_embeddings.weight"]=emb[0][1]; origin["tok_embeddings.weight"]=emb[0][0]

    per={i:{} for i in range(8)}
    layer_norm_candidates={i:[] for i in range(8)}
    outside_norm=[]
    for k,v in tensors.items():
        if k==emb[0][0]: continue
        m=LAYER_RE.search(k)
        shape=tuple(v.shape)
        if m:
            i=int(m.group(1))
            if i not in per: continue
            kind=semantic_kind(k,shape)
            if kind:
                if kind in per[i]: raise SystemExit(f"duplicate layer {i} {kind}: {per[i][kind][0]} and {k}")
                per[i][kind]=(k,v)
            elif shape==(384,):
                layer_norm_candidates[i].append((k,v))
        elif shape==(384,):
            outside_norm.append((k,v))
    for i in range(8):
        # deterministic fallback for norms only when names unambiguous by ordering is NOT allowed.
        for k,v in layer_norm_candidates[i]:
            kind=semantic_kind(k,tuple(v.shape))
            if kind and kind not in per[i]: per[i][kind]=(k,v)
        need=["attn_norm","q_proj","k_proj","v_proj","o_proj","ffn_norm","gate_proj","up_proj","down_proj"]
        missing=[x for x in need if x not in per[i]]
        if missing:
            keys=[k for k,v in tensors.items() if (mm:=LAYER_RE.search(k)) and int(mm.group(1))==i]
            raise SystemExit(f"cannot semantically map layer {i}: missing={missing}; keys={keys}")
        for kind in need:
            src,v=per[i][kind]; dst=f"layers.{i}.{kind}.weight"; out[dst]=v; origin[dst]=src
    # final norm: exactly one outside-layer [384] tensor with norm-ish name.
    finals=[(k,v) for k,v in outside_norm if "norm" in k.lower()]
    if len(finals)!=1:
        raise SystemExit(f"expected exactly one final norm outside blocks, got {[k for k,_ in finals]}")
    out["final_norm.weight"]=finals[0][1]; origin["final_norm.weight"]=finals[0][0]

    unique=sum(v.numel() for v in out.values())
    if unique != EXPECTED["params"]:
        raise SystemExit(f"normalized unique params={unique:,} != {EXPECTED['params']:,}")
    return out,origin

def detect_rope_style(engine_text:str) -> str:
    t=engine_text.lower()
    # half-split Llama rotate_half: split last dim in halves then cat(-x2,x1)
    if "rotate_half" in t or ("chunk(2" in t and "cat" in t):
        return "half_split"
    # pairwise rotary: even/odd slicing.
    if ("::2" in t and "1::2" in t) or ("0::2" in t and "1::2" in t):
        return "interleaved"
    raise SystemExit("cannot prove RoPE layout from script17 source; refusing to guess")

def optimizer_hparams(ck:dict, args) -> dict:
    opt=ck.get("optimizer")
    if isinstance(opt,dict) and isinstance(opt.get("param_groups"),list) and opt["param_groups"]:
        gs=opt["param_groups"]
        # Gate requires a single uniform AdamW semantic contract. Different groups are allowed
        # only if core hyperparams match; weight decay group mapping would need explicit export.
        vals=[]
        for g in gs:
            vals.append((tuple(g.get("betas",(args.beta1,args.beta2))),float(g.get("eps",args.eps)),float(g.get("weight_decay",args.weight_decay))))
        if len(set(vals))!=1:
            raise SystemExit(f"optimizer has non-uniform groups {vals}; exporter refuses silent flattening")
        betas,eps,wd=vals[0]
        return {"beta1":float(betas[0]),"beta2":float(betas[1]),"eps":eps,"weight_decay":wd,
                "gate_lr":float(args.gate_lr),"source_optimizer_groups":len(gs)}
    return {"beta1":args.beta1,"beta2":args.beta2,"eps":args.eps,"weight_decay":args.weight_decay,
            "gate_lr":args.gate_lr,"source_optimizer_groups":0}

def probe_indices(name:str, n:int, count=16):
    seed=int(hashlib.sha256(name.encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(seed)
    return sorted(set(int(x) for x in rng.integers(0,n,size=count)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--train-bin",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--run-config")
    ap.add_argument("--window-index",type=int,default=0)
    ap.add_argument("--gate-lr",type=float,default=1e-4)
    ap.add_argument("--beta1",type=float,default=0.9)
    ap.add_argument("--beta2",type=float,default=0.999)
    ap.add_argument("--eps",type=float,default=1e-8)
    ap.add_argument("--weight-decay",type=float,default=0.1)
    a=ap.parse_args()

    project=Path(a.project).resolve(); ckpt=Path(a.checkpoint).resolve(); train_bin=Path(a.train_bin).resolve()
    eng=import_engine(project)
    ck=torch.load(ckpt,map_location="cpu",weights_only=False)
    if "model" not in ck: raise SystemExit("checkpoint has no model state")
    rc=json.loads(Path(a.run_config).read_text()) if a.run_config else None
    cfg=extract_cfg(ck,rc,eng)
    rope_style=detect_rope_style((project/"scripts"/"17_pretrain_model0001.py").read_text(errors="replace"))

    model=eng.Model0001(dict(cfg))
    model.load_state_dict(ck["model"],strict=True); model.float(); model.train()
    nparams=sum(p.numel() for p in model.parameters())
    if nparams!=EXPECTED["params"]: raise SystemExit(f"model params {nparams} != expected")
    norm,origin=normalize_state(model.state_dict())

    raw=np.memmap(train_bin,dtype="<u2",mode="r")
    start=a.window_index*EXPECTED["seq_len"]
    arr=np.asarray(raw[start:start+EXPECTED["seq_len"]+1],dtype=np.int64)
    if arr.size!=257: raise SystemExit("selected train window incomplete")
    if arr.max()>=14000: raise SystemExit("train window contains OOV token")
    x=torch.from_numpy(arr[:-1].copy()).unsqueeze(0)
    y=torch.from_numpy(arr[1:].copy()).unsqueeze(0)

    model.zero_grad(set_to_none=True)
    logits,loss=model(x,y)
    if not torch.isfinite(loss): raise SystemExit("reference loss is nonfinite")
    loss.backward()
    # Reference raw global gradient norm (before clipping).
    sq=torch.zeros((),dtype=torch.float64)
    named_params=dict(model.named_parameters())
    for p in model.parameters():
        if p.grad is not None: sq += p.grad.detach().double().pow(2).sum()
    global_norm=float(sq.sqrt())
    clip_coef=min(1.0,1.0/(global_norm+1e-6))

    # Map normalized slots back to actual Parameter objects.
    state_to_param={k:v for k,v in model.named_parameters()}
    grad_meta={}
    for slot,src in origin.items():
        p=state_to_param.get(src)
        if p is None or p.grad is None: raise SystemExit(f"missing grad for {src}")
        g=p.grad.detach().cpu().float().contiguous().view(-1)
        inds=probe_indices(slot,g.numel())
        grad_meta[slot]={
            "l2":float(g.double().norm()),"max_abs":float(g.abs().max()),
            "probe_indices":inds,"probe_values":[float(g[i]) for i in inds],
        }

    # Logit probes across multiple positions/classes, deterministic and compact.
    flat=logits.detach().cpu().float()[0]
    pos=[0,1,63,127,191,255]
    cls=[3,17,101,997,4096,8191,13999]
    logit_probe=[{"position":p,"token":c,"value":float(flat[p,c])} for p in pos for c in cls]

    hp=optimizer_hparams(ck,a)

    # Fresh AdamW one-step reference, keeping update semantics independent from backend state format.
    ref_model=eng.Model0001(dict(cfg)); ref_model.load_state_dict(ck["model"],strict=True); ref_model.float(); ref_model.train()
    opt=torch.optim.AdamW(ref_model.parameters(),lr=hp["gate_lr"],betas=(hp["beta1"],hp["beta2"]),
                          eps=hp["eps"],weight_decay=hp["weight_decay"])
    opt.zero_grad(set_to_none=True); _,l2=ref_model(x,y); l2.backward()
    torch.nn.utils.clip_grad_norm_(ref_model.parameters(),1.0); opt.step()
    ref_state=ref_model.state_dict()
    adam_probe={}
    for slot,src in origin.items():
        before=norm[slot].view(-1); after=ref_state[src].detach().cpu().float().contiguous().view(-1)
        inds=probe_indices(slot,before.numel())
        adam_probe[slot]={
          "probe_indices":inds,
          "before":[float(before[i]) for i in inds],
          "after":[float(after[i]) for i in inds],
        }

    manifest={
      "schema":"android_trainer_bundle_v1",
      "checkpoint_sha256":sha256_file(ckpt),
      "train_bin_sha256":sha256_file(train_bin),
      "config":cfg,
      "rope_style":rope_style,
      "parameter_count":nparams,
      "origin_state_keys":origin,
      "optimizer":hp,
      "sample":{"window_index":a.window_index,"tokens_file":"sample/tokens_i32.bin","token_count":257},
      "reference":{
        "loss":float(loss.detach()),"global_grad_norm":global_norm,"clip_coef":clip_coef,
        "logit_probe":logit_probe,"gradient":grad_meta,"adamw_step1":adam_probe,
      },
      "tensors":{},
    }

    out=Path(a.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        tb=arr.astype("<i4",copy=False).tobytes(); z.writestr("sample/tokens_i32.bin",tb)
        for slot,t in norm.items():
            b=t.numpy().astype("<f4",copy=False).tobytes(order="C")
            path="tensors/"+slot.replace(".","/")+".f32"
            manifest["tensors"][slot]={"path":path,"shape":list(t.shape),"dtype":"f32","nbytes":len(b),"sha256":sha256_bytes(b)}
            z.writestr(path,b)
        mb=json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
        z.writestr("manifest.json",mb)
    print(json.dumps({
      "status":"PASS","output":str(out),"bundle_sha256":sha256_file(out),
      "bundle_mib":out.stat().st_size/(1024**2),"checkpoint_sha256":manifest["checkpoint_sha256"],
      "reference_loss":manifest["reference"]["loss"],"global_grad_norm":global_norm,
      "rope_style":rope_style,"params":nparams,
    },indent=2))

if __name__=="__main__": main()
