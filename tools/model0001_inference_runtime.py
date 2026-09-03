#!/usr/bin/env python3
"""CPU inference runtime for frozen Model #0001 native checkpoints/bundles.

Purpose: post-training behavioral evaluation and exploratory chat on the same
Android phone. This is inference only: no optimizer, no gradient, no test split.

The implementation mirrors the accepted native architecture:
- decoder-only pre-norm transformer
- RMSNorm
- interleaved-pair RoPE
- 6 Q heads / 2 KV heads GQA
- SwiGLU
- tied embedding / LM head
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

EXPECTED_GEOMETRY={
    "seq_len":256,"vocab_size":14000,"d_model":384,"n_layers":8,
    "n_heads":6,"n_kv_heads":2,"head_dim":64,"d_ff":1152,
}
TOKENIZER_SHA="3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
FOUNDATION_MODEL_SHA="10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
FOUNDATION_CKPT_SHA="773d685b81a736de795e8b3d93cf1833dc01a1f6a7e0fd6edfd9edefd7a36a67"
F2_FILE_SHA="ed6556dbe293e9bb78af82f5ce410e3f37ad8f529b5b1cd9b84b4883f078d9d6"
F2_MODEL_SHA="d09d31d9759790c12ba62b4ae101c53807ffc2a95fe452c2706abe4a09ea0e11"
F2_COMMIT="51761fcf7aa9dc7c589cabc855a1798366378716"
BOS=1
EOS=2

ROLE_PREFIX={
    "system":"\nSystem: ",
    "user":"\nUser: ",
    "assistant":"\nAssistant: ",
    "tool":"\nTool: ",
}

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def normalized_model_sha(slots:dict[str,tuple[list[int],bytes]])->str:
    h=hashlib.sha256()
    for name in sorted(slots):
        shape,data=slots[name]
        h.update(name.encode("utf-8"))
        h.update(b"float32")
        h.update(struct.pack("<I",len(shape)))
        for d in shape: h.update(struct.pack("<Q",int(d)))
        h.update(data)
    return h.hexdigest()

def _read_exact(f,n):
    b=f.read(n)
    if len(b)!=n: raise RuntimeError("truncated native checkpoint")
    return b

def _u32(f): return struct.unpack("<I",_read_exact(f,4))[0]
def _u64(f): return struct.unpack("<Q",_read_exact(f,8))[0]
def _f64(f): return struct.unpack("<d",_read_exact(f,8))[0]
def _string(f):
    n=_u32(f)
    if n>4096: raise RuntimeError("native checkpoint string too long")
    return _read_exact(f,n).decode("utf-8")

@dataclass
class NativeCheckpointMeta:
    file_sha256:str
    model_state_sha256:str
    optimizer_step:int
    parameter_count:int
    geometry:list[int]
    beta1:float
    beta2:float
    eps:float
    lr:float
    commit:str
    parent_checkpoint_sha256:str
    parent_model_state_sha256:str

def load_native_checkpoint(path:Path, *, expected_file_sha:str|None=None,
                           expected_model_sha:str|None=None):
    path=path.resolve()
    got=sha256_file(path)
    if expected_file_sha and got!=expected_file_sha:
        raise RuntimeError(f"checkpoint SHA mismatch: {got}")
    slots={}
    with path.open("rb") as f:
        if _read_exact(f,8)!=b"ATNCL01\x00":
            raise RuntimeError("bad native checkpoint magic")
        version=_u32(f); endian=_u32(f); count=_u32(f); reserved=_u32(f)
        step=_u64(f); params=_u64(f)
        geometry=[_u32(f) for _ in range(8)]
        beta1=_f64(f); beta2=_f64(f); eps=_f64(f); lr=_f64(f)
        commit=_string(f); parent_ckpt=_string(f); parent_model=_string(f)
        if (version,endian,reserved)!=(1,0x01020304,0):
            raise RuntimeError("native checkpoint header drift")
        if count!=74 or params!=19_145_088:
            raise RuntimeError("native checkpoint geometry/slot-count drift")
        if geometry!=[256,14000,384,8,6,2,64,1152]:
            raise RuntimeError(f"native checkpoint frozen geometry drift: {geometry}")
        total=0
        for _ in range(count):
            name=_string(f)
            rank=_u32(f)
            shape=[_u32(f) for _ in range(rank)]
            elements=_u64(f)
            _wd=_f64(f)
            want=1
            for d in shape: want*=d
            if want!=elements: raise RuntimeError(f"shape mismatch: {name}")
            nbytes=elements*4
            pb=_read_exact(f,nbytes)
            # Evaluation uses parameters only; moments are read/skipped but
            # validated as finite to detect a corrupt final checkpoint.
            m1=_read_exact(f,nbytes); m2=_read_exact(f,nbytes)
            if not np.isfinite(np.frombuffer(pb,dtype="<f4")).all():
                raise RuntimeError(f"nonfinite parameter: {name}")
            if not np.isfinite(np.frombuffer(m1,dtype="<f4")).all():
                raise RuntimeError(f"nonfinite m1: {name}")
            if not np.isfinite(np.frombuffer(m2,dtype="<f4")).all():
                raise RuntimeError(f"nonfinite m2: {name}")
            slots[name]=(shape,pb)
            total+=elements
        if f.read(1): raise RuntimeError("trailing native checkpoint bytes")
        if total!=params: raise RuntimeError("parameter total mismatch")
    model_sha=normalized_model_sha(slots)
    if expected_model_sha and model_sha!=expected_model_sha:
        raise RuntimeError(f"model-state SHA mismatch: {model_sha}")
    meta=NativeCheckpointMeta(
        got,model_sha,step,params,geometry,beta1,beta2,eps,lr,
        commit,parent_ckpt,parent_model
    )
    return slots,meta

def load_bundle(path:Path, *, expected_model_sha:str|None=None):
    path=path.resolve()
    with zipfile.ZipFile(path,"r") as z:
        manifest=json.loads(z.read("manifest.json"))
        if manifest.get("schema")!="android_trainer_bundle_v2":
            raise RuntimeError("wrong .atb schema")
        if expected_model_sha and manifest.get("model_state_sha256")!=expected_model_sha:
            raise RuntimeError("bundle model-state SHA mismatch")
        cfg=dict(manifest["config"])
        if manifest.get("rope_style")!="interleaved":
            raise RuntimeError(
                "behavior runtime requires the accepted interleaved RoPE source"
            )
        slots={}
        for name,spec in manifest["tensors"].items():
            data=z.read(spec["path"])
            if hashlib.sha256(data).hexdigest()!=spec["sha256"]:
                raise RuntimeError(f"bundle tensor SHA mismatch: {name}")
            slots[name]=(list(spec["shape"]),data)
    if normalized_model_sha(slots)!=manifest["model_state_sha256"]:
        raise RuntimeError("bundle normalized model SHA mismatch")
    return slots,cfg,manifest

def slots_to_tensors(slots,device="cpu"):
    out={}
    for name,(shape,data) in slots.items():
        arr=np.frombuffer(data,dtype="<f4").copy().reshape(shape)
        out[name]=torch.from_numpy(arr).to(device=device,dtype=torch.float32)
    return out

class Model0001Inference(torch.nn.Module):
    def __init__(self, tensors:dict[str,torch.Tensor], cfg:dict):
        super().__init__()
        # State is frozen and intentionally registered as buffers rather than
        # trainable Parameters. This runtime cannot accidentally optimize.
        self.tensors={}
        for name,t in tensors.items():
            key="w_"+name.replace(".","__")
            self.register_buffer(key,t,persistent=False)
            self.tensors[name]=getattr(self,key)

        self.vocab=int(cfg["vocab_size"])
        self.max_seq=int(cfg["seq_len"])
        self.d=int(cfg["d_model"])
        self.layers=int(cfg["n_layers"])
        self.hq=int(cfg["n_heads"])
        self.hkv=int(cfg["n_kv_heads"])
        self.hd=int(cfg.get("head_dim",self.d//self.hq))
        self.ff=int(cfg["d_ff"])
        self.theta=float(cfg["rope_theta"])
        self.eps=float(cfg["rms_norm_eps"])
        if {
            "seq_len":self.max_seq,"vocab_size":self.vocab,"d_model":self.d,
            "n_layers":self.layers,"n_heads":self.hq,"n_kv_heads":self.hkv,
            "head_dim":self.hd,"d_ff":self.ff,
        }!=EXPECTED_GEOMETRY:
            raise RuntimeError("inference geometry drift")

    def w(self,name): return self.tensors[name]

    def rms(self,x,w):
        inv=torch.rsqrt(x.float().pow(2).mean(dim=-1,keepdim=True)+self.eps)
        return x*inv*w

    def rope(self,x):
        # x: [B,H,T,HD], accepted native interleaved pair layout.
        t=x.shape[2]
        pair=torch.arange(self.hd//2,device=x.device,dtype=torch.float32)
        inv=torch.pow(
            torch.tensor(self.theta,device=x.device,dtype=torch.float32),
            -2.0*pair/self.hd
        )
        pos=torch.arange(t,device=x.device,dtype=torch.float32)
        ang=pos[:,None]*inv[None,:]
        c=torch.cos(ang)[None,None,:,:]
        s=torch.sin(ang)[None,None,:,:]
        p=x.reshape(*x.shape[:-1],self.hd//2,2)
        a=p[...,0]; b=p[...,1]
        y=torch.stack((a*c-b*s,b*c+a*s),dim=-1)
        return y.reshape_as(x)

    @torch.inference_mode()
    def forward(self,ids):
        if ids.ndim!=2 or ids.shape[1]<=0 or ids.shape[1]>self.max_seq:
            raise RuntimeError(f"bad inference token geometry {tuple(ids.shape)}")
        b,t=ids.shape
        emb=self.w("tok_embeddings.weight")
        x=F.embedding(ids,emb)
        causal=torch.ones((t,t),device=ids.device,dtype=torch.bool).tril()
        repeat=self.hq//self.hkv

        for li in range(self.layers):
            p=f"layers.{li}."
            n=self.rms(x,self.w(p+"attn_norm.weight"))
            q=F.linear(n,self.w(p+"q_proj.weight"))
            k=F.linear(n,self.w(p+"k_proj.weight"))
            v=F.linear(n,self.w(p+"v_proj.weight"))
            q=q.view(b,t,self.hq,self.hd).transpose(1,2)
            k=k.view(b,t,self.hkv,self.hd).transpose(1,2)
            v=v.view(b,t,self.hkv,self.hd).transpose(1,2)
            q=self.rope(q); k=self.rope(k)
            k=k.repeat_interleave(repeat,dim=1)
            v=v.repeat_interleave(repeat,dim=1)
            scores=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.hd)
            scores=scores.masked_fill(~causal[None,None,:,:],float("-inf"))
            prob=torch.softmax(scores.float(),dim=-1)
            ctx=torch.matmul(prob,v)
            ctx=ctx.transpose(1,2).contiguous().view(b,t,self.d)
            x=x+F.linear(ctx,self.w(p+"o_proj.weight"))

            n=self.rms(x,self.w(p+"ffn_norm.weight"))
            gate=F.linear(n,self.w(p+"gate_proj.weight"))
            up=F.linear(n,self.w(p+"up_proj.weight"))
            ff=F.silu(gate)*up
            x=x+F.linear(ff,self.w(p+"down_proj.weight"))

        x=self.rms(x,self.w("final_norm.weight"))
        return F.linear(x,emb)

def load_tokenizer(project:Path):
    path=project/"artifacts"/"tokenizer_v1"/"tokenizer.json"
    if not path.is_file(): raise RuntimeError(f"tokenizer missing: {path}")
    if sha256_file(path)!=TOKENIZER_SHA:
        raise RuntimeError("frozen tokenizer SHA mismatch")
    return Tokenizer.from_file(str(path)),path

def serialize_messages(tokenizer:Tokenizer,messages:list[dict],assistant_prompt=True):
    ids=[BOS]
    for m in messages:
        role=m["role"]
        if role not in ROLE_PREFIX: raise RuntimeError(f"unsupported role {role}")
        ids.extend(tokenizer.encode(ROLE_PREFIX[role],add_special_tokens=False).ids)
        ids.extend(tokenizer.encode(m["content"],add_special_tokens=False).ids)
        ids.extend(tokenizer.encode("\n",add_special_tokens=False).ids)
    if assistant_prompt:
        ids.extend(tokenizer.encode(
            ROLE_PREFIX["assistant"],add_special_tokens=False
        ).ids)
    return ids

def clean_assistant_text(text:str)->str:
    # Generation is one assistant turn. Stop leaked next-role continuation.
    cut=len(text)
    for marker in ("\nUser:","\nSystem:","\nTool:","\nAssistant:"):
        i=text.find(marker)
        if i>=0: cut=min(cut,i)
    return text[:cut].strip()

@torch.inference_mode()
def generate(model:Model0001Inference,tokenizer:Tokenizer,messages:list[dict],
             *,max_new_tokens=48,temperature=0.0,top_p=1.0,seed=20260903):
    ids=serialize_messages(tokenizer,messages,assistant_prompt=True)
    prompt_tokens=len(ids)
    generated=[]
    gen=torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for _ in range(max_new_tokens):
        context=ids[-model.max_seq:]
        x=torch.tensor(
            [context],
            dtype=torch.long,
            device=next(model.buffers()).device
        )
        logits=model(x)[0,-1].float().cpu()
        if temperature<=0:
            nxt=int(torch.argmax(logits))
        else:
            probs=torch.softmax(logits/temperature,dim=-1)
            if top_p<1.0:
                sp,si=torch.sort(probs,descending=True)
                cum=torch.cumsum(sp,0)
                remove=cum>top_p
                if remove.numel()>1:
                    remove[1:]=remove[:-1].clone()
                    remove[0]=False
                sp[remove]=0
                sp/=sp.sum()
                pick=torch.multinomial(sp,1,generator=gen)
                nxt=int(si[pick])
            else:
                nxt=int(torch.multinomial(probs,1,generator=gen))
        if nxt==EOS: break
        ids.append(nxt); generated.append(nxt)
        decoded=tokenizer.decode(generated,skip_special_tokens=True)
        if any(x in decoded for x in ("\nUser:","\nSystem:","\nTool:","\nAssistant:")):
            break
    text=clean_assistant_text(
        tokenizer.decode(generated,skip_special_tokens=True)
    )
    return {
        "text":text,
        "prompt_tokens":prompt_tokens,
        "generated_tokens":len(generated),
        "token_ids":generated,
    }

@torch.inference_mode()
def masked_validation_ce(model:Model0001Inference,token_windows:np.ndarray,
                         masks:np.ndarray,indices:Iterable[int],batch_size=4):
    indices=list(indices)
    weighted=0.0
    active_total=0
    device=next(model.buffers()).device
    for start in range(0,len(indices),batch_size):
        idx=indices[start:start+batch_size]
        w=torch.from_numpy(token_windows[idx].astype(np.int64,copy=False)).to(device)
        m=torch.from_numpy(masks[idx].astype(np.float32,copy=False)).to(device)
        inp=w[:,:256]
        tgt=w[:,1:257]
        logits=model(inp)
        logp=F.log_softmax(logits.float(),dim=-1)
        nll=-logp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        weighted += float((nll*m).sum().cpu())
        active_total += int(m.sum().cpu())
    if active_total<=0: raise RuntimeError("masked validation has zero active targets")
    return weighted/active_total
