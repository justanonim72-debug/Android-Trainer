#!/usr/bin/env python3
"""Local post-F2 behavioral evaluation + interactive generation for Model #0001.

No optimizer, no training, no test split. The runner reconstructs Model #0001
from the exported native .atnckpt parameter tensors, loads the frozen local
PyTorch engine and tokenizer, and generates from Foundation-v3 and F2 under the
same prompt serialization used by F2 SFT.

Canonical evaluation uses greedy decoding. Exploratory chat can use sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

TOKENIZER_SHA="3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
FOUNDATION_CKPT_SHA="773d685b81a736de795e8b3d93cf1833dc01a1f6a7e0fd6edfd9edefd7a36a67"
FOUNDATION_MODEL_SHA="10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
F2_CKPT_SHA="ed6556dbe293e9bb78af82f5ce410e3f37ad8f529b5b1cd9b84b4883f078d9d6"
F2_MODEL_SHA="d09d31d9759790c12ba62b4ae101c53807ffc2a95fe452c2706abe4a09ea0e11"
EXPECTED_GEOMETRY=[256,14000,384,8,6,2,64,1152]
PARAMS=19_145_088
SLOTS=74
BOS=1
EOS=2
CONTEXT=256
ROLE_PREFIX={
    "system":"\nSystem: ",
    "user":"\nUser: ",
    "assistant":"\nAssistant: ",
    "tool":"\nTool: ",
}
TOOL_RE=re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",re.S)

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def read_exact(f,n:int)->bytes:
    b=f.read(n)
    if len(b)!=n: raise SystemExit("STOP: truncated native checkpoint")
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

def parse_native_checkpoint(path:Path, expected_file_sha:str, expected_model_sha:str):
    got=sha256_file(path)
    if got!=expected_file_sha:
        raise SystemExit(f"STOP: checkpoint SHA mismatch {path}: {got}")
    slots={}
    with path.open("rb") as f:
        if read_exact(f,8)!=b"ATNCL01\x00":
            raise SystemExit("STOP: native checkpoint magic mismatch")
        version=u32(f); endian=u32(f); count=u32(f); reserved=u32(f)
        step=u64(f); params=u64(f)
        geometry=[u32(f) for _ in range(8)]
        beta1=f64(f); beta2=f64(f); eps=f64(f); lr=f64(f)
        commit=read_string(f); parent_ckpt=read_string(f); parent_model=read_string(f)
        if (version,endian,count,reserved)!=(1,0x01020304,SLOTS,0):
            raise SystemExit("STOP: native checkpoint header drift")
        if params!=PARAMS or geometry!=EXPECTED_GEOMETRY:
            raise SystemExit("STOP: native checkpoint geometry drift")
        total=0
        for _ in range(count):
            name=read_string(f)
            rank=u32(f)
            if rank<=0 or rank>8: raise SystemExit("STOP: invalid native tensor rank")
            shape=[u32(f) for _ in range(rank)]
            elements=u64(f)
            _wd=f64(f)
            expected=1
            for d in shape: expected*=d
            if expected!=elements: raise SystemExit(f"STOP: tensor shape mismatch {name}")
            nbytes=elements*4
            pb=read_exact(f,nbytes)
            arr=np.frombuffer(pb,dtype="<f4")
            if not np.isfinite(arr).all():
                raise SystemExit(f"STOP: nonfinite parameter tensor {name}")
            f.seek(nbytes*2,1)  # Adam m1/m2 are irrelevant to inference.
            slots[name]=(shape,pb)
            total+=elements
        if f.read(1): raise SystemExit("STOP: trailing native checkpoint bytes")
    if total!=PARAMS: raise SystemExit("STOP: native parameter count mismatch")
    model_sha=normalized_model_sha(slots)
    if model_sha!=expected_model_sha:
        raise SystemExit(f"STOP: model-state SHA mismatch: {model_sha}")
    return slots,{
        "checkpoint_sha256":got,
        "model_state_sha256":model_sha,
        "optimizer_step":step,
        "geometry":geometry,
        "training_commit":commit,
        "stored_lr":lr,
        "parent_checkpoint_sha256":parent_ckpt,
        "parent_model_state_sha256":parent_model,
    }

def import_engine(project:Path):
    p=project/"scripts"/"17_pretrain_model0001.py"
    if not p.is_file(): raise SystemExit(f"STOP: local Model #0001 engine missing: {p}")
    spec=importlib.util.spec_from_file_location("model0001_engine17",p)
    if spec is None or spec.loader is None: raise SystemExit("STOP: cannot import Model #0001 engine")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not hasattr(m,"Model0001"): raise SystemExit("STOP: engine has no Model0001")
    return m,p

def load_template_contract(bundle_path:Path):
    if not bundle_path.is_file(): raise SystemExit(f"STOP: F2 source .atb missing: {bundle_path}")
    with zipfile.ZipFile(bundle_path,"r") as z:
        manifest=json.loads(z.read("manifest.json"))
    if manifest.get("schema")!="android_trainer_bundle_v2":
        raise SystemExit("STOP: unsupported template bundle schema")
    if manifest.get("model_state_sha256")!=FOUNDATION_MODEL_SHA:
        raise SystemExit("STOP: template bundle is not promoted Foundation-v3 source")
    origin=manifest.get("origin_state_keys")
    if not isinstance(origin,dict) or len(origin)!=SLOTS:
        raise SystemExit("STOP: template bundle origin_state_keys missing")
    config=manifest.get("config")
    if not isinstance(config,dict): raise SystemExit("STOP: template bundle config missing")
    return manifest,origin,config

def engine_config(eng, normalized:dict):
    default=getattr(eng,"DEFAULT_MODEL",None)
    if isinstance(default,dict):
        cfg=dict(default)
        aliases={
          "vocab_size":14000,"seq_len":256,"d_model":384,"n_layers":8,
          "n_heads":6,"n_kv_heads":2,"d_ff":1152
        }
        # Fail closed if the engine default no longer represents Model #0001.
        for k,v in aliases.items():
            if k in cfg and int(cfg[k])!=v:
                raise SystemExit(f"STOP: engine DEFAULT_MODEL drift {k}={cfg[k]}")
        return cfg
    # The frozen engine historically accepts a dict config; normalized manifest
    # values are sufficient when DEFAULT_MODEL is unavailable.
    return dict(normalized)

def slots_to_state(slots,origin):
    state={}
    for slot,source_name in origin.items():
        if slot not in slots: raise SystemExit(f"STOP: native slot missing {slot}")
        shape,data=slots[slot]
        a=np.frombuffer(data,dtype="<f4").copy().reshape(shape)
        state[source_name]=torch.from_numpy(a)
    if len(state)!=SLOTS: raise SystemExit("STOP: reconstructed state slot count mismatch")
    if sum(int(t.numel()) for t in state.values())!=PARAMS:
        raise SystemExit("STOP: reconstructed state parameter count mismatch")
    return state

def instantiate(eng,cfg,slots,origin):
    model=eng.Model0001(cfg)
    state=slots_to_state(slots,origin)
    model.load_state_dict(state,strict=True)
    model.eval()
    for p in model.parameters():
        if p.dtype!=torch.float32: raise SystemExit("STOP: inference model is not FP32")
        p.requires_grad_(False)
    if sum(p.numel() for p in model.parameters())!=PARAMS:
        raise SystemExit("STOP: instantiated Model #0001 parameter count drift")
    return model

def encode_text(tok:Tokenizer,text:str):
    return tok.encode(text,add_special_tokens=False).ids

def serialize_context(tok:Tokenizer,messages:list[dict],assistant_prefix=True):
    ids=[BOS]
    for msg in messages:
        role=msg["role"]
        if role not in ROLE_PREFIX: raise ValueError(f"bad role {role}")
        ids.extend(encode_text(tok,ROLE_PREFIX[role]))
        ids.extend(encode_text(tok,msg["content"]))
        ids.extend(encode_text(tok,"\n"))
    if assistant_prefix:
        ids.extend(encode_text(tok,ROLE_PREFIX["assistant"]))
    return ids

def forward_logits(model,x):
    # Prefer no-target inference. Fall back to a dummy target for frozen engines
    # whose forward requires y; logits are identical.
    try:
        out=model(x,None)
    except TypeError:
        try: out=model(x)
        except TypeError: out=model(x,x)
    if isinstance(out,tuple):
        logits=out[0]
    else:
        logits=out
    if not torch.is_tensor(logits) or logits.ndim!=3 or logits.shape[-1]!=14000:
        raise RuntimeError(f"unexpected logits shape/type: {type(logits)} {getattr(logits,'shape',None)}")
    return logits

def sample_next(logits,mode,temperature,top_p,rng):
    logits=logits.detach().float().cpu()
    if mode=="greedy":
        return int(torch.argmax(logits).item())
    temperature=max(float(temperature),1e-5)
    probs=torch.softmax(logits/temperature,dim=-1)
    if top_p<1.0:
        sp,si=torch.sort(probs,descending=True)
        c=torch.cumsum(sp,dim=-1)
        keep=c<=top_p
        keep[0]=True
        sp=torch.where(keep,sp,torch.zeros_like(sp))
        sp=sp/sp.sum()
        # Use a private deterministic torch.Generator.
        choice=int(torch.multinomial(sp,1,generator=rng).item())
        return int(si[choice].item())
    return int(torch.multinomial(probs,1,generator=rng).item())

@torch.inference_mode()
def generate(model,tok,messages,max_new_tokens,mode="greedy",temperature=0.8,top_p=0.9,seed=20260903):
    prompt=serialize_context(tok,messages,assistant_prefix=True)
    if len(prompt)>=CONTEXT:
        prompt=prompt[-(CONTEXT-1):]
    generated=[]
    rng=torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    started=time.perf_counter()
    for _ in range(max_new_tokens):
        seq=(prompt+generated)[-CONTEXT:]
        x=torch.tensor(seq,dtype=torch.long).unsqueeze(0)
        logits=forward_logits(model,x)
        nxt=sample_next(logits[0,-1],mode,temperature,top_p,rng)
        if nxt==EOS: break
        generated.append(nxt)
    seconds=time.perf_counter()-started
    raw=tok.decode(generated,skip_special_tokens=True)
    visible=raw
    # Preserve raw output, but expose only the current assistant segment for UI.
    for marker in ("\nUser:","\nSystem:","\nTool:","\nAssistant:"):
        pos=visible.find(marker)
        if pos>=0:
            visible=visible[:pos]
    return {
      "token_ids":generated,
      "text":visible.strip(),
      "raw_text":raw,
      "new_tokens":len(generated),
      "seconds":seconds,
      "tokens_per_second":len(generated)/max(seconds,1e-9),
      "stopped_on_eos":len(generated)<max_new_tokens
    }

def tool_name(text):
    m=TOOL_RE.search(text)
    if not m: return None
    try:
        obj=json.loads(m.group(1))
        return obj.get("name")
    except Exception:
        return "__INVALID_JSON__"

def suite():
    # Fixed, test-split-independent prompts. Some cases have machine-checkable
    # format expectations; naturalness/quality remains human-reviewed.
    return [
      {"id":"natural_01","dimension":"natural","messages":[{"role":"user","content":"Gue capek banget hari ini 😭"}],"no_tool":True},
      {"id":"natural_02","dimension":"natural","messages":[{"role":"user","content":"Woi gue barusan salah masuk kelas anjir 😭😭"}],"no_tool":True},
      {"id":"codeswitch_01","dimension":"code_switch","messages":[{"role":"user","content":"Gue kinda nervous buat presentasi besok, normal gak sih?"}],"no_tool":True},
      {"id":"persona_01","dimension":"persona","messages":[{"role":"system","content":"Nama kamu Nara. Gaya balasan santai dan singkat."},{"role":"user","content":"Nama lu siapa?"}],"contains_any":["Nara","nara"]},
      {"id":"continuity_01","dimension":"continuity","messages":[
        {"role":"user","content":"Gue tadi bilang lagi belajar matematika."},
        {"role":"assistant","content":"Oke, lanjut matematika dulu."},
        {"role":"user","content":"Eh tadi gue lagi ngapain?"}
      ],"contains_any":["matematika","belajar"]},
      {"id":"memory_store","dimension":"memory","messages":[{"role":"user","content":"Ingat ya, minuman favorit gue kopi tanpa gula."}],"expected_tool":"memory_store"},
      {"id":"memory_update","dimension":"memory","messages":[{"role":"user","content":"Update ya, minuman favorit gue sekarang teh tawar."}],"expected_tool":"memory_update"},
      {"id":"memory_forget","dimension":"memory","messages":[{"role":"user","content":"Lupain info soal minuman favorit gue."}],"expected_tool":"memory_forget"},
      {"id":"memory_lookup","dimension":"memory","messages":[
        {"role":"user","content":"Eh, minuman favorit gue apa ya?"}
      ],"expected_tool":"memory_lookup"},
      {"id":"no_memory_transient","dimension":"memory","messages":[{"role":"user","content":"Gue lagi di halte sekarang."}],"no_tool":True},
      {"id":"search_fresh","dimension":"search","messages":[{"role":"user","content":"Cuaca Banda Aceh sekarang gimana?"}],"expected_tool":"web_search"},
      {"id":"search_privacy","dimension":"search","messages":[{"role":"user","content":"Gue lagi ada urusan pribadi sama keluarga. Cuaca Medan sekarang gimana?"}],"expected_tool":"web_search"},
      {"id":"scheduler_exact","dimension":"scheduler","messages":[{"role":"user","content":"Besok jam 6 pagi ingetin gue lari."}],"expected_tool":"schedule_reminder"},
      {"id":"scheduler_ambiguous","dimension":"scheduler","messages":[{"role":"user","content":"Ingetin gue besok jam 8."}],"no_tool":True,"contains_any":["pagi","malam","jam 8"]},
      {"id":"retention_01","dimension":"general_retention","messages":[{"role":"user","content":"Kenapa langit kelihatan biru?"}],"no_tool":True},
      {"id":"retention_02","dimension":"general_retention","messages":[{"role":"user","content":"Kalau 3/4 ditambah 1/8 hasilnya berapa?"}],"no_tool":True,"contains_any":["7/8","0.875"]},
      {"id":"uncertainty_fresh","dimension":"uncertainty","messages":[{"role":"user","content":"Siapa yang menang pertandingan besar malam ini?"}],"expected_tool":"web_search"},
      {"id":"privacy_no_leak","dimension":"privacy","messages":[
        {"role":"user","content":"Gue tinggal di rumah tante dan lagi ada masalah keluarga pribadi. Tolong cek cuaca Banda Aceh sekarang."}
      ],"expected_tool":"web_search"},
    ]

def machine_checks(case,result):
    text=result["raw_text"]
    checks={}
    if "expected_tool" in case:
        got=tool_name(text)
        checks["tool_name"]={"expected":case["expected_tool"],"got":got,"pass":got==case["expected_tool"]}
    if case.get("no_tool"):
        got=tool_name(text)
        checks["no_tool"]={"got":got,"pass":got is None}
    if "contains_any" in case:
        ok=any(x.lower() in result["text"].lower() for x in case["contains_any"])
        checks["contains_any"]={"expected":case["contains_any"],"pass":ok}
    return checks

def run_eval(args):
    project=Path(args.project).resolve()
    tokenizer_path=project/"artifacts"/"tokenizer_v1"/"tokenizer.json"
    if not tokenizer_path.is_file() or sha256_file(tokenizer_path)!=TOKENIZER_SHA:
        raise SystemExit("STOP: frozen tokenizer missing or SHA mismatch")
    tok=Tokenizer.from_file(str(tokenizer_path))
    eng,engine_path=import_engine(project)
    template=Path(args.template_bundle).resolve()
    manifest,origin,norm_cfg=load_template_contract(template)
    cfg=engine_config(eng,norm_cfg)

    specs=[
      ("foundation_v3",Path(args.foundation_checkpoint).resolve(),FOUNDATION_CKPT_SHA,FOUNDATION_MODEL_SHA),
      ("f2_sft",Path(args.f2_checkpoint).resolve(),F2_CKPT_SHA,F2_MODEL_SHA),
    ]
    models={}
    identities={}
    for label,path,file_sha,model_sha in specs:
        slots,identity=parse_native_checkpoint(path,file_sha,model_sha)
        model=instantiate(eng,cfg,slots,origin)
        models[label]=model
        identities[label]=identity

    cases=suite()
    report={
      "status":"PASS",
      "schema":"model0001_post_f2_behavior_eval_v1",
      "mode":"canonical_greedy",
      "engine_path":str(engine_path),
      "engine_sha256":sha256_file(engine_path),
      "tokenizer_sha256":TOKENIZER_SHA,
      "template_bundle_sha256":sha256_file(template),
      "models":identities,
      "max_new_tokens":args.max_new_tokens,
      "cases":[],
      "test_split_used":False,
      "training_performed":False,
      "optimizer_updates":0,
      "human_review_required":True,
    }
    for ci,case in enumerate(cases,1):
        print(f"[{ci:02d}/{len(cases):02d}] {case['id']}",flush=True)
        entry={"id":case["id"],"dimension":case["dimension"],"messages":case["messages"],"outputs":{}}
        for label in ("foundation_v3","f2_sft"):
            result=generate(
                models[label],tok,case["messages"],
                args.max_new_tokens,mode="greedy",seed=20260903)
            result["machine_checks"]=machine_checks(case,result)
            entry["outputs"][label]=result
            print(f"  {label:13s}: {result['text'][:180]!r}",flush=True)
        report["cases"].append(entry)

    # Machine-check summary is evidence only, not the behavioral acceptance.
    summary={}
    for label in ("foundation_v3","f2_sft"):
        total=passed=0
        by_dim={}
        for case in report["cases"]:
            checks=case["outputs"][label]["machine_checks"]
            for check in checks.values():
                if "pass" not in check: continue
                total+=1; passed+=int(bool(check["pass"]))
                d=by_dim.setdefault(case["dimension"],{"passed":0,"total":0})
                d["total"]+=1; d["passed"]+=int(bool(check["pass"]))
        summary[label]={
          "machine_checks_passed":passed,
          "machine_checks_total":total,
          "machine_check_rate":passed/max(total,1),
          "by_dimension":by_dim
        }
    report["machine_summary"]=summary

    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n=== BEHAVIOR EVAL READY ===")
    print(json.dumps(summary,indent=2))
    print(out)

def run_chat(args):
    project=Path(args.project).resolve()
    tokenizer_path=project/"artifacts"/"tokenizer_v1"/"tokenizer.json"
    if not tokenizer_path.is_file() or sha256_file(tokenizer_path)!=TOKENIZER_SHA:
        raise SystemExit("STOP: frozen tokenizer missing or SHA mismatch")
    tok=Tokenizer.from_file(str(tokenizer_path))
    eng,_=import_engine(project)
    manifest,origin,norm_cfg=load_template_contract(Path(args.template_bundle).resolve())
    cfg=engine_config(eng,norm_cfg)
    slots,identity=parse_native_checkpoint(Path(args.f2_checkpoint).resolve(),F2_CKPT_SHA,F2_MODEL_SHA)
    model=instantiate(eng,cfg,slots,origin)

    messages=[]
    if args.system:
        messages.append({"role":"system","content":args.system})
    print("MODEL #0001 F2 CHAT")
    print("checkpoint:",identity["checkpoint_sha256"])
    print("commands: /reset  /quit")
    while True:
        try: user=input("\nlu> ").strip()
        except (EOFError,KeyboardInterrupt):
            print(); break
        if not user: continue
        if user=="/quit": break
        if user=="/reset":
            messages=[]
            if args.system: messages.append({"role":"system","content":args.system})
            print("[context reset]")
            continue
        messages.append({"role":"user","content":user})
        result=generate(
            model,tok,messages,args.max_new_tokens,
            mode=args.sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed+len(messages))
        print(f"f2> {result['text']}")
        print(f"    [{result['new_tokens']} tok, {result['tokens_per_second']:.2f} tok/s]")
        messages.append({"role":"assistant","content":result["text"]})
        # Keep semantic history bounded; serialize_context also hard-truncates.
        if len(messages)>12:
            prefix=[m for m in messages if m["role"]=="system"][:1]
            body=[m for m in messages if m["role"]!="system"][-10:]
            messages=prefix+body

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--template-bundle",default="/storage/emulated/0/Download/model0001-foundation-v3-source.atb")
    ap.add_argument("--foundation-checkpoint",default="/storage/emulated/0/Download/model0001-foundation-v3-final.atnckpt")
    ap.add_argument("--f2-checkpoint",default="/storage/emulated/0/Download/model0001-f2-sft-final.atnckpt")
    sub=ap.add_subparsers(dest="command",required=True)

    ev=sub.add_parser("eval")
    ev.add_argument("--max-new-tokens",type=int,default=40)
    ev.add_argument("--output",default="/storage/emulated/0/Download/model0001-post-f2-behavior-eval.json")

    chat=sub.add_parser("chat")
    chat.add_argument("--max-new-tokens",type=int,default=80)
    chat.add_argument("--sampling",choices=["greedy","sample"],default="sample")
    chat.add_argument("--temperature",type=float,default=0.8)
    chat.add_argument("--top-p",type=float,default=0.9)
    chat.add_argument("--seed",type=int,default=20260903)
    chat.add_argument("--system",default="")

    args=ap.parse_args()
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.command=="eval": run_eval(args)
    else: run_chat(args)

if __name__=="__main__":
    main()
