#!/usr/bin/env python3
"""Pack Friend-Core F2 SFT into fixed 257-token windows + 256-target masks.

Every model input token remains visible as causal context. Only assistant
CONTENT tokens (plus a terminal EOS after an assistant-ended record) receive
loss. Role labels, system/user text, and tool-return text are context-only.

No tokenizer changes; no padding token is trained. Final partial windows are
right-filled with EOS while all fill targets stay masked out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_SHA="3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
VOCAB=14000
BOS=1
EOS=2
UNK=3
SEQ=256
WINDOW=257
SEED="20260903-f2-pack-v1"
ROLE_PREFIX={
  "system":"\nSystem: ",
  "user":"\nUser: ",
  "assistant":"\nAssistant: ",
  "tool":"\nTool: ",
}

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def stable_rank(rid:str)->str:
    return hashlib.sha256((SEED+":"+rid).encode()).hexdigest()

def encode(tok:Tokenizer,text:str)->list[int]:
    ids=tok.encode(text,add_special_tokens=False).ids
    if any(x<0 or x>=VOCAB for x in ids):
        raise SystemExit("STOP: tokenizer produced out-of-vocab id")
    return ids

def serialize_record(tok:Tokenizer,row:dict):
    ids=[BOS]
    score=[0]  # score flag belongs to the token itself as a prediction target.
    assistant_tokens=0
    for msg in row["messages"]:
        role=msg["role"]
        prefix=encode(tok,ROLE_PREFIX[role])
        ids.extend(prefix); score.extend([0]*len(prefix))
        content=encode(tok,msg["content"])
        active=1 if role=="assistant" else 0
        ids.extend(content); score.extend([active]*len(content))
        if active: assistant_tokens+=len(content)
        newline=encode(tok,"\n")
        ids.extend(newline); score.extend([0]*len(newline))
    terminal_active=1 if row["messages"][-1]["role"]=="assistant" else 0
    ids.append(EOS); score.append(terminal_active)
    assistant_tokens+=terminal_active
    return ids,score,assistant_tokens

def build_split(tok:Tokenizer,records:list[dict],tokens_path:Path,mask_path:Path):
    records=sorted(records,key=lambda r:stable_rank(r["id"]))
    stream_ids=[]
    stream_score=[]
    family_records=Counter()
    source_tokens=Counter()
    scored_source_tokens=Counter()
    serialized_records=0
    for row in records:
        ids,score,assistant=serialize_record(tok,row)
        if assistant<=0:
            continue
        stream_ids.extend(ids)
        stream_score.extend(score)
        family=row.get("source_family","unknown")
        family_records[family]+=1
        source_tokens[family]+=len(ids)
        scored_source_tokens[family]+=assistant
        serialized_records+=1

    if len(stream_ids)!=len(stream_score) or len(stream_ids)<2:
        raise SystemExit("STOP: invalid serialized SFT stream")

    windows=[]
    masks=[]
    active_counts=[]
    start=0
    while start<len(stream_ids)-1:
        end=min(start+WINDOW,len(stream_ids))
        ids=stream_ids[start:end]
        # target j predicts token start+j+1, therefore mask is score of the
        # NEXT token in the serialized stream.
        target_score=stream_score[start+1:min(start+WINDOW,len(stream_score))]
        if len(ids)<WINDOW:
            ids=ids+[EOS]*(WINDOW-len(ids))
        if len(target_score)<SEQ:
            target_score=target_score+[0]*(SEQ-len(target_score))
        if len(ids)!=WINDOW or len(target_score)!=SEQ:
            raise SystemExit("STOP: SFT window geometry bug")
        active=sum(target_score)
        if active>0:
            windows.append(ids)
            masks.append(target_score)
            active_counts.append(active)
        start+=SEQ

    if not windows:
        raise SystemExit("STOP: SFT split produced zero scored windows")

    with tokens_path.open("wb") as tf, mask_path.open("wb") as mf:
        unk=0
        for ids,mask in zip(windows,masks):
            for x in ids:
                tf.write(struct.pack("<H",x))
                if x==UNK: unk+=1
            mf.write(bytes(mask))

    return {
      "records":serialized_records,
      "windows":len(windows),
      "input_tokens_per_window":WINDOW,
      "targets_per_window":SEQ,
      "scored_target_tokens":sum(active_counts),
      "all_target_positions":len(windows)*SEQ,
      "mask_density":sum(active_counts)/(len(windows)*SEQ),
      "min_scored_targets_per_window":min(active_counts),
      "max_scored_targets_per_window":max(active_counts),
      "mean_scored_targets_per_window":sum(active_counts)/len(active_counts),
      "unk_count":unk,
      "source_family_records":dict(family_records),
      "serialized_tokens_by_family":dict(source_tokens),
      "scored_tokens_by_family":dict(scored_source_tokens),
      "tokens_path":str(tokens_path),
      "tokens_sha256":sha256(tokens_path),
      "mask_path":str(mask_path),
      "mask_sha256":sha256(mask_path),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    args=ap.parse_args()
    project=Path(args.project).resolve()
    source=project/"data"/"f2_sft"/"friend_f2_sft_source.jsonl"
    source_audit=source.with_suffix(source.suffix+".audit.json")
    tok_path=project/"artifacts"/"tokenizer_v1"/"tokenizer.json"
    if not source.is_file() or not source_audit.is_file():
        raise SystemExit("STOP: validate F2 source JSONL before packing")
    audit=json.loads(source_audit.read_text(encoding="utf-8"))
    if audit.get("status")!="PASS" or audit.get("sha256")!=sha256(source):
        raise SystemExit("STOP: F2 source validator audit mismatch")
    if audit.get("test_split_used") is not False:
        raise SystemExit("STOP: F2 source validator reports test use")
    if not tok_path.is_file() or sha256(tok_path)!=TOKENIZER_SHA:
        raise SystemExit("STOP: frozen tokenizer identity mismatch")

    records={"train":[],"validation":[]}
    seen=set()
    with source.open("r",encoding="utf-8") as f:
        for raw in f:
            if not raw.strip(): continue
            row=json.loads(raw)
            rid=row["id"]
            if rid in seen: raise SystemExit("STOP: duplicate SFT id")
            seen.add(rid)
            split=row["split"]
            if split not in records: raise SystemExit("STOP: forbidden SFT split")
            records[split].append(row)

    tok=Tokenizer.from_file(str(tok_path))
    outdir=project/"artifacts"/"model0001_f2_sft"
    outdir.mkdir(parents=True,exist_ok=True)
    train=build_split(
        tok,records["train"],
        outdir/"train.tokens.u16",
        outdir/"train.lossmask.u8")
    val=build_split(
        tok,records["validation"],
        outdir/"validation.tokens.u16",
        outdir/"validation.lossmask.u8")

    if train["unk_count"]!=0 or val["unk_count"]!=0:
        raise SystemExit("STOP: frozen ByteLevel tokenizer unexpectedly produced UNK")
    if train["mask_density"]<=0 or val["mask_density"]<=0:
        raise SystemExit("STOP: empty assistant-only objective")
    if val["scored_target_tokens"]<256:
        raise SystemExit("STOP: F2 validation has too few scored tokens")

    report={
      "status":"PASS",
      "schema":"model0001_f2_sft_pack_report_v1",
      "objective":"assistant_content_only_cross_entropy",
      "format":"fixed_257_u16_tokens_plus_256_u8_target_mask",
      "source_jsonl":str(source),
      "source_jsonl_sha256":sha256(source),
      "source_validator_audit_sha256":sha256(source_audit),
      "tokenizer_sha256":TOKENIZER_SHA,
      "context_tokens":SEQ,
      "role_prefixes":ROLE_PREFIX,
      "train":train,
      "validation":val,
      "hard_guards":{
        "assistant_only_loss":True,
        "system_tokens_scored":False,
        "user_tokens_scored":False,
        "tool_return_tokens_scored":False,
        "tokenizer_changed":False,
        "architecture_changed":False,
        "test_split_created":False,
        "test_split_used":False,
        "foundation_v3_validation_used_for_train":False,
        "openai_teacher_outputs_used":False,
        "production_lr_locked":False,
        "training_started":False
      }
    }
    rp=outdir/"F2_SFT_PACK_REPORT.json"
    rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    lock={
      "status":"LOCKED",
      "schema":"model0001_f2_sft_dataset_lock_v1",
      "report_sha256":sha256(rp),
      "source_sha256":sha256(source),
      "train_tokens_sha256":train["tokens_sha256"],
      "train_mask_sha256":train["mask_sha256"],
      "validation_tokens_sha256":val["tokens_sha256"],
      "validation_mask_sha256":val["mask_sha256"],
      "tokenizer_sha256":TOKENIZER_SHA,
      "test_split_used":False,
    }
    lp=outdir/"DATASET_LOCK.json"
    lp.write_text(json.dumps(lock,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
