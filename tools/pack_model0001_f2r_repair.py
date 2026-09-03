#!/usr/bin/env python3
"""Pack F2R replacement SFT with record-isolated assistant-only windows.

Key difference from frozen F2 packer:
- each conversation is serialized and chunked independently;
- unrelated records are NEVER concatenated inside one 256-target window;
- assistant-content-only loss is preserved;
- repair behavior core must remain 22..32% of scored assistant targets;
- human dialogue remains >=68% of scored assistant targets.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_SHA="3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
VOCAB=14000
BOS=1
EOS=2
UNK=3
SEQ=256
WINDOW=257
ROLE_PREFIX={
  "system":"\nSystem: ",
  "user":"\nUser: ",
  "assistant":"\nAssistant: ",
  "tool":"\nTool: ",
}

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def encode(tok:Tokenizer,text:str)->list[int]:
    ids=tok.encode(text,add_special_tokens=False).ids
    if any(x<0 or x>=VOCAB for x in ids):
        raise SystemExit("STOP: tokenizer produced out-of-vocab token")
    return ids

def serialize_record(tok,row):
    ids=[BOS]
    score=[0]
    assistant_scored=0
    for msg in row["messages"]:
        prefix=encode(tok,ROLE_PREFIX[msg["role"]])
        ids.extend(prefix); score.extend([0]*len(prefix))
        content=encode(tok,msg["content"])
        active=1 if msg["role"]=="assistant" else 0
        ids.extend(content); score.extend([active]*len(content))
        assistant_scored+=active*len(content)
        nl=encode(tok,"\n")
        ids.extend(nl); score.extend([0]*len(nl))
    terminal=1 if row["messages"][-1]["role"]=="assistant" else 0
    ids.append(EOS); score.append(terminal)
    assistant_scored+=terminal
    return ids,score,assistant_scored

def pack_split(tok,records,tokens_path,mask_path):
    windows=[]
    masks=[]
    active_counts=[]
    family_scored=collections.Counter()
    family_records=collections.Counter()
    record_window_counts=collections.Counter()
    padded_targets=0
    long_record_windows=0

    for row in records:
        ids,score,scored=serialize_record(tok,row)
        if scored<=0: continue
        family=row["source_family"]
        family_scored[family]+=scored
        family_records[family]+=1

        # Record isolation: chunk THIS record only.
        start=0
        local_windows=0
        while start<len(ids)-1:
            chunk=ids[start:start+WINDOW]
            target_score=score[start+1:start+WINDOW]
            if len(chunk)<WINDOW:
                missing=WINDOW-len(chunk)
                chunk=chunk+[EOS]*missing
            if len(target_score)<SEQ:
                missing=SEQ-len(target_score)
                padded_targets+=missing
                target_score=target_score+[0]*missing
            if len(chunk)!=WINDOW or len(target_score)!=SEQ:
                raise SystemExit("STOP: F2R pack geometry bug")
            active=sum(target_score)
            if active>0:
                windows.append(chunk)
                masks.append(target_score)
                active_counts.append(active)
                local_windows+=1
            start+=SEQ
        record_window_counts[row["id"]]=local_windows
        if local_windows>1: long_record_windows+=local_windows

    if not windows:
        raise SystemExit("STOP: F2R split produced zero windows")

    unk=0
    with tokens_path.open("wb") as tf, mask_path.open("wb") as mf:
        for ids,mask in zip(windows,masks):
            for x in ids:
                if x==UNK: unk+=1
                tf.write(struct.pack("<H",x))
            mf.write(bytes(mask))

    total_scored=sum(active_counts)
    core_scored=family_scored.get("repair_behavior_core",0)
    human_scored=sum(
        v for k,v in family_scored.items()
        if k in ("repair_human_natural","repair_human_task")
    )
    return {
      "records":sum(family_records.values()),
      "windows":len(windows),
      "targets_per_window":SEQ,
      "input_tokens_per_window":WINDOW,
      "scored_target_tokens":total_scored,
      "mask_density":total_scored/(len(windows)*SEQ),
      "min_scored_targets_per_window":min(active_counts),
      "max_scored_targets_per_window":max(active_counts),
      "mean_scored_targets_per_window":total_scored/len(active_counts),
      "unk_count":unk,
      "source_family_records":dict(family_records),
      "scored_tokens_by_family":dict(family_scored),
      "behavior_core_scored_fraction":core_scored/max(1,total_scored),
      "human_scored_fraction":human_scored/max(1,total_scored),
      "record_isolated_packing":True,
      "cross_record_windows":0,
      "padded_unscored_target_positions":padded_targets,
      "multiwindow_record_windows":long_record_windows,
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
    source=project/"data"/"f2r_repair"/"friend_f2r_repair_source.jsonl"
    source_report_path=project/"data"/"f2r_repair"/"F2R_SOURCE_REPORT.json"
    tokenizer_path=project/"artifacts"/"tokenizer_v1"/"tokenizer.json"
    for p in (source,source_report_path,tokenizer_path):
        if not p.is_file(): raise SystemExit(f"STOP: missing F2R pack input {p}")
    if sha256(tokenizer_path)!=TOKENIZER_SHA:
        raise SystemExit("STOP: frozen tokenizer SHA drift")
    source_report=json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("status")!="PASS" or source_report.get("sha256")!=sha256(source):
        raise SystemExit("STOP: F2R source report mismatch")

    records={"train":[],"validation":[]}
    seen=set()
    with source.open("r",encoding="utf-8") as f:
        for raw in f:
            if not raw.strip(): continue
            row=json.loads(raw)
            rid=row["id"]
            if rid in seen: raise SystemExit("STOP: duplicate F2R id")
            seen.add(rid)
            split=row["split"]
            if split not in records: raise SystemExit("STOP: forbidden F2R split")
            records[split].append(row)

    tok=Tokenizer.from_file(str(tokenizer_path))
    outdir=project/"artifacts"/"model0001_f2r_repair"
    outdir.mkdir(parents=True,exist_ok=True)
    train=pack_split(
        tok,records["train"],
        outdir/"train.tokens.u16",
        outdir/"train.lossmask.u8")
    val=pack_split(
        tok,records["validation"],
        outdir/"validation.tokens.u16",
        outdir/"validation.lossmask.u8")

    if train["unk_count"]!=0 or val["unk_count"]!=0:
        raise SystemExit("STOP: ByteLevel tokenizer produced UNK")
    if train["cross_record_windows"]!=0 or val["cross_record_windows"]!=0:
        raise SystemExit("STOP: F2R cross-record window detected")
    if not (0.22<=train["behavior_core_scored_fraction"]<=0.32):
        raise SystemExit(
            "STOP: F2R train behavior-core fraction outside 0.22..0.32: "
            f"{train['behavior_core_scored_fraction']:.6f}"
        )
    if train["human_scored_fraction"]<0.68:
        raise SystemExit("STOP: F2R human scored fraction below 0.68")
    if val["scored_target_tokens"]<512:
        raise SystemExit("STOP: F2R validation too small")

    report={
      "status":"PASS",
      "schema":"model0001_f2r_repair_pack_report_v1",
      "stage_name":"friend_f2r_repair_sft",
      "source_policy":"RETRAIN_FROM_FOUNDATION_V3_NOT_F2_FINAL",
      "objective":"assistant_content_only_cross_entropy",
      "format":"record_isolated_257_u16_plus_256_u8_mask",
      "source_jsonl":str(source),
      "source_jsonl_sha256":sha256(source),
      "source_report_sha256":sha256(source_report_path),
      "tokenizer_sha256":TOKENIZER_SHA,
      "train":train,
      "validation":val,
      "hard_guards":{
        "assistant_only_loss":True,
        "record_isolated_packing":True,
        "cross_record_windows":0,
        "human_dialogue_majority":True,
        "behavior_core_scored_fraction_locked":[0.22,0.32],
        "tokenizer_changed":False,
        "architecture_changed":False,
        "old_f2_validation_used_for_train":False,
        "behavior_eval_prompts_used_for_train":False,
        "test_split_used":False,
        "training_started":False
      }
    }
    rp=outdir/"F2R_PACK_REPORT.json"
    rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    lock={
      "status":"LOCKED",
      "schema":"model0001_f2r_dataset_lock_v1",
      "report_sha256":sha256(rp),
      "train_tokens_sha256":train["tokens_sha256"],
      "train_mask_sha256":train["mask_sha256"],
      "validation_tokens_sha256":val["tokens_sha256"],
      "validation_mask_sha256":val["mask_sha256"],
      "tokenizer_sha256":TOKENIZER_SHA,
      "test_split_used":False
    }
    (outdir/"DATASET_LOCK.json").write_text(
        json.dumps(lock,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
