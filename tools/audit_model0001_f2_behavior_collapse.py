#!/usr/bin/env python3
"""Empirical audit for Model #0001 post-F2 generation collapse.

Reads ONLY already-created F2 SFT source/pack artifacts plus the canonical
post-F2 behavior report. It does not train, mutate artifacts, or touch test data.

The audit measures:
- assistant response exact-duplicate pressure;
- assistant start-prefix concentration;
- repeated n-gram concentration;
- source-family contribution;
- tool-call supervision distribution;
- generic-acknowledgement pressure;
- whether collapsed F2 generation n-grams overlap highly frequent training
  assistant n-grams;
- whether old packing mixed unrelated records inside a single 256-token window.

It is deliberately diagnostic: root-cause flags are emitted only when measured
thresholds are crossed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_SHA="3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
EXPECTED_F2_MODEL_SHA="d09d31d9759790c12ba62b4ae101c53807ffc2a95fe452c2706abe4a09ea0e11"
EXPECTED_BEHAVIOR_SCHEMA="model0001_post_f2_behavior_eval_report_v1"
EXPECTED_SOURCE_SCHEMA="friend_core_f2_sft_source_audit_v1"
SEQ=256
BOS=1
EOS=2
ROLE_PREFIX={
  "system":"\nSystem: ",
  "user":"\nUser: ",
  "assistant":"\nAssistant: ",
  "tool":"\nTool: ",
}
TOOL_RE=re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",re.S)
WORD_RE=re.compile(r"[A-Za-zÀ-ÿ0-9]+",re.UNICODE)

GENERIC_ACK_PREFIXES={
    "sama sama","oke","ok","sip","siap","iya","ya","baik","wkwk","waduh",
    "makasih","terima kasih"
}

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def norm_text(text:str)->str:
    return " ".join(WORD_RE.findall(text.lower()))

def words(text:str)->list[str]:
    return WORD_RE.findall(text.lower())

def ngrams(ws:list[str],n:int):
    for i in range(0,max(0,len(ws)-n+1)):
        yield " ".join(ws[i:i+n])

def pct(x,y):
    return float(x)/float(y) if y else 0.0

def top(counter:collections.Counter,n=20):
    total=sum(counter.values())
    return [
        {"value":k,"count":v,"fraction":pct(v,total)}
        for k,v in counter.most_common(n)
    ]

def encode(tok:Tokenizer,text:str)->list[int]:
    return tok.encode(text,add_special_tokens=False).ids

def serialize_record(tok:Tokenizer,row:dict):
    ids=[BOS]
    record_ids=[row["id"]]
    for msg in row["messages"]:
        ids.extend(encode(tok,ROLE_PREFIX[msg["role"]]))
        ids.extend(encode(tok,msg["content"]))
        ids.extend(encode(tok,"\n"))
    ids.append(EOS)
    return ids

def tool_name(text:str):
    m=TOOL_RE.search(text)
    if not m: return None
    try:
        obj=json.loads(m.group(1))
        return str(obj.get("name"))
    except Exception:
        return "__INVALID_JSON__"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--behavior-report",default="/storage/emulated/0/Download/model0001-post-f2-behavior-eval.json")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-f2-collapse-audit.json")
    args=ap.parse_args()

    project=Path(args.project).resolve()
    source=project/"data"/"f2_sft"/"friend_f2_sft_source.jsonl"
    source_audit_path=source.with_suffix(source.suffix+".audit.json")
    pack_report_path=project/"artifacts"/"model0001_f2_sft"/"F2_SFT_PACK_REPORT.json"
    tokenizer_path=project/"artifacts"/"tokenizer_v1"/"tokenizer.json"
    behavior_path=Path(args.behavior_report).resolve()

    for p in (source,source_audit_path,pack_report_path,tokenizer_path,behavior_path):
        if not p.is_file():
            raise SystemExit(f"STOP: missing audit input {p}")
    if sha256(tokenizer_path)!=TOKENIZER_SHA:
        raise SystemExit("STOP: tokenizer SHA mismatch")

    source_audit=json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("schema")!=EXPECTED_SOURCE_SCHEMA or source_audit.get("status")!="PASS":
        raise SystemExit("STOP: F2 source audit not PASS")
    if source_audit.get("sha256")!=sha256(source):
        raise SystemExit("STOP: F2 source changed since validation")

    pack=json.loads(pack_report_path.read_text(encoding="utf-8"))
    if pack.get("status")!="PASS":
        raise SystemExit("STOP: F2 pack report not PASS")
    if pack.get("hard_guards",{}).get("test_split_used") is not False:
        raise SystemExit("STOP: pack claims test usage")

    behavior=json.loads(behavior_path.read_text(encoding="utf-8"))
    if behavior.get("schema")!=EXPECTED_BEHAVIOR_SCHEMA or behavior.get("status")!="PASS":
        raise SystemExit("STOP: behavior report not PASS")
    if behavior.get("f2",{}).get("model_state_sha256")!=EXPECTED_F2_MODEL_SHA:
        raise SystemExit("STOP: behavior report is not the frozen F2 model")
    pre=behavior.get("numerical_preflight",{})
    if pre.get("pass") is not True:
        raise SystemExit("STOP: behavior inference numerical preflight failed")
    if behavior.get("test_split_used") is not False:
        raise SystemExit("STOP: behavior report touched test split")

    tok=Tokenizer.from_file(str(tokenizer_path))

    assistant_exact=collections.Counter()
    assistant_prefix1=collections.Counter()
    assistant_prefix2=collections.Counter()
    assistant_prefix3=collections.Counter()
    assistant_bigram=collections.Counter()
    assistant_trigram=collections.Counter()
    assistant_first_token=collections.Counter()
    generic_ack=0
    assistant_turns=0
    assistant_words=0
    assistant_tokens=0
    family_assistant_tokens=collections.Counter()
    family_records=collections.Counter()
    tool_names=collections.Counter()
    tool_turns=0
    tool_records=set()
    record_lengths=[]
    train_records=[]
    validation_records=[]

    with source.open("r",encoding="utf-8") as f:
        for raw in f:
            if not raw.strip(): continue
            row=json.loads(raw)
            if row["split"]=="train": train_records.append(row)
            elif row["split"]=="validation": validation_records.append(row)
            family=row.get("source_family","unknown")
            family_records[family]+=1
            rec_ids=serialize_record(tok,row)
            record_lengths.append(len(rec_ids))
            for msg in row["messages"]:
                if msg["role"]!="assistant": continue
                text=msg["content"]
                ws=words(text)
                ids=encode(tok,text)
                if not ids: continue
                assistant_turns+=1
                assistant_words+=len(ws)
                assistant_tokens+=len(ids)
                family_assistant_tokens[family]+=len(ids)
                nt=norm_text(text)
                assistant_exact[nt]+=1
                if ws:
                    assistant_prefix1[" ".join(ws[:1])]+=1
                    assistant_prefix2[" ".join(ws[:2])]+=1
                    assistant_prefix3[" ".join(ws[:3])]+=1
                assistant_first_token[str(ids[0])]+=1
                assistant_bigram.update(ngrams(ws,2))
                assistant_trigram.update(ngrams(ws,3))
                p2=" ".join(ws[:2])
                if p2 in GENERIC_ACK_PREFIXES or (ws and ws[0] in GENERIC_ACK_PREFIXES):
                    generic_ack+=1
                tn=tool_name(text)
                if tn is not None:
                    tool_turns+=1
                    tool_names[tn]+=1
                    tool_records.add(row["id"])

    if assistant_turns<=0:
        raise SystemExit("STOP: source has no assistant turns")

    exact_duplicate_turns=sum(v-1 for v in assistant_exact.values() if v>1)
    exact_duplicate_fraction=pct(exact_duplicate_turns,assistant_turns)
    top1_prefix2_fraction=(assistant_prefix2.most_common(1)[0][1]/assistant_turns
                          if assistant_prefix2 else 0.0)
    top10_prefix2_fraction=sum(v for _,v in assistant_prefix2.most_common(10))/assistant_turns
    top1_trigram_fraction=(assistant_trigram.most_common(1)[0][1]/sum(assistant_trigram.values())
                           if assistant_trigram else 0.0)
    generic_ack_fraction=pct(generic_ack,assistant_turns)

    # Reconstruct old continuous packing and measure how often a 256-target
    # training window spans more than one independent record.
    stream_record_index=[]
    stream_tokens=[]
    for ri,row in enumerate(sorted(train_records,key=lambda r:hashlib.sha256(
            ("20260903-f2-pack-v1:"+r["id"]).encode()).hexdigest())):
        ids=serialize_record(tok,row)
        stream_tokens.extend(ids)
        stream_record_index.extend([ri]*len(ids))
    cross_windows=0
    old_windows=0
    for start in range(0,max(0,len(stream_tokens)-1),SEQ):
        end=min(start+257,len(stream_tokens))
        if end-start<2: continue
        old_windows+=1
        ids=set(stream_record_index[start:end])
        if len(ids)>1: cross_windows+=1
    cross_window_fraction=pct(cross_windows,old_windows)

    # Behavior-output overlap: discover whether repeated collapse n-grams are
    # also high-frequency training target n-grams.
    f2_responses=[
        c.get("response","")
        for c in behavior.get("f2_result",{}).get("cases",[])
        if isinstance(c.get("response"),str)
    ]
    gen_bigram=collections.Counter()
    gen_trigram=collections.Counter()
    for text in f2_responses:
        ws=words(text)
        gen_bigram.update(ngrams(ws,2))
        gen_trigram.update(ngrams(ws,3))

    train_bigram_total=sum(assistant_bigram.values())
    train_trigram_total=sum(assistant_trigram.values())
    overlap=[]
    for n,label,gc,tc,total in [
        (2,"bigram",gen_bigram,assistant_bigram,train_bigram_total),
        (3,"trigram",gen_trigram,assistant_trigram,train_trigram_total),
    ]:
        for phrase,count in gc.most_common(25):
            overlap.append({
                "n":n,
                "kind":label,
                "phrase":phrase,
                "generated_count":count,
                "train_count":tc.get(phrase,0),
                "train_fraction":pct(tc.get(phrase,0),total),
            })
    overlap.sort(key=lambda x:(x["generated_count"],x["train_count"]),reverse=True)

    flags={
      "assistant_exact_duplicate_pressure":
          exact_duplicate_fraction>=0.10,
      "assistant_prefix_concentration":
          top1_prefix2_fraction>=0.04 or top10_prefix2_fraction>=0.25,
      "generic_acknowledgement_pressure":
          generic_ack_fraction>=0.08,
      "continuous_cross_record_packing_pressure":
          cross_window_fraction>=0.25,
      "tool_supervision_sparse_by_turn":
          pct(tool_turns,assistant_turns)<0.12,
      "behavior_generation_collapse_present":
          behavior.get("f2_result",{}).get("auto_pass_rate",1.0)<0.30,
    }

    # Training-overlap flag: at least one phrase emitted >=10 times during the
    # fixed eval is also within the top 1% mass of training n-grams.
    overlap_flag=False
    for x in overlap:
        if x["generated_count"]>=10 and x["train_fraction"]>=0.01:
            overlap_flag=True; break
    flags["collapsed_generation_ngram_overlaps_training_mass"]=overlap_flag

    root_cause_candidates=[
        name for name,value in flags.items() if value
    ]

    report={
      "status":"PASS",
      "schema":"model0001_f2_behavior_collapse_audit_v1",
      "inputs":{
        "source_jsonl":str(source),
        "source_sha256":sha256(source),
        "source_audit_sha256":sha256(source_audit_path),
        "pack_report_sha256":sha256(pack_report_path),
        "behavior_report":str(behavior_path),
        "behavior_report_sha256":sha256(behavior_path),
        "tokenizer_sha256":TOKENIZER_SHA,
      },
      "behavior":{
        "f2_auto_pass_rate":behavior.get("f2_result",{}).get("auto_pass_rate"),
        "foundation_auto_pass_rate":behavior.get("foundation_result",{}).get("auto_pass_rate"),
        "numerical_preflight":pre,
      },
      "assistant_target_distribution":{
        "assistant_turns":assistant_turns,
        "assistant_words":assistant_words,
        "assistant_tokens":assistant_tokens,
        "exact_unique_assistant_texts":len(assistant_exact),
        "exact_duplicate_turns":exact_duplicate_turns,
        "exact_duplicate_fraction":exact_duplicate_fraction,
        "generic_ack_turns":generic_ack,
        "generic_ack_fraction":generic_ack_fraction,
        "top1_prefix2_fraction":top1_prefix2_fraction,
        "top10_prefix2_fraction":top10_prefix2_fraction,
        "top1_trigram_fraction":top1_trigram_fraction,
        "top_prefix1":top(assistant_prefix1,20),
        "top_prefix2":top(assistant_prefix2,20),
        "top_prefix3":top(assistant_prefix3,20),
        "top_bigrams":top(assistant_bigram,30),
        "top_trigrams":top(assistant_trigram,30),
        "top_first_token_ids":top(assistant_first_token,20),
      },
      "source_families":{
        "record_counts":dict(family_records),
        "assistant_token_counts":dict(family_assistant_tokens),
        "assistant_token_fractions":{
            k:pct(v,assistant_tokens) for k,v in family_assistant_tokens.items()
        }
      },
      "tool_supervision":{
        "assistant_tool_call_turns":tool_turns,
        "assistant_tool_call_turn_fraction":pct(tool_turns,assistant_turns),
        "records_with_tool_call":len(tool_records),
        "tool_name_counts":dict(tool_names),
      },
      "old_packing":{
        "train_records":len(train_records),
        "record_serialized_length_mean":sum(record_lengths)/max(1,len(record_lengths)),
        "continuous_windows":old_windows,
        "cross_record_windows":cross_windows,
        "cross_record_window_fraction":cross_window_fraction,
      },
      "generation_training_ngram_overlap":overlap[:50],
      "root_cause_flags":flags,
      "root_cause_candidates":root_cause_candidates,
      "repair_policy":{
        "do_not_resume_f2":True,
        "replacement_source":"Foundation-v3 promoted source",
        "dedupe_exact_assistant_targets":True,
        "cap_high_frequency_assistant_prefixes":True,
        "eliminate_cross_record_packing":True,
        "raise_behavior_protocol_supervision":True,
        "keep_human_dialogue_majority":True,
        "do_not_use_behavior_eval_prompts_as_training_examples":True,
        "test_split_used":False,
      },
      "test_split_used":False,
      "training_performed":False,
    }

    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
