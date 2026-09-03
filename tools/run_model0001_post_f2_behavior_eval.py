#!/usr/bin/env python3
"""Canonical post-F2 behavior evaluation: Foundation-v3 vs F2 final.

No training and no held-out project test split. Before generation, the runtime
must numerically reproduce the completed native stage's masked SFT validation
CE for BOTH the Foundation-v3 source and F2 final model on the exact same
locked eval indices.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch

from model0001_inference_runtime import (
    F2_COMMIT,F2_FILE_SHA,F2_MODEL_SHA,FOUNDATION_CKPT_SHA,
    FOUNDATION_MODEL_SHA,generate,load_bundle,load_native_checkpoint,
    load_tokenizer,masked_validation_ce,slots_to_tensors,Model0001Inference,
    sha256_file,
)

def parse_tool(text):
    m=re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",text,re.S)
    if not m: return None
    try:
        obj=json.loads(m.group(1))
    except Exception:
        return None
    return obj if isinstance(obj,dict) else None

def score(text,rule):
    typ=rule["type"]
    low=text.strip().lower()
    tool=parse_tool(text)
    if typ=="human_review":
        return None,"human_review"
    if typ=="tool_name":
        ok=isinstance(tool,dict) and tool.get("name")==rule["name"]
        return ok,("tool_name_match" if ok else "tool_name_mismatch")
    if typ=="no_tool":
        ok=tool is None and "<tool_call>" not in text
        return ok,("no_tool" if ok else "unexpected_tool")
    if typ=="contains_any":
        ok=any(str(x).lower() in low for x in rule["values"])
        return ok,("contains_expected" if ok else "missing_expected")
    if typ=="normalized_exact_any":
        normalized=re.sub(r"[^\w]+"," ",low).strip()
        ok=normalized in [str(x).lower() for x in rule["values"]]
        return ok,("exact_expected" if ok else f"got:{normalized}")
    if typ=="tool_query_excludes":
        if not isinstance(tool,dict) or tool.get("name")!=rule["name"]:
            return False,"missing_expected_tool"
        query=json.dumps(tool.get("args",{}),ensure_ascii=False).lower()
        ok=all(x.lower() not in query for x in rule["excluded"])
        return ok,("private_context_excluded" if ok else "private_context_leaked")
    if typ=="clarify_no_tool":
        no_tool=tool is None and "<tool_call>" not in text
        asks="?" in text or any(x.lower() in low for x in rule["keywords"])
        ok=no_tool and asks
        return ok,("clarified" if ok else "did_not_clarify")
    raise RuntimeError(f"unknown score rule: {typ}")

def load_stage_eval(stage_package:Path):
    with zipfile.ZipFile(stage_package,"r") as z:
        manifest=json.loads(z.read("manifest.json"))
        if manifest.get("schema")!="model0001_f2_sft_stage_package_v1":
            raise RuntimeError("wrong F2 stage package schema")
        if manifest.get("source_model_state_sha256")!=FOUNDATION_MODEL_SHA:
            raise RuntimeError("F2 stage package source SHA mismatch")
        spec=manifest["data"]["sft_validation"]
        windows=int(spec["windows"])
        tb=z.read(spec["tokens_path"])
        mb=z.read(spec["mask_path"])
        tokens=np.frombuffer(tb,dtype="<u2").copy().reshape(windows,257)
        masks=np.frombuffer(mb,dtype=np.uint8).copy().reshape(windows,256)
        indices=list(manifest["eval_indices"]["sft_validation"])
    return manifest,tokens,masks,indices

def make_model(slots,cfg,threads):
    torch.set_num_threads(threads)
    try: torch.set_num_interop_threads(max(1,min(threads,4)))
    except RuntimeError: pass
    model=Model0001Inference(slots_to_tensors(slots),cfg)
    model.eval()
    return model

def evaluate_model(label,model,tokenizer,cases,generation):
    outputs=[]
    auto_pass=0
    auto_total=0
    categories=defaultdict(lambda:{"pass":0,"total":0,"human_review":0})
    started=time.perf_counter()
    for i,case in enumerate(cases,1):
        t0=time.perf_counter()
        gen=generate(
            model,tokenizer,case["messages"],
            max_new_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            seed=int(generation["seed"])+i
        )
        elapsed=time.perf_counter()-t0
        passed,reason=score(gen["text"],case["rule"])
        cat=categories[case["category"]]
        if passed is None:
            cat["human_review"]+=1
        else:
            auto_total+=1
            cat["total"]+=1
            if passed:
                auto_pass+=1
                cat["pass"]+=1
        outputs.append({
          "id":case["id"],"category":case["category"],
          "response":gen["text"],
          "prompt_tokens":gen["prompt_tokens"],
          "generated_tokens":gen["generated_tokens"],
          "seconds":elapsed,
          "generated_tokens_per_second":(
              gen["generated_tokens"]/elapsed if elapsed>0 else None
          ),
          "auto_pass":passed,"auto_reason":reason
        })
        print(
            f"[{label} {i:02d}/{len(cases)}] {case['id']}: "
            f"{'HUMAN' if passed is None else 'PASS' if passed else 'FAIL'} "
            f"→ {gen['text'][:120]!r}",
            flush=True
        )
    seconds=time.perf_counter()-started
    return {
      "label":label,
      "auto_pass":auto_pass,
      "auto_total":auto_total,
      "auto_pass_rate":auto_pass/auto_total if auto_total else None,
      "wall_seconds":seconds,
      "categories":dict(categories),
      "cases":outputs
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--foundation-bundle",default="/storage/emulated/0/Download/model0001-foundation-v3-source.atb")
    ap.add_argument("--f2-checkpoint",default="/storage/emulated/0/Download/model0001-f2-sft-final.atnckpt")
    ap.add_argument("--stage-package",default="/storage/emulated/0/Download/model0001-f2-sft.atsftstage")
    ap.add_argument("--stage-report",default="/storage/emulated/0/Download/model0001-f2-sft-stage-report.json")
    ap.add_argument("--suite",default=None)
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-post-f2-behavior-eval.json")
    ap.add_argument("--threads",type=int,default=min(8,os.cpu_count() or 4))
    ap.add_argument("--ce-tolerance",type=float,default=5e-3)
    args=ap.parse_args()

    project=Path(args.project).resolve()
    foundation=Path(args.foundation_bundle).resolve()
    f2=Path(args.f2_checkpoint).resolve()
    stage_package=Path(args.stage_package).resolve()
    stage_report_path=Path(args.stage_report).resolve()
    suite_path=Path(args.suite).resolve() if args.suite else (
        Path(__file__).resolve().parents[1]/
        "eval"/"model0001_post_f2_behavior_suite_v1.json"
    )
    for p in (foundation,f2,stage_package,stage_report_path,suite_path):
        if not p.is_file(): raise SystemExit(f"STOP: required eval input missing: {p}")

    suite=json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("schema")!="model0001_post_f2_behavior_suite_v1" or suite.get("test_split_used") is not False:
        raise SystemExit("STOP: behavior suite contract drift")
    stage_report=json.loads(stage_report_path.read_text(encoding="utf-8"))
    if stage_report.get("schema")!="model0001_f2_sft_stage_report_v1" or stage_report.get("status")!="PASS":
        raise SystemExit("STOP: completed F2 stage report not PASS")
    if stage_report.get("commit")!=F2_COMMIT or stage_report.get("ending_optimizer_step")!=2786:
        raise SystemExit("STOP: completed F2 report provenance mismatch")
    if stage_report.get("test_split_used") is not False:
        raise SystemExit("STOP: F2 report says test was used")

    fslots,fmeta=load_native_checkpoint(
        f2,expected_file_sha=F2_FILE_SHA,expected_model_sha=F2_MODEL_SHA
    )
    if fmeta.optimizer_step!=2786 or fmeta.commit!=F2_COMMIT:
        raise SystemExit("STOP: F2 final checkpoint completion mismatch")
    if fmeta.parent_checkpoint_sha256!=FOUNDATION_CKPT_SHA or fmeta.parent_model_state_sha256!=FOUNDATION_MODEL_SHA:
        raise SystemExit("STOP: F2 final checkpoint lineage mismatch")

    bslots,cfg,bmanifest=load_bundle(
        foundation,expected_model_sha=FOUNDATION_MODEL_SHA
    )
    tokenizer,tok_path=load_tokenizer(project)
    stage_manifest,val_tokens,val_masks,eval_indices=load_stage_eval(stage_package)

    print("=== NUMERICAL INFERENCE PREFLIGHT ===",flush=True)
    foundation_model=make_model(bslots,cfg,args.threads)
    foundation_ce=masked_validation_ce(
        foundation_model,val_tokens,val_masks,eval_indices,batch_size=4
    )
    native_foundation_ce=float(stage_report["baseline"]["sft_validation_ce"])
    foundation_err=abs(foundation_ce-native_foundation_ce)
    print(
        f"Foundation-v3 SFT-val CE python={foundation_ce:.9f} "
        f"native={native_foundation_ce:.9f} err={foundation_err:.3g}",
        flush=True
    )
    if foundation_err>args.ce_tolerance:
        raise SystemExit("STOP: Foundation inference reconstruction failed numerical preflight")

    f2_model=make_model(fslots,cfg,args.threads)
    f2_ce=masked_validation_ce(
        f2_model,val_tokens,val_masks,eval_indices,batch_size=4
    )
    native_f2_ce=float(stage_report["final"]["sft_validation_ce"])
    f2_err=abs(f2_ce-native_f2_ce)
    print(
        f"F2 SFT-val CE python={f2_ce:.9f} "
        f"native={native_f2_ce:.9f} err={f2_err:.3g}",
        flush=True
    )
    if f2_err>args.ce_tolerance:
        raise SystemExit("STOP: F2 inference reconstruction failed numerical preflight")

    generation=suite["generation"]
    cases=suite["cases"]

    print("\n=== CANONICAL GREEDY: FOUNDATION-v3 ===",flush=True)
    foundation_result=evaluate_model(
        "Foundation-v3",foundation_model,tokenizer,cases,generation
    )
    del foundation_model
    print("\n=== CANONICAL GREEDY: F2 FINAL ===",flush=True)
    f2_result=evaluate_model(
        "F2",f2_model,tokenizer,cases,generation
    )

    f_by={x["id"]:x for x in foundation_result["cases"]}
    s_by={x["id"]:x for x in f2_result["cases"]}
    improved=[]; regressed=[]; both_pass=[]; both_fail=[]
    for case in cases:
        a=f_by[case["id"]]["auto_pass"]
        b=s_by[case["id"]]["auto_pass"]
        if a is None or b is None: continue
        if not a and b: improved.append(case["id"])
        elif a and not b: regressed.append(case["id"])
        elif a and b: both_pass.append(case["id"])
        else: both_fail.append(case["id"])

    report={
      "status":"PASS",
      "schema":"model0001_post_f2_behavior_eval_report_v1",
      "mode":"canonical_greedy_comparison",
      "test_split_used":False,
      "training_performed":False,
      "suite":{"path":str(suite_path),"sha256":sha256_file(suite_path),"cases":len(cases)},
      "tokenizer":{"path":str(tok_path),"sha256":sha256_file(tok_path)},
      "foundation":{
        "bundle":str(foundation),
        "bundle_sha256":sha256_file(foundation),
        "model_state_sha256":FOUNDATION_MODEL_SHA
      },
      "f2":{
        "checkpoint":str(f2),
        "checkpoint_sha256":F2_FILE_SHA,
        "model_state_sha256":F2_MODEL_SHA,
        "optimizer_step":fmeta.optimizer_step
      },
      "numerical_preflight":{
        "tolerance":args.ce_tolerance,
        "foundation_python_sft_ce":foundation_ce,
        "foundation_native_sft_ce":native_foundation_ce,
        "foundation_abs_error":foundation_err,
        "f2_python_sft_ce":f2_ce,
        "f2_native_sft_ce":native_f2_ce,
        "f2_abs_error":f2_err,
        "pass":True
      },
      "generation":generation,
      "foundation_result":foundation_result,
      "f2_result":f2_result,
      "paired_auto_score":{
        "improved_after_f2":improved,
        "regressed_after_f2":regressed,
        "both_pass":both_pass,
        "both_fail":both_fail
      },
      "next_decision":"human_review_required_before_preference_stage"
    }
    out=Path(args.output).resolve()
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8")
    print("\n=== BEHAVIOR EVAL READY ===")
    print(json.dumps({
      "status":"PASS",
      "output":str(out),
      "foundation_auto":f"{foundation_result['auto_pass']}/{foundation_result['auto_total']}",
      "f2_auto":f"{f2_result['auto_pass']}/{f2_result['auto_total']}",
      "improved_after_f2":improved,
      "regressed_after_f2":regressed
    },indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
