#!/usr/bin/env python3
"""Exploratory local chat with the frozen Model #0001 F2 final checkpoint.

This is NOT the canonical behavioral score and performs no optimizer updates.
It exists so the user can feel the post-SFT model directly on the Android
phone after the canonical greedy comparison is captured.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from model0001_inference_runtime import (
    F2_COMMIT,F2_FILE_SHA,F2_MODEL_SHA,FOUNDATION_CKPT_SHA,
    FOUNDATION_MODEL_SHA,Model0001Inference,generate,load_bundle,
    load_native_checkpoint,load_tokenizer,slots_to_tensors
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--foundation-bundle",default="/storage/emulated/0/Download/model0001-foundation-v3-source.atb")
    ap.add_argument("--f2-checkpoint",default="/storage/emulated/0/Download/model0001-f2-sft-final.atnckpt")
    ap.add_argument("--temperature",type=float,default=0.8)
    ap.add_argument("--top-p",type=float,default=0.9)
    ap.add_argument("--max-new-tokens",type=int,default=64)
    ap.add_argument("--seed",type=int,default=20260903)
    ap.add_argument("--threads",type=int,default=min(8,os.cpu_count() or 4))
    ap.add_argument("--transcript",default="/storage/emulated/0/Download/model0001-f2-exploratory-chat.json")
    args=ap.parse_args()

    project=Path(args.project).resolve()
    foundation=Path(args.foundation_bundle).resolve()
    f2=Path(args.f2_checkpoint).resolve()
    if not foundation.is_file() or not f2.is_file():
        raise SystemExit("STOP: Foundation source bundle or F2 final checkpoint missing")

    slots,meta=load_native_checkpoint(
        f2,expected_file_sha=F2_FILE_SHA,expected_model_sha=F2_MODEL_SHA
    )
    if meta.optimizer_step!=2786 or meta.commit!=F2_COMMIT:
        raise SystemExit("STOP: wrong/incomplete F2 checkpoint")
    if meta.parent_checkpoint_sha256!=FOUNDATION_CKPT_SHA or meta.parent_model_state_sha256!=FOUNDATION_MODEL_SHA:
        raise SystemExit("STOP: F2 checkpoint lineage mismatch")
    _,cfg,_=load_bundle(foundation,expected_model_sha=FOUNDATION_MODEL_SHA)
    tokenizer,_=load_tokenizer(project)

    torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(max(1,min(args.threads,4)))
    except RuntimeError: pass
    model=Model0001Inference(slots_to_tensors(slots),cfg)
    model.eval()

    messages=[]
    transcript={
      "schema":"model0001_f2_exploratory_chat_v1",
      "model_state_sha256":F2_MODEL_SHA,
      "checkpoint_sha256":F2_FILE_SHA,
      "sampling":{
        "temperature":args.temperature,"top_p":args.top_p,
        "max_new_tokens":args.max_new_tokens,"base_seed":args.seed
      },
      "turns":[]
    }

    print("\nMODEL #0001 F2 — exploratory chat 😆")
    print("Commands: :reset  :system <text>  :save  :quit")
    print("Tool calls are shown as raw <tool_call> JSON; no real tool is executed.\n")

    turn=0
    while True:
        try:
            user=input("lu> ").strip()
        except (EOFError,KeyboardInterrupt):
            print()
            break
        if not user: continue
        if user==":quit": break
        if user==":reset":
            messages=[]
            print("[context reset]\n")
            continue
        if user.startswith(":system "):
            text=user[len(":system "):].strip()
            messages=[m for m in messages if m["role"]!="system"]
            messages.insert(0,{"role":"system","content":text})
            print("[system updated]\n")
            continue
        if user==":save":
            Path(args.transcript).write_text(
                json.dumps(transcript,indent=2,ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"[saved {args.transcript}]\n")
            continue

        messages.append({"role":"user","content":user})
        t0=time.perf_counter()
        out=generate(
            model,tokenizer,messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed+turn
        )
        dt=time.perf_counter()-t0
        text=out["text"]
        print(f"0001> {text}")
        if out["generated_tokens"]:
            print(f"      [{out['generated_tokens']/dt:.2f} tok/s, {out['generated_tokens']} tok]\n")
        else:
            print("      [0 generated tokens]\n")
        messages.append({"role":"assistant","content":text})
        transcript["turns"].append({
          "user":user,"assistant":text,
          "prompt_tokens":out["prompt_tokens"],
          "generated_tokens":out["generated_tokens"],
          "seconds":dt
        })
        turn+=1

    Path(args.transcript).write_text(
        json.dumps(transcript,indent=2,ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Transcript saved: {args.transcript}")

if __name__=="__main__":
    main()
