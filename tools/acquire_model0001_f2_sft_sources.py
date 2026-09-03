#!/usr/bin/env python3
"""Acquire only the pinned external TRAIN source needed for Friend-Core F2 SFT.

mDIA raw dialogue is already present locally from Dataset-v3 acquisition and is
reused explicitly under a NEW supervised objective. This downloader adds only
IndoSMD TRAIN from IndoToD; external dev/test files are never downloaded.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, tempfile, urllib.request
from pathlib import Path

INDOTOD_COMMIT="236b81c24403ff77c38ebd5408e79184b632a766"
INDOSMD_URL=(
    "https://raw.githubusercontent.com/dehanalkautsar/IndoToD/"
    +INDOTOD_COMMIT+
    "/IndoSMD/IndoSMD_split/IndoSMD_train.json"
)

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def download(url:str,dest:Path):
    req=urllib.request.Request(url,headers={"User-Agent":"Android-Trainer-F2/1.0"})
    tmp=None
    try:
        with urllib.request.urlopen(req,timeout=180) as r:
            with tempfile.NamedTemporaryFile(dir=str(dest.parent),delete=False) as f:
                tmp=Path(f.name)
                shutil.copyfileobj(r,f,length=1<<20)
                f.flush(); os.fsync(f.fileno())
        os.replace(tmp,dest); tmp=None
    finally:
        if tmp is not None:
            try: tmp.unlink()
            except FileNotFoundError: pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    args=ap.parse_args()
    project=Path(args.project).resolve()
    mdia=project/"data"/"raw_v3_sources"/"mdia_raw.zip"
    if not mdia.is_file():
        raise SystemExit("STOP: existing pinned mDIA raw.zip missing; run v3 acquisition first")

    out=project/"data"/"raw_f2_sft_sources"
    out.mkdir(parents=True,exist_ok=True)
    indosmd=out/"indotod_indosmd_train.json"
    if not indosmd.is_file():
        print(f"DOWNLOAD IndoSMD TRAIN: {INDOSMD_URL}",flush=True)
        download(INDOSMD_URL,indosmd)

    try:
        rows=json.loads(indosmd.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"STOP: IndoSMD TRAIN JSON invalid: {e}")
    if not isinstance(rows,list) or not rows:
        raise SystemExit("STOP: IndoSMD TRAIN empty")

    manifest={
      "status":"PASS",
      "schema":"model0001_f2_sft_acquisition_manifest_v1",
      "sources":[
        {
          "id":"mdia_raw_reuse",
          "path":str(mdia),
          "sha256":sha256(mdia),
          "license":"CC-BY-4.0",
          "provenance":"DoctorDream/mDIA real-life dialogue; reused from F1 source under supervised objective",
          "external_split":"raw_non_eval_selected_by_parser"
        },
        {
          "id":"indotod_indosmd_train",
          "path":str(indosmd),
          "sha256":sha256(indosmd),
          "license":"CC-BY-SA-4.0 dataset",
          "provenance":"dehanalkautsar/IndoToD IndoSMD native-speaker annotated Indonesian dialogue",
          "commit":INDOTOD_COMMIT,
          "external_split":"train"
        }
      ],
      "hard_guards":{
        "external_dev_downloaded":False,
        "external_test_downloaded":False,
        "project_test_split_touched":False,
        "openai_teacher_outputs_used":False,
        "llm_generated_dialogue_used":False
      }
    }
    mp=out/"SOURCE_MANIFEST.json"
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=="__main__": main()
