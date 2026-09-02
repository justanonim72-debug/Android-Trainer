#!/usr/bin/env python3
"""Download pinned Model #0001 Dataset-v3 candidate sources.

Public repo contains only provenance and download instructions. Data stays local
on the user's phone. This tool does not tokenize, pack, train, or touch any test
split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

SOURCES = [
    {
        "id": "mdia_raw",
        "family": "dialogue_like_id",
        "url": "https://raw.githubusercontent.com/DoctorDream/mDIA/684c6f93a0f8c6ca904e1b0ceeacfb95ea34647b/datasets/raw.zip",
        "filename": "mdia_raw.zip",
        "pin": "684c6f93a0f8c6ca904e1b0ceeacfb95ea34647b",
        "license": "CC-BY-4.0",
        "provenance": "DoctorDream/mDIA real-life Reddit dialogue corpus",
        "redistribution": "follow_source_terms",
    },
    {
        "id": "emotcmt",
        "family": "code_switch_id_en",
        "url": "https://raw.githubusercontent.com/ir-nlp-csui/emotcmt/d1ad01b073570b1aa23d41574b9c7f94b42854c2/codeswitch_emotion.csv",
        "filename": "emotcmt_codeswitch_emotion.csv",
        "pin": "d1ad01b073570b1aa23d41574b9c7f94b42854c2",
        "license": "dataset README: free use + citation; no copied-dataset redistribution without permission",
        "provenance": "ir-nlp-csui/emotcmt, real Indonesian-English code-mixed tweets",
        "redistribution": "NO_DATA_REDISTRIBUTION",
    },
    {
        "id": "frog_spoken",
        "family": "spoken_narrative_id",
        "url": "https://github.com/davidmoeljadi/corpus-frog-storytelling/archive/ff35f69ea8b612627ac0bf2e654ef7039696550e.zip",
        "filename": "frog_storytelling_ff35f69.zip",
        "pin": "ff35f69ea8b612627ac0bf2e654ef7039696550e",
        "license": "CC-BY-SA-4.0",
        "provenance": "davidmoeljadi/corpus-frog-storytelling spoken transcripts",
        "redistribution": "CC-BY-SA-4.0",
    },
    {
        "id": "talpco_ind",
        "family": "neutral_id",
        "url": "https://github.com/matbahasa/TALPCo/archive/eb4746249830e2c0a8b192a464a74616da3e0453.zip",
        "filename": "talpco_eb47462.zip",
        "pin": "eb4746249830e2c0a8b192a464a74616da3e0453",
        "license": "CC-BY-4.0",
        "provenance": "matbahasa/TALPCo Indonesian parallel-corpus text",
        "redistribution": "CC-BY-4.0",
    },
    {
        "id": "reddit_indonesia_train",
        "family": "colloquial_id",
        "url": "https://huggingface.co/datasets/w11wo/reddit_indonesia_sarcastic/resolve/77e64b52405753abd887c813e4de219ff0abf6e1/data/train.json?download=true",
        "filename": "reddit_indonesia_sarcastic_train.json",
        "pin": "77e64b52405753abd887c813e4de219ff0abf6e1",
        "license": "Apache-2.0 dataset card; underlying Reddit UGC provenance retained",
        "provenance": "w11wo/reddit_indonesia_sarcastic TRAIN only; real Reddit comments, PII-masked by authors",
        "redistribution": "follow_dataset_and_platform_terms",
    },
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Android-Trainer-Model0001/1.0"},
    )
    tmp_path = None
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            with tempfile.NamedTemporaryFile(
                dir=str(dest.parent), prefix=dest.name + ".", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                shutil.copyfileobj(r, tmp, length=1 << 20)
                tmp.flush()
                os.fsync(tmp.fileno())
        os.replace(tmp_path, dest)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    if not project.is_dir():
        raise SystemExit(f"STOP: project missing: {project}")
    out = project / "data" / "raw_v3_sources"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for src in SOURCES:
        dest = out / src["filename"]
        if args.force or not dest.is_file():
            print(f"DOWNLOAD {src['id']}: {src['url']}", flush=True)
            download(src["url"], dest)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise SystemExit(f"STOP: empty/missing download: {dest}")
        record = dict(src)
        record.update({
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": sha256(dest),
        })
        records.append(record)
        print(
            f"OK {src['id']}: {dest.stat().st_size / (1<<20):.2f} MiB "
            f"sha256={record['sha256']}"
        )

    manifest = {
        "status": "PASS",
        "schema": "model0001_v3_acquisition_manifest_v1",
        "project": str(project),
        "test_split_touched": False,
        "dataset_v2_train_bin_reused": False,
        "sources": records,
    }
    path = out / "SOURCE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWROTE: {path}")

if __name__ == "__main__":
    main()
