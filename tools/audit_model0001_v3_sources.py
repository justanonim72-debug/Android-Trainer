#!/usr/bin/env python3
"""
Read-only inventory for Model #0001 Foundation/Dataset-v3 planning.

This does not pack, tokenize, mutate artifacts, or touch any test split.
It inventories likely source corpora, historical dataset metadata, and builder
scripts from the local phone project so Dataset-v3 can be designed from
evidence instead of guessed paths or guessed mixture percentages.
"""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path

TEXT_EXTS = {".jsonl", ".json", ".txt", ".md", ".csv", ".tsv"}
SCRIPT_HINTS = ("dataset", "corpus", "pack", "pretrain", "token", "model0001")
SKIP_DIRS = {".git", "__pycache__", "model0001_runs"}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def line_count(path: Path):
    if path.suffix.lower() not in TEXT_EXTS:
        return None
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def rel(project: Path, path: Path) -> str:
    try: return str(path.relative_to(project))
    except Exception: return str(path)

def classify_candidate(project: Path, p: Path) -> tuple[str, bool, str]:
    r = rel(project, p).replace("\\", "/")
    low = r.lower()

    # Never candidate training data.
    if "/test." in "/" + low or low.startswith("data/splits/test"):
        return "test_split", False, "test data is forbidden"
    if low.startswith(("reports/", "benchmarks/")):
        return "report_or_benchmark", False, "metadata/eval artifact"
    if low.startswith(("data/tokenizer/", "data/tokenizer_corpus/",
                       "artifacts/tokenizer_v1/")):
        return "tokenizer_artifact", False, "tokenizer/training metadata"
    if low.startswith(("artifacts/", "config/", "scripts/", "provenance/")):
        return "project_artifact", False, "project metadata/code, not corpus"
    if low.startswith(".") or low.endswith((".md",)):
        return "project_note", False, "notes/patch documentation"
    if low.startswith("data/raw_v2/"):
        return "prior_v2_source", False, "old v2 source; retention-only if explicitly selected"
    if low.startswith("data/corpus_v2/"):
        return "prior_v2_corpus", False, "old v2 corpus; never new-v3"
    if low.startswith("data/raw/") or low.startswith("data/splits/"):
        return "prior_v1_source", False, "old v1 source; retention-only if explicitly selected"

    # A file outside all known historical/project buckets is only a NEW SOURCE
    # CANDIDATE. It is not approved until provenance/license/content-family
    # metadata and prior-corpus dedupe are supplied.
    return "unclassified_new_candidate", True, "needs provenance/license/content audit"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project", default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--output", default="/storage/emulated/0/Download/model0001-v3-source-audit.json")
    args=ap.parse_args()
    project=Path(args.project).resolve()
    if not project.is_dir():
        raise SystemExit(f"STOP: project missing: {project}")

    v2_train=project/"artifacts"/"model0001_dataset_v2"/"train.bin"
    v2_sha=sha256(v2_train) if v2_train.is_file() else None

    historical=[]
    for dname in ("model0001_dataset_v1","model0001_dataset_v2"):
        root=project/"artifacts"/dname
        if not root.is_dir(): continue
        for p in sorted(root.rglob("*")):
            if not p.is_file(): continue
            if p.suffix.lower() in {".json",".md",".txt"}:
                historical.append({
                    "path":rel(project,p),"bytes":p.stat().st_size,
                    "sha256":sha256(p)
                })

    scripts=[]
    sroot=project/"scripts"
    if sroot.is_dir():
        for p in sorted(sroot.rglob("*.py")):
            low=p.name.lower()
            if any(h in low for h in SCRIPT_HINTS):
                scripts.append({
                    "path":rel(project,p),"bytes":p.stat().st_size,
                    "sha256":sha256(p)
                })

    candidates=[]
    for root, dirs, files in os.walk(project):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rp=Path(root)
        # Frozen packed artifacts are evidence, not candidate v3 source.
        if "artifacts/model0001_dataset_v2" in str(rp).replace("\\","/"):
            continue
        if "artifacts/model0001_dataset_v1" in str(rp).replace("\\","/"):
            continue
        for name in files:
            p=rp/name
            ext=p.suffix.lower()
            if ext not in TEXT_EXTS:
                continue
            low=str(p).lower()
            if "/scripts/" in low or low.endswith(".audit.json"):
                continue
            size=p.stat().st_size
            if size == 0:
                continue
            source_class, new_candidate, reason = classify_candidate(project, p)
            candidates.append({
                "path":rel(project,p),
                "bytes":size,
                "lines":line_count(p),
                "sha256":sha256(p),
                "source_class":source_class,
                "new_v3_candidate":new_candidate,
                "classification_reason":reason,
            })

    report={
      "status":"PASS",
      "schema":"model0001_v3_source_inventory_v1",
      "project":str(project),
      "frozen_dataset_v2_train_sha256":v2_sha,
      "historical_dataset_metadata":historical,
      "builder_scripts":scripts,
      "candidate_source_files":candidates,
      "candidate_source_file_count":len(candidates),
      "eligible_new_source_candidates":[x for x in candidates if x["new_v3_candidate"]],
      "eligible_new_source_candidate_count":sum(1 for x in candidates if x["new_v3_candidate"]),
      "classification_counts":dict(__import__("collections").Counter(
          x["source_class"] for x in candidates
      )),
      "inventory_interpretation":"inventory entries are not approved training sources; only eligible_new_source_candidates may proceed to provenance/license/content audit",
      "hard_guards":{
        "read_only":True,
        "packed_dataset_v2_not_candidate_source":True,
        "test_split_touched":False,
        "tokenizer_changed":False
      }
    }
    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    print(f"\nWROTE: {out}")

if __name__=="__main__":
    main()
