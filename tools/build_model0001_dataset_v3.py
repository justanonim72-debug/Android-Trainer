#!/usr/bin/env python3
"""Build immutable Model #0001 Foundation/Dataset-v3 token packs.

Inputs are LOCAL phone artifacts only:
- audited NEW v3 candidate pool
- frozen v1 TRAIN source text for an explicit retention slice
- frozen tokenizer v1

The script never reads any test split and never reuses v1/v2 packed bins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TOKENIZER_SHA = "3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
CANDIDATE_POOL_SHA = "e3e4174f9fc4dadcb4751f33dd7f51b15ccc3bdd3dd8cb2c3622db347071ee55"
FROZEN_V2_TRAIN_SHA = "86468e6511f9c7a145983268f9fe479bff24a5e6292e45b22b0c40b01717908e"

SEED = 20260903
VALIDATION_FRACTION = 0.03
RETENTION_FINAL_FRACTION = 0.15
SEQ = 256
VOCAB = 14000
BOS = 1
EOS = 2
UNK = 3

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?62|0)[\s.-]?(?:\d[\s.-]?){8,13}(?!\d)")
SPACE_RE = re.compile(r"[ \t]+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rank(label: str) -> str:
    return hashlib.sha256(f"{SEED}:{label}".encode("utf-8")).hexdigest()


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u0000", " ")
    s = s.replace("[USERNAME]", " ").replace("[URL]", " ")
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = HANDLE_RE.sub(" ", s)
    s = PHONE_RE.sub(" ", s)
    lines = []
    for line in s.splitlines():
        line = SPACE_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def text_key(s: str) -> str:
    s = re.sub(r"\s+", " ", norm_text(s).lower())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_text(obj):
    if isinstance(obj, dict):
        for key in ("text", "content", "body", "document", "raw_text", "sentence"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def tokenize_document(tok: Tokenizer, text: str) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False).ids
    if any((x < 0 or x >= VOCAB) for x in ids):
        raise SystemExit("STOP: tokenizer produced OOV id")
    return [BOS, *ids, EOS]


def read_new_pool(pool: Path):
    records = []
    seen_ids = set()
    with pool.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            rid = row.get("id")
            family = row.get("family")
            text = row.get("text")
            if not isinstance(rid, str) or not rid or rid in seen_ids:
                raise SystemExit(f"STOP: bad/duplicate new-pool id at line {line_no}")
            if not isinstance(family, str) or not family:
                raise SystemExit(f"STOP: missing family at line {line_no}")
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f"STOP: missing text at line {line_no}")
            seen_ids.add(rid)
            records.append({
                "id": rid,
                "family": family,
                "source_id": row.get("source_id"),
                "text": text,
                "text_sha256": row.get("text_sha256") or text_key(text),
                "audit_token_count": int(row.get("token_count", 0)),
            })
    if not records:
        raise SystemExit("STOP: new pool is empty")
    return records


def split_new_records(records):
    by_family = defaultdict(list)
    for row in records:
        by_family[row["family"]].append(row)

    train, validation = [], []
    split_report = {}
    for family, rows in sorted(by_family.items()):
        total = sum(max(1, int(x["audit_token_count"])) for x in rows)
        target = max(1, int(round(total * VALIDATION_FRACTION)))
        ordered = sorted(rows, key=lambda x: stable_rank("val:" + x["id"]))
        selected = []
        acc = 0
        for row in ordered:
            if acc >= target and selected:
                break
            selected.append(row)
            acc += max(1, int(row["audit_token_count"]))
        selected_ids = {x["id"] for x in selected}
        validation.extend(selected)
        train.extend(x for x in rows if x["id"] not in selected_ids)
        split_report[family] = {
            "source_records": len(rows),
            "source_audit_tokens": total,
            "validation_records": len(selected),
            "validation_audit_tokens": acc,
            "target_validation_audit_tokens": target,
        }
    return train, validation, split_report


def read_v1_retention_candidates(path: Path, tok: Tokenizer, forbidden_keys: set[str]):
    if "test" in path.name.lower() or "/test" in str(path).replace("\\", "/").lower():
        raise SystemExit("STOP: retention path unexpectedly points to test data")
    out = []
    seen = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            text = extract_text(row)
            if not text:
                continue
            text = norm_text(text)
            if len(text) < 20:
                continue
            key = text_key(text)
            if key in forbidden_keys or key in seen:
                continue
            seen.add(key)
            ids = tokenize_document(tok, text)
            if len(ids) < 6:
                continue
            out.append({
                "id": f"v1ret:{line_no:08d}:{key[:12]}",
                "family": "retention_v1",
                "source_id": str(row.get("source", row.get("source_id", "v1_train"))),
                "text": text,
                "text_sha256": key,
                "packed_token_count": len(ids),
            })
    return out


def pack_records(tok: Tokenizer, records, out_bin: Path, manifest_path: Path):
    tokens = []
    family_tokens = Counter()
    family_docs = Counter()
    unk = 0
    manifests = []

    for row in records:
        ids = tokenize_document(tok, row["text"])
        tokens.extend(ids)
        family_tokens[row["family"]] += len(ids)
        family_docs[row["family"]] += 1
        unk += sum(1 for x in ids if x == UNK)
        manifests.append({
            "id": row["id"],
            "family": row["family"],
            "source_id": row.get("source_id"),
            "text_sha256": row["text_sha256"],
            "packed_tokens": len(ids),
        })

    if not tokens:
        raise SystemExit(f"STOP: no tokens for {out_bin}")
    arr = np.asarray(tokens, dtype="<u2")
    if int(arr.max()) >= VOCAB:
        raise SystemExit("STOP: packed token >= vocab")
    arr.tofile(out_bin)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifests:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    return {
        "documents": len(records),
        "packed_tokens": int(arr.size),
        "full_256_target_windows": max(0, (int(arr.size) - 1) // SEQ),
        "scored_target_tokens": max(0, (int(arr.size) - 1) // SEQ) * SEQ,
        "unk_count": int(unk),
        "family_tokens": dict(family_tokens),
        "family_docs": dict(family_docs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1",
    )
    args = ap.parse_args()
    project = Path(args.project).resolve()

    pool = project / "data" / "corpus_v3_candidates" / "new_pool.jsonl"
    pool_audit = project / "data" / "corpus_v3_candidates" / "CANDIDATE_POOL_AUDIT.json"
    tokenizer_path = project / "artifacts" / "tokenizer_v1" / "tokenizer.json"
    v1_train = project / "data" / "splits" / "train.jsonl"
    v2_train = project / "artifacts" / "model0001_dataset_v2" / "train.bin"

    for required in (pool, pool_audit, tokenizer_path, v1_train, v2_train):
        if not required.is_file():
            raise SystemExit(f"STOP: required file missing: {required}")

    if sha256(pool) != CANDIDATE_POOL_SHA:
        raise SystemExit("STOP: v3 new_pool SHA mismatch")
    audit = json.loads(pool_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("pool_sha256") != CANDIDATE_POOL_SHA:
        raise SystemExit("STOP: candidate-pool audit mismatch")
    guards = audit.get("hard_guards", {})
    if guards.get("test_split_touched") is not False:
        raise SystemExit("STOP: candidate audit test-split guard failed")
    if sha256(tokenizer_path) != TOKENIZER_SHA:
        raise SystemExit("STOP: tokenizer SHA mismatch")
    if sha256(v2_train) != FROZEN_V2_TRAIN_SHA:
        raise SystemExit("STOP: frozen Dataset-v2 train.bin identity changed")

    tok = Tokenizer.from_file(str(tokenizer_path))
    new_records = read_new_pool(pool)
    new_train, validation, split_report = split_new_records(new_records)

    # Tokenize new train now so the retention target is based on the ACTUAL
    # frozen tokenizer and includes BOS/EOS overhead.
    for row in new_train:
        row["packed_token_count"] = len(tokenize_document(tok, row["text"]))

    new_train_tokens = sum(x["packed_token_count"] for x in new_train)
    retention_target = int(round(
        new_train_tokens * RETENTION_FINAL_FRACTION / (1.0 - RETENTION_FINAL_FRACTION)
    ))

    forbidden = {text_key(x["text"]) for x in new_records}
    retention_candidates = read_v1_retention_candidates(v1_train, tok, forbidden)
    retention_candidates.sort(key=lambda x: stable_rank("retention:" + x["id"]))

    retention = []
    retention_tokens = 0
    for row in retention_candidates:
        if retention_tokens >= retention_target and retention:
            break
        retention.append(row)
        retention_tokens += int(row["packed_token_count"])

    if retention_tokens < retention_target * 0.95:
        raise SystemExit(
            f"STOP: insufficient v1 retention text: {retention_tokens} < target {retention_target}"
        )

    train = new_train + retention
    train.sort(key=lambda x: stable_rank("train:" + x["id"]))
    validation.sort(key=lambda x: stable_rank("validation:" + x["id"]))

    outdir = project / "artifacts" / "model0001_dataset_v3"
    outdir.mkdir(parents=True, exist_ok=True)

    train_bin = outdir / "train.bin"
    val_bin = outdir / "validation.bin"
    train_manifest = outdir / "train_manifest.jsonl"
    val_manifest = outdir / "validation_manifest.jsonl"

    train_stats = pack_records(tok, train, train_bin, train_manifest)
    val_stats = pack_records(tok, validation, val_bin, val_manifest)

    if train_stats["unk_count"] != 0 or val_stats["unk_count"] != 0:
        raise SystemExit("STOP: frozen ByteLevel tokenizer unexpectedly produced UNK")
    if sha256(train_bin) == FROZEN_V2_TRAIN_SHA:
        raise SystemExit("STOP: Dataset-v3 train.bin unexpectedly equals Dataset-v2")

    realized = {}
    for family, count in train_stats["family_tokens"].items():
        realized[family] = {
            "packed_tokens": count,
            "fraction": count / train_stats["packed_tokens"],
        }

    retention_fraction = (
        train_stats["family_tokens"].get("retention_v1", 0)
        / train_stats["packed_tokens"]
    )

    report = {
        "status": "PASS",
        "schema": "model0001_dataset_v3_report_v1",
        "stage_objective": "friend_foundation_v3_cpt",
        "packing_seed": SEED,
        "validation_fraction_target": VALIDATION_FRACTION,
        "retention_fraction_target": RETENTION_FINAL_FRACTION,
        "retention_fraction_realized": retention_fraction,
        "new_pool_sha256": CANDIDATE_POOL_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "dataset_v2_train_sha256_guard": FROZEN_V2_TRAIN_SHA,
        "train": {
            **train_stats,
            "path": str(train_bin),
            "sha256": sha256(train_bin),
            "manifest": str(train_manifest),
            "manifest_sha256": sha256(train_manifest),
        },
        "validation": {
            **val_stats,
            "path": str(val_bin),
            "sha256": sha256(val_bin),
            "manifest": str(val_manifest),
            "manifest_sha256": sha256(val_manifest),
        },
        "new_source_split": split_report,
        "realized_train_mix": realized,
        "hard_guards": {
            "source_oversampling": False,
            "dataset_v2_train_bin_reused": False,
            "v1_packed_bin_reused": False,
            "v1_retention_source_text_only": True,
            "validation_new_v3_only": True,
            "test_split_created": False,
            "test_split_used": False,
            "tokenizer_changed": False,
            "architecture_changed": False,
            "production_lr_locked": False,
            "training_started": False,
        },
    }

    report_path = outdir / "DATASET_V3_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lock = {
        "status": "LOCKED",
        "schema": "model0001_dataset_v3_lock_v1",
        "report_sha256": sha256(report_path),
        "train_sha256": sha256(train_bin),
        "validation_sha256": sha256(val_bin),
        "tokenizer_sha256": TOKENIZER_SHA,
        "new_pool_sha256": CANDIDATE_POOL_SHA,
        "packing_seed": SEED,
        "test_split_used": False,
    }
    lock_path = outdir / "DATASET_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")

    sums = outdir / "SHA256SUMS.txt"
    with sums.open("w", encoding="utf-8") as f:
        for p in (
            train_bin, val_bin, train_manifest, val_manifest, report_path, lock_path
        ):
            f.write(f"{sha256(p)}  {p.name}\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nDATASET-v3 LOCKED: {outdir}")
    print(f"REPORT: {report_path}")


if __name__ == "__main__":
    main()
