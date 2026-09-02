#!/usr/bin/env python3
"""Prepare and audit a NEW Friend-Core Dataset-v3 source pool.

Read-only with respect to v1/v2 artifacts. Never reads any project test split.
It normalizes pinned v3 sources, exact-deduplicates against prior TRAIN source
text, counts frozen-tokenizer tokens, and writes a candidate pool + audit.
It does NOT choose mixture weights, pack bins, or train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

TOKENIZER_SHA = "3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?62|0)[\s.-]?(?:\d[\s.-]?){8,13}(?!\d)")
SPACE_RE = re.compile(r"[ \t]+")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u0000", " ")
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

def key_text(s: str) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"\s+", " ", s)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def acceptable(s: str) -> bool:
    if len(s) < 20 or len(s) > 5000:
        return False
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    if printable / max(1, len(s)) < 0.98:
        return False
    alpha = sum(1 for ch in s if ch.isalpha())
    if alpha < 8:
        return False
    return True

def iter_json_strings(obj):
    if isinstance(obj, str):
        if len(obj.strip()) >= 20:
            yield obj
    elif isinstance(obj, dict):
        for name in ("text", "content", "body", "document", "raw_text", "sentence"):
            v = obj.get(name)
            if isinstance(v, str) and len(v.strip()) >= 20:
                yield v
                return
        for v in obj.values():
            yield from iter_json_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_json_strings(v)

def load_prior_train_keys(project: Path) -> set[str]:
    keys = set()
    paths = [
        project / "data" / "corpus_v2" / "train_pool.jsonl",
        project / "data" / "splits" / "train.jsonl",
    ]
    for path in paths:
        if not path.is_file():
            continue
        if "test" in path.name.lower():
            raise RuntimeError("internal guard: attempted test read")
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                for text in iter_json_strings(obj):
                    n = norm_text(text)
                    if acceptable(n):
                        keys.add(key_text(n))
    return keys

def source_record(source_id, family, license_name, provenance, text, ordinal):
    return {
        "id": f"{source_id}:{ordinal:08d}",
        "source_id": source_id,
        "family": family,
        "license": license_name,
        "provenance": provenance,
        "text": text,
    }

def parse_emot(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            t = row.get("tweet") or ""
            yield source_record(
                "emotcmt",
                "code_switch_id_en",
                "dataset README free-use+citation; no copied-dataset redistribution",
                "ir-nlp-csui/emotcmt real code-mixed tweets",
                t,
                i,
            )

def parse_reddit(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return
    if raw.startswith("["):
        rows = json.loads(raw)
    else:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        t = row.get("text") or ""
        yield source_record(
            "reddit_indonesia_train",
            "colloquial_id",
            "Apache-2.0 dataset card; Reddit UGC provenance retained",
            "w11wo/reddit_indonesia_sarcastic train text; PII-masked by authors",
            t,
            i,
        )

def parse_talpco(path: Path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("/ind/data_ind.txt")]
        if len(names) != 1:
            raise RuntimeError(f"TALPCo Indonesian file ambiguity: {names}")
        text = z.read(names[0]).decode("utf-8-sig", "replace")
    i = 0
    for line in text.splitlines():
        parts = line.split("\t", 1)
        t = parts[1] if len(parts) == 2 else line
        if t.strip():
            yield source_record(
                "talpco_ind",
                "neutral_id",
                "CC-BY-4.0",
                "matbahasa/TALPCo Indonesian text",
                t,
                i,
            )
            i += 1

def parse_frog(path: Path):
    with zipfile.ZipFile(path) as z:
        names = sorted(
            n for n in z.namelist()
            if "/data/spoken/" in n and n.endswith(".txt")
        )
        if not names:
            raise RuntimeError("frog spoken transcripts missing")
        ordinal = 0
        for name in names:
            text = z.read(name).decode("utf-8-sig", "replace")
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    yield source_record(
                        "frog_spoken",
                        "spoken_narrative_id",
                        "CC-BY-SA-4.0",
                        f"davidmoeljadi/corpus-frog-storytelling:{name}",
                        para,
                        ordinal,
                    )
                    ordinal += 1

def parse_mdia(path: Path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        candidates = [
            n for n in names
            if n.lower().endswith(".csv")
            and ("id_" in Path(n).name.lower() or "indones" in n.lower())
            and "translated" not in n.lower()
            and "test" not in n.lower()
            and "eval" not in n.lower()
            and "valid" not in n.lower()
        ]
        preferred = [n for n in candidates if "cleaned_dialogue" in n.lower()]
        if preferred:
            candidates = preferred
        if not candidates:
            for n in names:
                if not n.lower().endswith(".csv"):
                    continue
                if any(x in n.lower() for x in ("translated", "test", "eval", "valid")):
                    continue
                try:
                    head = z.read(n)[:4096].decode("utf-8-sig", "replace")
                except Exception:
                    continue
                if (
                    "source_body" in head
                    and "target_body" in head
                    and ("/id" in n.lower() or "id_" in n.lower())
                ):
                    candidates.append(n)
        if not candidates:
            raise RuntimeError(
                "mDIA Indonesian raw/train CSV not found; archive members sample="
                + repr(names[:50])
            )
        ordinal = 0
        for name in sorted(set(candidates)):
            with z.open(name) as bf:
                txt = io.TextIOWrapper(
                    bf,
                    encoding="utf-8-sig",
                    errors="replace",
                    newline="",
                )
                for row in csv.DictReader(txt):
                    src = row.get("source_body") or ""
                    tgt = row.get("target_body") or ""
                    if src.strip() and tgt.strip():
                        t = src.strip() + "\n" + tgt.strip()
                        yield source_record(
                            "mdia_raw",
                            "dialogue_like_id",
                            "CC-BY-4.0",
                            f"DoctorDream/mDIA raw Indonesian dialogue:{name}",
                            t,
                            ordinal,
                        )
                        ordinal += 1

def load_tokenizer(project: Path):
    tok_path = project / "artifacts" / "tokenizer_v1" / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"STOP: tokenizer missing: {tok_path}")
    if sha256(tok_path) != TOKENIZER_SHA:
        raise SystemExit("STOP: frozen tokenizer SHA mismatch")
    try:
        from tokenizers import Tokenizer
    except Exception as e:
        raise SystemExit(f"STOP: python tokenizers package unavailable: {e}")
    return Tokenizer.from_file(str(tok_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1",
    )
    args = ap.parse_args()

    project = Path(args.project).resolve()
    raw = project / "data" / "raw_v3_sources"
    manifest_path = raw / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit("STOP: run acquire_model0001_v3_sources.py first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("test_split_touched") is not False
    ):
        raise SystemExit("STOP: invalid acquisition manifest")

    by_id = {x["id"]: x for x in manifest["sources"]}
    required = {
        "mdia_raw",
        "emotcmt",
        "frog_spoken",
        "talpco_ind",
        "reddit_indonesia_train",
    }
    if set(by_id) != required:
        raise SystemExit(f"STOP: source set drift: {sorted(by_id)}")

    for sid, spec in by_id.items():
        p = Path(spec["path"])
        if not p.is_file() or sha256(p) != spec["sha256"]:
            raise SystemExit(f"STOP: source changed after acquisition: {sid}")

    prior = load_prior_train_keys(project)
    tokenizer = load_tokenizer(project)

    parsers = [
        parse_mdia(Path(by_id["mdia_raw"]["path"])),
        parse_emot(Path(by_id["emotcmt"]["path"])),
        parse_frog(Path(by_id["frog_spoken"]["path"])),
        parse_talpco(Path(by_id["talpco_ind"]["path"])),
        parse_reddit(Path(by_id["reddit_indonesia_train"]["path"])),
    ]

    outdir = project / "data" / "corpus_v3_candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    pool = outdir / "new_pool.jsonl"

    seen = set()
    counts = Counter()
    tokens = Counter()
    rejected = Counter()
    chars = Counter()
    written = 0

    with pool.open("w", encoding="utf-8") as out:
        for iterator in parsers:
            for rec in iterator:
                t = norm_text(rec["text"])
                fam = rec["family"]

                if not acceptable(t):
                    rejected[f"{fam}:quality"] += 1
                    continue

                k = key_text(t)
                if k in prior:
                    rejected[f"{fam}:prior_exact"] += 1
                    continue
                if k in seen:
                    rejected[f"{fam}:new_exact"] += 1
                    continue

                ids = tokenizer.encode(t, add_special_tokens=False).ids
                if len(ids) < 4:
                    rejected[f"{fam}:too_few_tokens"] += 1
                    continue

                seen.add(k)
                rec["text"] = t
                rec["text_sha256"] = hashlib.sha256(
                    t.encode("utf-8")
                ).hexdigest()
                rec["token_count"] = len(ids)

                out.write(
                    json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                counts[fam] += 1
                tokens[fam] += len(ids)
                chars[fam] += len(t)
                written += 1

    report = {
        "status": "PASS",
        "schema": "model0001_v3_candidate_pool_audit_v1",
        "pool": str(pool),
        "pool_sha256": sha256(pool),
        "frozen_tokenizer_sha256": TOKENIZER_SHA,
        "records": written,
        "family_records": dict(counts),
        "family_tokens": dict(tokens),
        "family_characters": dict(chars),
        "total_tokens": sum(tokens.values()),
        "prior_train_exact_keys": len(prior),
        "rejected": dict(rejected),
        "hard_guards": {
            "test_split_touched": False,
            "dataset_v2_train_bin_reused": False,
            "tokenizer_changed": False,
            "mixture_not_yet_locked": True,
            "packing_not_started": True,
            "training_not_started": True,
        },
    }

    rp = outdir / "CANDIDATE_POOL_AUDIT.json"
    rp.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWROTE: {rp}")

if __name__ == "__main__":
    main()
