#!/usr/bin/env python3
"""
Strict Friend-Core F2 SFT JSONL validator.

This validates source records only. It does not generate examples, tokenize,
pack, train, or touch the frozen test split. The accepted base schema follows
the project's Local AI Training Blueprint Appendix B.1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
ALLOWED_SPLITS = {"train", "validation"}
REQUIRED = {"id", "language", "style", "messages", "source", "license", "quality_score", "split"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--output")
    args = ap.parse_args()
    path = Path(args.jsonl).resolve()
    if not path.is_file():
        raise SystemExit(f"STOP: missing SFT JSONL {path}")

    ids = set()
    split_counts = Counter()
    language_counts = Counter()
    style_counts = Counter()
    assistant_turns = 0
    user_turns = 0
    records = 0
    errors = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            records += 1
            try:
                row = json.loads(raw)
            except Exception as e:
                errors.append(f"line {line_no}: invalid JSON: {e}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_no}: record must be object")
                continue
            missing = sorted(REQUIRED - set(row))
            if missing:
                errors.append(f"line {line_no}: missing {missing}")
                continue
            rid = row["id"]
            if not isinstance(rid, str) or not rid.strip() or rid in ids:
                errors.append(f"line {line_no}: id missing/duplicate")
            else:
                ids.add(rid)
            if not isinstance(row["language"], str) or not row["language"].strip():
                errors.append(f"line {line_no}: language invalid")
            else:
                language_counts[row["language"]] += 1
            if not isinstance(row["style"], list) or not all(isinstance(x, str) and x for x in row["style"]):
                errors.append(f"line {line_no}: style must be list[str]")
            else:
                style_counts.update(row["style"])
            if not isinstance(row["source"], str) or not row["source"].strip():
                errors.append(f"line {line_no}: source/provenance missing")
            if not isinstance(row["license"], str) or not row["license"].strip():
                errors.append(f"line {line_no}: license/provenance missing")
            q = row["quality_score"]
            if not isinstance(q, (int, float)) or not math.isfinite(float(q)) or not 0.0 <= float(q) <= 1.0:
                errors.append(f"line {line_no}: quality_score must be finite 0..1")
            split = row["split"]
            if split not in ALLOWED_SPLITS:
                errors.append(f"line {line_no}: split must be train|validation; test is forbidden here")
            else:
                split_counts[split] += 1

            messages = row["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"line {line_no}: messages must contain >=2 turns")
                continue
            local_user = local_assistant = 0
            for mi, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"line {line_no}: message {mi} must be object")
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if role not in ALLOWED_ROLES:
                    errors.append(f"line {line_no}: message {mi} role invalid")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"line {line_no}: message {mi} content empty")
                if role == "user":
                    local_user += 1
                if role == "assistant":
                    local_assistant += 1
            if local_user == 0 or local_assistant == 0:
                errors.append(f"line {line_no}: record needs both user and assistant turns")
            user_turns += local_user
            assistant_turns += local_assistant

    if records == 0:
        errors.append("dataset has zero records")
    if split_counts["train"] == 0:
        errors.append("dataset has zero train records")
    if split_counts["validation"] == 0:
        errors.append("dataset has zero validation records; immutable SFT holdout required")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "schema": "friend_core_f2_sft_source_audit_v1",
        "input": str(path),
        "sha256": sha256(path),
        "records": records,
        "unique_ids": len(ids),
        "split_counts": dict(split_counts),
        "language_counts": dict(language_counts),
        "style_counts": dict(style_counts),
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "test_split_used": False,
        "errors": errors[:200],
    }
    out = Path(args.output).resolve() if args.output else path.with_suffix(path.suffix + ".audit.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
