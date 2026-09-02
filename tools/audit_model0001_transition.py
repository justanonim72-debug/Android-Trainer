#!/usr/bin/env python3
"""
Fail-closed transition audit for moving Friend-Core Model #0001 from the
completed CPT-v2 CPU stage onto the validated native OpenCL trainer.

This tool does NOT train. It extracts every fact needed to define the next
semantic stage and reports any missing production recipe fields instead of
inventing them.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import torch

EXPECTED_MODEL_SHA = "047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
EXPECTED_STAGE_STEPS = 15624
EXPECTED_STAGE_TOKENS = 3_999_744
EXPECTED_LIFETIME_TOKENS = 5_535_744


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def first(obj: dict, names: list[str]):
    for name in names:
        if name in obj:
            return obj[name]
    return None


def load_exporter(path: Path):
    spec = importlib.util.spec_from_file_location("model0001_exporter", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"STOP: cannot import exporter {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_completion_marker(ckpt: Path):
    for base in (ckpt.parent, ckpt.parent.parent):
        for name in ("COMPLETED.json", "completed.json", "RUN_COMPLETE.json", "run_complete.json"):
            p = base / name
            if p.is_file():
                try:
                    value = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        return p, value
                except Exception:
                    pass
    return None, None



def extract_engine_recipe_evidence(engine, engine_path: Path):
    """Collect literal training-policy evidence from the real script-17 engine.

    This intentionally returns candidates rather than deciding a new stage.
    Values are accepted only when they are literal module globals or literal
    argparse defaults in the source file.
    """
    out = {}
    wanted = re.compile(
        r"(?:^|_)(?:lr|learning_rate|max_lr|min_lr|warmup|epoch|steps?|"
        r"checkpoint|save_every|eval_every|log_every|seed|shuffle|schedule|"
        r"scheduler)(?:_|$)",
        re.I,
    )
    for name in dir(engine):
        if not wanted.search(name):
            continue
        try:
            value = getattr(engine, name)
        except Exception:
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            out[f"engine_global.{name}"] = jsonable(value)

    try:
        tree = ast.parse(engine_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_add_argument = (
            isinstance(fn, ast.Attribute) and fn.attr == "add_argument"
        )
        if not is_add_argument or not node.args:
            continue
        flags = []
        for arg in node.args:
            try:
                value = ast.literal_eval(arg)
            except Exception:
                continue
            if isinstance(value, str):
                flags.append(value)
        if not flags:
            continue
        dest = None
        for flag in flags:
            if flag.startswith("--"):
                dest = flag[2:].replace("-", "_")
                break
        if not dest or not wanted.search(dest):
            continue
        default_node = None
        for kw in node.keywords:
            if kw.arg == "default":
                default_node = kw.value
                break
        if default_node is None:
            continue
        try:
            default = ast.literal_eval(default_node)
        except Exception:
            # Common pattern: default=DEFAULT_LR. Resolve only literal globals
            # already observed from the imported real engine module.
            if isinstance(default_node, ast.Name):
                key = f"engine_global.{default_node.id}"
                if key in out:
                    default = out[key]
                else:
                    continue
            else:
                continue
        if default is None or isinstance(default, (bool, int, float, str)):
            out[f"argparse_default.{dest}"] = jsonable(default)
    return out


def extract_scheduler_evidence(ck: dict, contract: dict):
    out = {}
    for key in (
        "scheduler", "lr_scheduler", "schedule", "lr_schedule",
        "warmup_steps", "warmup_ratio", "min_lr", "min_lr_ratio",
        "max_lr", "peak_lr", "learning_rate", "lr", "epochs",
        "total_steps", "max_steps", "steps", "eval_every",
        "checkpoint_every", "save_every", "seed", "shuffle_seed",
    ):
        if key in ck:
            out[f"checkpoint.{key}"] = jsonable(ck[key])
        if key in contract:
            out[f"contract.{key}"] = jsonable(contract[key])
    run_cfg = ck.get("run_config")
    if isinstance(run_cfg, dict):
        for key in (
            "scheduler", "lr_scheduler", "schedule", "lr_schedule",
            "warmup_steps", "warmup_ratio", "min_lr", "min_lr_ratio",
            "max_lr", "peak_lr", "learning_rate", "lr", "epochs",
            "total_steps", "max_steps", "steps", "eval_every",
            "checkpoint_every", "save_every", "seed", "shuffle_seed",
        ):
            if key in run_cfg:
                out[f"run_config.{key}"] = jsonable(run_cfg[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--exporter", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--train-bin", default=None)
    ap.add_argument("--output", default="/storage/emulated/0/Download/model0001-transition-audit.json")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    exporter_path = Path(args.exporter).resolve() if args.exporter else Path(__file__).with_name("export_model0001_bundle.py")
    exporter = load_exporter(exporter_path)
    engine, engine_path = exporter.import_engine(project)
    ckpt = exporter.discover_checkpoint(project, args.checkpoint, engine)
    train_bin = Path(args.train_bin).resolve() if args.train_bin else project / "artifacts" / "model0001_dataset_v2" / "train.bin"

    if not train_bin.is_file():
        raise SystemExit(f"STOP: Dataset-v2 train.bin missing: {train_bin}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get("model"), dict):
        raise SystemExit("STOP: checkpoint structure invalid")

    boundary = exporter.verify_completed_boundary(project, ckpt, train_bin, ck, engine)
    model_sha = exporter.engine_state_hash(engine, ck["model"])
    if model_sha != EXPECTED_MODEL_SHA:
        raise SystemExit(f"STOP: wrong source model SHA {model_sha}")

    contract = ck.get("contract") or ck.get("run_contract") or {}
    if not isinstance(contract, dict):
        contract = {}

    marker_path, marker = find_completion_marker(ckpt)
    marker = marker if isinstance(marker, dict) else {}

    opt = ck.get("optimizer")
    if not isinstance(opt, dict) or not isinstance(opt.get("param_groups"), list):
        raise SystemExit("STOP: source checkpoint optimizer state missing")
    groups = []
    for i, group in enumerate(opt["param_groups"]):
        groups.append({
            "index": i,
            "lr": group.get("lr"),
            "betas": jsonable(group.get("betas")),
            "eps": group.get("eps"),
            "weight_decay": group.get("weight_decay"),
            "param_count": len(group.get("params", [])),
            "has_param_names": isinstance(group.get("param_names"), list),
        })

    train_bytes = train_bin.stat().st_size
    if train_bytes % 2 != 0:
        raise SystemExit("STOP: train.bin byte size is not uint16-aligned")
    packed_tokens = train_bytes // 2
    full_windows = max(0, (packed_tokens - 1) // 256)

    stage_step = first(ck, ["stage_step", "stage_steps", "step", "optimizer_step"])
    stage_tokens = first(ck, ["stage_tokens_seen", "stage_tokens", "stage_scored_train_tokens"])
    lifetime_tokens = first(ck, ["lifetime_tokens_seen", "lifetime_tokens", "cumulative_tokens_seen"])

    engine_recipe_evidence = extract_engine_recipe_evidence(engine, engine_path)
    scheduler_evidence = extract_scheduler_evidence(ck, contract)
    scheduler_evidence.update(engine_recipe_evidence)

    known = {
        "source_weights": {
            "checkpoint": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "model_state_sha256": model_sha,
            "required_model_state_sha256": EXPECTED_MODEL_SHA,
            "source_stage_steps": int(stage_step) if stage_step is not None else EXPECTED_STAGE_STEPS,
            "source_stage_tokens": int(stage_tokens) if stage_tokens is not None else EXPECTED_STAGE_TOKENS,
            "source_lifetime_tokens": int(lifetime_tokens) if lifetime_tokens is not None else EXPECTED_LIFETIME_TOKENS,
        },
        "dataset_v2": {
            "train_bin": str(train_bin),
            "train_bin_sha256": sha256_file(train_bin),
            "uint16_tokens": packed_tokens,
            "full_256_target_windows": full_windows,
            "test_split_used": False,
        },
        "engine": {
            "path": str(engine_path),
            "sha256": sha256_file(engine_path),
        },
        "optimizer_source": {
            "groups": groups,
            "state_entry_count": len(opt.get("state", {})),
            "production_migration_policy": "fresh_zero_moments",
            "reason": "semantic backend-migration boundary; do not inherit gate benchmark state",
        },
        "resume_metadata_present": {
            "epoch": ck.get("epoch"),
            "cursor": ck.get("cursor"),
            "optimizer_step": ck.get("optimizer_step"),
            "stage_step": ck.get("stage_step"),
            "stage_tokens_seen": ck.get("stage_tokens_seen"),
            "lifetime_tokens_seen": ck.get("lifetime_tokens_seen"),
        },
        "completion_marker": {
            "path": str(marker_path) if marker_path else None,
            "sha256": sha256_file(marker_path) if marker_path else None,
            "status": marker.get("status"),
        },
        "scheduler_and_cadence_evidence": scheduler_evidence,
        "engine_recipe_evidence": engine_recipe_evidence,
    }

    # These fields are mandatory before production training is allowed to mutate
    # the completed source model. They are deliberately not inferred from the
    # 1e-4 gate LR or from the 20-step benchmark.
    ev = known["scheduler_and_cadence_evidence"]
    requirements = {
        "next_stage_name": bool(first(contract, ["stage_name", "run_name", "name"])) or
            any(k.endswith((".run_name", ".stage_name")) for k in ev),
        "next_stage_total_updates_or_epochs": any(
            k.endswith((".total_steps", ".max_steps", ".steps", ".epochs", ".epoch"))
            for k in ev
        ),
        "production_lr_or_schedule": any(
            any(token in k.lower() for token in (".lr", "learning_rate", "max_lr", "peak_lr", "schedule", "scheduler"))
            for k in ev
        ),
        "sample_order_seed_or_explicit_cursor_policy": any(
            ("seed" in k.lower() or "shuffle" in k.lower()) for k in ev
        ) or ck.get("cursor") is not None,
        "checkpoint_cadence": any(
            ("checkpoint_every" in k.lower() or "save_every" in k.lower()) for k in ev
        ),
        "evaluation_cadence": any("eval_every" in k.lower() for k in ev),
    }
    missing = [name for name, present in requirements.items() if not present]

    report = {
        "status": "READY_FOR_PRODUCTION_IMPLEMENTATION" if not missing else "NEEDS_STAGE_RECIPE",
        "schema": "model0001_native_transition_audit_v1",
        "known": known,
        "requirements": requirements,
        "missing_required_recipe_fields": missing,
        "hard_guards": {
            "gate_checkpoint_is_production_source": False,
            "gate_lr_1e_4_is_production_lr": False,
            "gate_benchmark_sample_reuse_allowed": False,
            "source_model_must_equal_completed_cpt_v2_sha": True,
            "optimizer_moments_start_fresh_at_migration_boundary": True,
            "lifetime_tokens_start_from": EXPECTED_LIFETIME_TOKENS,
            "test_split_must_remain_untouched": True,
        },
        "raw_run_contract": jsonable(contract),
    }

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWROTE: {out}")


if __name__ == "__main__":
    main()
