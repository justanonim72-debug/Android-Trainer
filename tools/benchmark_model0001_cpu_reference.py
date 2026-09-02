#!/usr/bin/env python3
"""
Apples-to-apples CPU reference benchmark for Model #0001.

Purpose: recover the missing denominator for the already-proven native OpenCL
sustained benchmark. This does not train a production stage and does not write
model checkpoints. It loads the immutable completed CPT-v2 weights, recreates
the exact fresh-state AdamW gate semantics, uses the same 257-token Dataset-v2
window, runs one warmup + N timed full updates, and reports target tok/s.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


def load_exporter(path: Path):
    spec = importlib.util.spec_from_file_location("model0001_exporter", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"STOP: cannot import exporter {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--exporter", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--train-bin", default=None)
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--gate-lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = preserve current PyTorch thread setting")
    ap.add_argument("--output", default="/storage/emulated/0/Download/model0001-cpu-reference-benchmark.json")
    args = ap.parse_args()

    if args.warmup < 0 or args.steps <= 0 or args.window_index < 0:
        raise SystemExit("STOP: invalid warmup/steps/window-index")
    if not math.isfinite(args.gate_lr) or args.gate_lr <= 0:
        raise SystemExit("STOP: gate LR must be finite and > 0")
    if args.threads < 0:
        raise SystemExit("STOP: --threads must be >= 0")
    if args.threads:
        torch.set_num_threads(args.threads)

    project = Path(args.project).resolve()
    exporter_path = (
        Path(args.exporter).resolve()
        if args.exporter else Path(__file__).with_name("export_model0001_bundle.py")
    )
    ex = load_exporter(exporter_path)
    eng, engine_path = ex.import_engine(project)
    ckpt = ex.discover_checkpoint(project, args.checkpoint, eng)
    train_bin = (
        Path(args.train_bin).resolve()
        if args.train_bin else project / "artifacts" / "model0001_dataset_v2" / "train.bin"
    )
    if not train_bin.is_file():
        raise SystemExit(f"STOP: missing Dataset-v2 train.bin: {train_bin}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get("model"), dict):
        raise SystemExit("STOP: checkpoint structure invalid")
    boundary = ex.verify_completed_boundary(project, ckpt, train_bin, ck, eng)

    contract = ck.get("contract") or ck.get("run_contract") or {}
    cfg = None
    for candidate in [
        ck.get("model_config"),
        contract.get("model") if isinstance(contract, dict) else None,
        ck.get("run_config", {}).get("model") if isinstance(ck.get("run_config"), dict) else None,
        getattr(eng, "DEFAULT_MODEL", None),
    ]:
        if isinstance(candidate, dict):
            cfg = dict(candidate)
            break
    if cfg is None:
        raise SystemExit("STOP: cannot locate exact Model #0001 config")
    ex.normalized_config(cfg)

    model = eng.Model0001(cfg)
    model.load_state_dict(ck["model"], strict=True)
    ex.assert_model_fp32(model, "CPU benchmark model")
    model.train()
    norm, origin = ex.exact_state(model)
    del norm
    hp = ex.optimizer_semantics(model, ck, origin, args.gate_lr, eng)
    optimizer = ex.make_fresh_reference_optimizer(model, hp)

    raw = np.memmap(train_bin, dtype="<u2", mode="r")
    start = args.window_index * 256
    arr = np.asarray(raw[start:start + 257], dtype=np.int64)
    if arr.size != 257 or int(arr.min()) < 0 or int(arr.max()) >= 14000:
        raise SystemExit("STOP: invalid benchmark Dataset-v2 window")
    x = torch.from_numpy(arr[:-1].copy()).unsqueeze(0)
    y = torch.from_numpy(arr[1:].copy()).unsqueeze(0)

    def step():
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("nonfinite CPU benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    losses = []
    for _ in range(args.warmup):
        losses.append(step())

    t0 = time.perf_counter()
    for _ in range(args.steps):
        losses.append(step())
    seconds = time.perf_counter() - t0
    tokps = args.steps * 256.0 / max(seconds, 1e-12)

    report = {
        "status": "PASS",
        "schema": "model0001_cpu_reference_benchmark_v1",
        "purpose": "GPU_ACCEPTANCE_DENOMINATOR_ONLY",
        "production_training": False,
        "source_model_state_sha256": boundary["model_state_sha256"],
        "source_checkpoint_sha256": boundary["checkpoint_sha256"],
        "train_bin_sha256": boundary["train_bin_sha256"],
        "window_index": args.window_index,
        "target_tokens_per_step": 256,
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "seconds": seconds,
        "tokens_per_second": tokps,
        "final_loss": losses[-1],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "gate_lr": args.gate_lr,
        "betas": [hp["beta1"], hp["beta2"]],
        "eps": hp["eps"],
        "grad_clip": 1.0,
        "optimizer_init": "fresh_zero_moments",
        "engine_path": str(engine_path),
        "pid": os.getpid(),
    }

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWROTE: {out}")


if __name__ == "__main__":
    main()
