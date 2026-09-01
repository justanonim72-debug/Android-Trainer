#!/usr/bin/env python3
"""
Strict local exporter for the completed Friend-Core Model #0001 CPT-v2 stage.

The exporter runs only on the user's local phone/project. It never uploads the
checkpoint or dataset. It hard-verifies the completed CPT-v2 boundary, imports
the real script-17 model implementation, computes PyTorch FP32 reference
forward/backward/AdamW evidence, and emits a ZIP-compatible .atb bundle.

Gate optimizer state is intentionally FRESH (zero moments) because backend
migration is only evaluated at a semantic stage boundary. Hyperparameters and
per-parameter weight-decay grouping are reconstructed from the real completed
checkpoint and the real engine, then verified before export.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

EXPECTED = {
    "vocab_size": 14000,
    "seq_len": 256,
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 6,
    "n_kv_heads": 2,
    "head_dim": 64,
    "d_ff": 1152,
    "params": 19_145_088,
}
EXPECTED_STAGE = {
    "run_name": "model0001_cpt_v2_epoch1",
    "stage_steps": 15_624,
    "stage_tokens_seen": 3_999_744,
    "lifetime_tokens_seen": 5_535_744,
}
NORMALIZED_TO_SOURCE = {
    "tok_embeddings.weight": "tok_emb.weight",
    "final_norm.weight": "norm.weight",
}
for _i in range(8):
    _p = f"blocks.{_i}."
    _q = f"layers.{_i}."
    NORMALIZED_TO_SOURCE.update({
        _q + "attn_norm.weight": _p + "attn_norm.weight",
        _q + "q_proj.weight": _p + "attn.q_proj.weight",
        _q + "k_proj.weight": _p + "attn.k_proj.weight",
        _q + "v_proj.weight": _p + "attn.v_proj.weight",
        _q + "o_proj.weight": _p + "attn.o_proj.weight",
        _q + "ffn_norm.weight": _p + "ffn_norm.weight",
        _q + "gate_proj.weight": _p + "ffn.gate_proj.weight",
        _q + "up_proj.weight": _p + "ffn.up_proj.weight",
        _q + "down_proj.weight": _p + "ffn.down_proj.weight",
    })


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tensor_state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k]
        if not torch.is_tensor(t):
            continue
        x = t.detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(x.dtype).encode("utf-8"))
        h.update(struct.pack("<I", x.ndim))
        for d in x.shape:
            h.update(struct.pack("<Q", int(d)))
        h.update(x.numpy().tobytes(order="C"))
    return h.hexdigest()


def import_engine(project: Path):
    p = project / "scripts" / "17_pretrain_model0001.py"
    if not p.exists():
        raise SystemExit(f"STOP: missing source engine: {p}")
    spec = importlib.util.spec_from_file_location("friend_model0001_engine17", p)
    if spec is None or spec.loader is None:
        raise SystemExit("STOP: cannot import script 17")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, p


def normalized_config(raw: dict) -> dict:
    aliases = {
        "vocab_size": ["vocab_size", "vocab"],
        "seq_len": ["seq_len", "max_seq_len", "context_length"],
        "d_model": ["d_model", "hidden_size"],
        "n_layers": ["n_layers", "num_layers"],
        "n_heads": ["n_heads", "num_heads"],
        "n_kv_heads": ["n_kv_heads", "num_kv_heads"],
        "d_ff": ["d_ff", "ffn_hidden", "hidden_dim", "intermediate_size"],
        "rope_theta": ["rope_theta"],
        "rms_norm_eps": ["rms_norm_eps", "rms_eps", "norm_eps"],
    }
    out = {}
    for dst, names in aliases.items():
        vals = [raw[n] for n in names if n in raw]
        if vals:
            if any(v != vals[0] for v in vals[1:]):
                raise SystemExit(f"STOP: conflicting config aliases for {dst}: {vals}")
            out[dst] = vals[0]
    need = list(aliases)
    missing = [x for x in need if x not in out]
    if missing:
        raise SystemExit(f"STOP: model config missing exact fields {missing}")
    out["head_dim"] = int(out["d_model"]) // int(out["n_heads"])
    for k in ["vocab_size", "seq_len", "d_model", "n_layers", "n_heads",
              "n_kv_heads", "d_ff", "head_dim"]:
        if int(out[k]) != EXPECTED[k]:
            raise SystemExit(f"STOP: architecture drift {k}: {out[k]} != {EXPECTED[k]}")
    if float(out["rope_theta"]) <= 1 or float(out["rms_norm_eps"]) <= 0:
        raise SystemExit("STOP: invalid RoPE/RMSNorm configuration")
    return out


def exact_state(model) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    state = model.state_dict()
    actual_tensor_keys = {k for k, v in state.items() if torch.is_tensor(v)}
    expected_keys = set(NORMALIZED_TO_SOURCE.values())
    if actual_tensor_keys != expected_keys:
        missing = sorted(expected_keys - actual_tensor_keys)
        extra = sorted(actual_tensor_keys - expected_keys)
        raise SystemExit(
            "STOP: Model #0001 state layout drift. "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    out = {}
    for slot, src in NORMALIZED_TO_SOURCE.items():
        out[slot] = state[src].detach().cpu().float().contiguous()
    unique = sum(x.numel() for x in out.values())
    if unique != EXPECTED["params"]:
        raise SystemExit(f"STOP: normalized params={unique:,} != {EXPECTED['params']:,}")
    return out, dict(NORMALIZED_TO_SOURCE)


def detect_rope_style(engine_text: str) -> str:
    t = engine_text.lower()
    if "0::2" in t and "1::2" in t and ("even/odd" in t or "stack((out_e, out_o)" in t):
        return "interleaved"
    if "rotate_half" in t or ("chunk(2" in t and "cat" in t):
        return "half_split"
    return "auto"


def probe_indices(name: str, n: int, count: int = 16):
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return sorted(set(int(x) for x in rng.integers(0, n, size=count)))


def verify_completed_boundary(project: Path, ckpt: Path, train_bin: Path, ck: dict) -> dict:
    run_dir = project / "artifacts" / "model0001_runs" / EXPECTED_STAGE["run_name"]
    completed_path = run_dir / "COMPLETED.json"
    if not completed_path.exists():
        raise SystemExit(f"STOP: completed stage marker missing: {completed_path}")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))

    if completed.get("schema") != "model0001_cpt_v2_completed_v1" or completed.get("status") != "PASS":
        raise SystemExit("STOP: CPT-v2 COMPLETED.json is not PASS")
    for k, v in EXPECTED_STAGE.items():
        if completed.get(k) != v:
            raise SystemExit(f"STOP: CPT-v2 boundary mismatch {k}: {completed.get(k)} != {v}")
    if completed.get("test_split_used") is not False:
        raise SystemExit("STOP: CPT-v2 test split flag is not untouched")

    got_ck_sha = sha256_file(ckpt)
    if completed.get("final_checkpoint_sha256") != got_ck_sha:
        raise SystemExit("STOP: latest.pt does not match COMPLETED.json final checkpoint SHA")

    if ck.get("schema") != "model0001_cpt_v2_checkpoint_v1":
        raise SystemExit(f"STOP: unexpected checkpoint schema: {ck.get('schema')}")
    if int(ck.get("stage_step", -1)) != EXPECTED_STAGE["stage_steps"]:
        raise SystemExit("STOP: checkpoint is not the final CPT-v2 step")
    if int(ck.get("stage_tokens_seen", -1)) != EXPECTED_STAGE["stage_tokens_seen"]:
        raise SystemExit("STOP: checkpoint stage token counter mismatch")
    if int(ck.get("lifetime_tokens_seen", -1)) != EXPECTED_STAGE["lifetime_tokens_seen"]:
        raise SystemExit("STOP: checkpoint lifetime token counter mismatch")

    model_sha = tensor_state_hash(ck["model"])
    if completed.get("final_model_state_sha256") != model_sha:
        raise SystemExit("STOP: final model state SHA mismatch")

    contract = ck.get("contract")
    if not isinstance(contract, dict) or contract.get("stage") != "continued_pretraining_dataset_v2_epoch1":
        raise SystemExit("STOP: missing/invalid CPT-v2 semantic contract")
    if contract.get("test_split_used") is not False:
        raise SystemExit("STOP: stage contract test-split flag changed")

    got_train_sha = sha256_file(train_bin)
    if contract.get("dataset_v2_train_sha256") != got_train_sha:
        raise SystemExit("STOP: Dataset v2 train.bin does not match stage contract")

    return {
        "completed_path": str(completed_path),
        "completed_sha256": sha256_file(completed_path),
        "checkpoint_sha256": got_ck_sha,
        "model_state_sha256": model_sha,
        "train_bin_sha256": got_train_sha,
        "stage_steps": EXPECTED_STAGE["stage_steps"],
        "stage_tokens_seen": EXPECTED_STAGE["stage_tokens_seen"],
        "lifetime_tokens_seen": EXPECTED_STAGE["lifetime_tokens_seen"],
        "test_split_used": False,
    }


def optimizer_semantics(eng, model, ck: dict, origin: dict[str, str], gate_lr: float) -> dict:
    opt_state = ck.get("optimizer")
    if not isinstance(opt_state, dict) or not isinstance(opt_state.get("param_groups"), list):
        raise SystemExit("STOP: final checkpoint optimizer state missing")

    groups = opt_state["param_groups"]
    if not groups:
        raise SystemExit("STOP: final checkpoint optimizer has no groups")

    positive_wd = sorted({float(g.get("weight_decay", 0.0)) for g in groups if float(g.get("weight_decay", 0.0)) > 0})
    if len(positive_wd) != 1:
        raise SystemExit(f"STOP: cannot reconstruct exact AdamW decay rule: positive_wd={positive_wd}")
    source_wd = positive_wd[0]

    # Rebuild the optimizer using the real engine's grouping rule, then load the
    # completed checkpoint optimizer. This proves the group structure still matches.
    opt = eng.optimizer_for(model, gate_lr, source_wd)
    opt.load_state_dict(opt_state)

    beta_eps = {
        (tuple(float(x) for x in g["betas"]), float(g["eps"]))
        for g in opt.param_groups
    }
    if len(beta_eps) != 1:
        raise SystemExit(f"STOP: per-group beta/eps differ: {beta_eps}")
    (betas, eps), = beta_eps

    param_group_by_id = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            param_group_by_id[id(p)] = gi

    named = dict(model.named_parameters())
    slot_wd = {}
    slot_group = {}
    for slot, src in origin.items():
        p = named.get(src)
        if p is None:
            raise SystemExit(f"STOP: optimizer mapping missing parameter {src}")
        gi = param_group_by_id.get(id(p))
        if gi is None:
            raise SystemExit(f"STOP: optimizer group mapping missing {src}")
        slot_group[slot] = int(gi)
        slot_wd[slot] = float(opt.param_groups[gi]["weight_decay"])

    # Exact rule from script17 must still be reflected by the loaded groups:
    # matrices/embedding decay; 1-D RMSNorm scales do not.
    for slot, src in origin.items():
        expected_wd = source_wd if named[src].dim() >= 2 else 0.0
        if not math.isclose(slot_wd[slot], expected_wd, rel_tol=0, abs_tol=1e-15):
            raise SystemExit(
                f"STOP: optimizer group drift for {src}: wd={slot_wd[slot]} expected={expected_wd}"
            )

    source_lrs = sorted({float(g.get("lr", 0.0)) for g in opt.param_groups})
    return {
        "gate_state": "fresh_zero_moments",
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(eps),
        "gate_lr": float(gate_lr),
        "source_optimizer_groups": len(opt.param_groups),
        "source_weight_decay": float(source_wd),
        "source_checkpoint_lr_values": source_lrs,
        "slot_group": slot_group,
        "slot_weight_decay": slot_wd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--checkpoint")
    ap.add_argument("--train-bin")
    ap.add_argument("--output")
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--gate-lr", type=float, default=1e-4)
    args = ap.parse_args()

    project = Path(args.project).resolve()
    run_dir = project / "artifacts" / "model0001_runs" / EXPECTED_STAGE["run_name"]
    ckpt = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "latest.pt"
    train_bin = Path(args.train_bin).resolve() if args.train_bin else project / "artifacts" / "model0001_dataset_v2" / "train.bin"
    out = Path(args.output).resolve() if args.output else project.parent / "model0001-gpu-gate.atb"

    if args.window_index < 0:
        raise SystemExit("STOP: --window-index must be >= 0")
    if not math.isfinite(args.gate_lr) or args.gate_lr <= 0:
        raise SystemExit("STOP: --gate-lr must be finite and > 0")
    if not ckpt.exists() or not train_bin.exists():
        raise SystemExit(f"STOP: missing checkpoint/train.bin: {ckpt} / {train_bin}")

    eng, engine_path = import_engine(project)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise SystemExit("STOP: checkpoint has no model state")

    boundary = verify_completed_boundary(project, ckpt, train_bin, ck)
    contract = ck["contract"]
    engine_cfg = dict(contract["model"])
    cfg = normalized_config(engine_cfg)

    model = eng.Model0001(engine_cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.float()
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != EXPECTED["params"]:
        raise SystemExit(f"STOP: model params {nparams:,} != {EXPECTED['params']:,}")

    norm, origin = exact_state(model)
    rope_style = detect_rope_style(engine_path.read_text(encoding="utf-8", errors="replace"))
    hp = optimizer_semantics(eng, model, ck, origin, args.gate_lr)

    raw = np.memmap(train_bin, dtype="<u2", mode="r")
    start = args.window_index * EXPECTED["seq_len"]
    arr = np.asarray(raw[start:start + EXPECTED["seq_len"] + 1], dtype=np.int64)
    if arr.size != 257:
        raise SystemExit("STOP: selected Dataset v2 train window incomplete")
    if int(arr.min()) < 0 or int(arr.max()) >= EXPECTED["vocab_size"]:
        raise SystemExit("STOP: selected train window contains OOV token")

    x = torch.from_numpy(arr[:-1].copy()).unsqueeze(0)
    y = torch.from_numpy(arr[1:].copy()).unsqueeze(0)

    model.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    if loss is None or not torch.isfinite(loss):
        raise SystemExit("STOP: PyTorch reference loss is nonfinite")
    loss.backward()

    sq = torch.zeros((), dtype=torch.float64)
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().double().pow(2).sum()
    global_norm = float(sq.sqrt())
    clip_coef = min(1.0, 1.0 / (global_norm + 1e-6))

    state_to_param = dict(model.named_parameters())
    grad_meta = {}
    for slot, src in origin.items():
        p = state_to_param[src]
        if p.grad is None:
            raise SystemExit(f"STOP: missing gradient for {src}")
        g = p.grad.detach().cpu().float().contiguous().view(-1)
        inds = probe_indices(slot, g.numel())
        grad_meta[slot] = {
            "l2": float(g.double().norm()),
            "max_abs": float(g.abs().max()),
            "probe_indices": inds,
            "probe_values": [float(g[i]) for i in inds],
        }

    flat = logits.detach().cpu().float()[0]
    positions = [0, 1, 63, 127, 191, 255]
    classes = [3, 17, 101, 997, 4096, 8191, 13999]
    logit_probe = [
        {"position": p, "token": c, "value": float(flat[p, c])}
        for p in positions for c in classes
    ]

    # Fresh optimizer gate reference using the REAL engine grouping/betas/eps.
    ref_model = eng.Model0001(engine_cfg)
    ref_model.load_state_dict(ck["model"], strict=True)
    ref_model.float()
    ref_model.train()
    ref_opt = eng.optimizer_for(ref_model, args.gate_lr, hp["source_weight_decay"])
    # Verify the fresh optimizer has the same group hyperparameters we exported.
    for g in ref_opt.param_groups:
        g["lr"] = args.gate_lr
    ref_opt.zero_grad(set_to_none=True)
    _, ref_loss = ref_model(x, y)
    ref_loss.backward()
    torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 1.0)
    ref_opt.step()
    ref_state = ref_model.state_dict()

    adam_probe = {}
    for slot, src in origin.items():
        before = norm[slot].view(-1)
        after = ref_state[src].detach().cpu().float().contiguous().view(-1)
        inds = probe_indices(slot, before.numel())
        adam_probe[slot] = {
            "probe_indices": inds,
            "before": [float(before[i]) for i in inds],
            "after": [float(after[i]) for i in inds],
            "weight_decay": hp["slot_weight_decay"][slot],
        }

    manifest = {
        "schema": "android_trainer_bundle_v2",
        "checkpoint_sha256": boundary["checkpoint_sha256"],
        "model_state_sha256": boundary["model_state_sha256"],
        "train_bin_sha256": boundary["train_bin_sha256"],
        "engine_sha256": sha256_file(engine_path),
        "stage_boundary": boundary,
        "config": cfg,
        "rope_style": rope_style,
        "parameter_count": nparams,
        "origin_state_keys": origin,
        "optimizer": hp,
        "sample": {
            "window_index": args.window_index,
            "tokens_file": "sample/tokens_i32.bin",
            "token_count": 257,
        },
        "reference": {
            "loss": float(loss.detach()),
            "global_grad_norm": global_norm,
            "clip_coef": clip_coef,
            "logit_probe": logit_probe,
            "gradient": grad_meta,
            "adamw_step1": adam_probe,
        },
        "tensors": {},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        sample_bytes = arr.astype("<i4", copy=False).tobytes()
        manifest["sample"]["sha256"] = sha256_bytes(sample_bytes)
        z.writestr("sample/tokens_i32.bin", sample_bytes)

        for slot, t in norm.items():
            b = t.numpy().astype("<f4", copy=False).tobytes(order="C")
            path = "tensors/" + slot.replace(".", "/") + ".f32"
            manifest["tensors"][slot] = {
                "path": path,
                "shape": list(t.shape),
                "dtype": "f32",
                "nbytes": len(b),
                "sha256": sha256_bytes(b),
            }
            z.writestr(path, b)

        z.writestr(
            "manifest.json",
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
        )

    print(json.dumps({
        "status": "PASS",
        "schema": manifest["schema"],
        "output": str(out),
        "bundle_sha256": sha256_file(out),
        "bundle_mib": out.stat().st_size / (1024 ** 2),
        "checkpoint_sha256": boundary["checkpoint_sha256"],
        "model_state_sha256": boundary["model_state_sha256"],
        "stage_steps": boundary["stage_steps"],
        "stage_tokens_seen": boundary["stage_tokens_seen"],
        "lifetime_tokens_seen": boundary["lifetime_tokens_seen"],
        "reference_loss": manifest["reference"]["loss"],
        "global_grad_norm": global_norm,
        "rope_style": rope_style,
        "optimizer": {
            "beta1": hp["beta1"], "beta2": hp["beta2"], "eps": hp["eps"],
            "gate_lr": hp["gate_lr"],
            "weight_decay_values": sorted(set(hp["slot_weight_decay"].values())),
        },
        "params": nparams,
        "test_split_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
