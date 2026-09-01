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
    "stage_steps": 15_624,
    "stage_tokens_seen": 3_999_744,
    "lifetime_tokens_seen": 5_535_744,
    "final_model_state_sha256": "047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d",
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


def assert_model_fp32(model, label: str) -> None:
    """Fail closed unless the loaded reference model is already FP32.

    The completed script-17 RotaryEmbedding intentionally defines a static
    helper named _apply(x, cos, sin). PyTorch module-wide dtype/device
    conversion helpers (including Module.float()) recursively dispatch the
    framework method nn.Module._apply(fn) and collide with that engine helper.
    The completed checkpoint is already FP32, so the exporter verifies dtype
    without mutating or converting the frozen engine.
    """
    bad_params = [
        (name, str(param.dtype))
        for name, param in model.named_parameters()
        if param.dtype != torch.float32
    ]
    bad_buffers = [
        (name, str(buf.dtype))
        for name, buf in model.named_buffers()
        if torch.is_floating_point(buf) and buf.dtype != torch.float32
    ]
    if bad_params or bad_buffers:
        raise SystemExit(
            f"STOP: {label} is not already FP32 after strict checkpoint load; "
            f"non_fp32_params={bad_params[:8]} "
            f"non_fp32_floating_buffers={bad_buffers[:8]}"
        )

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


def _semantic_kind(name: str, shape: tuple[int, ...]) -> str | None:
    n = name.lower()
    if shape == (14000, 384):
        return "tok_embeddings"
    if shape == (384, 384):
        if re.search(r"(^|[._])(q|query)([._]|$)", n):
            return "q_proj"
        if re.search(r"(^|[._])(o|out|output)([._]|$)", n):
            return "o_proj"
    if shape == (128, 384):
        if re.search(r"(^|[._])(k|key)([._]|$)", n):
            return "k_proj"
        if re.search(r"(^|[._])(v|value)([._]|$)", n):
            return "v_proj"
    if shape == (1152, 384):
        if "gate" in n or re.search(r"(^|[._])w1([._]|$)", n):
            return "gate_proj"
        if re.search(r"(^|[._])up([._]|$)", n) or re.search(r"(^|[._])w3([._]|$)", n):
            return "up_proj"
    if shape == (384, 1152):
        return "down_proj"
    if shape == (384,):
        if any(x in n for x in ["attn_norm", "attention_norm", "input_layernorm", "ln1", "norm1"]):
            return "attn_norm"
        if any(x in n for x in ["ffn_norm", "post_attention", "post_attn", "ln2", "norm2"]):
            return "ffn_norm"
    return None


def exact_state(model) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    params = {k: v for k, v in model.named_parameters()}
    if sum(p.numel() for p in params.values()) != EXPECTED["params"]:
        raise SystemExit("STOP: Model #0001 trainable parameter count drift")

    # Fast exact-name path for the reference implementation.
    expected_keys = set(NORMALIZED_TO_SOURCE.values())
    if set(params) == expected_keys:
        out = {
            slot: params[src].detach().cpu().float().contiguous()
            for slot, src in NORMALIZED_TO_SOURCE.items()
        }
        return out, dict(NORMALIZED_TO_SOURCE)

    # Strict semantic fallback. Every mapping must be unique and every trainable
    # parameter must be consumed; otherwise export stops rather than guessing.
    layer_re = re.compile(r"(?:^|\.)(?:blocks|layers|h)\.(\d+)(?:\.|$)", re.I)
    embedding = [(k, p) for k, p in params.items() if tuple(p.shape) == (14000, 384)]
    if len(embedding) != 1:
        raise SystemExit(f"STOP: expected exactly one 14000x384 tied embedding, got {[k for k,_ in embedding]}")

    out = {"tok_embeddings.weight": embedding[0][1].detach().cpu().float().contiguous()}
    origin = {"tok_embeddings.weight": embedding[0][0]}
    consumed = {embedding[0][0]}
    layers = {i: {} for i in range(8)}
    outside_norm = []

    for name, p in params.items():
        if name in consumed:
            continue
        shape = tuple(p.shape)
        m = layer_re.search(name)
        if m:
            li = int(m.group(1))
            if li not in layers:
                raise SystemExit(f"STOP: unexpected layer index in parameter {name}")
            kind = _semantic_kind(name, shape)
            if kind is None:
                raise SystemExit(f"STOP: cannot semantically classify trainable parameter {name} shape={shape}")
            if kind in layers[li]:
                raise SystemExit(f"STOP: ambiguous layer {li} {kind}: {layers[li][kind][0]} vs {name}")
            layers[li][kind] = (name, p)
        elif shape == (384,) and "norm" in name.lower():
            outside_norm.append((name, p))
        else:
            raise SystemExit(f"STOP: unexpected trainable parameter outside blocks: {name} shape={shape}")

    need = ["attn_norm", "q_proj", "k_proj", "v_proj", "o_proj",
            "ffn_norm", "gate_proj", "up_proj", "down_proj"]
    for li in range(8):
        missing = [k for k in need if k not in layers[li]]
        if missing:
            raise SystemExit(f"STOP: layer {li} semantic mapping missing {missing}")
        for kind in need:
            src, p = layers[li][kind]
            slot = f"layers.{li}.{kind}.weight"
            out[slot] = p.detach().cpu().float().contiguous()
            origin[slot] = src
            consumed.add(src)

    if len(outside_norm) != 1:
        raise SystemExit(f"STOP: expected exactly one final norm outside blocks, got {[k for k,_ in outside_norm]}")
    src, p = outside_norm[0]
    out["final_norm.weight"] = p.detach().cpu().float().contiguous()
    origin["final_norm.weight"] = src
    consumed.add(src)

    if consumed != set(params):
        raise SystemExit(f"STOP: unconsumed trainable parameters: {sorted(set(params)-consumed)}")
    if sum(x.numel() for x in out.values()) != EXPECTED["params"]:
        raise SystemExit("STOP: normalized parameter count drift")
    return out, origin


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


def engine_state_hash(eng, state: dict) -> str:
    fn = getattr(eng, "tensor_state_hash", None)
    if callable(fn):
        return str(fn(state))
    # Fallback used only when the training engine exposes no hash helper.
    return tensor_state_hash(state)


def _first_int(d: dict, names: list[str]):
    for n in names:
        if n in d:
            try:
                return int(d[n])
            except Exception:
                pass
    return None


def discover_checkpoint(project: Path, explicit: str | None, eng) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise SystemExit(f"STOP: checkpoint not found: {p}")
        return p

    roots = [
        project / "artifacts" / "model0001_runs",
        project / "runs",
        project / "artifacts",
    ]
    candidates = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for pat in ("latest.pt", "final_model.pt"):
            for p in root.rglob(pat):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                candidates.append(rp)

    matches = []
    inspected = []
    for p in candidates:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            state = ck.get("model") if isinstance(ck, dict) else None
            if not isinstance(state, dict):
                continue
            sha = engine_state_hash(eng, state)
            step = _first_int(ck, ["stage_step", "stage_steps", "step", "optimizer_step"])
            stage_tok = _first_int(ck, ["stage_tokens_seen", "stage_tokens", "stage_scored_train_tokens"])
            life_tok = _first_int(ck, ["lifetime_tokens_seen", "lifetime_tokens", "cumulative_tokens_seen"])
            opt = ck.get("optimizer")
            has_optimizer = isinstance(opt, dict) and isinstance(opt.get("param_groups"), list) and len(opt["param_groups"]) > 0
            rec = {
                "path": p,
                "model_sha": sha,
                "step": step,
                "stage_tokens": stage_tok,
                "lifetime_tokens": life_tok,
                "has_optimizer": has_optimizer,
                "is_latest": p.name == "latest.pt",
                "looks_cpt_v2": "cpt_v2" in str(p).lower(),
            }
            inspected.append(rec)
            if sha == EXPECTED_STAGE["final_model_state_sha256"]:
                matches.append(rec)
        except Exception as e:
            inspected.append({"path": p, "error": type(e).__name__})

    # It is normal for a completed run to contain both latest.pt and final_model.pt
    # with the SAME final model-state hash. The gate needs optimizer metadata, so
    # select the unique checkpoint that can prove the completed stage semantics.
    eligible = []
    for rec in matches:
        if not rec["has_optimizer"]:
            continue
        if rec["step"] is not None and rec["step"] != EXPECTED_STAGE["stage_steps"]:
            continue
        if rec["stage_tokens"] is not None and rec["stage_tokens"] != EXPECTED_STAGE["stage_tokens_seen"]:
            continue
        if rec["lifetime_tokens"] is not None and rec["lifetime_tokens"] != EXPECTED_STAGE["lifetime_tokens_seen"]:
            continue
        eligible.append(rec)

    # Prefer the canonical resumable latest.pt from the CPT-v2 run. This is not a
    # guess: final_model.pt may intentionally duplicate weights while omitting
    # optimizer/RNG state, whereas latest.pt is the full stage-boundary checkpoint.
    ranked = sorted(
        eligible,
        key=lambda r: (
            1 if r["is_latest"] else 0,
            1 if r["looks_cpt_v2"] else 0,
            1 if r["step"] == EXPECTED_STAGE["stage_steps"] else 0,
            1 if r["stage_tokens"] == EXPECTED_STAGE["stage_tokens_seen"] else 0,
            1 if r["lifetime_tokens"] == EXPECTED_STAGE["lifetime_tokens_seen"] else 0,
        ),
        reverse=True,
    )

    if ranked:
        best_key = (
            ranked[0]["is_latest"],
            ranked[0]["looks_cpt_v2"],
            ranked[0]["step"] == EXPECTED_STAGE["stage_steps"],
            ranked[0]["stage_tokens"] == EXPECTED_STAGE["stage_tokens_seen"],
            ranked[0]["lifetime_tokens"] == EXPECTED_STAGE["lifetime_tokens_seen"],
        )
        tied = [
            r for r in ranked
            if (
                r["is_latest"],
                r["looks_cpt_v2"],
                r["step"] == EXPECTED_STAGE["stage_steps"],
                r["stage_tokens"] == EXPECTED_STAGE["stage_tokens_seen"],
                r["lifetime_tokens"] == EXPECTED_STAGE["lifetime_tokens_seen"],
            ) == best_key
        ]
        if len(tied) == 1:
            return tied[0]["path"]

    detail = "\n".join(
        "  " + str(r.get("path")) +
        (f" model_sha={r.get('model_sha')} optimizer={r.get('has_optimizer')} "
         f"step={r.get('step')} stage_tok={r.get('stage_tokens')} life_tok={r.get('lifetime_tokens')}"
         if "model_sha" in r else f" error={r.get('error')}")
        for r in inspected[-20:]
    )
    raise SystemExit(
        "STOP: could not uniquely identify the full completed CPT-v2 checkpoint. "
        f"model_matches={len(matches)} eligible_full_checkpoints={len(eligible)}\n{detail}"
    )


def discover_completed_marker(ckpt: Path) -> tuple[Path | None, dict | None]:
    names = ["COMPLETED.json", "completed.json", "RUN_COMPLETE.json", "run_complete.json"]
    for base in [ckpt.parent, ckpt.parent.parent]:
        for n in names:
            p = base / n
            if p.exists():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        return p, obj
                except Exception:
                    pass
    return None, None


def verify_completed_boundary(project: Path, ckpt: Path, train_bin: Path, ck: dict, eng) -> dict:
    state = ck.get("model")
    if not isinstance(state, dict):
        raise SystemExit("STOP: checkpoint has no model state")

    model_sha = engine_state_hash(eng, state)
    if model_sha != EXPECTED_STAGE["final_model_state_sha256"]:
        raise SystemExit(
            "STOP: checkpoint is not the completed CPT-v2 model. "
            f"model_sha={model_sha}"
        )

    stage_step = _first_int(ck, ["stage_step", "stage_steps", "step", "optimizer_step"])
    stage_tokens = _first_int(ck, ["stage_tokens_seen", "stage_tokens", "stage_scored_train_tokens"])
    lifetime_tokens = _first_int(ck, ["lifetime_tokens_seen", "lifetime_tokens", "cumulative_tokens_seen"])

    if stage_step is not None and stage_step != EXPECTED_STAGE["stage_steps"]:
        raise SystemExit(f"STOP: checkpoint step mismatch: {stage_step}")
    if stage_tokens is not None and stage_tokens != EXPECTED_STAGE["stage_tokens_seen"]:
        raise SystemExit(f"STOP: checkpoint stage-token mismatch: {stage_tokens}")
    if lifetime_tokens is not None and lifetime_tokens != EXPECTED_STAGE["lifetime_tokens_seen"]:
        raise SystemExit(f"STOP: checkpoint lifetime-token mismatch: {lifetime_tokens}")

    completed_path, completed = discover_completed_marker(ckpt)
    if completed is not None:
        if completed.get("status") not in (None, "PASS"):
            raise SystemExit(f"STOP: completion marker status is not PASS: {completed.get('status')}")
        c_model_sha = completed.get("final_model_state_sha256") or completed.get("model_state_sha256") or completed.get("final_model_sha")
        if c_model_sha is not None and c_model_sha != EXPECTED_STAGE["final_model_state_sha256"]:
            raise SystemExit("STOP: completion marker model SHA does not match the completed CPT-v2 model")

        c_step = _first_int(completed, ["stage_steps", "stage_step", "optimizer_steps", "optimizer_step", "steps"])
        c_stage_tokens = _first_int(completed, ["stage_tokens_seen", "stage_tokens", "stage_scored_train_tokens"])
        c_life = _first_int(completed, ["lifetime_tokens_seen", "lifetime_tokens", "cumulative_tokens_seen"])
        if c_step is not None and c_step != EXPECTED_STAGE["stage_steps"]:
            raise SystemExit(f"STOP: completion marker step mismatch: {c_step}")
        if c_stage_tokens is not None and c_stage_tokens != EXPECTED_STAGE["stage_tokens_seen"]:
            raise SystemExit(f"STOP: completion marker stage-token mismatch: {c_stage_tokens}")
        if c_life is not None and c_life != EXPECTED_STAGE["lifetime_tokens_seen"]:
            raise SystemExit(f"STOP: completion marker lifetime-token mismatch: {c_life}")

        for key in ("test_split_used", "test_used"):
            if key in completed and completed[key] not in (False, 0, "false", "UNTOUCHED"):
                raise SystemExit(f"STOP: completion marker says test split was used: {completed[key]}")

    got_ck_sha = sha256_file(ckpt)
    got_train_sha = sha256_file(train_bin)

    # Contract linkage is checked when the real checkpoint exposes the field.
    contract = ck.get("contract") or ck.get("run_contract")
    if isinstance(contract, dict):
        for key in ("dataset_v2_train_sha256", "train_bin_sha256"):
            if key in contract and contract[key] != got_train_sha:
                raise SystemExit(f"STOP: Dataset v2 train.bin hash differs from checkpoint contract ({key})")
        for key in ("test_split_used", "test_used"):
            if key in contract and contract[key] not in (False, 0, "false", "UNTOUCHED"):
                raise SystemExit("STOP: checkpoint contract says test split was used")

    return {
        "completed_path": str(completed_path) if completed_path else None,
        "completed_sha256": sha256_file(completed_path) if completed_path else None,
        "checkpoint_sha256": got_ck_sha,
        "model_state_sha256": model_sha,
        "train_bin_sha256": got_train_sha,
        "stage_steps": EXPECTED_STAGE["stage_steps"],
        "stage_tokens_seen": EXPECTED_STAGE["stage_tokens_seen"],
        "lifetime_tokens_seen": EXPECTED_STAGE["lifetime_tokens_seen"],
        "test_split_used": False,
        "checkpoint_metadata_stage_step": stage_step,
        "checkpoint_metadata_stage_tokens": stage_tokens,
        "checkpoint_metadata_lifetime_tokens": lifetime_tokens,
    }


def optimizer_semantics(model, ck: dict, origin: dict[str, str], gate_lr: float, eng) -> dict:
    opt_state = ck.get("optimizer")
    if not isinstance(opt_state, dict) or not isinstance(opt_state.get("param_groups"), list):
        raise SystemExit("STOP: final checkpoint optimizer state missing")
    groups = opt_state["param_groups"]
    if not groups:
        raise SystemExit("STOP: final checkpoint optimizer has no parameter groups")

    named_items = list(model.named_parameters())
    named = dict(named_items)
    if set(origin.values()) != set(named):
        raise SystemExit("STOP: normalized parameter mapping does not cover the exact trainable model")

    beta_eps = set()
    for g in groups:
        betas = tuple(float(x) for x in g.get("betas", (0.9, 0.999)))
        eps = float(g.get("eps", 1e-8))
        beta_eps.add((betas, eps))
    if len(beta_eps) != 1:
        raise SystemExit(f"STOP: per-group beta/eps differ: {beta_eps}")
    (betas, eps), = beta_eps

    source_lrs = sorted({float(g.get("lr", 0.0)) for g in groups})
    if len(source_lrs) != 1:
        raise SystemExit(
            "STOP: checkpoint optimizer groups have different learning rates; "
            f"cannot define one exact fresh gate LR mapping: {source_lrs}"
        )

    slot_wd = {}
    slot_group = {}
    grouping_proof = None
    source_weight_decay = None
    group_param_counts = []

    if len(groups) == 1:
        wd = float(groups[0].get("weight_decay", 0.0))
        got_ids = [int(x) for x in groups[0].get("params", [])]
        if got_ids != list(range(len(named_items))):
            raise SystemExit(
                "STOP: single optimizer-group parameter IDs do not match the "
                "complete model parameter order"
            )
        for slot in origin:
            slot_wd[slot] = wd
            slot_group[slot] = 0
        group_param_counts = [len(got_ids)]
        grouping_proof = "single_group_complete_model"
        source_weight_decay = wd
    else:
        # The completed script-17 engine exposes optimizer_for(model, lr, weight_decay).
        # Re-run THAT exact grouping function on this freshly loaded model instead of
        # guessing names from opaque optimizer state_dict IDs.  PyTorch assigns those
        # IDs in flattened optimizer-group order, not global model.named_parameters()
        # order, so a naive integer->model-index mapping would be wrong.
        factory = getattr(eng, "optimizer_for", None)
        if not callable(factory):
            raise SystemExit(
                "STOP: multiple optimizer groups require the frozen engine's "
                "optimizer_for() grouping function"
            )

        run_cfg = ck.get("run_config")
        if not isinstance(run_cfg, dict) or "weight_decay" not in run_cfg:
            raise SystemExit(
                "STOP: checkpoint run_config.weight_decay missing; cannot "
                "reconstruct the frozen optimizer grouping exactly"
            )
        source_weight_decay = float(run_cfg["weight_decay"])
        if not math.isfinite(source_weight_decay) or source_weight_decay < 0:
            raise SystemExit(
                f"STOP: invalid checkpoint run_config.weight_decay={source_weight_decay}"
            )

        source_lr = source_lrs[0]
        reconstructed = factory(model, source_lr, source_weight_decay)
        recon_groups = reconstructed.param_groups
        if len(recon_groups) != len(groups):
            raise SystemExit(
                "STOP: frozen engine optimizer_for() group count differs from "
                f"checkpoint: engine={len(recon_groups)} checkpoint={len(groups)}"
            )

        name_by_obj = {id(p): name for name, p in named_items}
        by_name = {}
        cursor = 0

        for gi, (rg, cg) in enumerate(zip(recon_groups, groups)):
            recon_names = []
            for p in rg.get("params", []):
                name = name_by_obj.get(id(p))
                if name is None:
                    raise SystemExit(
                        f"STOP: engine optimizer group {gi} contains a parameter "
                        "outside the exact model"
                    )
                recon_names.append(name)

            got_ids = [int(x) for x in cg.get("params", [])]
            expected_ids = list(range(cursor, cursor + len(recon_names)))
            if got_ids != expected_ids:
                raise SystemExit(
                    "STOP: checkpoint optimizer parameter-ID packing differs from "
                    f"the frozen engine grouping in group {gi}; "
                    f"expected={expected_ids[:8]}... got={got_ids[:8]}..."
                )
            cursor += len(recon_names)
            group_param_counts.append(len(recon_names))

            recon_betas = tuple(float(x) for x in rg.get("betas", (0.9, 0.999)))
            ck_betas = tuple(float(x) for x in cg.get("betas", (0.9, 0.999)))
            recon_eps = float(rg.get("eps", 1e-8))
            ck_eps = float(cg.get("eps", 1e-8))
            recon_wd = float(rg.get("weight_decay", 0.0))
            ck_wd = float(cg.get("weight_decay", 0.0))

            if recon_betas != ck_betas or recon_eps != ck_eps or recon_wd != ck_wd:
                raise SystemExit(
                    "STOP: frozen engine optimizer semantics differ from checkpoint "
                    f"in group {gi}: engine(betas={recon_betas},eps={recon_eps},wd={recon_wd}) "
                    f"checkpoint(betas={ck_betas},eps={ck_eps},wd={ck_wd})"
                )

            for key in ("amsgrad", "maximize", "decoupled_weight_decay"):
                if key in rg or key in cg:
                    if rg.get(key) != cg.get(key):
                        raise SystemExit(
                            f"STOP: optimizer semantic flag {key} differs in group {gi}: "
                            f"engine={rg.get(key)} checkpoint={cg.get(key)}"
                        )

            for name in recon_names:
                if name in by_name:
                    raise SystemExit(f"STOP: parameter appears in multiple optimizer groups: {name}")
                by_name[name] = (gi, ck_wd)

        if cursor != len(named_items):
            raise SystemExit(
                "STOP: frozen engine optimizer grouping does not consume every "
                f"trainable parameter: grouped={cursor} model={len(named_items)}"
            )
        if set(by_name) != set(named):
            raise SystemExit(
                "STOP: frozen engine optimizer grouping names do not exactly match "
                "the trainable model"
            )

        for slot, src in origin.items():
            gi, wd = by_name[src]
            slot_group[slot] = int(gi)
            slot_wd[slot] = float(wd)

        grouping_proof = "frozen_engine_optimizer_for+checkpoint_group_order"

    return {
        "gate_state": "fresh_zero_moments",
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(eps),
        "gate_lr": float(gate_lr),
        "source_optimizer_groups": len(groups),
        "source_checkpoint_lr_values": source_lrs,
        "source_weight_decay": float(source_weight_decay),
        "source_group_param_counts": group_param_counts,
        "grouping_proof": grouping_proof,
        "slot_group": slot_group,
        "slot_weight_decay": slot_wd,
        "slot_source_name": dict(origin),
    }


def make_fresh_reference_optimizer(model, hp: dict):
    # Exact support for the common single-group checkpoint. Multi-group requires
    # param_names and is reconstructed by name.
    if hp["source_optimizer_groups"] == 1:
        only_wd = next(iter(hp["slot_weight_decay"].values()))
        return torch.optim.AdamW(
            model.parameters(),
            lr=hp["gate_lr"],
            betas=(hp["beta1"], hp["beta2"]),
            eps=hp["eps"],
            weight_decay=only_wd,
        )

    named = dict(model.named_parameters())
    groups = {}
    for slot, src in hp["slot_source_name"].items():
        gi = hp["slot_group"][slot]
        groups.setdefault(gi, {"params": [], "weight_decay": hp["slot_weight_decay"][slot]})
        groups[gi]["params"].append(named[src])
    ordered = []
    for gi in sorted(groups):
        g = groups[gi]
        ordered.append({
            "params": g["params"],
            "weight_decay": g["weight_decay"],
            "lr": hp["gate_lr"],
            "betas": (hp["beta1"], hp["beta2"]),
            "eps": hp["eps"],
        })
    return torch.optim.AdamW(ordered)


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
    eng, engine_path = import_engine(project)
    ckpt = discover_checkpoint(project, args.checkpoint, eng)
    train_bin = Path(args.train_bin).resolve() if args.train_bin else project / "artifacts" / "model0001_dataset_v2" / "train.bin"
    out = Path(args.output).resolve() if args.output else project.parent / "model0001-gpu-gate.atb"

    if args.window_index < 0:
        raise SystemExit("STOP: --window-index must be >= 0")
    if not math.isfinite(args.gate_lr) or args.gate_lr <= 0:
        raise SystemExit("STOP: --gate-lr must be finite and > 0")
    if not ckpt.exists() or not train_bin.exists():
        raise SystemExit(f"STOP: missing checkpoint/train.bin: {ckpt} / {train_bin}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise SystemExit("STOP: checkpoint has no model state")

    boundary = verify_completed_boundary(project, ckpt, train_bin, ck, eng)
    contract = ck.get("contract") or ck.get("run_contract") or {}
    engine_cfg = None
    for candidate in [
        ck.get("model_config"),
        contract.get("model") if isinstance(contract, dict) else None,
        ck.get("run_config", {}).get("model") if isinstance(ck.get("run_config"), dict) else None,
        getattr(eng, "DEFAULT_MODEL", None),
    ]:
        if isinstance(candidate, dict):
            engine_cfg = dict(candidate)
            break
    if engine_cfg is None:
        raise SystemExit("STOP: cannot locate the exact Model #0001 config")
    cfg = normalized_config(engine_cfg)

    model = eng.Model0001(engine_cfg)
    model.load_state_dict(ck["model"], strict=True)
    assert_model_fp32(model, "PyTorch reference model")
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != EXPECTED["params"]:
        raise SystemExit(f"STOP: model params {nparams:,} != {EXPECTED['params']:,}")

    norm, origin = exact_state(model)
    rope_style = detect_rope_style(engine_path.read_text(encoding="utf-8", errors="replace"))
    hp = optimizer_semantics(model, ck, origin, args.gate_lr, eng)

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
    assert_model_fp32(ref_model, "fresh AdamW reference model")
    ref_model.train()
    ref_opt = make_fresh_reference_optimizer(ref_model, hp)
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
