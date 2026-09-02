#!/usr/bin/env python3
"""
Build a private Model #0001 native-stage package after the transition audit
and an explicit stage recipe are both available.

The recipe is intentionally external to the public repo so decisions recovered
from the Training-project session can be supplied without inventing defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

SOURCE_MODEL_SHA = "047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
START_LIFETIME_TOKENS = 5_535_744
SEQ = 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(obj, key, typ):
    if key not in obj or not isinstance(obj[key], typ):
        raise SystemExit(f"STOP: recipe missing/invalid {key}")
    return obj[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--output", default="/storage/emulated/0/Download/model0001-native-stage.atstage")
    args = ap.parse_args()

    audit_path = Path(args.audit).resolve()
    recipe_path = Path(args.recipe).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

    if audit.get("schema") != "model0001_native_transition_audit_v1":
        raise SystemExit("STOP: wrong transition-audit schema")
    known = audit["known"]
    source = known["source_weights"]
    data = known["dataset_v2"]
    if source["model_state_sha256"] != SOURCE_MODEL_SHA:
        raise SystemExit("STOP: transition audit is for the wrong model")
    train_bin = Path(data["train_bin"]).resolve()
    if not train_bin.is_file() or sha256(train_bin) != data["train_bin_sha256"]:
        raise SystemExit("STOP: Dataset-v2 train.bin changed after audit")

    if recipe.get("schema") != "model0001_native_stage_recipe_v1":
        raise SystemExit("STOP: recipe schema must be model0001_native_stage_recipe_v1")
    if recipe.get("source_model_state_sha256") != SOURCE_MODEL_SHA:
        raise SystemExit("STOP: recipe source model SHA mismatch")
    if recipe.get("train_bin_sha256") != data["train_bin_sha256"]:
        raise SystemExit("STOP: recipe Dataset-v2 SHA mismatch")
    if recipe.get("optimizer_init") != "fresh_zero_moments":
        raise SystemExit("STOP: migration recipe must use fresh_zero_moments")
    if recipe.get("test_split_used") is not False:
        raise SystemExit("STOP: test split must remain untouched")

    stage_name = need(recipe, "stage_name", str)
    total_updates = need(recipe, "total_updates", int)
    if not stage_name.strip() or total_updates <= 0:
        raise SystemExit("STOP: stage_name/total_updates invalid")
    if need(recipe, "target_tokens_per_update", int) != SEQ:
        raise SystemExit("STOP: target_tokens_per_update must be 256")
    if need(recipe, "start_lifetime_tokens", int) != START_LIFETIME_TOKENS:
        raise SystemExit("STOP: lifetime-token boundary mismatch")

    optimizer = need(recipe, "optimizer", dict)
    if optimizer.get("name") != "AdamW":
        raise SystemExit("STOP: optimizer must be AdamW")
    betas = need(optimizer, "betas", list)
    if len(betas) != 2 or [float(x) for x in betas] != [0.9, 0.95]:
        raise SystemExit("STOP: AdamW betas must remain [0.9,0.95]")
    if float(optimizer.get("eps", 0)) != 1e-8:
        raise SystemExit("STOP: AdamW eps must remain 1e-8")
    if float(optimizer.get("grad_clip", 0)) != 1.0:
        raise SystemExit("STOP: grad_clip must remain 1.0")

    schedule = need(recipe, "lr_schedule", dict)
    if schedule.get("type") not in ("constant", "linear_warmup_cosine"):
        raise SystemExit("STOP: unsupported lr_schedule.type")
    for key in ("checkpoint_every", "log_every"):
        if need(recipe, key, int) <= 0:
            raise SystemExit(f"STOP: {key} must be > 0")
    if need(recipe, "eval_every", int) < 0:
        raise SystemExit("STOP: eval_every must be >= 0")

    manifest = {
        "schema": "model0001_native_stage_package_v1",
        "stage_name": stage_name,
        "source_model_state_sha256": SOURCE_MODEL_SHA,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "train_bin_sha256": data["train_bin_sha256"],
        "train_uint16_tokens": data["uint16_tokens"],
        "full_256_target_windows": data["full_256_target_windows"],
        "recipe": recipe,
        "audit_sha256": sha256(audit_path),
        "recipe_sha256": sha256(recipe_path),
    }

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr(
            "manifest.json",
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        )
        z.write(train_bin, "data/train.bin")
    print(json.dumps({
        "status": "PASS",
        "output": str(out),
        "sha256": sha256(out),
        "stage_name": stage_name,
        "total_updates": total_updates,
        "train_bin_sha256": data["train_bin_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
