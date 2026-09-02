#!/usr/bin/env python3
"""Lock Model #0001 Foundation-v3 production recipe from the physical LR pilot.

This is fail-closed: it verifies the exact accepted backend/source model,
three-candidate pilot structure, and the measured safety trade-off before
writing the long-run recipe. It never reads test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCE_MODEL_SHA = "047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"
ACCEPTED_BACKEND_COMMIT = "660638e350f3190a9578e7de9a0c2c26fd8a6cf9"
V3_TRAIN_SHA = "19c7d23661aee08d11ac243347d4b943661084f1dc6fa740a222de01ae970975"
V3_VAL_SHA = "90dde5eeb7d7934c08213d73bd2d79abd69639c9d00bbfc2bc497137f59096c1"

TOTAL_UPDATES = 4013
TARGET_TOKENS_PER_UPDATE = 256
START_LIFETIME_TOKENS = 5_535_744

# Locked from the physical pilot:
PEAK_LR = 1.0e-4
MIN_LR = 2.0e-5
WARMUP_STEPS = 128
CHECKPOINT_EVERY = 500
EVAL_EVERY = 500
LOG_EVERY = 25
EVAL_WINDOWS_PER_SET = 64
SEED = 20260903

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20), b""):
            h.update(c)
    return h.hexdigest()

def close(a: float, b: float, tol: float=1e-12) -> bool:
    return math.isfinite(a) and abs(a-b) <= tol

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot-report", default="/storage/emulated/0/Download/model0001-v3-lr-pilot-report.json")
    ap.add_argument("--output", default="/storage/emulated/0/Download/model0001-v3-production-recipe.json")
    args=ap.parse_args()

    pilot_path=Path(args.pilot_report).resolve()
    if not pilot_path.is_file():
        raise SystemExit(f"STOP: physical LR pilot report missing: {pilot_path}")
    pilot=json.loads(pilot_path.read_text(encoding="utf-8"))

    if pilot.get("schema")!="model0001_v3_lr_pilot_report_v1" or pilot.get("status")!="PASS" or pilot.get("pass") is not True:
        raise SystemExit("STOP: physical LR pilot is not PASS")
    if pilot.get("backend")!="PURE_OPENCL_C_1_2_FP32_BUFFER":
        raise SystemExit("STOP: wrong pilot backend")
    if pilot.get("commit")!=ACCEPTED_BACKEND_COMMIT:
        raise SystemExit("STOP: pilot was not produced by accepted #74 build")
    if pilot.get("source_model_state_sha256")!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: pilot source-model SHA mismatch")
    if pilot.get("fresh_zero_moments_each_candidate") is not True:
        raise SystemExit("STOP: pilot candidates were not fresh-zero")
    if pilot.get("test_split_used") is not False:
        raise SystemExit("STOP: pilot touched test split")
    if pilot.get("train_steps_per_candidate")!=96 or pilot.get("warmup_steps")!=3:
        raise SystemExit("STOP: pilot protocol drift")

    candidates=pilot.get("candidates")
    if not isinstance(candidates,list) or len(candidates)!=3:
        raise SystemExit("STOP: expected exactly three pilot candidates")
    by_lr={}
    for row in candidates:
        if row.get("pass") is not True:
            raise SystemExit("STOP: a pilot candidate failed")
        lr=float(row["lr"])
        by_lr[lr]=row
        for key in ("v3_validation_ce","v3_validation_delta","v1_validation_ce","v1_validation_delta","max_global_grad_norm"):
            if not math.isfinite(float(row[key])):
                raise SystemExit(f"STOP: nonfinite pilot metric {key}")

    for lr in (5e-5,1e-4,2e-4):
        if lr not in by_lr:
            raise SystemExit(f"STOP: missing LR candidate {lr}")

    low=by_lr[5e-5]
    mid=by_lr[1e-4]
    high=by_lr[2e-4]

    # Selection contract:
    # - 2e-4 is rejected because it regresses frozen V1 validation.
    # - 1e-4 must beat 5e-5 on V3 validation while not regressing V1.
    if float(high["v1_validation_delta"]) <= 0:
        raise SystemExit("STOP: expected 2e-4 retention regression evidence missing")
    if float(mid["v1_validation_delta"]) > 0:
        raise SystemExit("STOP: 1e-4 regressed V1; cannot lock it")
    if float(mid["v3_validation_ce"]) >= float(low["v3_validation_ce"]):
        raise SystemExit("STOP: 1e-4 did not beat 5e-5 on V3 validation")

    recipe={
      "schema":"model0001_native_stage_recipe_v1",
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "train_bin_sha256":V3_TRAIN_SHA,
      "validation_bin_sha256":V3_VAL_SHA,
      "stage_name":"friend_foundation_v3_cpt",
      "total_updates":TOTAL_UPDATES,
      "target_tokens_per_update":TARGET_TOKENS_PER_UPDATE,
      "start_lifetime_tokens":START_LIFETIME_TOKENS,
      "optimizer_init":"fresh_zero_moments",
      "optimizer":{
        "name":"AdamW",
        "betas":[0.9,0.95],
        "eps":1e-8,
        "grad_clip":1.0
      },
      "lr_schedule":{
        "type":"linear_warmup_cosine",
        "peak_lr":PEAK_LR,
        "min_lr":MIN_LR,
        "warmup_steps":WARMUP_STEPS
      },
      "sample_order":{
        "type":"sequential_packed_windows",
        "seed":SEED
      },
      "checkpoint_every":CHECKPOINT_EVERY,
      "eval_every":EVAL_EVERY,
      "log_every":LOG_EVERY,
      "eval_windows_per_set":EVAL_WINDOWS_PER_SET,
      "eval_seed":SEED,
      "test_split_used":False,
      "selection_evidence":{
        "pilot_report_sha256":sha256(pilot_path),
        "accepted_backend_commit":ACCEPTED_BACKEND_COMMIT,
        "baseline_v3_validation_ce":float(pilot["baseline"]["v3_validation_ce"]),
        "baseline_v1_validation_ce":float(pilot["baseline"]["v1_validation_ce"]),
        "lr_5e_5":{
          "v3_delta":float(low["v3_validation_delta"]),
          "v1_delta":float(low["v1_validation_delta"])
        },
        "lr_1e_4":{
          "v3_delta":float(mid["v3_validation_delta"]),
          "v1_delta":float(mid["v1_validation_delta"])
        },
        "lr_2e_4":{
          "v3_delta":float(high["v3_validation_delta"]),
          "v1_delta":float(high["v1_validation_delta"])
        },
        "decision":"peak 1e-4: best V3 pilot CE without V1 regression; cosine decay reduces long-run forgetting risk"
      }
    }

    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(recipe,indent=2,sort_keys=True),encoding="utf-8")

    end_lifetime=START_LIFETIME_TOKENS + TOTAL_UPDATES*TARGET_TOKENS_PER_UPDATE
    result={
      "status":"LOCKED",
      "schema":"model0001_v3_production_recipe_lock_v1",
      "recipe":str(out),
      "recipe_sha256":sha256(out),
      "pilot_report_sha256":sha256(pilot_path),
      "peak_lr":PEAK_LR,
      "min_lr":MIN_LR,
      "warmup_steps":WARMUP_STEPS,
      "total_updates":TOTAL_UPDATES,
      "stage_target_tokens":TOTAL_UPDATES*TARGET_TOKENS_PER_UPDATE,
      "end_lifetime_tokens":end_lifetime,
      "checkpoint_every":CHECKPOINT_EVERY,
      "eval_every":EVAL_EVERY,
      "log_every":LOG_EVERY
    }
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
