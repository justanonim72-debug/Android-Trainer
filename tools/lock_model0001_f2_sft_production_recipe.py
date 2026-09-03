#!/usr/bin/env python3
"""Lock Model #0001 Friend-Core F2 SFT production recipe from the accepted #80 physical pilot.

The policy is intentionally conservative:
- choose the lowest candidate that captures most of the SFT gain while minimizing
  V3/V1 retention damage;
- run exactly one full masked-SFT epoch;
- keep fresh-zero AdamW;
- warm up using the same ~3.125% fraction as the 3/96-step pilot;
- cosine-decay to 1e-6 to reduce long-run forgetting risk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_PILOT_SHA="6415f8335e41f57e1c3d3dd7544e1d29d95a78a48def58bad1d434e6e822fe95"
EXPECTED_COMMIT="51761fcf7aa9dc7c589cabc855a1798366378716"
SOURCE_MODEL_SHA="10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b"
PACK_SCHEMA="model0001_f2_sft_pack_report_v1"
SEED=20260903

PEAK_LR=1e-5
MIN_LR=1e-6
CHECKPOINT_EVERY=500
EVAL_EVERY=500
LOG_EVERY=25
EVAL_WINDOWS=64

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def finite(x)->float:
    v=float(x)
    if not math.isfinite(v): raise SystemExit("STOP: nonfinite pilot metric")
    return v

def find_candidate(rows,lr):
    for row in rows:
        if abs(float(row["lr"])-lr)<=1e-12:
            return row
    raise SystemExit(f"STOP: missing pilot candidate {lr}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot-report",default="/storage/emulated/0/Download/model0001-f2-sft-lr-pilot-report.json")
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--output",default="/storage/emulated/0/Download/model0001-f2-sft-production-recipe.json")
    args=ap.parse_args()

    pilot_path=Path(args.pilot_report).resolve()
    project=Path(args.project).resolve()
    if not pilot_path.is_file():
        raise SystemExit(f"STOP: pilot report missing: {pilot_path}")
    got_pilot_sha=sha256(pilot_path)
    if got_pilot_sha!=EXPECTED_PILOT_SHA:
        raise SystemExit(
            f"STOP: accepted F2 pilot SHA mismatch: {got_pilot_sha} != {EXPECTED_PILOT_SHA}"
        )

    p=json.loads(pilot_path.read_text(encoding="utf-8"))
    if p.get("schema")!="model0001_f2_sft_lr_pilot_report_v1" or p.get("status")!="PASS" or p.get("pass") is not True:
        raise SystemExit("STOP: F2 physical pilot is not PASS")
    if p.get("backend")!="PURE_OPENCL_C_1_2_FP32_BUFFER":
        raise SystemExit("STOP: F2 pilot backend drift")
    if p.get("commit")!=EXPECTED_COMMIT:
        raise SystemExit("STOP: F2 pilot was not produced by accepted #80 build")
    if p.get("source_model_state_sha256")!=SOURCE_MODEL_SHA:
        raise SystemExit("STOP: F2 pilot source-model SHA mismatch")
    if p.get("objective")!="assistant_content_only_cross_entropy":
        raise SystemExit("STOP: F2 pilot objective drift")
    if p.get("fresh_zero_moments_each_candidate") is not True:
        raise SystemExit("STOP: F2 pilot candidates were not fresh-zero")
    if p.get("test_split_used") is not False:
        raise SystemExit("STOP: F2 pilot touched test split")
    if int(p.get("train_steps_per_candidate",-1))!=96 or int(p.get("warmup_steps",-1))!=3:
        raise SystemExit("STOP: F2 pilot protocol drift")

    rows=p.get("candidates")
    if not isinstance(rows,list) or len(rows)!=3 or any(r.get("pass") is not True for r in rows):
        raise SystemExit("STOP: F2 pilot candidate set invalid")
    low=find_candidate(rows,1e-5)
    mid=find_candidate(rows,2e-5)
    high=find_candidate(rows,5e-5)

    # Require all candidates to improve SFT, and require retention damage to
    # monotonically worsen with higher LR as observed in the accepted pilot.
    for row in (low,mid,high):
        if finite(row["sft_validation_delta"])>=0:
            raise SystemExit("STOP: a candidate did not improve SFT validation")
    if not (
        finite(low["v3_validation_delta"]) <
        finite(mid["v3_validation_delta"]) <
        finite(high["v3_validation_delta"])
    ):
        raise SystemExit("STOP: V3 retention tradeoff does not match accepted evidence")
    if not (
        finite(low["v1_validation_delta"]) <
        finite(mid["v1_validation_delta"]) <
        finite(high["v1_validation_delta"])
    ):
        raise SystemExit("STOP: V1 retention tradeoff does not match accepted evidence")

    best_gain=max(-finite(r["sft_validation_delta"]) for r in rows)
    low_gain=-finite(low["sft_validation_delta"])
    if low_gain/best_gain < 0.85:
        raise SystemExit("STOP: 1e-5 no longer captures enough of best SFT gain")
    if finite(low["v3_validation_delta"])>0.08 or finite(low["v1_validation_delta"])>0.03:
        raise SystemExit("STOP: 1e-5 retention damage exceeds conservative lock")

    pack_path=project/"artifacts"/"model0001_f2_sft"/"F2_SFT_PACK_REPORT.json"
    if not pack_path.is_file():
        raise SystemExit("STOP: audited F2 pack report missing")
    pack=json.loads(pack_path.read_text(encoding="utf-8"))
    if pack.get("schema")!=PACK_SCHEMA or pack.get("status")!="PASS":
        raise SystemExit("STOP: F2 pack report not PASS")
    if pack.get("objective")!="assistant_content_only_cross_entropy":
        raise SystemExit("STOP: F2 pack objective drift")
    guards=pack.get("hard_guards",{})
    if guards.get("assistant_only_loss") is not True or guards.get("test_split_used") is not False:
        raise SystemExit("STOP: F2 pack hard guard failed")
    protocol_fraction=float(pack.get("train_deterministic_protocol_scored_fraction",-1))
    if not (0.08<=protocol_fraction<=0.15):
        raise SystemExit("STOP: F2 protocol scored fraction outside locked range")

    train_windows=int(pack["train"]["windows"])
    if train_windows<=0:
        raise SystemExit("STOP: F2 train window count invalid")

    total_updates=train_windows       # exactly one full epoch
    max_epochs=1
    warmup_steps=max(3,int(round(total_updates*(3.0/96.0))))
    if warmup_steps>=total_updates:
        raise SystemExit("STOP: F2 warmup invalid")

    recipe={
      "schema":"model0001_f2_sft_stage_recipe_v1",
      "source_model_state_sha256":SOURCE_MODEL_SHA,
      "stage_name":"friend_f2_sft",
      "objective":"assistant_content_only_cross_entropy",
      "source_foundation_lifetime_tokens":6563072,
      "total_updates":total_updates,
      "max_epochs":max_epochs,
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
        "warmup_steps":warmup_steps
      },
      "sample_order":{
        "type":"sequential_masked_windows",
        "seed":SEED
      },
      "checkpoint_every":CHECKPOINT_EVERY,
      "eval_every":EVAL_EVERY,
      "log_every":LOG_EVERY,
      "eval_windows_per_set":EVAL_WINDOWS,
      "eval_seed":SEED,
      "test_split_used":False,
      "selection_evidence":{
        "pilot_report_sha256":got_pilot_sha,
        "accepted_backend_commit":EXPECTED_COMMIT,
        "baseline":p["baseline"],
        "candidate_1e_5":{
          "sft_validation_delta":finite(low["sft_validation_delta"]),
          "v3_validation_delta":finite(low["v3_validation_delta"]),
          "v1_validation_delta":finite(low["v1_validation_delta"])
        },
        "candidate_2e_5":{
          "sft_validation_delta":finite(mid["sft_validation_delta"]),
          "v3_validation_delta":finite(mid["v3_validation_delta"]),
          "v1_validation_delta":finite(mid["v1_validation_delta"])
        },
        "candidate_5e_5":{
          "sft_validation_delta":finite(high["sft_validation_delta"]),
          "v3_validation_delta":finite(high["v3_validation_delta"]),
          "v1_validation_delta":finite(high["v1_validation_delta"])
        },
        "low_lr_fraction_of_best_sft_gain":low_gain/best_gain,
        "decision":"1e-5 peak: captures >=85% of best SFT gain with the smallest V3/V1 retention damage; one epoch with cosine decay to 1e-6"
      }
    }

    out=Path(args.output).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(recipe,indent=2,sort_keys=True),encoding="utf-8")

    print(json.dumps({
      "status":"LOCKED",
      "schema":"model0001_f2_sft_production_recipe_lock_v1",
      "recipe":str(out),
      "recipe_sha256":sha256(out),
      "pilot_report_sha256":got_pilot_sha,
      "total_updates":total_updates,
      "max_epochs":max_epochs,
      "peak_lr":PEAK_LR,
      "min_lr":MIN_LR,
      "warmup_steps":warmup_steps,
      "checkpoint_every":CHECKPOINT_EVERY,
      "eval_every":EVAL_EVERY,
      "log_every":LOG_EVERY,
      "test_split_used":False
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
