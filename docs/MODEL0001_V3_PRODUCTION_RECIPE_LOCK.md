# Model #0001 — Foundation-v3 production recipe lock

Date: 2026-09-03
Status: LOCKED FROM PHYSICAL LR PILOT

Physical pilot evidence:

- 5e-5:
  - V3 delta: -0.8791344166
  - V1 delta: -0.1123262048
- 1e-4:
  - V3 delta: -0.9477328062
  - V1 delta: -0.0408748984
- 2e-4:
  - V3 delta: -0.9282646775
  - V1 delta: +0.2027033567

Decision:

- reject 2e-4 because frozen V1 validation regressed;
- use 1e-4 as the production PEAK because it produced the best V3 validation
  CE while V1 still improved;
- do not keep 1e-4 constant for the whole stage;
- use linear warmup + cosine decay to 2e-5 for the 4,013-update run.

Locked recipe:

- stage: friend_foundation_v3_cpt
- total updates: 4,013
- stage target tokens: 1,027,328
- start lifetime tokens: 5,535,744
- expected end lifetime tokens: 6,563,072
- AdamW: beta1 0.9, beta2 0.95, eps 1e-8, grad clip 1.0
- optimizer init: fresh zero moments
- peak LR: 1e-4
- minimum LR: 2e-5
- warmup: 128 updates
- schedule: linear warmup -> cosine decay
- sample order: sequential packed windows, seed 20260903
- checkpoint every: 500 updates
- eval every: 500 updates
- log every: 25 updates
- eval windows: 64 per V3/V1 set
- test split: untouched

Warmup 128 is approximately the same ~3.1% fraction used in the 3/96-step
pilot, scaled to the 4,013-update production stage.

No further LR pilot is required.
