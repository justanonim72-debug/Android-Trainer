# ADR — Model #0001 post-CPT-v2 semantic-stage decision

Date: 2026-09-03
Status: DECISION PENDING TRANSITION AUDIT

## Why this document was corrected

An earlier revision prematurely labeled Friend Core F2 SFT as the selected
next objective. That was stricter than the project handoff actually supports.

The locked handoff says that after the GPU backend gate succeeds we must first
run the production transition audit, then choose the next semantic objective:

1. freeze Foundation v2 and design a genuine SFT stage; or
2. create a genuinely new Foundation/Dataset v3 if additional pretraining
   breadth/retention repair is required.

Another Dataset-v2 epoch is not an allowed substitute for either choice and
must never be renamed "SFT".

## Completed immutable source boundary

- CPT-v2: COMPLETE
- source stage updates: 15,624
- source stage target tokens: 3,999,744
- lifetime tokens at boundary: 5,535,744
- source model-state SHA256:
  `047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d`
- architecture/tokenizer: frozen
- Dataset-v2: frozen/audited
- test split: untouched
- CPT-v2 must not be resumed.

Validation evidence at the completed boundary:

- Dataset-v2 CE: 6.9012 -> 5.1321 (improved by 1.7691)
- Dataset-v1 CE: 6.2810 -> 6.6379 (regressed by 0.3568)

That combination is why the next semantic objective must be selected from
evidence instead of automatically repeating Dataset-v2 or automatically
jumping to SFT.

## GPU backend status

Physical-phone native OpenCL gate is now accepted for the next semantic stage.

Accepted build:

- commit: `9201b188c13f3b5ac2f4dc790d3dcd7e1a45abc2`
- backend: PURE_OPENCL_C_1_2_FP32_BUFFER
- device: Mali-G610 MC6
- sustained throughput: 172.285 target tok/s
- CPU production reference: 78.38313427620264 target tok/s
- ratio: about 2.198x CPU
- forward parity: PASS
- backward parity: PASS
- AdamW parity: PASS
- checkpoint clear/reload: PASS
- fresh-state sustained benchmark: PASS
- FP16: disabled
- fast math: disabled
- Vulkan: disabled
- CPU fallback: disabled

This authorizes GPU as the production backend candidate at the semantic-stage
boundary. It does NOT define the next dataset or training recipe.

## Mandatory next action

Run `tools/audit_model0001_transition.py` on the physical-phone project
using the completed CPT-v2 source.

The audit must verify/record:

- exact source checkpoint and model-state SHA;
- frozen Dataset-v2 train identity;
- completed stage/lifetime counters;
- optimizer contract evidence;
- scheduler/cadence evidence;
- migration policy.

The migration optimizer starts with fresh zero moments. Gate Adam state and the
gate LR 1e-4 are not production state or production LR.

## Decision after the audit

Only after the transition audit:

### Candidate A — freeze Foundation v2 -> genuine SFT

Choose this if the completed foundation is judged sufficiently broad/stable to
stop continued pretraining.

A genuine SFT stage requires its own dataset, train/validation split, objective
and recipe. Dataset-v2 replay is not SFT.

### Candidate B — genuine Foundation/Dataset v3

Choose this if evidence says more foundation breadth or retention repair is
needed before SFT.

Dataset v3 must be genuinely new/expanded data. It must not regenerate or
silently reuse the frozen Dataset-v2 pack as a second epoch.

## Production recipe fields that remain forbidden to invent

Before long training, explicitly lock:

- next stage name;
- total updates or epochs;
- production LR and schedule;
- sample order / seed / cursor policy;
- checkpoint cadence;
- evaluation cadence.

Optimizer invariants:

- AdamW
- betas [0.9, 0.95]
- eps 1e-8
- grad clip 1.0
- migration starts with fresh zero moments
- lifetime counter starts at 5,535,744

Allowed schedule implementations currently supported by the stage-package
builder are constant or linear_warmup_cosine.

## Hard exclusions

Do not:

- resume CPT-v2;
- alter the frozen architecture or tokenizer;
- regenerate frozen Dataset-v2 bins;
- call another Dataset-v2 epoch SFT;
- use gate checkpoint/Adam state as production source state;
- use gate LR 1e-4 as production LR;
- change backend in the middle of a semantic stage;
- re-open Vulkan/MNN-GPU paths;
- touch the test split before meaningful stage/model selection freeze;
- use OpenAI/ChatGPT output as bulk teacher/distillation corpus.
