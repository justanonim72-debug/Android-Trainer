# ADR — Model #0001 next semantic stage: Friend Core F2 SFT

Date: 2026-09-02
Status: SELECTED OBJECTIVE / DATASET+LR PILOT NOT YET LOCKED

## Source evidence

The completed CPT-v2 source is immutable at 15,624 steps, 3,999,744 stage
target tokens, 5,535,744 lifetime tokens, model-state SHA256
`047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d`.

The project handoff explicitly says another Dataset-v2 epoch is not
automatically the next stage. Dataset-v2 was already consumed for essentially
one epoch, V2 validation improved, and V1 validation regressed.

The locked Local AI Training Blueprint defines the Friend Core ladder:

- F1 CPT: distribution shift toward Indonesian/slang/code-switch;
- F2 SFT: natural chat, persona, memory/tool formats;
- F3 Preference;
- F4 Quant;
- F5 Device.

Therefore the next semantic objective is **F2 SFT**, not CPT-v2 resume/replay.

## What is locked

- source weights = exact completed CPT-v2 model above;
- tokenizer/model architecture unchanged;
- test split remains untouched;
- SFT source records require explicit provenance/license metadata;
- immutable SFT validation split must exist before a training pilot;
- production/gate benchmark state is not a training source;
- OpenAI/ChatGPT output is not accepted as bulk teacher/distillation corpus.

## What is deliberately NOT invented

The blueprint labels hyperparameter ranges as starting points only. It says
short pilot runs must identify a stable LR and exact sequence/batch settings.
Therefore the following remain pilot-gated:

- exact full-SFT vs adapter choice;
- exact SFT LR within the allowed experimental search range;
- warmup/scheduler;
- total updates;
- final data mixture;
- checkpoint/eval cadence.

The blueprint's initial full-SFT LR range is approximately 1e-5..5e-5 and
grad-clip ~1.0, but those are not a production recipe until measured on this
exact 19,145,088-parameter Model #0001 and the final SFT dataset.

## GPU acceptance status

Native OpenCL numerical correctness and sustained training are physically
proven on the Mali-G610. The current GPU gate reports about 97-101 target
tok/s. The in-app CPU denominator is null because the retired MNN static CPU
loop is invalid.

The backend switch threshold remains >=1.5x a defensible CPU baseline. The
historical CPT-v2 CPU log often showed ~108-115 tok/s earlier/mid-run and
~40-90 tok/s late-run, but no full-stage aggregate was retained. Therefore
GPU speed-worth is still formally unresolved.

Use `tools/phone_benchmark_cpu_reference.sh` exactly once to recover an
apples-to-apples 20-step CPU denominator before authorizing production GPU.

## Immediate implementation order

1. recover CPU reference denominator;
2. validate/lock the SFT source dataset and immutable SFT validation split;
3. run short LR/format pilot(s), not a long production stage;
4. lock exact F2 recipe;
5. only then implement/enable long production training.
