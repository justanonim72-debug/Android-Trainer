# Model #0001 — Foundation-v3 production runner

Status: IMPLEMENTED / RECIPE VALUES PENDING PHYSICAL LR PILOT

The same Android build supports both:
1. the physical three-candidate Foundation-v3 LR pilot; and
2. the final recipe-driven Foundation-v3 training stage.

## Resume contract

Production training persists a native checkpoint in app-internal storage.
Checkpoint writes are atomic. If Android kills the process after a completed
checkpoint, reopen the SAME installed APK build and press Run / resume.

Do not update/uninstall the APK mid-stage: the native checkpoint deliberately
binds to the exact Android-Trainer commit.

The deterministic production cursor is the optimizer step itself because:
- Dataset-v3 train.bin is already deterministically shuffled;
- sample order is sequential packed windows;
- one semantic stage is capped at one Dataset-v3 pass.

Therefore resume requires no hidden RNG state.

## Progress

The native runner atomically writes:
`model0001-production-progress.json`

The app polls it and displays:
- optimizer step / total
- percent complete
- LR
- last train loss
- latest V3 validation CE
- latest V1 retention validation CE

## Evaluation

The stage package contains:
- Dataset-v3 train
- Dataset-v3 validation
- frozen Dataset-v1 validation

No test split and no Dataset-v2 train.bin are packaged.

## Final artifact

At completion, the app exposes the final native checkpoint for export.
The stage report records final V3 and V1 validation deltas against the immutable
CPT-v2 source baseline measured by the same native backend.
