# Model #0001 — post-F2 behavior failure and F2R repair decision

Date: 2026-09-03
Status: F2 FROZEN / PREFERENCE BLOCKED / F2R DATA REPAIR SELECTED

## Behavioral evidence

Canonical post-F2 behavior evaluation passed numerical preflight, so the
generation failure is not explained by a Python/native reconstruction mismatch.

Observed automatic behavior score:

- Foundation-v3: 3 / 23
- F2 final: 4 / 23
- F2 persona: 0 / 2
- F2 relationship continuity: 0 / 1
- F2 memory routing: 0 / 4
- F2 scheduler: 0 / 3
- F2 basic Foundation retention checks: 0 / 2
- F2 instruction following: 0 / 1

F2 greedy generation repeatedly collapsed into narrow phrases such as
"Sama-sama", "di rumah", and "ngerasain" instead of following the prompt.

## Decision

Preference optimization is BLOCKED.

Do not resume frozen F2 and do not add an F2 second epoch.

The next path is a replacement supervised stage named:

`friend_f2r_repair_sft`

Its SOURCE is the promoted Foundation-v3 model, not the collapsed F2 final
checkpoint.

Reason: F2 final is retained as evidence and rollback material, but it is not a
healthy behavioral source checkpoint. Replacement SFT lets the repaired
objective be tested without compounding the F2 attractor state.

## Empirical audit first

The phone repair command first measures the old F2 target distribution and
continuous-packing behavior. Root-cause flags are data-derived; they are not
silently assumed.

Measured audit includes:

- exact assistant-target duplicate fraction;
- first-response prefix concentration;
- generic-acknowledgement concentration;
- assistant n-gram concentration;
- tool-call turn share;
- source-family target-token share;
- generation/training n-gram overlap;
- fraction of old windows spanning unrelated records.

## F2R source policy

Only original F2 TRAIN records are eligible for recycled human dialogue.
Original F2 validation remains holdout and is never moved into repair train.

Repair filtering:

- exact assistant targets deduplicated;
- very short generic acknowledgements filtered;
- high-frequency first-two-word response prefixes capped;
- human dialogue stays the majority;
- project-authored deterministic behavior core contributes 22–32% of scored
  assistant tokens;
- fixed behavior-eval user prompts are explicitly excluded from repair train;
- no OpenAI/LLM teacher corpus is added;
- project/external test split remains untouched.

## F2R packing policy

Each conversation is packed independently.

Unlike frozen F2, unrelated conversations cannot share one training window.
Cross-record windows are hard-locked to zero.

Assistant-content-only loss remains unchanged.

## Stop condition before training

No F2R optimizer step may run until these three phone reports are reviewed:

1. `model0001-f2-collapse-audit.json`
2. `F2R_SOURCE_REPORT.json`
3. `F2R_PACK_REPORT.json`

After those pass, define a new physical LR pilot from Foundation-v3. Do not
reuse the old F2 production LR automatically.
