# Model #0001 — F2 SFT completed / frozen

Date: 2026-09-03
Status: COMPLETE / FROZEN

## Final checkpoint identity

- file: `model0001-f2-sft-final.atnckpt`
- file SHA256:
  `ed6556dbe293e9bb78af82f5ce410e3f37ad8f529b5b1cd9b84b4883f078d9d6`
- normalized model-state SHA256:
  `d09d31d9759790c12ba62b4ae101c53807ffc2a95fe452c2706abe4a09ea0e11`
- native format: ATNCL01 v1
- tensors: 74
- parameters: 19,145,088
- optimizer step: 2,786
- geometry: seq 256 / vocab 14,000 / d_model 384 / layers 8 /
  heads 6 / kv-heads 2 / head_dim 64 / d_ff 1152
- training commit:
  `51761fcf7aa9dc7c589cabc855a1798366378716`
- parent Foundation-v3 checkpoint SHA256:
  `773d685b81a736de795e8b3d93cf1833dc01a1f6a7e0fd6edfd9edefd7a36a67`
- parent Foundation-v3 promoted model-state SHA256:
  `10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b`
- stored final LR: approximately 1e-6
- all parameter tensors and Adam moments: finite
- checkpoint has no trailing bytes

## Completed semantic stage

- stage: friend_f2_sft
- objective: assistant_content_only_cross_entropy
- updates: 2,786 / 2,786
- epochs: 1 / 1
- scored assistant tokens: 283,957
- optimizer init: fresh_zero_moments
- test split used: false

Validation evidence from the completed stage report:

- SFT validation CE:
  6.3353170385 -> 5.4073699413
  delta -0.9279470971
- V3 retention CE:
  5.8148588091 -> 5.9734331891
  delta +0.1585743800
- V1 retention CE:
  6.4440121039 -> 6.5769556486
  delta +0.1329435446

Interpretation:

F2 substantially improved the intended assistant-only supervised objective.
The V3/V1 increases are real retention regressions and must remain explicit in
the model record. They are not grounds to silently continue F2 or replay
Foundation data. The next action is behavioral evaluation before any preference
optimization.

## Freeze rules

Do not:
- resume F2 SFT;
- add a second F2 epoch under the same stage name;
- alter architecture/tokenizer;
- overwrite this final checkpoint;
- delete the Foundation-v3 parent checkpoint;
- touch the held-out test split before behavioral model selection is frozen;
- start preference training before behavioral evaluation evidence is reviewed.

The next semantic decision is BEHAVIOR EVALUATION -> preference-stage decision.
