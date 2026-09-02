# Model #0001 — Post-GPU transition order

This file mirrors the locked Training-project handoff.

1. CPT-v2 remains immutable and completed.
2. Dataset-v2 remains frozen/audited; do not run a second epoch automatically.
3. Physical native OpenCL gate is complete and accepted at commit
   `9201b188c13f3b5ac2f4dc790d3dcd7e1a45abc2`, 172.285 tok/s.
4. Run `tools/phone_audit_transition.sh` on the physical-phone project.
5. Use the audit plus the completed V2/V1 validation evidence to choose exactly
   one next semantic objective:
   - freeze Foundation v2 and design genuine SFT; or
   - build genuinely new Foundation/Dataset v3.
6. Lock the exact production recipe.
7. Implement/enable the recipe-driven long native GPU trainer.
8. Start the next semantic stage from the immutable CPT-v2 model with fresh
   AdamW moments and lifetime counter 5,535,744.
9. Keep the test split untouched until meaningful stage/model selection freeze.

No additional GPU benchmarking is part of this sequence.
