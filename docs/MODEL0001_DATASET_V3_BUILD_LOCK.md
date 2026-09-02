# Model #0001 — Dataset-v3 build lock

Date: 2026-09-03
Status: PACKING POLICY LOCKED

Source candidate audit:
- records: 21,697
- unique new tokens: 856,838
- pool SHA256:
  `e3e4174f9fc4dadcb4751f33dd7f51b15ccc3bdd3dd8cb2c3622db347071ee55`

Measured new-source mix:
- dialogue_like_id: 543,338 tokens (~63.4%)
- colloquial_id: 253,822 tokens (~29.6%)
- code_switch_id_en: 25,307 tokens (~3.0%)
- spoken_narrative_id: 22,730 tokens (~2.7%)
- neutral_id: 11,641 tokens (~1.4%)

This is intentionally Friend-heavy, not encyclopedic.

## Locked packing policy

- frozen tokenizer SHA256:
  `3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7`
- vocab 14,000; BOS=1; EOS=2; UNK=3
- packing seed: 20260903
- no source oversampling
- deterministic family-stratified v3 validation holdout:
  approximately 3% of NEW tokens per family
- validation contains NEW v3 data only
- all remaining NEW records are used once in train
- explicit retention target: 15% of final train tokens
- retention source: `data/splits/train.jsonl` from frozen v1 source text
- retention is selected deterministically by document hash
- no Dataset-v1 or Dataset-v2 packed bin is replayed
- Dataset-v2 train.bin is never an input to the packer
- project/external test splits are never read
- document format: BOS + frozen-tokenizer(text) + EOS
- train documents are deterministically shuffled before concatenation
- validation documents are deterministically shuffled separately
- uint16 little-endian token bins
- context/target window remains 256

## Why 15% retention

CPT-v2 improved v2 validation substantially but v1 CE regressed by +0.3568.
A small explicit v1 source-text retention slice is therefore justified.

15% is deliberately subordinate to the new Friend-focused pool. It prevents a
second broad-data stage from dominating the new behavioral distribution while
still giving the old foundation signal a meaningful presence.

## What is NOT locked yet

Production LR/scheduler is still pilot-gated. Dataset-v3 must be packed and
audited first, then the accepted GPU backend runs the short LR pilot. Long
training starts only after that pilot locks the recipe.
