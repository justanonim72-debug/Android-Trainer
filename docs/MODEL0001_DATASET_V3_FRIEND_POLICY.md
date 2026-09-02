# Model #0001 — Dataset-v3 Friend-Core source policy

Status: LOCKED BEHAVIOR POLICY / NEW SOURCE SET PENDING

## Product identity

Model #0001 is Friend Core foundation research. It is not being trained to be
an encyclopedia, academic tutor, or broad-current-world knowledge store.

Priority order:
1. natural Indonesian conversation;
2. Indonesian slang / abbreviations / short-message rhythm;
3. Indonesian-English code-switch;
4. social/context continuity;
5. enough neutral Indonesian + English + basic reasoning for robust fallback;
6. current/specialized knowledge delegated to search/tools/retrieval later.

## What Dataset-v2 taught us

Dataset-v2 realized:
- general_id 53.1499%
- natural_id 18.9961%
- conversational_id 3.0646%
- English 24.1020%
- formal_id 0.6730%
- code_switch_id_en 0.0144%
- informal_id 0%

Dataset-v2 improved its own validation strongly, but v1 validation regressed.
Dataset-v3 must therefore NOT simply add more general/encyclopedic material.

## Eligibility rules for NEW Dataset-v3 source files

Eligible source material must:
- be genuinely new to v1/v2 or explicitly labeled retention-only;
- have source/provenance + license recorded;
- fit at least one stage-relevant family:
  natural_id, colloquial_id, dialogue_like_id, code_switch_id_en,
  neutral_id, english_retention, reasoning_utility_retention;
- pass privacy/PII and quality filters;
- be deduplicated against prior source text before packing.

Automatic exclusions:
- test split files;
- tokenizer files;
- reports/benchmarks;
- project README/config/notes;
- generated audit/manifest files;
- frozen Dataset-v1/v2 packed bins;
- raw_v1/raw_v2/corpus_v2 files as "new" sources;
- random private chats/customer logs;
- unlicensed copyrighted books/school materials;
- OpenAI/ChatGPT output as bulk teacher/distillation corpus.

Prior v1/v2 source material may appear only through an explicit retention
selection policy, never by silently replaying the old packed train.bin.

## CPT-v3 vs SFT boundary

CPT-v3:
- natural/colloquial/code-switch distribution shift;
- basic language/reasoning retention;
- plain text or dialogue-like text is fine.

F2 SFT later:
- assistant/user role supervision;
- persona behavior;
- memory decisions;
- tool/scheduler schemas;
- concise response preference.

Do not use CPT-v3 to prematurely bake all tool/persona behavior into the
foundation weights.
