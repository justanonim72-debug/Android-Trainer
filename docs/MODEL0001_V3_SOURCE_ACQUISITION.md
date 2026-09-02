# Model #0001 — Foundation/Dataset-v3 source acquisition plan

Date: 2026-09-03
Status: SOURCE ACQUISITION PHASE

## Target behavior

Model #0001 is Friend Core. Dataset-v3 must improve:
- natural Indonesian;
- colloquial/slang rhythm;
- Indonesian-English code-switching;
- dialogue-like/social continuity;
- enough neutral Indonesian/English/basic reasoning for retention.

It must NOT turn the 19M-parameter model into a tiny encyclopedia.

## Sources selected for local research acquisition

### A. mDIA raw dialogue — PRIMARY dialogue-like source

- repository: DoctorDream/mDIA
- pin: 684c6f93a0f8c6ca904e1b0ceeacfb95ea34647b
- artifact: datasets/raw.zip
- data type: real-life Reddit parent/response dialogue pairs
- family: dialogue_like_id
- license evidence: CC-BY-4.0 in the project paper / SEACrowd record
- use: Indonesian raw dialogue only; translated English fields are excluded
- test/eval files are excluded by the local processor

### B. EmotCMT — PRIMARY natural code-switch seed

- repository: ir-nlp-csui/emotcmt
- pin: d1ad01b073570b1aa23d41574b9c7f94b42854c2
- artifact: codeswitch_emotion.csv
- data type: 825 real Indonesian-English code-mixed tweets
- family: code_switch_id_en
- use: local research only
- license note: dataset README permits free use with citation and forbids
  redistributing a copied dataset without permission. The public Android-Trainer
  repository contains only the downloader/manifest, never the data.

### C. Indonesian Frog Storytelling — spoken/narrative texture

- repository: davidmoeljadi/corpus-frog-storytelling
- pin: ff35f69ea8b612627ac0bf2e654ef7039696550e
- family: spoken_narrative_id
- license: CC-BY-SA-4.0
- use: spoken transcripts only, paragraph-level units

### D. TALPCo Indonesian — small neutral-retention support

- repository: matbahasa/TALPCo
- pin: eb4746249830e2c0a8b192a464a74616da3e0453
- family: neutral_id
- license: CC-BY-4.0
- use: Indonesian text only; intentionally small because this is not the
  Friend-Core identity source

### E. Reddit Indonesia Sarcastic TRAIN only — colloquial UGC support

- dataset: w11wo/reddit_indonesia_sarcastic
- pin: 77e64b52405753abd887c813e4de219ff0abf6e1
- artifact: data/train.json only
- family: colloquial_id
- dataset-card license: Apache-2.0
- source: real Reddit comments, minHash deduplicated and PII-masked by dataset
  authors
- use: TRAIN split only; validation/test are never downloaded by our tool
- note: social-UGC terms/provenance stay recorded in the manifest for later
  deployment/license review

## Explicitly NOT acquired automatically

- OpenSubtitles / movie subtitles: conversational but licensing of underlying
  subtitle text is not sufficiently clean for automatic production inclusion.
- Kamus Alay: useful lexicon but source license is reported as unknown.
- LLM-generated dialogue datasets: excluded from this F1 CPT source stage.
- SEADialogues / synthetic chat collections: excluded where model-generated
  provenance is material.
- project test split, external benchmark test/validation splits: excluded.
- Dataset-v2 train.bin: forbidden replay.

## Preparation rules

The local preparation tool:
1. downloads only pinned artifacts above;
2. extracts Indonesian raw text, never translated variants;
3. strips URLs/emails/obvious user handles and control characters;
4. never reads project test files;
5. exact-deduplicates against prior v1/v2 TRAIN source text;
6. exact-deduplicates inside the new pool;
7. counts tokens using the frozen tokenizer v1 SHA256
   3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7;
8. writes a source-family audit before any mixture/packing is allowed.

No source-family percentage or Dataset-v3 token budget is locked until this
audit reports the actual usable token volume.
