# ADR — Model #0001 next semantic objective: Foundation / Dataset v3

Date: 2026-09-03
Status: OBJECTIVE SELECTED / DATASET RECIPE PENDING SOURCE INVENTORY

## Decision

After the completed CPT-v2 boundary, accepted physical GPU backend, and
transition audit, Model #0001 will do one genuinely new Foundation/Dataset-v3
continued-pretraining stage before SFT.

This is NOT a second Dataset-v2 epoch and MUST NOT reuse the frozen Dataset-v2
train.bin as production input.

## Evidence

Completed CPT-v2:
- stage updates: 15,624
- stage target tokens: 3,999,744
- lifetime tokens: 5,535,744
- Dataset-v2 validation CE: 6.9012 -> 5.1321
- Dataset-v1 validation CE: 6.2810 -> 6.6379
- Dataset-v2 CE was still improving late in the run (5.1702 at step 14,000
  -> 5.1321 final)

Interpretation:
- the new domain was still learning at the stage boundary;
- the older validation domain regressed, so retention/breadth needs attention;
- therefore freezing the foundation immediately for SFT is premature;
- another identical Dataset-v2 epoch would risk deeper data reuse/domain shift.

## Locked source boundary

- source model-state SHA256:
  `047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d`
- source checkpoint SHA256:
  `cbc6dec84e51d2a19e50ea38607e64cef78e62c627e0aa44f3dbe838d100ddf9`
- architecture/tokenizer frozen
- migration optimizer starts fresh zero moments
- AdamW betas [0.9, 0.95], eps 1e-8, grad clip 1.0
- lifetime token counter starts at 5,535,744
- test split remains untouched

## GPU production backend

Accepted physical build:
- commit `9201b188c13f3b5ac2f4dc790d3dcd7e1a45abc2`
- Mali-G610 MC6
- 172.285 target tok/s
- full numerical/checkpoint gate PASS

No more GPU microbenchmarking is required before the Dataset-v3 stage.

## Friend-Core objective guardrail

Dataset-v3 is still an F1/Foundation continued-pretraining stage. It exists to
move Model #0001 toward the locked Friend Core target, NOT toward encyclopedic
memorization.

The Training Blueprint explicitly defines Friend Core as:
- personal companion / natural chat first;
- Indonesian + Indonesian-English code-switch;
- short natural replies, slang/abbreviations, relationship continuity;
- basic reasoning/general knowledge sufficient for conversation;
- current/fresh facts delegated to search/tools;
- NOT a tiny encyclopedia that wastes limited capacity on broad knowledge.

The Blueprint's Friend data-mixture starting experiment is:
- natural Indonesian chat/slang: 30-40%;
- formal/neutral Indonesian: 10-15%;
- English + Indo-English code-switch: 15-20%;
- multi-turn social/relationship dialogue: 10-15%;
- tool/memory/scheduler examples: 10-15%;
- reasoning/general utility/safety: 10-15%.

Those shares overlap by stage and MUST NOT be blindly copied into one CPT
epoch. In particular, structured tool/memory/scheduler supervision belongs
primarily to F2 SFT, not to this F1 CPT pack.

For Dataset-v3 CPT, the stage objective is therefore:
1. substantially increase natural/colloquial Indonesian coverage;
2. add real code-switch coverage (v2 effectively had almost none);
3. add dialogue-like/social text without turning CPT into instruction tuning;
4. preserve enough English/basic reasoning/neutral Indonesian for retention;
5. keep encyclopedic/formal factual text a supporting component, never the
   dominant identity of the pack.

## Dataset-v3 requirements

Before packing:
- inventory the local project sources and the exact v1/v2 builder history;
- identify genuinely NEW/expanded source data;
- exclude project docs, reports, tokenizer artifacts, benchmark files and test
  data from candidate training sources;
- preserve an explicit retention component without silently replaying the
  frozen Dataset-v2 train.bin;
- deduplicate new source content against prior v1/v2 source content where
  provenance permits;
- create immutable train/validation identities;
- preserve tokenizer v1 exactly;
- do not create/use a test split for this stage.

Exact Dataset-v3 CPT mixture percentages and total token budget are locked only
after NEW source acquisition/quality audit. The overall Friend blueprint ranges
above are behavior guardrails, not permission to mix every family into CPT.

## Next action

Run:
`tools/phone_audit_v3_sources.sh`

Then lock:
- Dataset-v3 source manifest and mixture
- token budget / updates
- LR schedule
- sample order seed
- checkpoint cadence
- evaluation cadence

Only then build the recipe-driven native GPU production package and start the
long Foundation-v3 stage.
