# Model #0001 — F2 SFT transition lock

Date: 2026-09-03
Status: F2 SELECTED / DATASET PREPARATION IN PROGRESS

## Completed source boundary

Foundation-v3 is complete and frozen:

- stage: friend_foundation_v3_cpt
- optimizer steps: 4,013 / 4,013
- stage target tokens: 1,027,328
- lifetime tokens: 6,563,072
- V3 validation CE: 7.5096210018 -> 5.8277540430
- V1 validation CE: 6.6378621933 -> 6.4440121039
- test split used: false
- final native checkpoint SHA256:
  `773d685b81a736de795e8b3d93cf1833dc01a1f6a7e0fd6edfd9edefd7a36a67`

The normalized parameter-only promoted source identity is expected to be:

`10836dbde12e6c1eb732c1b6695ed248af5754d038011058250e81593287d00b`

The promotion tool verifies this from the native checkpoint and does NOT carry
Foundation-v3 Adam moments into F2.

## Next semantic stage

F2 SFT is now selected.

The objective follows the locked Training Blueprint:

1. natural Indonesian conversation;
2. short/slang/code-switch conversational rhythm;
3. persona/config consistency without hard-coding one customer identity;
4. relationship/context continuity;
5. memory operations: remember/update/recall/forget only when appropriate;
6. tool routing, including explicit no-tool behavior;
7. search for fresh facts with minimal private query context;
8. scheduler intent parsing; Android handles actual time;
9. concise uncertainty/refusal behavior without turning Friend Core into an
   encyclopedia.

## Genuine-SFT objective

F2 MUST use assistant-only supervised loss.

Context-only tokens:
- system messages
- user messages
- tool-return messages
- role labels

Scored tokens:
- assistant message CONTENT
- terminal EOS after assistant-ended records

The native OpenCL CE path now supports a binary 256-target loss mask. The
legacy CPT/gate path supplies an all-one mask, preserving the old objective.

## Source policy

Human conversational style:
- mDIA human dialogue, explicitly reused under a new supervised objective;
- IndoToD IndoSMD TRAIN only, native-speaker annotated.

Hard exclusions:
- external IndoSMD dev/test;
- project test split;
- Foundation-v3 validation examples from F2 train;
- random private chats/customer logs;
- OpenAI/ChatGPT outputs as bulk teacher data;
- LLM-generated dialogue collections as default style source.

A small bounded deterministic protocol slice may be generated locally for
machine-verifiable tool/memory/scheduler serialization. It is protocol
supervision, not a natural-language teacher corpus.

Canonical tool-call surface for F2 v1:

`<tool_call>{"name":"...","args":{...}}</tool_call>`

Tool results use role `tool`; the subsequent visible assistant reply is scored
normally.

## Hyperparameter rule

The Blueprint full-SFT LR starting range is 1e-5 .. 5e-5. Exact F2 production
LR/schedule is NOT locked here. After the SFT pack is audited, run a physical
masked-SFT LR pilot from the promoted Foundation-v3 source with fresh-zero
AdamW moments.

No Foundation-v3 continuation is part of F2.
