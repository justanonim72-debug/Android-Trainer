# Model #0001 — post-F2 behavioral evaluation contract

Status: NEXT REQUIRED STAGE / NO TRAINING

Behavior evaluation happens before preference optimization.

Required dimensions:

1. natural Indonesian conversation;
2. short/slang/code-switch conversational rhythm;
3. persona/config consistency without hard-coding one customer identity;
4. relationship/context continuity across turns;
5. memory STORE / UPDATE / LOOKUP / FORGET correctness;
6. explicit do-not-store behavior for transient chatter;
7. tool routing and explicit no-tool cases;
8. fresh-information search decisions with minimal private query context;
9. scheduler intent parsing and ambiguity clarification;
10. concise uncertainty / refusal behavior;
11. regression checks for basic general knowledge and Foundation retention.

Evaluation policy:

- no optimizer updates;
- no training-data replay;
- no preference pairs created until raw behavioral outputs are reviewed;
- use fixed prompts and deterministic sampling settings for the canonical pass;
- keep a separate exploratory sampling pass for naturalness;
- do not use the held-out project test split yet;
- compare at minimum Foundation-v3 source vs F2 final for the same prompts;
- record exact checkpoint/model-state SHA for every evaluated model.

Pass/fail is multidimensional. A lower SFT CE alone is not sufficient to start
preference optimization.
