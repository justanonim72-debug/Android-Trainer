# Model #0001 — Friend-Core F2 SFT production recipe lock

Date: 2026-09-03
Status: LOCKED FROM PHYSICAL #80 ASSISTANT-ONLY LR PILOT

Accepted pilot report SHA256:
`6415f8335e41f57e1c3d3dd7544e1d29d95a78a48def58bad1d434e6e822fe95`

Accepted Android-Trainer commit:
`51761fcf7aa9dc7c589cabc855a1798366378716`

Pilot tradeoff:

- 1e-5: SFT -0.5147487, V3 +0.0562408, V1 +0.0170227
- 2e-5: SFT -0.5613290, V3 +0.0964970, V1 +0.0428207
- 5e-5: SFT -0.5804139, V3 +0.1888886, V1 +0.1088348

Decision:
- choose 1e-5 as production PEAK;
- 1e-5 captures ~88.7% of the best observed SFT validation gain while causing
  much less V3/V1 regression than 2e-5 or 5e-5;
- use exactly one full F2 masked-SFT epoch;
- use the same ~3.125% warmup fraction as the 3/96-step pilot;
- cosine-decay to 1e-6 to reduce long-run retention pressure.

Locked long-run policy:
- stage: friend_f2_sft
- objective: assistant_content_only_cross_entropy
- optimizer init: fresh_zero_moments
- AdamW: beta1 0.9, beta2 0.95, eps 1e-8, grad clip 1.0
- total updates: exactly current audited F2 train-window count (one epoch)
- max epochs: 1
- peak LR: 1e-5
- min LR: 1e-6
- warmup: round(total_updates * 3/96)
- checkpoint every: 500
- eval every: 500
- log every: 25
- eval windows per SFT/V3/V1 set: 64
- test split: untouched

No additional LR pilot is required.
