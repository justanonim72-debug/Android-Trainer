# Android-Trainer

Native Android **exact-model GPU gate** for Friend-Core Model #0001.

This repository separates cross-compilation from private model execution:

```
GitHub Actions
  Android SDK 35 + NDK r27c + Gradle 8.9
  MNN 3.6.1 @ d407447ed56c4121a11ccbd266dc184ca1ead0c2
  arm64-v8a / FP32 / MNNTrain / OpenCL / Vulkan-buffer
                         │
                         ▼
                signed debug APK
                         │
                         ▼
                physical Android phone
                         │
       local completed CPT-v2 checkpoint
                         │
              strict local .atb exporter
                         │
                         ▼
      CPU parity → OpenCL/Vulkan training gate
```

The public repository does **not** contain the user's checkpoint, Dataset v2,
or exported gate bundle.

## Frozen Model #0001 boundary

The exporter is locked to the completed CPT-v2 stage:

- 19,145,088 trainable parameters
- tokenizer vocabulary 14,000
- context 256
- 8 decoder blocks
- d_model 384
- 6 query heads / 2 KV heads / head_dim 64
- SwiGLU width 1,152
- RMSNorm + RoPE + GQA
- tied token embedding / LM head
- CPT-v2 stage steps 15,624
- CPT-v2 stage tokens 3,999,744
- lifetime tokens 5,535,744
- final model-state SHA256
  `047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d`
- test split remains untouched

The exporter auto-discovers the local completed checkpoint by that exact
model-state SHA. An explicit `--checkpoint` can be supplied, but it must hash
to the same completed model.

## What the phone gate proves

The APK does not promote a backend because it merely starts successfully.

It performs:

1. local bundle integrity validation;
2. exact PyTorch-reference → MNN CPU FP32 forward/loss/logit parity;
3. raw gradient and global gradient-norm parity;
4. one-step **fresh-state** decoupled AdamW parity with checkpoint-derived
   per-parameter weight-decay semantics;
5. actual MNN backend scheduling/fallback profiling;
6. independent CPU, OpenCL and Vulkan-buffer training sessions;
7. sustained 20-update exact-model throughput;
8. finite-loss checks;
9. atomic MNN checkpoint persistence;
10. checkpoint reload verification;
11. start/end Android thermal-headroom sampling;
12. CPU/GPU target-token throughput ratios.

OpenCL is the primary GPU candidate. Vulkan buffer is a secondary empirical
candidate and failure of one GPU backend does not invalidate evidence from the
other.

A backend is reported as:

- **not useful** below 1.5× CPU throughput;
- **useful** at ≥1.5× when numerical and checkpoint gates pass;
- **canonical candidate** only at ≥2.0× with all correctness gates passing.

A real backend migration still occurs only at a semantic training-stage
boundary.

## Build

Pushes to `gpu-gate-v1` run the complete GitHub Actions build. CI verifies:

- pinned MNN commit;
- Android SDK/NDK versions;
- exporter Python syntax;
- no private `.pt`, `.atb`, `.bin`, or `.mnn` files in project source;
- arm64-v8a APK build;
- APK signature;
- Android 16-KiB zip alignment;
- AArch64 native libraries;
- packaged MNN, MNNTrain, OpenCL and Vulkan libraries;
- build manifest and SHA256 evidence.

## Export the completed local CPT-v2 bundle

Copy `tools/export_model0001_bundle.py` from this repository into the
Friend-Core project or invoke it from any local path. Run it from the Ubuntu
environment that contains the project's PyTorch installation:

```bash
/root/mobilellm-ref/.venv/bin/python /path/to/export_model0001_bundle.py \
  --project /storage/emulated/0/Download/friend_core_corpus_bootstrap_v1 \
  --output /storage/emulated/0/Download/model0001-gpu-gate.atb
```

By default the exporter uses:

`artifacts/model0001_dataset_v2/train.bin`

and finds the completed checkpoint by the frozen final model-state SHA.
Ambiguous tensor names, architecture drift, optimizer semantics that cannot be
proven, dataset-contract mismatches, non-finite reference values, or the wrong
checkpoint all stop export instead of falling back to guesses.

See `docs/GATE_SPEC.md` for the complete evidence contract.
