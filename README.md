# Android-Trainer

Native Android **exact-model GPU gate** for Friend-Core Model #0001.

This repository deliberately separates **build** from **training**:

```
GitHub Actions (x86_64)
        │
        ├─ Android SDK / NDK r27c
        ├─ MNN 3.6.1 pinned commit
        └─ arm64-v8a APK
                 │
                 ▼
Infinix X6871 (physical phone)
        │
        ├─ import local .atb gate bundle
        ├─ MNN CPU FP32 reference
        ├─ MNN OpenCL FP32 IMAGE candidate
        └─ MNN Vulkan FP32 BUFFER candidate
```

GitHub never receives the user's checkpoint or dataset. The gate bundle is
created locally after a CPU training stage and imported into the APK.

## Frozen safety rules

- Current CPU training remains canonical until it ends.
- Never switch backend in the middle of a semantic training stage.
- FP32 correctness before FP16/mixed precision.
- Built-in MNN ADAM is not treated as PyTorch AdamW; the gate verifies a
  decoupled AdamW implementation.
- A GPU backend must prove loss/gradient/update parity and sustained speed.
- CPU fallback is measured; a merely non-crashing GPU run is not enough.
- Dataset test split remains untouched.

## Build

GitHub Actions performs a clean arm64-v8a build from pinned MNN 3.6.1.
The workflow publishes the debug APK, SHA256SUMS, and a build manifest.

## Export a local gate bundle

Run inside the actual Friend-Core project Ubuntu environment after a stage
checkpoint is frozen:

```bash
/root/mobilellm-ref/.venv/bin/python tools/export_model0001_bundle.py \
  --project /storage/emulated/0/Download/friend_core_corpus_bootstrap_v1 \
  --checkpoint /path/to/frozen/latest.pt \
  --train-bin /storage/emulated/0/Download/friend_core_corpus_bootstrap_v1/artifacts/model0001_dataset_v2/train.bin \
  --output /storage/emulated/0/Download/model0001-gpu-gate.atb
```

The exporter imports the real `scripts/17_pretrain_model0001.py`, verifies the
19,145,088-parameter geometry, refuses ambiguous tensor mapping, extracts exact
RoPE/RMS settings, and stores PyTorch FP32 reference evidence in the bundle.

See `docs/GATE_SPEC.md` for the acceptance contract.
