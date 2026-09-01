# Android-Trainer

Native Android GPU-training research harness for **Model #0001**.

## Locked target

- Device: Infinix X6871 / Dimensity 8200 / Mali-G610 MC6
- GPU API: OpenCL from a native Android app
- Runtime candidate: MNN 3.6.1 + MNNTrain
- Training precision gate: FP32 / high precision first
- Vulkan is not used for training
- CPU remains canonical until the exact-model GPU gate passes
- Backend migration only happens at a training-stage boundary
- Optimizer semantics must match PyTorch AdamW exactly
- The gate must expose CPU fallback, numerical parity, checkpoint integrity, thermal behavior, and sustained throughput

## Build/runtime split

GitHub Actions is only the reproducible Android build machine.

```
GitHub Actions
  -> Android SDK/NDK
  -> arm64-v8a APK

Infinix X6871
  -> Android native process
  -> MNNTrain
  -> OpenCL
  -> Mali-G610 MC6
  -> real Model #0001 forward/backward/update
```

No training dataset or private checkpoint is committed to this public repository. Runtime files are imported locally on the phone.
