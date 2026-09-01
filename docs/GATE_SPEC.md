# Model #0001 GPU Gate v1

The gate measures whether a native mobile GPU backend is correct and materially
faster than the canonical PyTorch CPU training path on the physical phone.

## Frozen geometry

- 19,145,088 unique trainable parameters
- vocabulary 14,000
- 256 target tokens / 257 packed tokens per update
- hidden width 384; 8 blocks
- 6 query heads; 2 KV heads; head dimension 64
- SwiGLU width 1,152
- RMSNorm, RoPE, GQA
- tied embedding / LM head

The exporter reads RoPE theta, RMS epsilon, exact tensors, optimizer groups and
AdamW state from the real checkpoint. Ambiguous state mapping is a hard error.

## Backends

The APK contains MNN CPU FP32, MNN OpenCL FP32 IMAGE mode, and MNN Vulkan
BUFFER FP32. OpenCL is primary. Vulkan buffer is secondary.

## Evidence required

The report contains bundle integrity, PyTorch -> MNN CPU loss/logit parity,
raw-gradient/global-norm parity, one-step decoupled AdamW update probes, actual
per-op backend scheduling, sustained finite-loss throughput, CPU/GPU speed
ratio, Android thermal headroom, atomic checkpoint write, and checkpoint reload.

The current CPU training run remains canonical; backend migration happens only
between semantic stages.
