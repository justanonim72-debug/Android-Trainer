# Model #0001 Pure Native OpenCL Gate

## Purpose

Determine, using measurements from the physical Android device, whether native
GPU training of the completed Friend-Core Model #0001 is both numerically
interchangeable with the canonical FP32 path and materially faster.

This is a backend migration gate, not a model-quality evaluation.

## Frozen completed source

- parameters: 19,145,088
- vocabulary: 14,000
- context: 256 target tokens, 257 packed tokens per update
- hidden width: 384
- blocks: 8
- query heads: 6
- KV heads: 2
- head dimension: 64
- SwiGLU width: 1,152
- normalization: RMSNorm
- positional encoding: RoPE
- attention: GQA
- embedding / LM head: tied
- CPT-v2 steps: 15,624
- CPT-v2 stage tokens: 3,999,744
- lifetime tokens: 5,535,744
- final model-state SHA256:
  `047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d`
- test split: untouched

## Bundle integrity

The local exporter validates the exact final model hash and, when present in
the real checkpoint/completion metadata, validates stage counters, dataset
hash linkage, completion status and test-split flags. The bundle stores hashes
for the checkpoint, Dataset-v2 train pack, engine, sample payload and every FP32
tensor.

No private model or dataset data is uploaded to GitHub.

## Numerical gate

The exported PyTorch FP32 reference contains:

- cross-entropy loss;
- deterministic logit probes;
- raw gradient probes;
- raw global gradient norm;
- clipping coefficient for max norm 1.0;
- one-step AdamW parameter probes.

The Android implementation must first pass MNN CPU parity. If MNN CPU parity
fails, all GPU evidence is rejected.

The GPU candidate is a pure native OpenCL C 1.2 implementation using only FP32
buffers. MNN is never scheduled on GPU and remains strictly the CPU oracle.

## Optimizer semantics

The migration benchmark starts with fresh Adam moments because it evaluates a
new semantic-stage boundary. It does **not** pretend to continue the previous
optimizer state.

Beta values, epsilon and weight-decay grouping are extracted from the completed
checkpoint. If exact parameter-group membership cannot be proven, bundle export
stops.

The update is decoupled AdamW:

`p <- p * (1 - lr * wd) - lr * Adam(clipped_gradient)`

with global max-norm clipping at 1.0.

## Backends and constraints

- MNN CPU FP32: mandatory oracle and CPU throughput baseline
- pure native OpenCL C 1.2 FP32 BUFFER: the only GPU training backend
- MNN GPU, Vulkan, OpenCL images, FP16, relaxed math, float atomics, and hidden
  CPU fallback: disabled

MNN is pinned to version 3.6.1 commit
`d407447ed56c4121a11ccbd266dc184ca1ead0c2` and built without its OpenCL or
Vulkan backends.

## Sustained benchmark

The native path performs one warm-up update, then times 20 full updates. A
`clFinish` occurs before the stop timestamp. Reported throughput is target
tokens per second and is compared with the MNN CPU-only baseline.

Before benchmarking, native parameters and Adam `m/v` are serialized in
`android_trainer_native_checkpoint_v1` form. Live buffers are cleared, the
checkpoint is reloaded, and compact probes must remain bit-identical.

## Thermal evidence

The app samples Android `PowerManager.getThermalHeadroom(0)` immediately
before and after the complete gate and records wall-clock duration. A missing
platform thermal value is recorded as unavailable rather than fabricated.

## Decision thresholds

Correctness gates are mandatory. Throughput is then classified:

- <1.5× CPU: not useful
- ≥1.5× CPU: useful
- ≥2.0× CPU: canonical candidate

The report chooses a GPU backend only when numerical parity, finite sustained
training, checkpoint reload and the ≥2.0× threshold all pass. Otherwise CPU
remains recommended.

## Non-goals

The gate does not touch the test split, does not measure downstream language
quality, does not enable FP16/mixed precision, and does not switch an already
running training stage in place.
