# Model #0001 GPU Gate v2

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

GPU candidates then run the same forward/backward/update graph. Candidate
failures are isolated: an unavailable/broken Vulkan path cannot erase valid
OpenCL evidence and vice versa.

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

## Backends

- MNN CPU FP32: mandatory correctness reference
- MNN OpenCL FP32 IMAGE: primary GPU candidate
- MNN Vulkan FP32 BUFFER: secondary candidate

MNN is pinned to version 3.6.1 commit
`d407447ed56c4121a11ccbd266dc184ca1ead0c2`.
Vulkan is compiled with `MNN_VULKAN_IMAGE=OFF`.

## Sustained benchmark

Each backend loads an independent copy of the same serialized training model.
A warm-up/compile update is excluded from timing, then 20 updates are timed.
Reported throughput is target tokens per second.

Each completed session is persisted atomically as an MNN model and re-opened
into a fresh Interpreter/Session. A candidate cannot become canonical if its
checkpoint cannot be reloaded.

## Backend scheduling evidence

MNN operation callbacks inspect actual tensor backends during execution.
Reports expose CPU/GPU/other backend hits, allowing hidden fallback to be
distinguished from a genuinely GPU-scheduled graph.

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
