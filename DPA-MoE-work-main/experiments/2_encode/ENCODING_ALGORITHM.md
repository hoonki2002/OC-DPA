# Encoding Algorithm

This document describes the current release encoding path in
`src/dpa_moe`.  The main target is `ss_anybcq`, which keeps the
runtime layout small by storing one shared scale table while preserving
the original AnyBCQ bitplane construction.  The `anybcq` path remains as
the concrete original AnyBCQ baseline for uniform-K evaluation.

The removed experimental paths are intentionally not part of this
document: freeze-previous-scale calibration, post-hoc alpha sorting, and
cross-GPU expert multiloading inside a single layer.

## Entry Points

The encoder is driven by:

```bash
.venv/bin/dpa-moe-encode <phase> \
    --config experiments/config/Qwen3-30B-A3B/base.yaml \
    --variant ss_anybcq
```

Important files:

| File | Role |
|------|------|
| `src/dpa_moe/cli.py` | Phase dispatch, variant registry, config lookup. |
| `src/dpa_moe/calibrate.py` | Layer-wise AnyBCQ calibration and per-layer safetensor writing. |
| `src/dpa_moe/packers.py` | Converts bitplanes, alpha, and beta into DPA-MoE byte streams. |
| `src/dpa_moe/assemble.py` | Builds the final hybrid GGUF from codebooks plus a fresh HF model. |
| `third_party/anybcq/anybcq/quantization/anybcq.py` | Patched upstream AnyBCQ layer loop. |

Each model has its own config, `experiments/config/<model>/base.yaml`,
whose `model.adapter` selects the architecture adapter:

| Config | Adapter | Layer count |
|--------|---------|-------------|
| `Qwen3-30B-A3B` | `qwen3_moe` | 48 |
| `deepseek-moe-16b-chat` | `deepseek_moe` | 28 (layer 0 dense) |

The adapter selects model-specific expert tensor names, HF unfusing
logic, and GGUF metadata.  The quantization algorithm is otherwise the
same across these MoE architectures.

## Phases

The CLI phases are:

| Phase | Input | Output |
|-------|-------|--------|
| `calibrate` | HF model, calibration text, encoder hyperparams | `layer_NN.safetensors` codebooks |
| `assemble` | Fresh HF model, per-layer codebooks | Final hybrid GGUF |
| `quantize` | Final GGUF | No-op; assembly already writes unmanaged weights inline |
| `finalize` | Final GGUF | Same GGUF with `dpa_moe.required_runtime=true` |
| `all` | Runs calibrate, assemble, finalize | Final runtime-required GGUF |

Calibration is the slow phase.  Assembly is deterministic packing and
GGUF writing.

## Tensor Model

Only managed MoE expert matrices are AnyBCQ-encoded.  For each decoder
layer, the adapter identifies expert projections such as:

| Kind | Meaning |
|------|---------|
| `gate` | Gating projection inside an expert MLP |
| `up` | Up projection inside an expert MLP |
| `down` | Down projection inside an expert MLP |

Each managed matrix `W` has shape:

```text
W: (out_features, in_features)
```

Input columns are grouped by `group_size = G`.  Let:

```text
n = out_features
m = in_features
n_groups_per_row = ceil(m / G)
d1 = n * n_groups_per_row
```

After padding the input dimension to a multiple of `G`, each expert
weight is represented per group as:

```text
W_hat[row_group, g] = beta[row_group]
                    + sum_{p=0}^{K-1} alpha[row_group, p] * B[row_group, g, p]
```

where:

```text
B     in {-1, +1}, shape (d1, G, K)
alpha is float,    shape (d1, K)
beta  is float,    shape (d1)
K     is the selected precision
```

The default config uses:

```text
base_precision   = 2
target_precision = 8
add_bits         = target_precision - base_precision = 6
group_size       = 128
```

So each layer runs one base reconstruction plus six add-one-bit
precision steps.

## Production Variant: `ss_anybcq`

`ss_anybcq` is the release path.  It stores:

```text
one beta table
one shared alpha table with target_precision columns
all target_precision sign bitplanes
```

At runtime, choosing precision `K` means reading the first `K` bitplane
chunks and using the first `K` alpha columns with the shared beta.

### High-Level Algorithm

For each decoder layer:

```text
1. Cache calibration activations for the current layer.
2. Create an independent reconstruction block on GPU.
3. Replace managed Linear modules in the block with BCQLinear modules
   initialized at base_precision.
4. Run block reconstruction at base_precision.
5. For each added bitplane:
   a. Extract one new sign bitplane from the residual.
   b. Run block reconstruction again with all active alpha planes
      trainable.
6. Convert BCQLinear modules to AnyBCQLinear.
7. Pack that layer's expert codebooks and write one safetensors file.
8. Continue to the next layer.
```

The layer output from the quantized block becomes the input cache for
the next layer in exact sequential mode.

### Base Precision Step

At `add_bit = 0`, upstream AnyBCQ initializes `base_precision`
bitplanes and learns alpha/beta using block reconstruction.

For one layer block `F` and its quantized copy `F_q`, the main loss is:

```text
L_block = mean_squared_error(F_q(x_q), F_fp(x_fp))
```

where:

```text
x_fp = full-precision cached input to the layer
x_q  = quantized-chain cached input to the layer
```

For base precision, trainable parameters are:

```text
alpha[0 .. base_precision-1]
```

`beta` is not trained inside the DPA-MoE preserve-joint override after
initial BCQ setup.

### Add-One-Bit Steps

For `add_bit > 0`, upstream AnyBCQ first calls its add-one-bit path.
That path extracts one new sign plane from the current residual using
alternating refinement.  The total active precision becomes:

```text
K_new = base_precision + add_bit
```

DPA-MoE then overrides `blockReconstruction` for `ss_anybcq`.
Unlike the failed freeze-previous-scale ablation, all active alpha
planes remain trainable:

```text
trainable alpha planes = alpha[0 .. K_new-1]
```

The signs `B` are fixed during block reconstruction; the optimizer only
updates active alpha parameters.

### Previous-K Preservation Loss

A shared-scale model can lose quality at lower K if the 8-bit scale
optimization ignores the intermediate K-bit reconstructions.  To keep
intermediate precisions aligned, the training loss adds a weight-space
preservation term for previous K values.

For each managed BCQLinear module and each previous precision
`k in [base_precision, K_new)`, reconstruct:

```text
W_k = beta + sum_{p=0}^{k-1} alpha[p] * B[p]
```

Then compare it to the original BF16 expert weight:

```text
L_preserve = mean_k_modules(mean_squared_error(W_k, W_fp))
```

The total loss is:

```text
L = L_block + lambda_preserve * L_preserve
lambda_preserve = 0.5
```

This is the key `ss_anybcq` scale calibration behavior: active scales
are trained jointly, but previous lower-K reconstructions are explicitly
kept close to the original weight.

## Original Baseline Variant: `anybcq`

`anybcq` keeps original AnyBCQ's per-precision snapshots:

```text
precision 2: alpha_2, beta_2, bitplanes up to 2
precision 3: alpha_3, beta_3, new bitplane 3
...
precision 8: alpha_8, beta_8, new bitplane 8
```

This produces a larger byte stream because every precision stores its
own alpha/beta table.  It is useful as a concrete original AnyBCQ
baseline, especially for uniform-K KLD evaluation.

The `anybcq` calibration path does not use the DPA-MoE
preserve-joint override.  It uses upstream block reconstruction and then
packs all per-precision alpha/beta snapshots.

## Calibration Data Flow

Calibration data is loaded from the config:

```yaml
encode:
  calibration:
    files:
      - cache/datasets/calibration_v3.txt
      - cache/datasets/calibration_v5.txt
      - cache/datasets/wikitext-2-raw-train.txt
```

The loader tokenizes the concatenated text and slices it into:

```text
num_samples sequences
seq_len tokens per sequence
batch_size sequences per outer calibration batch
```

The current config defaults are:

```text
num_samples = 512
seq_len     = 2048
batch_size  = 2
iters       = 640
```

`DPA_MOE_ITERS` can override `iters` at runtime for quick sweeps.

## Unmanaged Weights

Unmanaged linear weights are attention projections, embeddings,
lm_head, and other non-MoE tensors.  They are not AnyBCQ encoded.

During calibration, unmanaged `nn.Linear` weights are RTN-quantized to
Q8_0 before block reconstruction.  This makes the MoE expert
calibration see the same non-MoE quantization noise that will exist in
the final GGUF.

During assembly, unmanaged tensors are written into the GGUF inline.
The default unmanaged qtype for `ss_anybcq` is:

```text
Q8_0
```

## Per-Layer Safetensor Output

Calibration writes one file per decoder layer:

```text
codebooks_dir/layer_00.safetensors
codebooks_dir/layer_01.safetensors
...
```

Each file contains one byte tensor per managed expert projection:

```text
e{xid:03d}.{kind}.bytes
```

Examples:

```text
e000.gate.bytes
e000.up.bytes
e000.down.bytes
e127.gate.bytes
```

This per-layer streaming output is deliberate:

| Reason | Effect |
|--------|--------|
| Memory | Does not keep full-model codebooks in RAM. |
| Restartability | Completed layers stay on disk after a crash. |
| Progress visibility | Layer files are a direct progress indicator. |
| Parallel ranges | Disjoint workers can write disjoint layer files. |

## Byte Stream Layout

### `ss_anybcq` Layout

`build_ss_anybcq_stream` writes:

```text
HEADER
beta table
chunk 0: bitplane 0 signs, alpha column 0
chunk 1: bitplane 1 signs, alpha column 1
...
chunk P-1: bitplane P-1 signs, alpha column P-1
```

Properties:

```text
FLAG_ANY_PREC = 0
alpha dtype   = fp16
beta dtype    = fp16
sign packing  = MSB-first, 1 bit per sign
sign value    = bit 1 means +1, bit 0 means -1
```

The fixed metadata is:

```text
HEADER + beta
```

Each precision chunk adds:

```text
one packed sign plane + one alpha column
```

To reconstruct at precision `K`, runtime uses:

```text
beta + chunks[0:K]
```

### `anybcq` Layout

`build_anybcq_stream` writes:

```text
HEADER
chunk 0 for precision base:
    base sign planes
    alpha^(base)
    beta^(base)
chunk 1 for precision base+1:
    new sign plane
    alpha^(base+1)
    beta^(base+1)
...
```

Properties:

```text
FLAG_ANY_PREC = 1
alpha/beta are per precision snapshot
runtime uses the snapshot for the selected uniform K
```

## Exact Sequential Mode

Sequential calibration is launched by:

```bash
CUDA_VISIBLE_DEVICES=0 DPA_MOE_FP_CACHING=0 .venv/bin/dpa-moe-encode calibrate \
    --config experiments/config/Qwen3-30B-A3B/base.yaml --calibration-mode sequential
```

This preserves the original layer-sequential AnyBCQ contract:

```text
layer 0 calibrates from BF16 model input
layer 1 calibrates from quantized layer 0 output
layer 2 calibrates from quantized layer 1 output
...
```

Quality-wise, this is the strictest path.  Speed-wise, it only uses one
GPU because each layer depends on the previous quantized layer output.

## Layer-Range Parallel Mode

Fast calibration is launched by:

```bash
GPUS=0,1,2,3 bash experiments/2_encode/calibrate_parallel.sh
```

It starts one process per listed GPU.  If there are `L` layers and `N`
GPUs, worker `i` owns:

```text
start = floor(i * L / N)
end   = floor((i + 1) * L / N) - 1
```

Each worker sets:

```text
CUDA_VISIBLE_DEVICES = assigned physical GPU
DPA_MOE_LAYER_RANGE = start:end
DPA_MOE_FP_CACHING = 1
--calibration-mode layer_fp_parallel
```

This mode does not split experts across GPUs.  Each process keeps the
original per-layer AnyBCQ execution on its own GPU and only reconstructs
its assigned layers.

Because `DPA_MOE_FP_CACHING=1`, each calibrated layer reconstructs
against BF16 cached activations rather than the output of previously
quantized layers.  That makes ranges independent and fast, but it is an
AdaQuant-style approximation:

```text
sequential:        layer i input = quantized output of layer i-1
layer_fp_parallel: layer i input = BF16 output of layer i-1
```

Use this mode for throughput.  Use sequential mode when measuring the
exact layer-sequential contract.

## Important Runtime Knobs

| Variable | Default | Applies to | Meaning |
|----------|---------|------------|---------|
| `DPA_MOE_ITERS` | config `iters` | calibration | Override Adam block-reconstruction iterations. |
| `DPA_MOE_INNER_BS` / `INNER_BS` | launcher default 2 | calibration | Inner DataLoader batch size inside block reconstruction. |
| `DPA_MOE_QUANT_ROUNDS` | 20 | upstream AnyBCQ swap | BCQ initialization/add-one-bit refinement rounds. |
| `DPA_MOE_FP_CACHING` | 0 sequential, 1 parallel | calibration | Whether the next-layer cache uses BF16 outputs. |
| `DPA_MOE_LAYER_RANGE` | unset | calibration | Inclusive layer range for one worker. |
| `DPA_MOE_EXPERT_BATCH` | 8 | upstream swap | Batch same-shape expert initialization work. |

## Assembly

Assembly is pure conversion.  It does not re-run calibration.

```bash
.venv/bin/dpa-moe-encode assemble \
    --config experiments/config/Qwen3-30B-A3B/base.yaml \
    --variant ss_anybcq \
    --force
```

Assembly steps:

```text
1. Load a fresh HF model for metadata and unmanaged tensors.
2. Read per-layer safetensors through the variant-specific reader.
3. Write managed expert tensors as dpa_moe byte streams.
4. Write unmanaged tensors inline, usually Q8_0.
5. Emit dpa_moe metadata describing chunk layout and residual data.
6. Write the final GGUF path from the config.
```

`finalize` then flips:

```text
dpa_moe.required_runtime = true
```

This prevents accidental loading through an incompatible vanilla runtime.

## Correctness Invariants

The release encoder should preserve these invariants:

1. `ss_anybcq` has exactly one shared alpha/beta scale set in the final
   byte stream.
2. `ss_anybcq` does not do post-hoc alpha sorting or BST sign
   re-derivation after block reconstruction.
3. `ss_anybcq` jointly retrains all active alpha planes at each
   precision tier.
4. Lower-K behavior is protected by the previous-K preservation loss.
5. `anybcq` remains the original per-precision alpha/beta baseline.
6. Multi-GPU calibration is process-level layer-range parallelism, not
   expert-level model parallelism inside a layer.
7. Per-layer codebook files are independent outputs and may be assembled
   after all expected layer files exist.

## Practical Run Recipes

Fast multi-GPU encode for Qwen3:

```bash
GPUS=0,1,2,3 INNER_BS=2 \
    bash experiments/2_encode/calibrate_parallel.sh

.venv/bin/dpa-moe-encode assemble \
    --config experiments/config/Qwen3-30B-A3B/base.yaml \
    --variant ss_anybcq \
    --force

.venv/bin/dpa-moe-encode finalize \
    --config experiments/config/Qwen3-30B-A3B/base.yaml \
    --variant ss_anybcq
```

Small debug run:

```bash
CUDA_VISIBLE_DEVICES=0 DPA_MOE_ITERS=2 \
    .venv/bin/dpa-moe-encode calibrate \
    --config experiments/config/Qwen3-30B-A3B/base.yaml \
    --variant ss_anybcq \
    --limit-layers 1 \
    --force
```

Original AnyBCQ uniform-K baseline:

```bash
GPUS=0,1,2,3 DPA_MOE_VARIANT=anybcq \
    bash experiments/2_encode/calibrate_parallel.sh

.venv/bin/dpa-moe-encode assemble \
    --config experiments/config/Qwen3-30B-A3B/base.yaml \
    --variant anybcq \
    --force
```
