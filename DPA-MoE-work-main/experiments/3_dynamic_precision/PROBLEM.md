# DPA-MoE Dynamic Precision Problem

This file records the allocator semantics used by the paper and by the runtime
implementation.

## Structural Issue

MoE routing chooses which experts are active for a token or batch, but a
uniform precision policy gives every active expert the same chunk count. This
is inefficient because active experts have different gate scores and different
reconstruction sensitivities. DPA-MoE keeps the pretrained router unchanged and
only changes how many prefix chunks are used inside the selected experts.

## Inputs

- Active experts for the current MoE dispatch.
- Live router gate scores for those active experts.
- A gate-free residual lookup table `R[layer, expert, k]` measured offline.
- A global average chunk budget `KBar`.

For batched prefill, the active expert set is the union of experts selected by
tokens in the batch, and gate energy is aggregated as the sum of squared gate
scores for each expert.

## Budget

The logical per-dispatch budget is:

```text
B = round(KBar * number_of_unique_active_experts)
```

Each selected expert receives one chunk count `K[e]` in the supported range.
The allocator must satisfy:

```text
sum_e K[e] = B
```

The runtime may load multiple physical tensors per logical expert chunk, for
example gate, up, and down FFN tensors, but the allocator budget is defined in
logical expert chunks.

## Objective

DPA-MoE minimizes the gate-weighted residual surrogate:

```text
minimize   sum_e w[e] * R_projected[layer, e, K[e]]
subject to sum_e K[e] = B
```

where `w[e]` is `gate[e]^2` for single-token decoding and the batch-aggregated
squared gate score for prefill. `R_projected` is the residual table after
monotone projection along `k`; the dynamic program is exact for this projected
surrogate objective.

Uniform-k is the controlled baseline: it uses the same encoded artifact and
execution path but assigns the same chunk count to every active expert.

## Invariants

- Residual profiling is gate-free; live gate scores are applied only at
  allocation time.
- DPA does not change router top-k expert identities.
- Dynamic and full-pinned/capture execution must use the same allocation
  semantics.
- Expert selection is the router's top-k; there is no score-threshold or expert-dropping policy.
