# OC-DPA

On-chip dynamic precision allocation for ADC-shared analog in-memory (AIMC) MoE accelerators.

## Encoding the experts

Everything in this repository starts from one encoding, so it comes first. Every routed expert matrix is stored in
the **any-precision encoding** of DPA-MoE (Kim et al.): the matrix is cut, per output row, into groups of 128 inputs,
and each group is written as

```
W^(k) ≈ β + Σ_{p=1..k} α_p · S_p ,     S_p ∈ {−1, +1}^128 ,   k = 2 … 8
```

with 8 sign planes `S_p` (one bit per weight and plane), 8 descending scales `α_p` and an offset `β` per group. The
scales are shared by all `k`, so the first `k` planes with their scales are a valid `k`-bit model: the number of planes
read is a run-time knob. The two base planes carry most of the signal and are always read.

`ocdpa/encoder.py` is the encoder and needs nothing outside this repository (PyTorch and NumPy). Per matrix it
1. initialises the two base planes greedily (each plane takes the sign of the residual and its mean magnitude as scale),
2. refines them in 20 rounds: scales and offset by least squares (batched conjugate gradient), then every sign is
   re-assigned to the nearest of the 2^k representable values (binary search),
3. adds the planes 3 … 8 one at a time (20 rounds each), re-fitting scales and offset while the earlier planes keep
   their meaning, and re-assigns all signs by the nearest-level search after each added plane.

`scripts/01_encode_experts.py` runs it over every routed expert of a checkpoint and packs the signs to one bit each
(`cache/planes_<model>/L<layer>.npz`; about 6.9 GB for OLMoE-1B-7B and 16 GB for DeepSeek-MoE-16B). The encoded
planes are not part of this repository: encode them once from the released checkpoints. The scales are stored in fp16
and rounded to **8-bit logarithmic codes** when the planes are loaded (`ocdpa/planes.py`, `SCALE_BITS`).
Encoding of one OLMoE layer takes about a minute on a single GPU.

The numerical kernels of the encoder (greedy initialisation, least-squares refinement, nearest-level search,
conjugate gradient) are adapted from [AnyBCQ](https://github.com/naver-aics/anybcq) (NAVER Cloud Corp., Apache-2.0;
license and notice in `licenses/`, summary in `NOTICE`).

## What the code does

The code has two halves that share the plane counts chosen by the allocator:

* **quality**: a simulated analog array (8-bit ADC, IR-drop, cell variation) runs the experts of
  OLMoE-1B-7B and DeepSeek-MoE-16B plane by plane; KL divergence is measured on both and downstream accuracy on OLMoE;
* **hardware**: the same plane counts are priced on a 16-tier 3D stack with ADCs shared on a logic
  tier (NeuroSim for arrays, converters and the allocator; a closed-form NPU model; a lane-queue model
  of prefill): TTFT, area and energy.

Reading the first `k` planes is a run-time knob, and the allocator spends a per-token plane budget on the experts
the router weights most. See `ocdpa/__init__.py` for the module map.

## Setup

```bash
pip install -e .                       # numpy, torch, transformers, datasets, safetensors, matplotlib
git clone https://github.com/neurosim/3D_NeuroSim_V1.0.git third_party/3D_NeuroSim_V1.0   # CC BY-NC 4.0, not redistributed here
scripts/05_build_neurosim.sh           # copies neurosim_patch/ in and builds probe, probe3d, allocator, ...
```
Models and WikiText-2 are read from the local Hugging Face cache. Outputs go to `./cache`
(`OCDPA_CACHE` and `NEUROSIM_DIR` override the locations; see `ocdpa/paths.py`).
Run every command from the repository root.

## Pipeline

Run in this order; each step reads what the previous ones wrote to `cache/`.

| step | script | writes |
|---|---|---|
| 1 | `01_encode_experts.py --model M --out cache/planes_X` | planes and scales of every routed expert |
| 2 | `02_record_routing.py` twice: measurement (`--start 0 --tokens 8192`, tag `X_long`) and calibration (`--start 0 --tokens 2048`, its first 2048 tokens) | routing traces |
| 3 | `03_profile_residual.py` (`03b_...large.py` when the planes do not fit on the GPU) | residual table `R[layer, expert, k]` |
| 4 | `04_calibrate_irdrop.py` | `cache/irdrop_gain.npz`, the column gains of the IR-drop model |
| 5 | `05_build_neurosim.sh` | NeuroSim driver programs |
| 6 | `10_hardware.py`, `16_sharing_sweep.py`, `17_adc_sweep.py` | design comparison, TTFT ablation, budget and sharing sweeps (`cache/hardware.json`); memory and logic die, TTFT and throughput per area against the sharing, with and without DPA (`cache/sharing_sweep.json`); the same over both lane-count knobs, tiers per lane and columns per ADC, with the best baseline and the best OC-DPA (`cache/adc_sweep.json`) |
| 7 | `11_allocator_ppa.py`, `14_check_allocator.py` | allocator cycles, area, energy; agreement of the on-chip allocator with the dynamic program |
| 8 | `13_encode_native_gptq.py --model allenai/OLMoE-1B-7B-0125 --out cache/planes_olmoe_gptq` | GPTQ 4- and 8-bit copies of every routed expert (`..._gptq4`, `..._gptq8`); OLMoE only, as in the paper |
| 9 | `12_precision_alternatives.py` | area, TTFT and throughput per area of the fixed-precision alternatives (OLMoE) |
| 10 | `20_kl_sweep.py` | KL divergence under a noise scenario (`--noise`, `--budgets`, `--dense`, `--native-planes`) |
| 11 | `21_downstream_choice.py`, `22_downstream_generation.py` | HellaSwag / WinoGrande / BoolQ; GSM8K / HumanEval (22 uses the instruct checkpoint, encoded with step 1) |
| 12 | `30_make_figures.py` | figures 1, 4 and 5 |

Task setting used for the reported results: 1T1R base planes with 1% and crossbar planes with 2% cell
variation (`--noise "mixed,0.01,0.02,cal"`: IR-drop at R_on = 300 kΩ with a global gain trim), prefill budget
k-bar = 4 and decode at k = 8. The sigma sweep uses `"mixed,s,2s,cal"` with s in {0, 1, 2, 5, 10} %.

Quality numbers depend on the encoder run (the any-precision planes of step 1 and the residual table of step 3)
and on the noise seed, so a fresh run reproduces the trends and the hardware numbers exactly, but the KL values only
approximately.

## Where each table and figure comes from

| in the paper | produced by |
|---|---|
| Table I (hardware comparison: 3D-AIMC, best baseline, OC-DPA) | `10_hardware.py` (the sharing of the best baseline and of OC-DPA per model is `BEST_SHARE` in that script, chosen with `16_sharing_sweep.py`) |
| Table II, TTFT ablation rows (OLMoE, s = 2) | mm², TTFT and throughput per area: `10_hardware.py`; KL: `20_kl_sweep.py --noise "mixed,0.01,0.02,cal" --budgets 4` |
| Table II, static 4-bit and 4 + 8-bit rows | `12_precision_alternatives.py --share 2` (mm², TTFT, throughput per area); KL: `13_encode_native_gptq.py`, then `20_kl_sweep.py --planes cache/planes_olmoe_gptq8 --native-planes cache/planes_olmoe_gptq4 --native-only --home cpu --noise "mixed,0.01,0.02,cal"` |
| Table III (downstream accuracy) | `21_downstream_choice.py`, `22_downstream_generation.py` at prefill k-bar = 4, decode k = 8 |
| Fig. 1 (decode latency; TTFT, memory and logic die against the sharing) | `10_hardware.py`, `30_make_figures.py` |
| Sharing sweep with and without DPA (memory and logic die, TTFT, throughput per area) | `16_sharing_sweep.py`, `17_adc_sweep.py` |
| Fig. 4 (budget sweep, TTFT at the sharing of OC-DPA) | `20_kl_sweep.py --noise "mixed,0.01,0.02,cal" --budgets 2,3,4,5,6,7`, `10_hardware.py`, `30_make_figures.py` |
| Fig. 5 (sigma sweep, dense layers on AIMC) | `20_kl_sweep.py --noise "mixed,s,2s,cal;..."` with and without `--dense attn,router,head`, `30_make_figures.py` |
| Sec. II-A (ADC share of a subarray read) | `python -m ocdpa.hw.neurosim` |
| Sec. II-B, III-B (scales per plane, share of the base planes) | `15_plane_statistics.py` |
| Sec. III-B (column gain variation, IR-drop error; mixed placement vs all-crossbar) | `04_calibrate_irdrop.py`; `12_precision_alternatives.py` |
| Sec. III-C (allocator cycles, area, energy) | `11_allocator_ppa.py` |
| Sec. III-C (the on-chip allocator reproduces the dynamic program) | `14_check_allocator.py` |
| Sec. IV-B (energy shares, decode speed and energy against NPU+flash) | printed by `10_hardware.py` |
| Figs. 2 and 3 (architecture, allocator datapath) | drawn by hand |

## Lane model

The timing model reads the planes of an expert one after the other through the ADC lanes of its sharing group
(Eq. 1 of the paper), so the area model provisions lanes the same way: all planes of a tile share one lane set
through a plane-select mux, sized for the wider of the two kinds of subarray (1T1R and crossbar), so the logic die is the larger of their lane
areas (`hetero_area(..., lane_sets="tile")`, `ocdpa/hw/stack.py`). The memory die still carries the TSV landings of every plane
subarray. `lane_sets="kind"` gives each kind its own lanes (which could then run in parallel; the timing model does not
assume that) and `"plane"` gives every plane its own lanes.

The simulated array takes its inputs at full precision (the bit-serial input is counted in time and energy, but its
quantisation is not simulated), and the converter range follows the largest input of each call.

## Static (fixed-precision) copies

`13_encode_native_gptq.py` quantises every routed expert matrix on its own with GPTQ (asymmetric, group
size 128, 4 and 8 bits). The Hessians come from the full-precision model on 32,768 tokens of WikiText-2
*train* text, disjoint from the test text of the KL experiments, over the tokens routed to each expert.
A b-bit integer is exactly b sign planes, so a GPTQ copy is written in the same plane format and runs
through the same analog read (planes 1-2 in 1T1R, the rest in crossbars, same noise). The two-copy
design reads the 4-bit copy in prefill and the 8-bit copy in decode.

## Layout

```
ocdpa/        library (see ocdpa/__init__.py)
scripts/      numbered entry points, one per pipeline step
neurosim_patch/  our additions to 3D+NeuroSim (driver programs); the upstream tree is not included
```

## License

The encoding method (`ocdpa/encoder.py`) follows DPA-MoE; its numerical kernels are adapted from AnyBCQ
(Apache License 2.0, Copyright (c) 2025-present NAVER Cloud Corp., [github.com/naver-aics/anybcq](https://github.com/naver-aics/anybcq)),
whose license and notice are in `licenses/` and summarised in `NOTICE`. The rest of the code is released under the
MIT License (`LICENSE`). The driver programs in `neurosim_patch/` are built on 3D+NeuroSim V1.0
(CC BY-NC 4.0, [github.com/neurosim/3D_NeuroSim_V1.0](https://github.com/neurosim/3D_NeuroSim_V1.0)), which is not
redistributed here; building and using them is subject to NeuroSim's license, including its non-commercial condition.
