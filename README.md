# OC-DPA

On-chip dynamic precision allocation for ADC-shared analog in-memory (AIMC) MoE accelerators.
The code has two halves that share the plane counts chosen by the allocator:

* **quality**: a simulated analog array (8-bit ADC, IR-drop, cell variation) runs the experts of
  OLMoE-1B-7B and DeepSeek-MoE-16B plane by plane; KL divergence is measured on both and downstream accuracy on OLMoE;
* **hardware**: the same plane counts are priced on a 16-tier 3D stack with ADCs shared on a logic
  tier (NeuroSim for arrays, converters and the allocator; a closed-form NPU model; a lane-queue model
  of prefill): TTFT, area and energy.

An expert matrix is stored as binary planes `S_p` with scales: `W^(k) ≈ β + Σ_{p≤k} α_p S_p`.
The scales are kept as **8-bit logarithmic codes** (`ocdpa/planes.py`, `SCALE_BITS`); reading the
first `k` planes is a run-time knob, and the allocator spends a per-token plane budget on the experts
the router weights most. See `ocdpa/__init__.py` for the module map.

## Setup

```bash
pip install -e .                       # numpy, torch, transformers, datasets, safetensors, matplotlib
git clone https://github.com/kimGreg/DPA-MoE.git ../DPA-MoE
export DPAMOE_DIR=$PWD/../DPA-MoE      # provides the any-precision encoder (no submodules needed)
git clone https://github.com/neurosim/3D_NeuroSim_V1.0.git third_party/3D_NeuroSim_V1.0   # CC BY-NC 4.0, not redistributed here
scripts/05_build_neurosim.sh           # copies neurosim_patch/ in and builds probe, probe3d, allocator, ...
```
Models and WikiText-2 are read from the local Hugging Face cache. Outputs go to `./cache`
(`OCDPA_CACHE`, `NEUROSIM_DIR`, `DPAMOE_DIR` override the locations; see `ocdpa/paths.py`).
Run every command from the repository root.

## Encoding the experts

The any-precision encoding is DPA-MoE's (Kim et al., [github.com/kimGreg/DPA-MoE](https://github.com/kimGreg/DPA-MoE)):
a SharedScale AnyBCQ fit of every expert matrix, per output row and group of 128 inputs, into 8 sign planes and
descending scales, `W^(k) ≈ β + Σ_{p≤k} α_p S_p`, whose first `k` planes are a valid `k`-bit model. We call its
optimizer (`src/chunked_encoder/optimizer.py`, `compute_codebook(W, base_precision=2, target_precision=8, group_size=128)`)
through `ocdpa/encoder.py`, which loads it by path so that none of DPA-MoE's other dependencies are needed;
`scripts/01_encode_experts.py` runs it over every routed expert and packs the signs to one bit each.
The encoded planes are not part of this repository: encode them once from the released checkpoints
(about 6.9 GB for OLMoE-1B-7B and 16 GB for DeepSeek-MoE-16B).

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

The timing model reads the planes of an expert one after the other through the ADC lanes of its sharing
group (Eq. 1 of the paper), so the area model provisions lanes the same way: the planes of one kind of subarray
(1T1R or crossbar) share one set of lanes per tile position through a plane-select mux
(`hetero_area(..., lane_sets="kind")`, `ocdpa/hw/stack.py`). `lane_sets="plane"` gives every plane its own lanes,
which fits hardware that reads planes in parallel but not this timing model.

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

The code is released under the MIT License (`LICENSE`). The driver programs in `neurosim_patch/` are built on
3D+NeuroSim V1.0 (CC BY-NC 4.0, [github.com/neurosim/3D_NeuroSim_V1.0](https://github.com/neurosim/3D_NeuroSim_V1.0)),
which is not redistributed here; building and using them is subject to NeuroSim's license, including its
non-commercial condition. The any-precision encoder comes from DPA-MoE and is used under that repository's license.
