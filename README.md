# OC-DPA

On-chip dynamic precision allocation for ADC-shared analog in-memory (AIMC) MoE accelerators.
Code behind the numbers of the paper *OC-DPA: On-Chip Dynamic Precision Allocation in ADC-Shared Analog In-Memory MoE Accelerators*.

```bash
pip install -e . && python scripts/00_verify.py     # ~30 s, CPU: one point per step of the paper
```

The scripts are numbered in the order of the data flow (checkpoint, encoding, residual table, LUT, routing, noise model, hardware, quality, figures).

## From zero

```bash
git clone https://github.com/hoonki2002/OC-DPA.git && cd OC-DPA
python -m venv env && source env/bin/activate      # Python >= 3.10
pip install -e .                                   # numpy, scipy, torch, transformers, datasets, safetensors, matplotlib
python scripts/00_verify.py                        # ~30 s, CPU: encoder, LUT, allocator, analog path, hardware numbers
```

**What each part needs.** The hardware half (Tables I, II hardware columns, Fig. 1) needs nothing else: its inputs are in `data/`.
The quality half needs the two OLMoE checkpoints and some datasets, which `scripts/00_download.py` fetches into the default
Hugging Face cache (`~/.cache/huggingface/hub`, where the scripts look for them; keep that location or link it there):

| part | model | datasets | size |
|---|---|---|---|
| hardware (Tables I, II, Fig. 1) | none | none | |
| quality: KL (Table II, Fig. 4), multiple-choice tasks of Table III | `allenai/OLMoE-1B-7B-0125` (base) | WikiText-2, HellaSwag, WinoGrande, BoolQ | 28 GB + 0.3 GB |
| downstream generation (GSM8K, HumanEval of Table III) | `allenai/OLMoE-1B-7B-0125-Instruct` | GSM8K, HumanEval | 14 GB + 0.1 GB |

```bash
python scripts/00_download.py --dry-run            # what would be fetched and how big
python scripts/00_download.py --what quality       # base checkpoint + WikiText-2 + the three multiple-choice datasets
python scripts/00_download.py --what downstream    # Instruct checkpoint + GSM8K + HumanEval
```

**What the runs need.** One GPU with 16 GB (we used an RTX 5070 Ti) and a host with a few tens of GB of RAM and about 60 GB of free disk:
each encoded OLMoE is 6.9 GB (`cache/planes_olmoe`), and the GPTQ copies of Table II are 6.9 GB each. The digital bf16 evaluation of the
Instruct model holds the whole model on the GPU (15 GB); the analog runs keep the planes in host memory and need 5 GB of GPU memory.

**Run.**
```bash
bash scripts/24_quality_run.sh                     # IR-drop gains, encoding, residual table, every KL measurement (~2.5 h); results in work/cache
# Table III (run from work/, where the first command left cache/): the Instruct model needs its own planes and residual table
cd work
python ../scripts/01_encode_experts.py --model allenai/OLMoE-1B-7B-0125-Instruct --out cache/planes_olmoe_instruct
python ../scripts/03_profile_residual.py --model allenai/OLMoE-1B-7B-0125-Instruct --planes cache/planes_olmoe_instruct --out cache/residual_olmoe_instruct.npy
python ../scripts/22_downstream_choice.py --model allenai/OLMoE-1B-7B-0125 --planes cache/planes_olmoe --residual cache/residual_olmoe.npy --home cpu --configs "DPA4-8,uni4-8" --out cache/mc_aimc.json
python ../scripts/22_downstream_choice.py --model allenai/OLMoE-1B-7B-0125 --planes cache/planes_olmoe --residual cache/residual_olmoe.npy --digital --configs "digital-digital" --tok-budget 1024 --out cache/mc_digital.json
python ../scripts/23_downstream_generation.py --mode digital --batch 8 --out cache/gen_digital.json
python ../scripts/23_downstream_generation.py --mode analog --configs "DPA4-8,uni4-8" --out cache/gen_aimc.json
```
Optional: new design points of the hardware model need the NeuroSim build (`git clone https://github.com/neurosim/3D_NeuroSim_V1.0.git
third_party/3D_NeuroSim_V1.0`, CC BY-NC 4.0, then `scripts/09_build_neurosim.sh`). `OCDPA_CACHE` and `NEUROSIM_DIR` override the
locations of the outputs and of NeuroSim (see `ocdpa/paths.py`). Run every command from the repository root.

## Paper to code

Every number of the paper, the script that produces it and the file that holds the value we measured (`results/`, or the
output of the hardware report). "GPU" marks the steps that need the models and a GPU; everything else runs on the CPU.

| in the paper | script | result | time |
|---|---|---|---|
| Eq. (2): the any-precision encoding; 8.56 bit per weight | `01_encode_experts.py` (GPU), `ocdpa/encoder.py`, `ocdpa/planes.py` | `cache/planes_<model>/` | 17.5 min (OLMoE) |
| alpha falls 1.7--1.9x per plane; base planes carry 89 % of the cell-error variance (Sec. III-B) | `02_plane_statistics.py` | printed | |
| residual table R[layer, expert, k] | `03_profile_residual.py` (GPU) | `data/residual_olmoe.npy` | 44 s |
| residual LUT: 6.75 KB, 9-bit codes (Sec. III-C, Fig. 3) | `04_export_lut.py` | `results/lut_olmoe.hex(.json)` | 1 s |
| greedy = dynamic program on all dispatches; the fixed-point datapath agrees on about 80 % of them and reaches an objective within 0.2 % of the program's | `05_check_allocator.py` | printed | 1 min |
| routing traces (hardware input) | `06_record_routing.py` (GPU) | `data/trace_*.npz` | 4 min (both traces) |
| column gain variation 0.7 % / 2.2 % (Sec. III-B) | `07_calibrate_irdrop.py` | `data/irdrop_gain.npz` (copy it to `cache/` to run the quality scripts) | 10 s |
| bit-serial array = what the hardware model prices | `08_check_analog_path.py` | printed | 1 s |
| Table I; Fig. 1; the numbers of Secs. I, IV-B (footprint 14.9x, TTFT 3.6x, lanes 91 %, 1.81x, 1.49x, 1.69x, energy 22 / 16 / 16 %, same-TTFT area 1.9x / 2.0x, decode 8--10x and 13--17x against flash) | `10_hardware.py`, `13_hardware_report.py`, `30_make_figures.py --only 1` | `results/hardware_report.md`, `hardware_claims.json` | 30 s |
| allocator: 19 cycles, 1.9 ns, 9.7 pJ, 604 / 10,145 um2, DP 1,859 cycles | `11_allocator_ppa.py` (NeuroSim build; shipped result) | `data/allocator_ppa.json` | |
| TTFT of DPA within 2 % of uniform at equal budget (Sec. II-B); the TTFT of every budget (Fig. 4, left) | `10_hardware.py` | `results/hardware_claims.json` (`budget.*`) | |
| Table II: mm2, TTFT, tok/s/mm2, b/w | `12_precision_alternatives.py` | `results/hardware_report.md` | |
| mixed placement 31 % larger memory die, 0.4 % of the footprint (Sec. III-B) | `12_precision_alternatives.py` | `results/hardware_claims.json` (`table2.*`) | |
| Table II: KL of uniform, OC-DPA, all planes | `21_kl_sweep.py` (GPU), stage `kl_table2` of `24_quality_run.sh` | `results/kl_olmoe_table2.json` | 9 min |
| Table II: KL of the static GPTQ rows | `20_encode_native_gptq.py`, `21_kl_sweep.py` (GPU), stages `gptq`, `kl_static` | `results/kl_olmoe_static.json` | 15 min |
| Fig. 4, left: KL against TTFT over the budget | stage `kl_task` | `results/kl_olmoe_task.json` | 35 min |
| Fig. 4, right: KL against cell variation (experts only, with OC-DPA, all on AIMC) | stages `sigma_experts`, `sigma_dense` | `results/sigma_olmoe_*.json` | 1 h |
| KL with all planes in crossbars against the mixed placement (Sec. III-B) | stage `kl_xbar` | `results/kl_olmoe_xbar.json` | 7 min |
| Fig. 4 | `30_make_figures.py --only q` | `figs/quality_olmoe.png` | |
| Table III: HellaSwag, WinoGrande, BoolQ | `22_downstream_choice.py` (GPU) | `results/downstream_mc_*.json` | 1.5 h per configuration |
| Table III: GSM8K, HumanEval | `23_downstream_generation.py` (GPU) | `results/downstream_gen_*.json` | several hours per configuration |

Models: KL divergence, the multiple-choice tasks and all hardware inputs are those of `allenai/OLMoE-1B-7B-0125` (base); GSM8K and
HumanEval use `allenai/OLMoE-1B-7B-0125-Instruct`. `bash scripts/24_quality_run.sh` runs the whole quality half (encoding to every KL measurement) in one resumable command (see "Run times").

## Encoding the experts

Everything in this repository starts from one encoding, so it comes first. Every routed expert matrix is stored in
a **any-precision encoding**: progressive binary-coded quantization in the style of AnyBCQ, with one set of scales shared by
all precisions. The matrix is cut, per output row, into groups of 128 inputs, and each group is written as

```
W^(k) ≈ β + Σ_{p=1..k} α_p · S_p ,     S_p ∈ {−1, +1}^128 ,   k = 2 … 8
```

with 8 sign planes `S_p` (one bit per weight and plane), 8 descending scales `α_p` and an offset `β` per group. The
scales are shared by all `k`, so the first `k` planes with their scales are a valid `k`-bit model: the number of planes
read is a run-time knob. The two base planes carry most of the signal and are always read.

`ocdpa/encoder.py` is the encoder (`scripts/01_encode_experts.py` runs it over a checkpoint). It is weight-only: no calibration
data, nothing outside this repository but PyTorch and NumPy, 65 s per OLMoE layer on one GPU (17.5 min for the 16 layers).
Per matrix it
1. initialises the two base planes greedily (each plane takes the sign of the residual and its mean magnitude as scale),
2. refines them in 20 rounds: scales and offset by least squares (batched conjugate gradient), then every sign is
   re-assigned to the nearest of the 2^k representable values (binary search),
3. adds the planes 3 … 8 one at a time (20 rounds each), re-fitting scales and offset while the earlier planes keep
   their meaning, and re-assigns all signs by the nearest-level search after each added plane.

It writes `cache/planes_<model>/L<layer>.npz` (signs packed to one bit; 6.9 GB for OLMoE-1B-7B, 16 GB for DeepSeek-MoE-16B).
The scales are stored in fp16 and rounded to **8-bit logarithmic codes** when the planes are loaded (`ocdpa/planes.py`,
`SCALE_BITS`), which is where the 0.56 bit per weight of scale storage comes from. The encoder is deterministic: the same
checkpoint gives the same planes bit for bit.

The idea of giving every expert its own number of planes is DPA-MoE's; the encoding is not taken from it, and DPA-MoE's code is
not used here. The numerical kernels of the encoder (greedy initialisation, least-squares refinement, nearest-level search,
conjugate gradient) are adapted from [AnyBCQ](https://github.com/naver-aics/anybcq) (NAVER Cloud Corp., Apache-2.0;
license and notice in `licenses/`, summary in `NOTICE`).

## What the code does

The code has two halves that share the plane counts chosen by the allocator:

* **quality**: a simulated analog array (8-bit ADC, IR-drop, cell variation) runs the experts of
  OLMoE-1B-7B plane by plane; KL divergence and downstream accuracy are measured on OLMoE only;
* **hardware** (OLMoE-1B-7B, DeepSeek-MoE-16B and Qwen3-30B-A3B): the same plane counts are priced on a 16-tier 3D stack with ADCs shared on a logic
  tier, a 22 nm die like the memory tiers (NeuroSim for arrays, converters, NPU and allocator; a closed-form NPU model; a lane-queue
  model of prefill): TTFT, area and energy. The design space is the tiers per ADC lane `s`, the columns per ADC `m` and, for models
  with always-on shared experts, the `m` of their lanes; each design takes its best point by prefill throughput per area.

Reading the first `k` planes is a run-time knob, and the allocator spends a per-token plane budget on the experts
the router weights most. See `ocdpa/__init__.py` for the module map.

## Settings of the reported results

Task setting used for the reported results: 1T1R base planes with 1% and crossbar planes with 2% cell
variation (`--noise "mixed,0.01,0.02,cal"`: IR-drop at R_on = 300 kΩ with a global gain trim), prefill budget
k-bar = 4 and decode at k = 8. The sigma sweep uses `"mixed,s,2s,cal"` with s in {0, 1, 2, 5, 10} % (Fig. 4 shows 1, 2 and 5 %). The KL is the one of the
256 continuation tokens after the 1024-token prompt (`kld_cont` in the JSON files; the stored per-window values allow confidence
intervals), averaged over four windows of the WikiText-2 test text.

**Models.** KL divergence and the hardware inputs (routing traces, residual table) are those of `allenai/OLMoE-1B-7B-0125` (base);
GSM8K and HumanEval of Table III use `allenai/OLMoE-1B-7B-0125-Instruct`.

Quality numbers depend on the encoder run (the any-precision planes of `01_encode_experts.py` and the residual table of `03_profile_residual.py`)
and on the noise seed; see "Results" for how closely a new run agrees.

## Hardware model

`scripts/13_hardware_report.py` is the one command for the hardware half:

```bash
python scripts/13_hardware_report.py --out cache/hardware     # ~30 s, CPU only
```

It runs `11_allocator_ppa.py`, `10_hardware.py`, `12_precision_alternatives.py` and the hardware figures, and writes
`hardware_report.md` (Tables I and II, the sweeps, the allocator) and `hardware_claims.json` (every number quoted in the text, by
name: footprint ratios, equal-TTFT and equal-area comparisons, the lane share of the TTFT, energy shares, decode ratios).

* **Inputs** (`data/`, a measurement in `OCDPA_CACHE` takes precedence): `trace_<model>[_long].npz` routing traces,
  `residual_<model>.npy` the residual tables `R[layer, expert, k]` (they depend on the encoding: regenerate them with `03_profile_residual.py` for
  new planes), `probe3d.json` the NeuroSim runs behind the arrays, ADC lanes and TSVs, `allocator_ppa.json`.
  A design point that is not in `probe3d.json` needs the NeuroSim build of `09_build_neurosim.sh`; the result is added to the cache.
* **Stack**: 16 memory tiers (22 nm RRAM; planes 1-2 in 1T1R 128 x 128, planes 3-8 in 256 x 256 crossbars) on a logic die
  that holds the ADC lanes, the allocator and the NPU. The footprint is the larger of the two dies. The logic die is 22 nm
  like the memory tiers; `OCDPA_LOGIC_NODE=7` prices ADC lanes, NPU primitives and the scale multiplier at 7 nm instead.
  The NPU (128 x 128 weight-stationary array, three 2 MB buffers) is 11.6 mm² at 22 nm and 1.3 mm² at 7 nm, from NeuroSim.
* **Design space** (`compare.grid`): `s` in {1, 2, 4, 8, 16} tiers per ADC lane, `m` in {8, 4, 2, 1} columns per ADC and, for
  DeepSeek, `m` of the shared-expert lanes in {4, 2, 1}. The best point of a design is the one with the highest prefill
  throughput per area (`compare.best`); the best baseline reads all planes there.
* **Not modeled**: the area of the per-column scale multiplier and accumulators, the input drivers and registers beyond NeuroSim's
  word-line switches, offset and noise of the converters. The multiplier's energy and latency are counted.

## How a hardware number is obtained

Every hardware number (area, TTFT, energy, throughput per area) is the output of one chain, and each link is a file or a
function that can be inspected:

1. **Cost of one column read, from NeuroSim** (`ocdpa/hw/neurosim.py`, driver programs in `neurosim_patch/`, results cached in
   `data/probe3d.json`, 35 design points). A single subarray (128 rows, 1T1R or 256 x 256 crossbar, RRAM, 22 nm) is built with our
   mapping and NeuroSim reports its area, latency and energy split into cell array, switch matrices, column mux, ADC and
   shift-add. The ADC resolution is not a knob: 128 rows of ±1 cells driven by a 1-bit input give 257 distinct sums, so a lossless
   read needs 8 bits. The same blocks are priced at the node they sit on (memory tier or logic tier, `OCDPA_LOGIC_NODE`).
2. **The stack** (`ocdpa/hw/stack.py`): T = 16 memory tiers, `s` tiers sharing one ADC lane, `m` columns per ADC. It gives the
   memory die (cells, switches, TSV landings, which grow with the number of lanes) and the logic die (lanes, allocator, NPU);
   the footprint is the larger of the two. The NPU (128 x 128 weight-stationary array, three 2 MB buffers; 11.6 mm² at 22 nm)
   comes from NeuroSim's primitive library (`ocdpa/hw/npu.py`).
3. **The workload** (`ocdpa/hw/workload.py`, `ocdpa/hw/npu.py`): attention, router, dense layers, the shared experts that stay on
   the NPU and the output head are priced as weight-stationary GEMMs with SCALE-Sim's cycle count (partial sums kept on chip) and
   LPDDR4 traffic. **Experts** are priced from real routing: `data/trace_<model>_long.npz` (8,192 tokens of WikiText-2, which
   expert each token chose and with what gate weight) and `trace_<model>.npz` (its first 2,048 tokens, used only to
   group experts by load, so the grouping never sees the text it is scored on). Record them with `06_record_routing.py`.
4. **Plane counts**: the allocator (marginal-gain greedy, equal to the dynamic program) spends the per-token budget on the experts
   with the largest gate-weighted residual reduction, from `data/residual_<model>.npy` (**depends on the encoding**; the shipped
   OLMoE table is that of the base checkpoint, encoded by `ocdpa/encoder.py`). Its own area, energy and cycles come from NeuroSim's
   `allocator` driver (`data/allocator_ppa.json`).
5. **Lane timing** (`ocdpa/compare.py: evaluate`, `ocdpa/hw/stack.py: stack_latency`): an expert's planes are read one after
   the other through the lanes of its sharing group, a plane read is 8 bit-serial input bits times the read time of the subarray,
   and the lane time of a layer is `m` x the jobs of its busiest group. The imbalance (busiest group over mean) is measured on the
   trace, not assumed. TTFT is the NPU time of the non-expert work plus the lane time of the experts summed over the layers (prefill of 1,024
   tokens, scored on the second window of the long trace; the NPU and the stack do not overlap in this number, `ttft_chunk_ms`
   gives the overlapped one); energy adds ADC conversions, array reads, scale multiplications and the NPU.
6. **Design choice** (`compare.grid`, `compare.best`): every design (3D-AIMC with all planes, best baseline, OC-DPA) is evaluated on
   the grid `s` in {1, 2, 4, 8, 16} x `m` in {8, 4, 2, 1} (and the `m` of the shared-expert lanes for DeepSeek) and takes its
   point with the highest prefill throughput per area. "Same TTFT" means within 1 %.
7. **Report** (`scripts/13_hardware_report.py`): runs `11`, `10` and `12`, writes every number quoted in the text by name to
   `hardware_claims.json` and the tables to `hardware_report.md`.

Not measured by the chain: the closed-form parts (NPU cycle counts, DRAM and flash traffic) are models, not simulation of RTL; the
absolute numbers depend on NeuroSim's 22 nm technology files, the ratios between designs are what the paper claims.

## Lane model

The timing model reads the planes of an expert one after the other through the ADC lanes of its sharing group
(Eq. 1 of the paper), so the area model provisions lanes the same way: all planes of a tile share one lane set
through a plane-select mux, sized for the wider of the two kinds of subarray (1T1R and crossbar), so the logic die is the larger of their lane
areas (`hetero_area(..., lane_sets="tile")`, `ocdpa/hw/stack.py`). The memory die still carries the TSV landings of every plane
subarray. `lane_sets="kind"` gives each kind its own lanes (which could then run in parallel; the timing model does not
assume that) and `"plane"` gives every plane its own lanes.

## Quality model of the array

The quality experiments (`ocdpa/analog.py`, `analog_matmul`) follow the path the hardware takes and the hardware model
prices:

1. the input is quantised to 8 bits (symmetric, one scale per group of 128 inputs, the granularity at which the array
   output is rescaled anyway) and applied bit-serially in two's complement;
2. every input bit is one read of the array: the column sum of the 0/1 input against the ±1 cells of a group, with the
   frozen cell variation and the IR-drop gain of the column position, goes through the 8-bit ADC, whose range is fixed by
   the array (128 rows of full-swing cells, not by the data), so every input bit is one conversion;
3. the digital shift-and-add recombines the bits, then the scales α_p and the offset β (8-bit logarithmic codes) are applied.

DeepSeek-MoE's always-on shared experts sit in the stack in the hardware model, so the quality runs put them on the array
too (`--dense shared`, eight planes, same read path); `--dense attn,router,head` additionally moves the attention projections,
the router and the output head, as in the sensitivity experiment of Fig. 4 (right).

The converters are ideal apart from quantisation and saturation (no offset, gain error, INL/DNL or thermal noise), and
the 1-bit input drive is ideal. `21_kl_sweep.py --ideal-inputs` runs the earlier idealisation (full-precision inputs, one
conversion per group, a converter range set by the largest input of the batch) for comparison. The residual table
`R[layer, expert, k]` of `03_profile_residual.py` is measured with full-precision inputs: it is the approximation error of the planes.

## Static (fixed-precision) copies

`20_encode_native_gptq.py` quantises every routed expert matrix on its own with GPTQ (asymmetric, group
size 128, 4 and 8 bits). The Hessians come from the full-precision model on 32,768 tokens of WikiText-2
*train* text, disjoint from the test text of the KL experiments, over the tokens routed to each expert.
A b-bit integer is exactly b sign planes, so a GPTQ copy is written in the same plane format and runs
through the same analog read (planes 1-2 in 1T1R, the rest in crossbars, same noise). The two-copy
design reads the 4-bit copy in prefill and the 8-bit copy in decode.

## Run times

One RTX 5070 Ti (16 GB), 12 CPU cores. `24_quality_run.sh` runs the stages in this order and skips a stage whose output exists.

| stage | what | time |
|---|---|---|
| `encode` | `ocdpa/encoder.py` on the 16 layers of OLMoE-1B-7B | 17.5 min (65 s per layer) |
| `residual` | residual table `R[layer, expert, k]` | 44 s |
| `kl_table2` | KL of all planes, uniform 4 and OC-DPA 4 (Table II) | 9 min |
| `gptq` | GPTQ 4- and 8-bit copies (Table II, static rows) | 6 min |
| `kl_static` | KL of GPTQ 8-bit, 4-bit and 4 + 8-bit | 9 min |
| `kl_task` | KL over the budgets 2 to 7 (Fig. 4, left) | 35 min |
| `sigma_experts` | KL against cell variation, experts on AIMC (Fig. 4, right) | 32 min |
| `sigma_dense` | the same with attention, router and head on AIMC | 29 min |
| `kl_xbar` | KL with all planes in crossbars (Sec. III-B) | 7 min |
| `13_hardware_report.py` | every hardware number (CPU only) | 25 s |

The KL runs read four windows of 1280 tokens; a row of `21_kl_sweep.py` takes about 2.5 min (all planes, one noise scenario about 4 min).
The sum of the quality stages is about 2.5 h.

## Downstream accuracy

Table III reports HellaSwag, WinoGrande and BoolQ (base model; 300 items per task, seed 0, 0-shot, prompt formats of lm-evaluation-harness)
and GSM8K (240 problems) and HumanEval (164; Instruct model, greedy, 0-shot chat prompt), each prefilled at the given budget and read at
k = 8 in decoding (the commands are in "From zero"). Digital rows run the bf16 model. The multiple-choice values are in
`results/downstream_mc_digital.json` and `results/downstream_mc_aimc.json`.

Speed: about 40 tokens/s of context at four planes in prefill, with the answers read at eight planes, which makes a multiple-choice
configuration take about 1.5 h; a generation configuration takes several hours (the 240 GSM8K problems about 3 h).

## Results

The values in `results/` are the ones behind the paper. The hardware numbers are deterministic and a new run reproduces them
exactly; the quality values (KL, accuracy) are measured with a fixed noise seed, and two runs of the same KL rows on our machine gave
identical values. Another GPU or library version can change the last digits, and greedy decoding of long generations can change
the accuracy by a problem or two, so a new run should not be expected to agree digit for digit.

`results/` has the measured values behind the tables and figures: the KL files of the stages above (`kl_olmoe_*.json`,
`sigma_olmoe_*.json`; each row carries `kld_cont`, the KL of the continuation, and `kld_cont_windows`, its value per window) and the
output of `13_hardware_report.py` (`hardware_report.md`, `hardware_claims.json`, `hardware.json`). `figs/` has Fig. 1 and Fig. 4.

An earlier version of the sigma sweep (full-precision inputs, one conversion per group, a converter range set by the largest
input of the batch) gave a much higher KL when attention, router and head are on the arrays; that
idealisation is still available as `21_kl_sweep.py --ideal-inputs` and reproduces the earlier values.

## Layout

```
ocdpa/        library (see ocdpa/__init__.py)
scripts/      numbered entry points, one per pipeline step
neurosim_patch/  our additions to 3D+NeuroSim (driver programs); the upstream tree is not included
```

## License

The encoder (`ocdpa/encoder.py`) follows AnyBCQ's progressive precision expansion; its numerical kernels are adapted from AnyBCQ
(Apache License 2.0, Copyright (c) 2025-present NAVER Cloud Corp., [github.com/naver-aics/anybcq](https://github.com/naver-aics/anybcq)),
whose license and notice are in `licenses/` and summarised in `NOTICE`. The rest of the code is released under the
MIT License (`LICENSE`). The driver programs in `neurosim_patch/` are built on 3D+NeuroSim V1.0
(CC BY-NC 4.0, [github.com/neurosim/3D_NeuroSim_V1.0](https://github.com/neurosim/3D_NeuroSim_V1.0)), which is not
redistributed here; building and using them is subject to NeuroSim's license, including its non-commercial condition.
