# OC-DPA

Code for the paper *OC-DPA: On-Chip Dynamic Precision Allocation in ADC-Shared Analog In-Memory MoE Accelerators*.

## Setup

```bash
git clone https://github.com/hoonki2002/OC-DPA.git && cd OC-DPA
python -m venv env && source env/bin/activate      # Python >= 3.10
pip install -e .
python scripts/00_verify.py                        # ~30 s, CPU: one point per step of the paper
```

Run every command from the repository root.

## Hardware numbers (Tables I, II)

No model or GPU needed; the inputs are in `data/`.

```bash
python scripts/13_hardware_report.py --out cache/hardware     # ~30 s, CPU
```

Writes `hardware_report.md` (tables) and `hardware_claims.json` (every hardware number quoted in the text, by name).

## Quality numbers (KL, Table III)

Needs one 16 GB GPU, ~60 GB free disk and two checkpoints:

| used for | model | datasets |
|---|---|---|
| KL (Table II, Fig. 4 data), HellaSwag / WinoGrande / BoolQ | `allenai/OLMoE-1B-7B-0125` (28 GB) | WikiText-2, HellaSwag, WinoGrande, BoolQ |
| GSM8K, HumanEval | `allenai/OLMoE-1B-7B-0125-Instruct` (14 GB) | GSM8K, HumanEval |

```bash
python scripts/00_download.py --dry-run            # sizes
python scripts/00_download.py --what quality       # base model + WikiText-2 + multiple-choice sets
python scripts/00_download.py --what downstream    # Instruct model + GSM8K + HumanEval

bash scripts/24_quality_run.sh                     # encoding, residual table, every KL measurement (~2.5 h, resumable); output in work/cache
```

Table III (run from `work/`, where the previous command left `cache/`):

```bash
cd work
python ../scripts/01_encode_experts.py --model allenai/OLMoE-1B-7B-0125-Instruct --out cache/planes_olmoe_instruct
python ../scripts/03_profile_residual.py --model allenai/OLMoE-1B-7B-0125-Instruct --planes cache/planes_olmoe_instruct --out cache/residual_olmoe_instruct.npy
python ../scripts/22_downstream_choice.py --model allenai/OLMoE-1B-7B-0125 --planes cache/planes_olmoe --residual cache/residual_olmoe.npy --home cpu --configs "DPA4-8,uni4-8" --out cache/mc_aimc.json
python ../scripts/22_downstream_choice.py --model allenai/OLMoE-1B-7B-0125 --planes cache/planes_olmoe --residual cache/residual_olmoe.npy --digital --configs "digital-digital" --tok-budget 1024 --out cache/mc_digital.json
python ../scripts/23_downstream_generation.py --mode digital --batch 8 --out cache/gen_digital.json
python ../scripts/23_downstream_generation.py --mode analog --configs "DPA4-8,uni4-8" --out cache/gen_aimc.json
```

## Paper to script

"GPU" = needs a model and a GPU; everything else runs on the CPU. Times are for an RTX 5070 Ti.

| in the paper | script | result | time |
|---|---|---|---|
| any-precision encoding, 8.56 bit/weight | `01_encode_experts.py` (GPU) | `cache/planes_<model>/` | 17.5 min |
| residual table R[layer, expert, k] | `03_profile_residual.py` (GPU) | `data/residual_olmoe.npy` | 44 s |
| residual LUT, 6.75 KB | `04_export_lut.py` | `results/lut_olmoe.hex` | 1 s |
| greedy = DP; fixed-point datapath vs DP | `05_check_allocator.py` | printed | 1 min |
| routing traces (hardware input) | `06_record_routing.py` (GPU) | `data/trace_*.npz` | 4 min |
| column gain variation 0.7 % / 2.2 % | `07_calibrate_irdrop.py` | `data/irdrop_gain.npz` | 10 s |
| analog path = what the hardware prices | `08_check_analog_path.py` | printed | 1 s |
| Table I, hardware numbers of the text | `13_hardware_report.py` | `results/hardware_report.md`, `hardware_claims.json` | 30 s |
| allocator 19 cycles, 9.7 pJ, 604 / 10,145 µm² | `11_allocator_ppa.py` (needs NeuroSim build) | `data/allocator_ppa.json` | |
| Table II: area, TTFT, tok/s/mm², b/w | `12_precision_alternatives.py` | `results/hardware_report.md` | |
| Table II: KL of uniform, OC-DPA, all planes | stage `kl_table2` of `24_quality_run.sh` | `results/kl_olmoe_table2.json` | 9 min |
| Table II: KL of static GPTQ rows | stages `gptq`, `kl_static` | `results/kl_olmoe_static.json` | 15 min |
| Fig. 4 left: KL vs budget | stage `kl_task` | `results/kl_olmoe_task.json` | 35 min |
| Fig. 4 right: KL vs cell variation | stages `sigma_experts`, `sigma_dense` | `results/sigma_olmoe_*.json` | 1 h |
| KL with all planes in crossbars | stage `kl_xbar` | `results/kl_olmoe_xbar.json` | 7 min |
| Table III: HellaSwag, WinoGrande, BoolQ | `22_downstream_choice.py` (GPU) | `results/downstream_mc_*.json` | 1.5 h / config |
| Table III: GSM8K, HumanEval | `23_downstream_generation.py` (GPU) | `cache/gen_*.json` | hours / config |

Settings of the reported results: `--noise "mixed,0.01,0.02,cal"` (1 % cell variation in 1T1R planes, 2 % in crossbar planes,
IR-drop with a global gain trim), prefill budget 4, decode at k = 8, KL of 256 continuation tokens after a 1024-token prompt,
averaged over four WikiText-2 windows (`kld_cont` in the JSON files). Multiple-choice: 300 items per task; GSM8K 240 problems;
HumanEval 164. KL, hardware inputs and multiple-choice use the base model; GSM8K and HumanEval use the Instruct model.

## Notes

* The hardware numbers are deterministic. The quality numbers use a fixed seed and were identical on repeat runs on our machine; another
  GPU or library version can change the last digits, and greedy generation can move the accuracy by a problem or two.
* New design points of the hardware model need the NeuroSim build:
  `git clone https://github.com/neurosim/3D_NeuroSim_V1.0.git third_party/3D_NeuroSim_V1.0`, then `scripts/09_build_neurosim.sh`.
  `OCDPA_CACHE` and `NEUROSIM_DIR` override the output and NeuroSim locations (`ocdpa/paths.py`).

## Layout

```
ocdpa/           library (module map in ocdpa/__init__.py)
scripts/         numbered entry points, in the order of the data flow
data/            hardware inputs: traces, residual tables, NeuroSim probes
results/         values behind the paper
neurosim_patch/  our driver programs for 3D+NeuroSim (upstream not included)
```

## License

MIT (`LICENSE`). The encoder's numerical kernels are adapted from [AnyBCQ](https://github.com/naver-aics/anybcq)
(Apache-2.0, NAVER Cloud Corp.; license and notice in `licenses/`, summary in `NOTICE`). The drivers in `neurosim_patch/` build on
[3D+NeuroSim V1.0](https://github.com/neurosim/3D_NeuroSim_V1.0) (CC BY-NC 4.0, not redistributed here).
