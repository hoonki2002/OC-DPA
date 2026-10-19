## Table I: hardware comparison at a 1024-token prefill

| model | design | s / m (shared m) | mm2 | TTFT (ms) | E_pre (mJ) | tok/s/mm2 |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 3D-AIMC | 16 / 8 | 147 | 1014 | 360 | 6.9 |
| OLMoE-1B-7B | Best baseline | 16 / 4 | 184 | 497 | 339 | 11.2 |
| OLMoE-1B-7B | OC-DPA | 16 / 4 | 184 | 275 | 263 | 20.3 |
| DeepSeek-MoE-16B | 3D-AIMC | 16 / 8 | 412 | 1252 | 754 | 2.0 |
| DeepSeek-MoE-16B | Best baseline | 16 / 4 (4) | 609 | 656 | 717 | 2.6 |
| DeepSeek-MoE-16B | OC-DPA | 16 / 8 (4) | 412 | 653 | 604 | 3.8 |
| Qwen3-30B-A3B | 3D-AIMC | 16 / 8 | 660 | 2236 | 1114 | 0.7 |
| Qwen3-30B-A3B | Best baseline | 16 / 4 | 787 | 995 | 1067 | 1.3 |
| Qwen3-30B-A3B | OC-DPA | 16 / 4 | 787 | 588 | 896 | 2.2 |

## Numbers quoted in the text

### OLMoE-1B-7B

- Fig. 1 (sweep of s at 4 columns per ADC): footprint 2744 -> 184 mm2 (14.9x); logic die 2744 -> 184, memory die 530 -> 153 mm2; TTFT 136 -> 497 ms (3.6x)
- 3D-AIMC -> best baseline: +25% area, TTFT 1014 -> 497 ms (2.04x)
- OC-DPA over the best baseline: 1.81x throughput per area; energy 22% lower (experts: 45% of the baseline's prefill energy, 50% lower)
- same TTFT (within 1%): the baseline needs 354 mm2 (s=16, m=2, TTFT 277 ms) against 184 mm2: 1.93x the area
- same area: baseline TTFT 497 ms against 275 ms: 1.81x
- prefill at 1024 tokens: NPU+DRAM 45 ms, lanes 452 ms (91% of the TTFT); OC-DPA halves the lane time (2.0x); lanes bound the prefill beyond about 44 tokens
- decode: experts on AIMC 9.5x faster and 17.1x less energy than NPU+flash; NPU 90% of the AIMC decode latency

### DeepSeek-MoE-16B

- Fig. 1 (sweep of s at 4 columns per ADC): footprint 6550 -> 609 mm2 (10.7x); logic die 6550 -> 609, memory die 1269 -> 393 mm2; TTFT 443 -> 656 ms (1.5x)
- 3D-AIMC -> best baseline: +48% area, TTFT 1252 -> 656 ms (1.91x)
- OC-DPA over the best baseline: 1.49x throughput per area; energy 16% lower (experts: 49% of the baseline's prefill energy, 32% lower)
- same TTFT (within 1%): the baseline needs 609 mm2 (s=16, m=4, TTFT 656 ms) against 412 mm2: 1.48x the area
- same area: baseline TTFT 1207 ms against 653 ms: 1.85x
- prefill at 1024 tokens: NPU+DRAM 88 ms, lanes 567 ms (87% of the TTFT); OC-DPA halves the lane time (1.6x); lanes bound the prefill beyond about 74 tokens
- decode: experts on AIMC 9.3x faster and 15.7x less energy than NPU+flash; NPU 93% of the AIMC decode latency

### Qwen3-30B-A3B

- Fig. 1 (sweep of s at 4 columns per ADC): footprint 12308 -> 787 mm2 (15.6x); logic die 12308 -> 787, memory die 2387 -> 688 mm2; TTFT 556 -> 995 ms (1.8x)
- 3D-AIMC -> best baseline: +19% area, TTFT 2236 -> 995 ms (2.25x)
- OC-DPA over the best baseline: 1.69x throughput per area; energy 16% lower (experts: 32% of the baseline's prefill energy, 50% lower)
- same TTFT (within 1%): the baseline needs 1553 mm2 (s=16, m=2, TTFT 593 ms) against 787 mm2: 1.97x the area
- same area: baseline TTFT 995 ms against 588 ms: 1.69x
- prefill at 1024 tokens: NPU+DRAM 170 ms, lanes 825 ms (83% of the TTFT); OC-DPA halves the lane time (2.0x); lanes bound the prefill beyond about 83 tokens
- decode: experts on AIMC 7.7x faster and 13.4x less energy than NPU+flash; NPU 92% of the AIMC decode latency

## Table II: sources of the TTFT reduction and static quantization (OLMoE; the KL column comes from scripts/21_kl_sweep.py)

| row | s / m | mm2 | memory die | logic die | TTFT (ms) | tok/s/mm2 | E_pre (mJ) |
|---|---|---|---|---|---|---|---|
| 8-bit (best baseline) | 16 / 4 | 184 | 153 | 184 | 497 | 11.2 | 339 |
| + uniform k=4 | 16 / 4 | 184 | 160 | 184 | 271 | 20.6 | 263 |
| + DPA (OC-DPA) | 16 / 4 | 184 | 160 | 184 | 275 | 20.3 | 263 |
| Static 4-bit | 16 / 8 | 99 | 93 | 99 | 491 | 21.1 | 274 |
| Static 4-bit + 8-bit | 16 / 2 | 354 | 277 | 354 | 161 | 18.0 | 258 |

- two-copy (4 + 8 bit) area 1.93x that of OC-DPA; OC-DPA throughput per area 1.13x
- 1T1R base planes: memory die 153 against 116 mm2 for all-crossbar (+31%); footprint 184 against 183 mm2

## On-chip allocator (marginal-gain rule)

| model | cycles | latency (ns) | energy (pJ) | datapath (um2) | with table (um2) | table (KB) |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 19 | 35.7 | 9.7 | 604 | 10145 | 6.75 |
| DeepSeek-MoE-16B | 15 | 24.0 | 9.6 | 413 | 16484 | 11.39 |
| Qwen3-30B-A3B | 19 | 39.2 | 27.5 | 604 | 53567 | 40.50 |

