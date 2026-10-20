## Table I: hardware comparison at a 1024-token prefill

| model | design | s / m (shared m) | mm2 | TTFT (ms) | E_pre (mJ) | tok/s/mm2 |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 3D-AIMC | 16 / 8 | 293 | 1014 | 402 | 3.4 |
| OLMoE-1B-7B | Best baseline | 16 / 2 | 354 | 277 | 339 | 10.5 |
| OLMoE-1B-7B | OC-DPA | 16 / 2 | 355 | 163 | 263 | 17.7 |
| DeepSeek-MoE-16B | 3D-AIMC | 16 / 8 | 758 | 1252 | 840 | 1.1 |
| DeepSeek-MoE-16B | Best baseline | 16 / 2 (4) | 1004 | 443 | 730 | 2.3 |
| DeepSeek-MoE-16B | OC-DPA | 16 / 4 (2) | 862 | 375 | 604 | 3.2 |
| Qwen3-30B-A3B | 3D-AIMC | 16 / 8 | 1320 | 2236 | 1209 | 0.3 |
| Qwen3-30B-A3B | Best baseline | 16 / 2 | 1565 | 593 | 1068 | 1.1 |
| Qwen3-30B-A3B | OC-DPA | 16 / 2 | 1598 | 384 | 897 | 1.7 |

## Numbers quoted in the text

### OLMoE-1B-7B

- Fig. 1 (sweep of s at 2 columns per ADC): footprint 5467 -> 354 mm2 (15.4x); logic die 5467 -> 354, memory die 1858 -> 348 mm2; TTFT 92 -> 277 ms (3.0x)
- 3D-AIMC -> best baseline: +21% area, TTFT 1014 -> 277 ms (3.66x)
- OC-DPA over the best baseline: 1.69x throughput per area; energy 22% lower (experts: 45% of the baseline's prefill energy, 50% lower)
- same TTFT (within 1%): the baseline needs 694 mm2 (s=16, m=1, TTFT 155 ms) against 355 mm2: 1.95x the area
- same area: baseline TTFT 277 ms against 163 ms: 1.70x
- prefill at 1024 tokens: NPU+DRAM 45 ms, lanes 232 ms (84% of the TTFT); OC-DPA halves the lane time (2.0x); lanes bound the prefill beyond about 96 tokens
- decode: experts on AIMC 9.5x faster and 17.1x less energy than NPU+flash; NPU 90% of the AIMC decode latency

### DeepSeek-MoE-16B

- Fig. 1 (sweep of s at 2 columns per ADC): footprint 12868 -> 1004 mm2 (12.8x); logic die 12868 -> 1004, memory die 4387 -> 884 mm2; TTFT 443 -> 443 ms (1.0x)
- 3D-AIMC -> best baseline: +33% area, TTFT 1252 -> 443 ms (2.82x)
- OC-DPA over the best baseline: 1.38x throughput per area; energy 17% lower (experts: 50% of the baseline's prefill energy, 35% lower)
- same TTFT (within 1%): the baseline needs 1991 mm2 (s=16, m=1, TTFT 270 ms) against 862 mm2: 2.31x the area
- same area: baseline TTFT 656 ms against 375 ms: 1.75x
- prefill at 1024 tokens: NPU+DRAM 88 ms, lanes 355 ms (80% of the TTFT); OC-DPA halves the lane time (1.0x); lanes bound the prefill beyond about 137 tokens
- decode: experts on AIMC 9.3x faster and 15.7x less energy than NPU+flash; NPU 93% of the AIMC decode latency

### Qwen3-30B-A3B

- Fig. 1 (sweep of s at 2 columns per ADC): footprint 24561 -> 1565 mm2 (15.7x); logic die 24561 -> 1553, memory die 8359 -> 1565 mm2; TTFT 368 -> 593 ms (1.6x)
- 3D-AIMC -> best baseline: +19% area, TTFT 2236 -> 593 ms (3.77x)
- OC-DPA over the best baseline: 1.51x throughput per area; energy 16% lower (experts: 32% of the baseline's prefill energy, 50% lower)
- same TTFT (within 1%): the baseline needs 3084 mm2 (s=16, m=1, TTFT 370 ms) against 1598 mm2: 1.93x the area
- same area: baseline TTFT 593 ms against 384 ms: 1.54x
- prefill at 1024 tokens: NPU+DRAM 170 ms, lanes 422 ms (71% of the TTFT); OC-DPA halves the lane time (2.0x); lanes bound the prefill beyond about 185 tokens
- decode: experts on AIMC 7.7x faster and 13.4x less energy than NPU+flash; NPU 92% of the AIMC decode latency

## Table II: sources of the TTFT reduction and static quantization (OLMoE; the KL column comes from scripts/21_kl_sweep.py)

| row | s / m | mm2 | memory die | logic die | TTFT (ms) | tok/s/mm2 | E_pre (mJ) |
|---|---|---|---|---|---|---|---|
| 8-bit (best baseline) | 16 / 2 | 354 | 348 | 354 | 277 | 10.5 | 339 |
| + uniform k=4 | 16 / 2 | 355 | 355 | 354 | 161 | 17.9 | 263 |
| + DPA (OC-DPA) | 16 / 2 | 355 | 355 | 354 | 163 | 17.7 | 263 |
| Static 4-bit | 16 / 4 | 189 | 189 | 184 | 271 | 20.0 | 274 |
| Static 4-bit + 8-bit | 16 / 1 | 699 | 699 | 694 | 100 | 14.7 | 258 |

- two-copy (4 + 8 bit) area 1.97x that of OC-DPA; OC-DPA throughput per area 1.21x
- 1T1R base planes: memory die 348 against 282 mm2 for all-crossbar (+23%); footprint 354 against 353 mm2

## On-chip allocator (marginal-gain rule)

| model | cycles | latency (ns) | energy (pJ) | datapath (um2) | with table (um2) | table (KB) |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 19 | 35.7 | 9.7 | 604 | 10145 | 6.75 |
| DeepSeek-MoE-16B | 15 | 24.0 | 9.6 | 413 | 16484 | 11.39 |
| Qwen3-30B-A3B | 19 | 39.2 | 27.5 | 604 | 53567 | 40.50 |

