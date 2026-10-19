# Step 0 — Build

Builds the llama.cpp fork in `third_party/llama.cpp` with the DPA-MoE
extension enabled (`-DGGML_CUDA=ON -DDPA_MOE_EXT=ON`).

```bash
bash experiments/0_build/run.sh
```

Overrides:

```bash
BUILD_DIR=third_party/llama.cpp/build-dpa-moe \
CUDA_ROOT=/usr/local/cuda \
CUDA_ARCHITECTURES=89 \
BUILD_TARGETS="llama-simple dpa-moe-gate-trace llama-perplexity llama-server llama-quantize" \
bash experiments/0_build/run.sh
```

`CUDA_ROOT` defaults to `/usr/local/cuda-12.8` when present, else
`/usr/local/cuda`. `CUDA_ARCHITECTURES` defaults to `80`; set it to your GPU's
compute capability (e.g. `89` for Ada, `90` for Hopper). The extension's unit
tests are built with the default target set of a full build
(`cmake --build third_party/llama.cpp/build-dpa-moe`) as `dpa_moe-*-test`.

Smoke test of an encoded model:

```bash
GPU=0 VRAM_CAP_MB=16384 \
bash experiments/smoke_test_dpa_moe.sh cache/models/Qwen3-30B-A3B/chunked/ss_anybcq.gguf
```
