#!/usr/bin/env bash
# Build the NeuroSim driver programs used for area / energy / latency.
#
# 3D+NeuroSim V1.0 (Georgia Tech, CC BY-NC 4.0) is not redistributed here. Clone it
# once, then this script copies our additions (neurosim_patch/) into the tree and
# compiles them against its unmodified sources:
#
#   probe      one analog column read of a 128x128 subarray, per cell type
#   probe3d    the same, split into memory-tier and logic-tier blocks, with TSV load
#   allocator  precision allocator PPA (marginal-gain rule and the DP it replaces)
#   npu_cost   NPU primitives of the logic-die node (MAC, SRAM; NPU_TECHNODE=7 or 22)   [constants are copied into hw/npu.py]
#   scaleunit  per-plane scale multiplier (SU_TECHNODE=7 or 22)                        [constants are copied into system.py]
#   tsvprobe   NeuroSim's TSVPath at the 7 nm logic tier (Heterogeneous3D)             [input TSVs only; negligible]
#
# usage: scripts/09_build_neurosim.sh [path/to/3D_NeuroSim_V1.0]    (default: $NEUROSIM_DIR)
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
NS="${1:-${NEUROSIM_DIR:-$HERE/third_party/3D_NeuroSim_V1.0}}"
[ -d "$NS/Monolithic3D/NeuroSIM" ] || { echo "3D_NeuroSim_V1.0 not found at $NS"; exit 1; }
CXXFLAGS="-fopenmp -O3 -std=c++0x -w"

# build <dir> <our mains...>: compile every other source once, link each main against them
build() {
  local dir="$1"; shift
  local mains=("$@")
  cd "$dir"
  local skip=" main.cpp"; for m in "${mains[@]}"; do skip="$skip $m.cpp"; done
  local objs=()
  for src in *.cpp; do
    case "$skip" in *" $src"*) continue;; esac
    obj="${src%.cpp}.o"
    [ "$obj" -nt "$src" ] || g++ -c $CXXFLAGS "$src" -o "$obj"
    objs+=("$obj")
  done
  for m in "${mains[@]}"; do
    g++ $CXXFLAGS "$m.cpp" "${objs[@]}" -o "$m"
    echo "built $dir/$m"
  done
}

for f in probe probe3d allocator npu_cost scaleunit; do cp "$HERE/neurosim_patch/$f.cpp" "$NS/Monolithic3D/NeuroSIM/"; done
cp "$HERE/neurosim_patch/tsvprobe.cpp" "$NS/Heterogeneous3D/NeuroSIM/"
build "$NS/Monolithic3D/NeuroSIM"   probe probe3d allocator npu_cost scaleunit
build "$NS/Heterogeneous3D/NeuroSIM" tsvprobe
