#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
NS="${1:-${NEUROSIM_DIR:-$HERE/third_party/3D_NeuroSim_V1.0}}"
[ -d "$NS/Monolithic3D/NeuroSIM" ] || { echo "3D_NeuroSim_V1.0 not found at $NS"; exit 1; }
CXXFLAGS="-fopenmp -O3 -std=c++0x -w"

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
