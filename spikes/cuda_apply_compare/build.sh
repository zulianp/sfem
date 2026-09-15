#!/usr/bin/env bash
# Build the driver twice, against the two generated trees.
#
# The driver declares the entry points it calls and includes nothing from
# either tree, so it compiles with no include path of its own; each generated
# operator compiles against its own material's root.  `shim/` supplies the
# scalar aliases and the two smesh enums that `op/*_c_abi.hpp` names and a
# standalone generation does not carry.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
MATERIALS=(laplace linear_elasticity neohookean_ogden mooney_rivlin_kelvin_voigt_newmark neumann)
rm -rf build && mkdir -p build

# The inexact-apply operator needs SFEM's own compressed-tangent aliases at
# their real widths and is not one of the kernels under comparison.
cpp_sources() { find "omp/$1" -name '*_operator.cpp' ! -name '*inexact_apply_operator.cpp' | sort; }
cu_sources()  { find "cuda/$1" -name '*_operator.cu'  ! -name '*inexact_apply_operator.cu'  | sort; }

host_objects=()
for material in "${MATERIALS[@]}"; do
    for source in $(cpp_sources "$material"); do
        object="build/host_$(echo "$source" | tr '/' '_' | sed 's/\.cpp$/.o/')"
        g++ -std=c++17 -O3 -march=native -fopenmp -DNDEBUG \
            -I "omp/$material" -I shim -c "$source" -o "$object"
        host_objects+=("$object")
    done
done
g++ -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -c apply_driver.cpp -o build/host_driver.o
g++ -O3 -fopenmp -o apply_cpu build/host_driver.o "${host_objects[@]}"

device_objects=()
for material in "${MATERIALS[@]}"; do
    for source in $(cu_sources "$material"); do
        object="build/device_$(echo "$source" | tr '/' '_' | sed 's/\.cu$/.o/')"
        nvcc -std=c++17 -O3 -arch=sm_90 -DNDEBUG -diag-suppress 177 \
            -I "cuda/$material" -I shim -c "$source" -o "$object"
        device_objects+=("$object")
    done
done
nvcc -std=c++17 -O3 -arch=sm_90 -DNDEBUG -diag-suppress 177 -x cu -c apply_driver.cpp -o build/device_driver.o
nvcc -O3 -arch=sm_90 -o apply_gpu build/device_driver.o "${device_objects[@]}"

echo "built apply_cpu and apply_gpu"
