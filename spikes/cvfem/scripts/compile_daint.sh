#!/usr/bin/env bash

SPIKE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rm -rf "$SPIKE_ROOT/build"
cmake -S "$SPIKE_ROOT" -B "$SPIKE_ROOT/build" \
    -DCMAKE_CXX_COMPILER=g++ \
    -DSFEM_DIR=$SCRATCH/installations/sfem/lib/cmake \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$SPIKE_ROOT/build" -j72