#!/usr/bin/env bash

rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DSFEM_DIR=$SCRATCH/installations/sfem/lib/cmake -DCMAKE_BUILD_TYPE=Release

make -j72