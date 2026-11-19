#!/usr/bin/env bash

set -e

HERE=$PWD

dir=`mktemp -d`
cd $dir


nvcc -arch=sm_60 -Xptxas=-O3,-v -use_fast_math $HERE/baseline.cu
./a.out

cd -

rm -rf $dir