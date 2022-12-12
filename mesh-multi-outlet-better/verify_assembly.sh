#!/usr/bin/env bash

set -e
set -x

HERE=$PWD

case_folder=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better
solver_exec=/Users/patrickzulian/Desktop/code/utopia/utopia/build/utopia_exec

# Scale factor in surface integral is missing in the case_folder, remove 0.5 once it is fixed
../python/fdiff.py condensed/rhs.raw $case_folder/rhs.raw float64 float32 0.5

../python/fdiff.py condensed/rowptr.raw $case_folder/lhs.rowindex.raw int32   int32
../python/fdiff.py condensed/colidx.raw $case_folder/lhs.colindex.raw int32   int32
../python/fdiff.py condensed/values.raw $case_folder/lhs.value.raw    float64 float32
