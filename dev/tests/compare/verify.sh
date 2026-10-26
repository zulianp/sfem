#!/usr/bin/env bash

set -e
set -x

HERE=$PWD

case_folder=./mesh-multi-outlet-better
solver_exec="mpirun /home/zulian/utopia/utopia/build/utopia_exec"
assemble="mpirun ../../assemble"
condense_matrix="mpirun ../../condense_matrix"
condense_vector="mpirun ../../condense_vector"

../../python/fdiff.py condensed/rowptr.raw $case_folder/lhs.rowindex.raw int32   int32
../../python/fdiff.py condensed/colidx.raw $case_folder/lhs.colindex.raw int32   int32
../../python/fdiff.py condensed/values.raw $case_folder/lhs.value.raw    float64 float32

cd $HERE
