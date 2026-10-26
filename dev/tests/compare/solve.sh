#!/usr/bin/env bash

set -e
set -x

case_folder=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better

PATH=/Users/patrickzulian/Desktop/code/utopia/utopia/build/:$PATH
PATH=../../:$PATH
PATH=../../python:$PATH

./convert.sh

mpiexec -np 8 utopia_exec -app ls_solve -A fp64/rowptr.raw -b fp64/rhs.raw -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out out.raw --verbose -ksp_monitor

remap_vector out.raw fp64/zd.raw ./fp64/sol.raw

raw2mesh.py -d $case_folder -f ./fp64/sol.raw
mv out.vtu fp64/sol.vtu
