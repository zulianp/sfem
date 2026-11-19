#!/usr/bin/env bash

set -e

mat=$1
rhs=$2
sol=$3

mpirun utopia_exec -app ls_solve -A $mat -b $rhs -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out $sol --verbose
