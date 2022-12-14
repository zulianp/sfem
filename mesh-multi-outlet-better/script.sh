#!/usr/bin/env bash

set -e
set -x

HERE=$PWD

case_folder=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better
solver_exec=/Users/patrickzulian/Desktop/code/utopia/utopia/build/utopia_exec

echo "Assemble system"
../assemble  $case_folder ./

ls -la 

echo "Convert rhs to vtu"
../python/raw2mesh.py -d $case_folder -f rhs.raw  
mv out.vtu rhs.vtu

echo "Condense system"
../condense_matrix ./  		 $case_folder/zd.raw ./condensed
../condense_vector ./rhs.raw $case_folder/zd.raw condensed/rhs.raw

../python/fp_convert.py condensed/rhs.raw condensed/rhs.raw float64 float32
../python/fp_convert.py condensed/rhs.raw condensed/rhs.raw float32 float64

../python/fp_convert.py condensed/values.raw condensed/values.raw float64 float32
../python/fp_convert.py condensed/values.raw condensed/values.raw float32 float64

ls -la ./condensed

cd condensed

echo "Solve condensed linear system"
$solver_exec -app ls_solve -A rowptr.raw -b rhs.raw -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out out.raw --verbose  -ksp_monitor

../../remap_vector out.raw $case_folder/zd.raw sol.raw
../../python/raw2mesh.py -d $case_folder -f sol.raw
mv out.vtu sol.vtu

cd $HERE
