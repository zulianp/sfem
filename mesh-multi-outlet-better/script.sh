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

ls -la ./condensed

echo "Solve linear system"
$solver_exec -app ls_solve -A rowptr.raw -b rhs.raw -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out out.raw --verbose  -ksp_monitor

../python/raw2mesh.py -d $case_folder -f out.raw 
mv out.vtu sol.vtu

cd condensed

echo "Solve condensed linear system"
$solver_exec -app ls_solve -A rowptr.raw -b rhs.raw -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out out.raw --verbose  -ksp_monitor
# ../python/raw2mesh.py -d $case_folder -f out.raw  

cd $HERE