#!/usr/bin/env bash

set -e

poisson_ratio=0.4
young_modulus=9174000.0
# db=/Users/patrickzulian/Desktop/code/sfem/data/FK1100_Prototype_design_simulation_results/res_solid.e 
# db=/Users/patrickzulian/Desktop/cloud/owncloud_HSLU/Patrick/membraneSetup_simplified/workspace/solid.e 
# db=./solid_test.e
young_modulus=10
poisson_ratio=0.3
# db=/Users/patrickzulian/Desktop/code/utopiatutorials/solidmech/uniaxial_tension/uniaxial_tension_out.e
db=/Users/patrickzulian/Desktop/code/utopiatutorials/solidmech/shear_test/shear_test_out.e

# mu=1e3; 
mu=`python3 -c	'young_modulus='$young_modulus';  poisson_ratio='$poisson_ratio'; print(young_modulus / (2 * (1 + poisson_ratio)))'`
lambda=`python3 -c 'poisson_ratio='$poisson_ratio'; mu='$mu'; print(2.0 * (mu * poisson_ratio) / (1 - 2 * poisson_ratio ))'`

workspace=dump

[ -d "$workspace" ] && rm -rf $workspace

ncdump -h $db

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

./stress.sh $db $workspace $mu $lambda
# ./strain.sh $db $workspace
# ./principal_strains.sh $db $workspace
# ./principal_stresses.sh $db $workspace $mu $lambda
