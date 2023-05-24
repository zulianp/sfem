#!/usr/bin/env bash

set -e

poisson_ratio=0.4
young_modulus=9174000.0

# mu=1e3; 
mu=`python3 -c	'young_modulus='$young_modulus';  poisson_ratio='$poisson_ratio'; print(young_modulus / (2 * (1 + poisson_ratio)))'`
lambda=`python3 -c 'poisson_ratio='$poisson_ratio'; mu='$mu'; print(2.0 * (mu * poisson_ratio) / (1 - 2 * poisson_ratio ))'`

db=/Users/patrickzulian/Desktop/code/sfem/data/FK1100_Prototype_design_simulation_results/res_solid.e 
workspace=dump

ncdump -h $db

# ./stress.sh $db $workspace $mu $lambda
# ./strain.sh $db $workspace
# ./principal_strains.sh $db $workspace
./principal_stresses.sh $db $workspace $mu $lambda
