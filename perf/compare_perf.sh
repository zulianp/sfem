#!/usr/bin/env bash

#SBATCH -J PerfHydroS
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --mem=12020

set -e
# set -x

#if ((1 != $#))
#then
#	printf "usage: $0 <path/to/mesh>\n" 1>&2
#	exit -1
#fi

#case_folder=$1
case_folder=/scratch/zulian/xdns/fe_hydros/sfem/tests/compare/mesh-multi-outlet-better

# Libraries
laplsoldir=/scratch/diegor/zu/laplsol/
sfemdir=/scratch/zulian/xdns/fe_hydros/sfem/
utopiadir=/home/zulian/utopia/utopia/build/

# Add executables to path
PATH=$sfemdir:$PATH
PATH=$laplsoldir:$PATH
PATH=$utopiadir:$PATH

if [ ! -d "./out/" ]; then
	mkdir out
fi

##############
# SFEM
#############

# Assemble operator
mpirun -np 1 assemble $case_folder ./out

# Remove dirichlet nodes
mpirun -np 1 condense_matrix ./out 	   $case_folder/zd.raw ./condensed
mpirun -np 1 condense_vector ./out/rhs.raw $case_folder/zd.raw condensed/rhs.raw

# Parallel linear solve
mpirun utopia_exec -app ls_solve -A ./condensed/rowptr.raw -b ./condensed/rhs.raw -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out ./condensed/out.raw

# Missing post processing of vector
# TODO

##############
# LAPLSOL
#############

# Assemble operator
pre $case_folder
laplsol-asm $case_folder

# Parallel linear solve
mpirun spsolve $case_folder $case_folder/rhs.raw $case_folder/sol.raw

# Post processing of solution vector
post $case_folder $case_folder/sol.raw

cd $HERE
