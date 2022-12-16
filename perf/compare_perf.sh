#!/usr/bin/env bash

#SBATCH -J PerfHydroS
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --time=00:40:00
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

module purge
module load petsc/3.13.5_gcc-10.1.0
module load git
module load cmake
module load boost

module list

case_folder=/scratch/zulian/xdns/fe_hydros/sfem/tests/compare/mesh-multi-outlet-better

# Libraries
laplsoldir=/scratch/diegor/zu/laplsol/
sfemdir=/scratch/zulian/xdns/fe_hydros/sfem/
utopiadir=./

# Add executables to path
PATH=$sfemdir:$PATH
PATH=$sfemdir/python:$PATH
PATH=$laplsoldir:$PATH
PATH=$utopiadir:$PATH


patdir=`mktemp -d`
patdircond=$patdir/condensed
mkdir $patdircond
pat32dir=$patdir/fp32
mkdir $pat32dir

diegodir=`mktemp -d`
cp -R $case_folder $diegodir

##############
# SFEM
#############
echo "software: SFEM"

# Assemble operator
mpirun -np 1 assemble $case_folder $patdir

# Remove dirichlet nodes
mpirun -np 1 condense_matrix $patdir 	      $case_folder/zd.raw $patdircond
mpirun -np 1 condense_vector $patdir/rhs.raw  $case_folder/zd.raw $patdircond/rhs.raw

# Parallel linear solve
mpirun utopia_exec -app ls_solve -A $patdircond/rowptr.raw -b $patdircond/rhs.raw -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out $patdircond/out.raw --verbose

# Post processing of vector
mpirun -np 1 remap_vector $patdircond/out.raw $case_folder/zd.raw $patdir/out.raw

# Convert output from fp64 to fp32
fp_convert.py $patdircond/rhs.raw 	  $pat32dir/rhs.fp32.raw 		float64 float32
fp_convert.py $patdircond/values.raw  $pat32dir/values.fp32.raw  	float64 float32
fp_convert.py $patdircond/sol.raw     $pat32dir/sol.fp32.raw 		float64 float32
fp_convert.py $patdir/sol.raw     	  $pat32dir/full_sol.fp32.raw 	float64 float32

##############
# LAPLSOL
#############
echo "software: LAPLSOL"

# Assemble operator
laplsol-bc $diegodir
laplsol-asm $diegodir

# Parallel linear solve
mpirun laplsol-solve $diegodir $diegodir/rhs.raw $diegodir/sol.raw

# Post processing of solution vector
laplsol-post $diegodir $diegodir/sol.raw

##############
# Compare
#############

# Int
fdiff.py $patdircond/rowptr.raw $diegodir/lhs.rowindex.raw 	int32 int32 1 ./rowptr_pat_vs_diego_rhs.png
fdiff.py $patdircond/colidx.raw $diegodir/lhs.colindex.raw 	int32 int32 1 ./colidx_pat_vs_diego_rhs.png

# FP
fdiff.py $pat32dir/values.fp32.raw 	$diegodir/lhs.value.raw float32 float32 1 ./lhs_pat_vs_diego_rhs.png
fdiff.py $pat32dir/rhs.fp32.raw 	$diegodir/rhs.raw 		float32 float32 1 ./rhs_pat_vs_diego_rhs.png
fdiff.py $pat32dir/full_sol.fp32.raw 	$diegodir/sol.raw  		float32 float32 1 ./sol_pat_vs_diego_sol.png

diffsol.py $diegodir/sol.raw $pat32dir/full_sol.fp32.raw ./diff.fp32.raw

# Remove temporaries
rm -rf $patdir
rm -rf $patdircond
rm -rf $pat32dir
rm -rf $diegodir
