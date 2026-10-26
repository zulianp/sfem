#!/usr/bin/env bash

#SBATCH -J PerfHydroS
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --exclusive
#SBATCH --mem=12020

set -e
#########################################
# Inputs
#########################################

sfemfp32=1
case_folder=/scratch/zulian/xdns/fe_hydros/sfem/tests/compare/mesh-multi-outlet-better

# Libraries
laplsoldir=/scratch/diegor/zu/laplsol/
sfemdir=/scratch/zulian/xdns/fe_hydros/sfem/
utopiadir=./

#########################################

module purge
module load petsc/3.13.5_gcc-10.1.0
module load git
module load cmake
module load boost

module list

# pip3 install --user matplotlib

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
diego64dir=$diegodir/fp64
mkdir $diego64dir

# Copy some stuff
# Coordinates
cp $case_folder/x.raw $diegodir/x.raw
cp $case_folder/y.raw $diegodir/y.raw
cp $case_folder/z.raw $diegodir/z.raw
# Mesh
cp $case_folder/i0.raw $diegodir/i0.raw
cp $case_folder/i1.raw $diegodir/i1.raw
cp $case_folder/i2.raw $diegodir/i2.raw
cp $case_folder/i3.raw $diegodir/i3.raw

# Boundary conditions
cp $case_folder/on.raw $diegodir/on.raw
cp $case_folder/zd.raw $diegodir/zd.raw

##############
# SFEM
#############
echo "software: SFEM"

# Assemble operator
mpirun -np 1 assemble $case_folder $patdir

# Remove dirichlet nodes
mpirun -np 1 condense_matrix $patdir 	      $case_folder/zd.raw $patdircond
mpirun -np 1 condense_vector $patdir/rhs.raw  $case_folder/zd.raw $patdircond/rhs.raw

# Convert output from fp64 to fp32
fp_convert.py $patdircond/rhs.raw 	  $pat32dir/rhs.fp32.raw 		float64 float32
fp_convert.py $patdircond/values.raw  $pat32dir/values.fp32.raw  	float64 float32

# Convert back to fp64 (for better comparison with fp32 software)
if [ "$sfemfp32" -eq "1" ]; then
	fp_convert.py  $pat32dir/rhs.fp32.raw 	 $patdircond/rhs.raw 		float32 float64
	fp_convert.py  $pat32dir/values.fp32.raw $patdircond/values.raw  	float32 float64
fi

# Parallel linear solve
usolve.sh $patdircond/rowptr.raw $patdircond/rhs.raw $patdircond/sol.raw

# Post processing of vector
mpirun -np 1 remap_vector $patdircond/sol.raw $case_folder/zd.raw $patdir/sol.raw

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

fp_convert.py $diegodir/lhs.value.raw  $diego64dir/values.raw float32 float64
fp_convert.py $diegodir/rhs.raw   	   $diego64dir/rhs.raw 	  float32 float64

cp $diegodir/lhs.rowindex.raw $diego64dir/rowptr.raw
cp $diegodir/lhs.colindex.raw $diego64dir/colidx.raw

usolve.sh $diego64dir/rowptr.raw  $diego64dir/rhs.raw $diego64dir/sol.raw
mpirun -np 1 remap_vector $diego64dir/sol.raw $case_folder/zd.raw $diego64dir/sol.full.raw
fp_convert.py $diego64dir/sol.full.raw $diegodir/sol.amg.f32.raw float64 float32

##############
# Compare
#############

echo "COMPARE"
echo "SFEM files"
ls -la $pat32dir/

echo "LAPLSOL files"
ls -la $diegodir

# Int
fdiff.py $patdircond/rowptr.raw $diegodir/lhs.rowindex.raw 	int32 int32 1 ./rowptr_pat_vs_diego_rhs.png
fdiff.py $patdircond/colidx.raw $diegodir/lhs.colindex.raw 	int32 int32 1 ./colidx_pat_vs_diego_rhs.png

# FP
fdiff.py $pat32dir/values.fp32.raw 	 $diegodir/lhs.value.raw 		float32 float32 1 ./lhs_pat_vs_diego_rhs.png
fdiff.py $pat32dir/rhs.fp32.raw 	 $diegodir/rhs.raw 				float32 float32 1 ./rhs_pat_vs_diego_rhs.png
fdiff.py $pat32dir/sol.fp32.raw 	 $diegodir/sol.raw 				float32 float32 1 ./sol_pat_vs_diego_sol.png
fdiff.py $pat32dir/full_sol.fp32.raw $diegodir/p.raw				float32 float32 1 ./full_sol_pat_vs_diego_sol.png

diffsol.py $diegodir/p.raw $pat32dir/full_sol.fp32.raw ./diff.fp32.raw
diffsol.py $diegodir/p.raw $diegodir/sol.amg.f32.raw   ./diff.amg.fp32.raw

# Remove temporaries
rm -rf $patdir
rm -rf $patdircond
rm -rf $pat32dir
rm -rf $diegodir
