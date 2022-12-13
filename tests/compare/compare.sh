#!/usr/bin/env bash

set -e
# set -x

HERE=$PWD

case_folder=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better
solver_exec=/Users/patrickzulian/Desktop/code/utopia/utopia/build/utopia_exec
patsystem=/Users/patrickzulian/Desktop/code/sfem/mesh-multi-outlet-better/

PATH=../../python:$PATH

if [ ! -d "./pat/" ]; then
	mkdir ./pat
else
	rm ./pat/*.raw
fi

if [ ! -d "./figures/" ]; then
	mkdir ./figures
else 
	rm ./figures/*.png
fi

if [ ! -d "./diego/" ]; then
	mkdir ./diego
fi

fp_convert.py $patsystem/condensed/rhs.raw 	   pat/rhs.fp32.raw 	float64 float32
fp_convert.py $patsystem/condensed/values.raw  pat/values.fp32.raw  float64 float32
fp_convert.py $patsystem/condensed/sol.raw     pat/sol.fp32.raw 	float64 float32
fp_convert.py $patsystem/condensed/sol.raw     pat/sol.fp32.raw 	float64 float32
fp_convert.py fp64/sol.raw  				   diego/sol.fp32.raw 	float64 float32

# Int
fdiff.py $patsystem/condensed/rowptr.raw downloads/lhs.rowindex.raw 	int32 int32 1 figures/rowptr_pat_vs_diego_rhs.png
fdiff.py $patsystem/condensed/colidx.raw downloads/lhs.colindex.raw 	int32 int32 1 figures/colidx_pat_vs_diego_rhs.png

# FP
fdiff.py pat/values.fp32.raw 	downloads/lhs.value.raw 	float32 float32 1 figures/lhs_pat_vs_diego_rhs.png
fdiff.py pat/rhs.fp32.raw 		downloads/rhs.raw 			float32 float32 1 figures/rhs_pat_vs_diego_rhs.png
fdiff.py pat/sol.fp32.raw 		diego/sol.fp32.raw  		float32 float32 1 figures/sol_pat_vs_diego_sol.png
fdiff.py diego/sol.fp32.raw 	downloads/p.raw 			float32 float32 1 figures/sol_diego_vs_diego_sol.png
