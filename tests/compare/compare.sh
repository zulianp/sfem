#!/usr/bin/env bash

set -e
set -x

HERE=$PWD

case_folder=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better
solver_exec=/Users/patrickzulian/Desktop/code/utopia/utopia/build/utopia_exec
patsystem=/Users/patrickzulian/Desktop/code/sfem/mesh-multi-outlet-better/condensed/

PATH=../../python:$PATH

if [ ! -d "./pat/" ]; then
	mkdir ./pat
else
	rm ./pat/*.raw
fi

if [ ! -d "./figures/" ]; then
	mkdir ./figures
fi

fp_convert.py $patsystem/rhs.raw 	 pat/rhs.fp32.raw 	  float64 float32
fp_convert.py $patsystem/values.raw  pat/values.fp32.raw  float64 float32
fp_convert.py $patsystem/out.raw     pat/sol.fp32.raw 	  float64 float32

fdiff.py pat/rhs.fp32.raw downloads/rhs.raw float32 float32 1 figures/pat_vs_diego_rhs.png
fdiff.py pat/sol.fp32.raw downloads/sol.raw float32 float32 1 figures/pat_vs_diego_sol.png
# fdiff.py pat/sol.fp32.raw downloads/sol.raw float32 float32 1 figures/pat_vs_diego_sol.png
