#!/usr/bin/env bash

set -e

host=`hostname -s`

if [[ "$host" == "daint-ln002" ]];
then
	echo "$host"
	export OMP_NUM_THREADS=72
	export OMP_PROC_BIND=true
else
	echo "$host"
fi

set -x

rm -rf output

SFEM_EXECUTION_SPACE=device     \
SFEM_COARSE_OP_TYPE=MF 			\
SFEM_MAX_INNER_IT=3 			\
SFEM_ENABLE_LINE_SEARCH=1		\
SFEM_MAX_IT=120					\
SFEM_ATOL=1e-7					\
SFEM_ELEMENT_REFINE_LEVEL=4	    \
SFEM_STAGNATION_THRESHOLD=10	\
SFEM_PENALTY_PARAM=100			\
SFEM_NL_SMOOTH_STEPS=27			\
SMESH_TRACE_FILE=obs.csv 		\
	$LAUNCH obs rock ./sdf dirichlet.yaml rock/contact_boundary output

rm -f output/out/contact_stress.{1,2}.*
rm -f output/out/rhs.{0,1,2}.*

SFEM_TRACE_FILE=hex8_cauchy_stress.csv \
	hex8_cauchy_stress output/mesh 1 1 output/out/disp.0.float64 output/out/disp.1.float64 output/out/disp.2.float64 output/out/cauchy_stress

raw_to_db output/mesh output.vtk -p 'output/out/*.float64' $EXTRA_OPTIONS
raw_to_db output/coarse_mesh macro_mesh.vtk
