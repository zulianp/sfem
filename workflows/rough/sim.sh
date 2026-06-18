#!/usr/bin/env bash

set -e



 # export PATH=$INSTALL_DIR/sfem/bin:$PATH

SFEM_MAX_INNER_IT=4 \
SFEM_SDF_SCALE=-1 				\
SFEM_ENABLE_LINE_SEARCH=1		\
SFEM_MAX_IT=20					\
SFEM_ATOL=1e-7					\
SFEM_ELEMENT_REFINE_LEVEL=16		\
SFEM_STAGNATION_THRESHOLD=10	\
SFEM_PENALTY_PARAM=100			\
SFEM_NL_SMOOTH_STEPS=27			\
SFEM_TRACE_FILE=obs.csv 		\
	$LAUNCH obs rock ./sdf  dirichlet.yaml rock/contact_boundary output

raw_to_db output/mesh output.vtk -p 'output/out/*.*' 

# # They are all zeros
# rm -f output/out/contact_stress.{1,2}.raw
# rm -f output/out/rhs.{0,1,2}.raw

# mv output/mesh/x0.raw output/mesh/x.raw
# mv output/mesh/x1.raw output/mesh/y.raw
# mv output/mesh/x2.raw output/mesh/z.raw

# SFEM_TRACE_FILE=hex8_cauchy_stress.csv \
# 	hex8_cauchy_stress output/mesh 1 1 output/out/disp.0.raw output/out/disp.1.raw output/out/disp.2.raw output/out/cauchy_stress

# raw_to_db.py output/mesh output.vtk -p 'output/out/*.raw' $EXTRA_OPTIONS
# raw_to_db.py output/coarse_mesh macro_mesh.vtk
