#!/usr/bin/env bash


set -e
# set -x

n=60

real_pattern="real.0000[0-3].raw"
imag_pattern="imag.0000[0-3].raw"
both_pattern="*.0000[0-3].raw"

max_eigs=$n

[ -d "directed_eigs" ]   && rm -rf directed_eigs   && rm -rf directed_matrix_1D   && rm *data.png && rm *data.raw
[ -d "undirected_eigs" ] && rm -rf undirected_eigs && rm -rf undirected_matrix_1D && rm -rf reconstructed
[ -d "laplacian_eigs" ]  && rm -rf laplacian_eigs  && rm -rf laplacian_matrix_1D

# python3 -c "import numpy as np; (np.cos((10*2 * np.pi /("$n")) * np.arange(0, "$n", dtype=np.float64))).tofile('rdata.raw')"
python3 -c "import numpy as np; angles=(2 * np.pi /("$n")) * np.arange(0, "$n", dtype=np.float64); (np.cos(2*angles) + np.sin(3*angles)).tofile('rdata.raw')"

./plot_vectors.py "rdata.raw" 1 "orginal_rdata.png"

function analyze()
{
	local name=$1
	local type=$2
	./create_adj_matrix.py $name $n $name"_matrix_1D"
	./dense_graph_analysis.py $name"_matrix_1D" $type $name"_eigs"

	./plot_vectors.py $name"_eigs/$real_pattern" $max_eigs $name"_eigs/real_dvecs.png"
	./plot_vectors.py $name"_eigs/$imag_pattern" $max_eigs $name"_eigs/imag_dvecs.png"
	./plot_vectors.py $name"_eigs/$both_pattern" $max_eigs $name"_eigs/both_dvecs.png"

	cp rdata.raw data.raw
	./project.py $name"_eigs/real.*.raw" -1 0 data.raw

	./plot_vectors.py "reconstructed/rdata.raw" 1 $name"_rdata.png"
	./plot_vectors.py "reconstructed/idata.raw" 1 $name"_idata.png"
}

analyze directed angle
analyze undirected LR
analyze laplacian SR

# open *rdata.png
# open *idata.png