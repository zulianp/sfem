#!/usr/bin/env bash


set -e
# set -x

nx=40
ny=40
n=$(( nx * ny ))

real_pattern="real.0000[0-0].raw"
imag_pattern="imag.0000[0-0].raw"
both_pattern="*.0000[0-0].raw"

max_eigs=$n

[ -d "directed_eigs_2D" ]   && rm -rf directed_eigs_2D   && rm -rf directed_matrix_2D   && rm *data.png && rm *data.raw
[ -d "undirected_eigs_2D" ] && rm -rf undirected_eigs_2D && rm -rf undirected_matrix_2D && rm -rf reconstructed
[ -d "laplacian_eigs_2D" ]  && rm -rf laplacian_eigs_2D  && rm -rf laplacian_matrix_2D
[ -d "odd_eigs_2D" ]  && rm -rf odd_eigs_2D  && rm -rf odd_matrix_2D

# python3 -c "import numpy as np; (np.cos((10*2 * np.pi /("$n")) * np.arange(0, "$n", dtype=np.float64))).tofile('rdata.raw')"
python3 -c "import numpy as np; X = np.arange(0, "$nx"); Y = np.arange(0, "$ny"); X, Y = np.meshgrid(X, Y); angles=(2 * np.pi /("$nx") * X); np.cos(angles).tofile('rdata.raw')"

./plot_surface.py "rdata.raw" $nx $ny 1 "orginal_rdata.png"

function analyze()
{
	local name=$1
	local type=$2
	./create_adj_matrix_2D.py $name $nx $ny $name"_matrix_2D"
	./dense_graph_analysis.py $name"_matrix_2D" $type $name"_eigs_2D"

	./plot_surface.py $name"_eigs_2D/$real_pattern" $nx $ny $max_eigs $name"_eigs_2D/real_dvecs.png"
	./plot_surface.py $name"_eigs_2D/$imag_pattern" $nx $ny $max_eigs $name"_eigs_2D/imag_dvecs.png"
	./plot_surface.py $name"_eigs_2D/$both_pattern" $nx $ny $max_eigs $name"_eigs_2D/both_dvecs.png"

	cp rdata.raw data.raw
	./project.py $name"_eigs_2D/real.*.raw" -1 0 data.raw

	./plot_surface.py "reconstructed/rdata.raw" $nx $ny 1 $name"_rdata.png"
	./plot_surface.py "reconstructed/idata.raw" $nx $ny 1 $name"_idata.png"
}

analyze directed SM
# analyze undirected LR
# analyze laplacian SR
# analyze laplacian SM
# analyze odd angle

# open *rdata.png
# open *idata.png