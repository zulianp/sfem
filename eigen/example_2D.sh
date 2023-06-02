#!/usr/bin/env bash


set -e
# set -x

nx=40
ny=40
n=$(( nx * ny ))

nmax=7
real_pattern="real.0000[0-$nmax].raw"
imag_pattern="imag.0000[0-$nmax].raw"
both_pattern="*.0000[0-$nmax].raw"

# real_pattern="real.*.raw"
# imag_pattern="imag.*.raw"
# both_pattern="*.*.raw"

max_eigs=$n

[ -d "directed_eigs_2D" ]   && rm -rf directed_eigs_2D   && rm -rf directed_matrix_2D   && rm *data.png && rm *data.raw && rm -rf eigs_2D  
[ -d "undirected_eigs_2D" ] && rm -rf undirected_eigs_2D && rm -rf undirected_matrix_2D && rm -rf reconstructed
[ -d "laplacian_eigs_2D" ]  && rm -rf laplacian_eigs_2D  && rm -rf laplacian_matrix_2D
[ -d "odd_eigs_2D" ]  && rm -rf odd_eigs_2D  && rm -rf odd_matrix_2D

./example_fun_2D.py $nx $ny "rdata.raw"

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

	./plot_images.py $name"_eigs_2D/$real_pattern" $nx $ny $max_eigs $name"_eigs_2D/image_real_dvecs.png"
	./plot_images.py $name"_eigs_2D/$imag_pattern" $nx $ny $max_eigs $name"_eigs_2D/image_imag_dvecs.png"
	# ./plot_images.py $name"_eigs_2D/$both_pattern" $nx $ny $max_eigs $name"_eigs_2D/image_both_dvecs.png"

	cp rdata.raw data.raw
	./project.py $name"_eigs_2D/real.*.raw" -1 0 data.raw

	./plot_surface.py "reconstructed/rdata.raw" $nx $ny 1 $name"_rdata.png"
	./plot_surface.py "reconstructed/idata.raw" $nx $ny 1 $name"_idata.png"
	./plot_surface.py "reconstructed/mdata.raw" $nx $ny 1 $name"_mdata.png"
}

# analyze yonly angle
analyze directed angle

# analyze undirected LR
# analyze laplacian SM
# analyze odd angle

# open *rdata.png
# open *idata.png