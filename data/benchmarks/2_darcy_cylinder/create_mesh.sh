#!/usr/bin/env bash


set -e
set -x


folder=cylinder
mkdir -p $folder
nrefs=1

./cylinder $folder/mesh.vtk $nrefs
