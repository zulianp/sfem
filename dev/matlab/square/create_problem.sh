#!/usr/bin/env bash

set -e

../../data/benchmarks/meshes/create_square.sh 5
SFEM_HANDLE_DIRICHLET=0 SFEM_HANDLE_NEUMANN=0 ../../assemble mesh crs