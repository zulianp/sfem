#!/usr/bin/env bash

set -e
set -x

HERE=$PWD

case_folder=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better
solver_exec=/Users/patrickzulian/Desktop/code/utopia/utopia/build/utopia_exec


../python/fdiff.py rhs_pat.raw rhs_diego.raw float64 float32
# ../python/fdiff.py pat.raw diego.raw float64 float32


# time ../python/raw2mesh.py -d $case_folder -f pat.raw --field_dtype=float64
# mv out.vtu pat.vtu

# time ../python/raw2mesh.py -d $case_folder -f diego.raw --field_dtype=float32
# mv out.vtu diego.vtu
