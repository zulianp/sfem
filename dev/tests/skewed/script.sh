#!/usr/bin/env bash

set -e
set -x

touch on.raw zd.raw

if [ ! -d "./out/" ]; then
	mkdir ./out
fi

../../python/create_test_mesh.py . 2.0 1.0 1.0
../../assemble . ./out
../../../matrix.io/print_crs out/rowptr.raw out/colidx.raw out/values.raw int int double
../../../matrix.io/print_array out/rhs.raw double

./oracle.py
