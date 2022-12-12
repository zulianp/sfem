#!/usr/bin/env bash

set -e
set -x

../../python/create_test_mesh.py . 1.0
../../assemble . ./out
../../../matrix.io/print_crs out/rowptr.raw out/colidx.raw out/values.raw int int double
../../../matrix.io/print_array out/rhs.raw double
