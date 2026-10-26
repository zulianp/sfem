#!/usr/bin/env bash

set -e
set -x

if [ ! -d "./fp64/" ]; then
	mkdir fp64
else
	rm fp64/*.raw
fi

PATH=../../python:$PATH

fp_convert.py downloads/lhs.value.raw fp64/values.raw float32 float64

cp downloads/lhs.colindex.raw fp64/colidx.raw
cp downloads/lhs.rowindex.raw fp64/rowptr.raw
cp downloads/on.raw fp64/on.raw
cp downloads/zd.raw fp64/zd.raw

fp_convert.py downloads/rhs.raw fp64/rhs.raw float32 float64
