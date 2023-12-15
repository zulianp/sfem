#!/usr/bin/env bash

if [[ -z "$MATRIXIO_DIR" ]]
then
	echo "Define MATRIXIO_DIR"
	exit 1
fi

export PATH=$MATRIXIO_DIR/python:$PATH

mesh=$1
idx_t=int32

idx=`ls $mesh/i*.raw`
blocks=`ls $mesh/blocks/*.int64.raw`

for b in ${blocks[@]}
do
	name=`basename $b | tr '.' ' ' | awk '{print $1}'`
	echo $name

	mkdir -p $mesh/blocks/$name

	range=`python3 -c "import numpy as np; a=np.fromfile('$b', dtype=np.int64); print(f'{a[0]} {a[1]}');"`

	for i in ${idx[@]}
	do
		idx_name=`basename $i`
		echo "rgather.py $range $idx_t $i "$mesh/blocks/$name/$idx_name".raw"
		rgather.py $range $idx_t $i "$mesh/blocks/$name/$idx_name".raw
	done
done