#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SCRIPTPATH/../:$PATH
export PATH=$SCRIPTPATH/../build/:$PATH
export PATH=$SCRIPTPATH/../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python/sfem:$PATH
PATH=$SCRIPTPATH/../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../matrix.io:$PATH

mesh=$1

rm -rf output
mkdir -p output

export OMP_NUM_THREADS=4 
export OMP_PROC_BIND=true 

dims=3
./test_obstacle.py gen:box output

files=`ls output/disp.*.raw`
mkdir -p output/soa

for f in ${files[@]}
do
	name=`basename $f`
	var=`echo $name | tr '.' ' ' | awk '{print $1}'`
	ts=`echo $name  | tr '.' ' ' | awk '{print $2}'`

	aos_to_soa $f 8 $dims output/soa/$name
	mv output/soa/$name".0.raw" output/soa/"$var".0."$ts".raw
	mv output/soa/$name".1.raw" output/soa/"$var".1."$ts".raw
	mv output/soa/$name".2.raw" output/soa/"$var".2."$ts".raw
done

raw_to_db.py output/rect_mesh out.xmf  \
 --transient \
 --point_data="output/soa/disp.0.*.raw,output/soa/disp.1.*.raw,output/soa/disp.2.*.raw" \
 --time_whole_txt="output/time.txt"
