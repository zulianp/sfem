#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
export PATH=$SCRIPTPATH/../../../matrix.io:$PATH

if [[ -z $SFEM_BIN_DIR ]]
then
	PATH=$SCRIPTPATH/../../build:$PATH
else
	PATH=$SFEM_BIN_DIR:$PATH
fi

mesh=mesh

rm -rf output
mkdir -p output

export OMP_NUM_THREADS=4 
export OMP_PROC_BIND=true 

if [[ -d "$mesh" ]] 
then
	echo "Reusing existing $mesh database"
else
	create_cylinder.sh 1
fi

dims=3
# python3 -m cProfile  -s time 

echo  "---------------------------------"
echo "Solving ostacle problem"
echo  "---------------------------------"
time ./obstacle.py $mesh output
echo  "---------------------------------"

files=`ls output/*.raw`
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

raw_to_db.py $mesh out.vtk  \
 --point_data="output/soa/*.raw" 
