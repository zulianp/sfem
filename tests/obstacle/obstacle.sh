#!/usr/bin/env bash

set -e

if [[ -z $SFEM_DIR ]]
then
	echo "SFEM_DIR must be defined with the installation prefix of sfem"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
export PATH=$SFEM_DIR/bin:$PATH
export PATH=$SFEM_DIR/scripts/sfem/mesh:$PATH
export PYTHONPATH=$SFEM_DIR/lib:$SFEM_DIR/scripts:$PYTHONPATH

# Clean-up prior output
rm -rf output
mkdir -p output
mesh=mesh

export OMP_NUM_THREADS=4 
export OMP_PROC_BIND=true 

if [[ -d "$mesh" ]] 
then
	echo "Reusing existing $mesh database"
else
	create_cylinder.sh 1
fi

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

	dims=3
	aos_to_soa $f 8 $dims output/soa/$name
	mv output/soa/$name".0.raw" output/soa/"$var".0."$ts".raw
	mv output/soa/$name".1.raw" output/soa/"$var".1."$ts".raw
	mv output/soa/$name".2.raw" output/soa/"$var".2."$ts".raw
done

raw_to_db.py $mesh out.vtk  \
 --point_data="output/soa/*.raw" 

raw_to_db.py $mesh/surface/outlet obstacle.vtk  \
	--coords=$mesh \
  	--point_data="output/soa/obs.0.raw.raw" 
