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
source $SFEM_DIR/workflows/sfem_config.sh

# Clean-up prior output
rm -rf output
mkdir -p output
mesh=mesh

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 


CASE=Box3D
# CASE=Cylinder
PROBLEM=LinearElasticity
# PROBLEM="Laplacian"
REFINEMENT=0

case $CASE in
	Cylinder)
		DIMS=3
		if [[ -d "$CASE" ]] 
		then
			echo "Reusing existing $CASE database"
		else
			mkdir workspace
			cd workspace
			create_cylinder.sh $REFINEMENT
			cd -
			mv workspace/mesh $CASE
			rm -rf workspace
		fi
	;;
	Box3D)
		DIMS=3
		if [[ -d "$CASE" ]] 
		then
			echo "Reusing existing $CASE database"
		else
			mkdir workspace
			cd workspace
			N=4
			box_mesh.py mesh -c tetra -x $N -y $N -z $N --height=1 --width=1 --depth=1
			cd -
			mv workspace/mesh $CASE
			raw_to_db.py $CASE $CASE/mesh.vtk

			rm -rf workspace
		fi
	;;
	*)
			echo "Wrong case: $CASE"
			exit 1
	;;
esac

export SFEM_FIRST_LAME_PARAMETER=3.333
export SFEM_SHEAR_MODULUS=0.357

echo  "---------------------------------"
echo "Solving ostacle problem"
echo  "---------------------------------"
time ./obstacle.py $CASE --problem=$PROBLEM --output=output
echo  "---------------------------------"

if [[ "$PROBLEM" == "LinearElasticity" ]]
then
	files=`ls output/*.raw`
	mkdir -p output/soa

	for f in ${files[@]}
	do
		name=`basename $f`
		var=`echo $name | tr '.' ' ' | awk '{print $1}'`
		ts=`echo $name  | tr '.' ' ' | awk '{print $2}'`

		
		aos_to_soa $f $SFEM_REAL_SIZE $DIMS output/soa/$name
		mv output/soa/$name".0.raw" output/soa/"$var".0."$ts".raw
		mv output/soa/$name".1.raw" output/soa/"$var".1."$ts".raw

		if [[ $DIMS == 3 ]]
		then
			mv output/soa/$name".2.raw" output/soa/"$var".2."$ts".raw
		fi
	done

	set -x

	raw_to_db.py $CASE out.vtk  \
	 --point_data="output/soa/*.raw" --point_data_type="$SFEM_REAL_T"

	raw_to_db.py $CASE/surface/outlet obstacle.vtk  \
		--coords=$CASE \
	  	--point_data="output/soa/obs.0.raw.raw" \
	  	--point_data_type="$SFEM_REAL_T"

else
	raw_to_db.py $CASE out.vtk  \
	 --point_data="output/*.raw" --point_data_type="$SFEM_REAL_T"
fi
