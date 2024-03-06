#!/usr/bin/env bash

set -e


SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH

if (($# != 4))
then
	printf "usage: $0 <simulation_db.e> <workspace_folder> <mu> <lambda>\n" 1>&2
	exit -1
fi

simulation_db=$1
workspace_folder=$2
mu=$3
lambda=$4
material="neohookean"


data_folder=$workspace_folder/data

platform=`uname -s`
if [[ "$platform" -eq "Darwin" ]]
then
	mesh_folder=$data_folder
else
	# On Linux: put mesh in shared memory for performance since we read it multiple times
	export TMPDIR=/dev/shm
	mesh_folder=`mktemp -d`
	cp -r $data_folder/*.raw $mesh_folder
	garbage="$mesh_folder"
fi

mkdir -p $workspace_folder

exodusII_to_raw.py $simulation_db $data_folder

dispx=(`ls $data_folder/point_data/disp_x*.raw`)
dispy=(`ls $data_folder/point_data/disp_y*.raw`)
dispz=(`ls $data_folder/point_data/disp_z*.raw`)

ndisps=${#dispx[@]}
max_steps=$ndisps
# max_steps=20

echo $ndisps

# set -x
padding=`python3 -c "import numpy as np; print(int(np.ceil(np.log10("$ndisps"))))"`

p0=$workspace_folder/p0
p1=$workspace_folder/p1

mkdir -p $p0
mkdir -p $p1

nsteps=`python3 -c "print(min("$ndisps","$max_steps"))"`

echo "Processing $nsteps steps out of $ndisps!"

for (( i=0; i < $nsteps; i++ ))
do
	ux=${dispx[$i]}
	uy=${dispy[$i]}
	uz=${dispz[$i]}

	stress_prefix=$p0"/principal_stress"
	stress_postfix=`printf ".%0."$padding"d" $i`

	echo $stress_prefix
	echo $stress_postfix

	SFEM_OUTPUT_POSTFIX=$stress_postfix cprincipal_stresses $material $mu $lambda $mesh_folder $ux $uy $uz $stress_prefix

	for(( d=0; d < 3; d++ ))
	do
		p0_file=$stress_prefix".$d"$stress_postfix".raw"
		p1_file=$p1"/principal_stress.$d"$stress_postfix".raw"
		projection_p0_to_p1 $mesh_folder $p0_file $p1_file
	done
done

disp_selector="$data_folder/point_data/disp_x*.raw,$data_folder/point_data/disp_y*.raw,$data_folder/point_data/disp_z*.raw"
stress_selector_0="$p1/principal_stress.0.*.raw,$p1/principal_stress.1.*.raw,$p1/principal_stress.2.*.raw"
# stress_selector_1="$p0/principal_stress.0.*.raw,$p0/principal_stress.1.*.raw,$p0/principal_stress.2.*.raw"

set -x
raw_to_db.py $mesh_folder principal_stress.xmf  \
 --transient --n_time_steps=$nsteps \
 --point_data="$disp_selector,$stress_selector_0" \
 --time_whole="$mesh_folder/time_whole.raw" 

if [[ -z "$garbage" ]]
then
	echo "No garbage to clean"
else
	echo "Removing garbage = $garbage"
	rm -r $garbage
fi

