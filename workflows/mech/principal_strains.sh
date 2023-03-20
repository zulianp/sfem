
#!/usr/bin/env bash

set -e


SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH

if (($# != 2))
then
	printf "usage: $0 <simulation_db.e> <output>\n" 1>&2
	exit -1
fi

simulation_db=$1
output_folder=$2
material="neohookean"

data_folder=$output_folder/data

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

mkdir -p $output_folder

exodusII_to_raw.py $simulation_db $data_folder

dispx=(`ls $data_folder/point_data/disp_x*.raw`)
dispy=(`ls $data_folder/point_data/disp_y*.raw`)
dispz=(`ls $data_folder/point_data/disp_z*.raw`)

ndisps=${#dispx[@]}
max_steps=$ndisps
# max_steps=100

echo $ndisps

# set -x
padding=`python3 -c "import numpy as np; print(int(np.ceil(np.log10("$ndisps"))))"`

p0=$output_folder/p0
p1=$output_folder/p1

mkdir -p $p0
mkdir -p $p1

nsteps=`python3 -c "print(min("$ndisps","$max_steps"))"`

echo "Processing $nsteps steps out of $ndisps!"

for (( i=0; i < $nsteps; i++ ))
do
	ux=${dispx[$i]}
	uy=${dispy[$i]}
	uz=${dispz[$i]}

	principal_strain_prefix=$p0"/principal_strain"
	principal_strain_postfix=`printf ".%0."$padding"d" $i`

	echo $principal_strain_prefix
	echo $principal_strain_postfix

	SFEM_OUTPUT_POSTFIX=$principal_strain_postfix cprincipal_strains $mesh_folder $ux $uy $uz $principal_strain_prefix

	for(( d = 0; d < 3; d++ ))
	do
		p0_file=$principal_strain_prefix".$d"$principal_strain_postfix".raw"
		p1_file=$p1"/principal_strain.$d"$principal_strain_postfix".raw"
		projection_p0_to_p1 $mesh_folder $p0_file $p1_file
	done
done

disp_selector="$data_folder/point_data/disp_x*.raw,$data_folder/point_data/disp_y*.raw,$data_folder/point_data/disp_z*.raw"
principal_strain_selector_0="$p1/principal_strain.0.*.raw,$p1/principal_strain.1.*.raw,$p1/principal_strain.2.*.raw"


raw_to_db.py $mesh_folder principal_strain.xmf  \
 --transient --n_time_steps=$nsteps \
 --point_data="$disp_selector,$principal_strain_selector_0"
