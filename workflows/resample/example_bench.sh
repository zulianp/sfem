#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../build/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

NCORES=8

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=$NCORES

field=field.raw

mesh=mesh
# mesh=impeller_tet4

out=resampled
skinned=skinned
sdf=sdf.float32.raw

mesh_sorted=sorted
# mesh_sorted=impeller_tet4

resample_target=$mesh_sorted

# resample_target=$skinned

if [[ -d "refined" ]]
then
	resample_target=refined
	echo "resample_target=$resample_target"
fi

if [[ -d "$skinned" ]] 
then
	echo "Reusing existing mesh $skinned and SDF!"
else
	create_sphere.sh 5
	
	refine $mesh refined
	# sfc $mesh $mesh_sorted
	sfc refined $mesh_sorted

	mkdir -p $skinned
	skin $mesh $skinned
	# mesh_to_sdf.py $skinned $sdf --hmax=0.01 --margin=0.1
	mesh_to_sdf.py $skinned $sdf --hmax=0.1 --margin=1
	# raw_to_xdmf.py $sdf
fi

## raw_to_xdmf.py $sdf
sdf_test.py $sdf

sizes=`head -3 metadata_sdf.float32.yml 			  | awk '{print $2}' | tr '\n' ' '`
origins=`head -8 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`
scaling=`head -11 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`

echo $sizes
echo $origins
echo $scaling


# n_procs=2
# n_procs=8

Nsight_PATH="/home/sriva/App/NVIDIA-Nsight-Compute-2024.3/"
Nsight_OUTPUT="/home/sriva/App/NVidia_prof_out/ncu_grid_to_mesh"


# LAUNCH="${Nsight_PATH}/ncu  --set roofline --print-details body  -f --section ComputeWorkloadAnalysis -o ${Nsight_OUTPUT} "
# LAUNCH="srun -p debug -n $n_procs -N 1 "
# LAUNCH=""

GRID_TO_MESH="grid_to_mesh"
#GRID_TO_MESH="perf record -o /tmp/out.perf grid_to_mesh"

# LAUNCH="lldb --"

# export OMP_NUM_THREADS=8
# export OMP_PROC_BIND=true

set -x

output_file="output_bench.log"
bench_file="tet4_bench.csv"

n_proc_max=18

export SFEM_INTERPOLATE=0
export SFEM_READ_FP32=1

for n_procs in $(seq 1 $n_proc_max); do
    LAUNCH="mpiexec -np $n_procs "
    $LAUNCH $GRID_TO_MESH $sizes $origins $scaling $sdf $resample_target $field TET4 CUDA > "$output_file" 2>&1

	if [ $n_procs -eq 1 ]; then
        # First iteration: capture lines beginning with <BenchH> and append to bench_file
        grep '^<BenchH>' "$output_file" | sed 's/^<BenchH>//' >> "$bench_file"
    fi

    grep '^<BenchR>' "$output_file" | sed 's/^<BenchR>//' >> "$bench_file"
done


raw_to_db.py $resample_target out.vtk --point_data=$field  --point_data_type=float32

