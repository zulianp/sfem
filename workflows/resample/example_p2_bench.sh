#!/usr/bin/env bash


LAUNCH_TEST="srun --nodes=1 --ntasks=1 " ### FOR HPC with SLURM ###
LAUNCH_TEST=" "

search_name_number_in_args() {
    local name="$1"
    shift
    for arg in "$@"
    do
        if [[ "$arg" =~ ^${name}([0-9]+)$ ]]
        then
            echo "${BASH_REMATCH[1]}"
            return 0
        fi
    done
    return 1
}

search_string_in_args() {
    local search_string="$1"
    shift
    for arg in "$@"
    do
        if [[ "$arg" == "$search_string" ]]
        then
            return 0
        fi
    done
    return 1
}

if search_name_number_in_args "np" "$@"
then
	n_procs=${BASH_REMATCH[1]}
else
	n_procs=1
fi

echo "example_p2.sh: n_procs=$n_procs"

export USE_MPI=0
export USE_MPI_GH200=0
export USE_MPI_NORMAL=0
export USE_NSIGNT=0

export FLOAT_64=0

export PERF="no"

if search_string_in_args "mpi" "$@"
then
    export USE_MPI=1
fi
    
if search_string_in_args "mpi_gh200" "$@"
then
	export USE_MPI_GH200=1
fi

if search_string_in_args "mpi_normal" "$@"
then
	export USE_MPI_NORMAL=1
fi

if search_string_in_args "perf" "$@"
then
	export PERF="yes"
fi

if search_string_in_args "nsight" "$@"
then
	export USE_NSIGNT=1
fi

if search_string_in_args "f64" "$@"
then
	export FLOAT_64=1
	echo "example_p2.sh: Using float64"
fi

# launcher

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

field=field.raw
mesh=mesh
p2_mesh=p2_mesh
out=resampled
skinned=skinned
sdf=sdf.float32.raw
mesh_sorted=sorted
resample_target=$p2_mesh

if [[ -d "refined" ]]
then
	resample_target=refined
	echo "resample_target=$resample_target"
fi

if [[ -d "$p2_mesh" ]] 
then
	echo "Reusing existing mesh $p2_mesh!"
else
	# create_sphere.sh 5
	
	create_sphere.sh 5 # Visibily see the curvy surface <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	${LAUNCH_TEST} sfc $mesh temp_mesh
	
	export SFEM_ORDER_WITH_COORDINATE=2 
	${LAUNCH_TEST} sfc temp_mesh $mesh_sorted
	
	# Project p2 nodes to sphere isosurfaces (to check if nonlinear map are creating errors)
	export SFEM_SPERE_TOL=1e-5 
	export SFEM_MAP_TO_SPHERE=1 
	${LAUNCH_TEST} mesh_p1_to_p2 $mesh_sorted $p2_mesh

	${LAUNCH_TEST} raw_to_db.py $p2_mesh test_mapping.vtk 
fi

if [[ -f "$sdf" ]]
then
	echo "Reusing existing sdf $sdf!"
else
	echo "Computing SDF!"
	mkdir -p $skinned
	${LAUNCH_TEST} skin $mesh_sorted $skinned
	${LAUNCH_TEST} mesh_to_sdf.py $skinned $sdf --hmax=0.01 --margin=0.1
	${LAUNCH_TEST} raw_to_xdmf.py $sdf
fi

${LAUNCH_TEST} sdf_test.py $sdf
# raw_to_xdmf.py $sdf

sizes=`head -3 metadata_sdf.float32.yml 			  | awk '{print $2}' | tr '\n' ' '`
origins=`head -8 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`
scaling=`head -11 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`

echo $sizes
echo $origins
echo $scaling


# export OMP_PROC_BIND=true
# export OMP_NUM_THREADS=8

Nsight_PATH="/home/sriva/App/NVIDIA-Nsight-Compute-2024.3/"
Nsight_OUTPUT="/home/sriva/App/NVidia_prof_out/ncu_grid_to_mesh"

if [[ "$USE_MPI" == "1" ]]
then
	LAUNCH="mpiexec -np $n_procs"
elif [[ "$USE_MPI_NORMAL" == "1" ]]
then
	LAUNCH="srun -n $n_procs -p debug "
elif [[ "$USE_MPI_GH200" == "1" ]]
then
	LAUNCH="srun -n $n_procs -p gh200 "
elif [[ "$USE_NSIGNT" == "1" ]]
then
	LAUNCH="${Nsight_PATH}/ncu --set roofline --print-details body  -f --section ComputeWorkloadAnalysis -o ${Nsight_OUTPUT} "
else
	LAUNCH=""
fi

# GRID_TO_MESH="perf record -o /tmp/out.perf grid_to_mesh"
GRID_TO_MESH="grid_to_mesh"

# chack if PERF == yes
if [[ "$PERF" == "yes" ]]
then
	GRID_TO_MESH="perf record -o /tmp/out.perf $GRID_TO_MESH"
	LAUNCH=""
fi


# To enable iso-parametric transformation of p2 meshes
# for the resampling

# Enable second order mesh parametrizations
export SFEM_ENABLE_ISOPARAMETRIC=1

output_file="output_Throughput.log"
bench_file="tet10_bench_p2.csv"

export SFEM_INTERPOLATE=0
export SFEM_READ_FP32=1

set -x

rm -vf $output_file $bench_file

################ run benchmark ################

n_proc_max=18

for n_procs in $(seq 1 $n_proc_max); do
	LAUNCH="srun --nodes=1 --ntasks=$n_procs " ### FOR HPC with SLURM ###
    LAUNCH="mpiexec -np $n_procs " ### 
    $LAUNCH $GRID_TO_MESH $sizes $origins $scaling $sdf $resample_target $field TET10 CUDA write

	if [ $n_procs -eq 1 ]; then
        # First iteration: capture lines beginning with <BenchH> and write to bench_file
        grep '^<BenchH>' "$output_file" | sed 's/^<BenchH>//' >> "$bench_file"
    fi

    grep '^<BenchR>' "$output_file" | sed 's/^<BenchR>//' >> "$bench_file"
done

if [[ "$FLOAT_64" == "1" ]]
then
	raw_to_db.py $resample_target out.vtk --point_data=$field --point_data_type=float64
else
	raw_to_db.py $resample_target out.vtk --point_data=$field --point_data_type=float32
fi

# raw_to_db.py $resample_target out.vtk --point_data=$field --point_data_type=float64
