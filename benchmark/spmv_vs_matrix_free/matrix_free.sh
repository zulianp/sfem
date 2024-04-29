#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../workflows/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/utils:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

if [[ -z $BENCHMARK_DIR ]]
then
	BENCHMARK_DIR="../db"
fi

if [[ -z $LAUNCH ]]
then
	LAUNCH=""
fi

# OpenMP
export OMP_PROC_BIND=true 
if [[ -z $OMP_NUM_THREADS ]]
then
	export OMP_NUM_THREADS=12
fi

if [[ -z $SFEM_REPEAT ]]
then
	export SFEM_REPEAT=10
fi

if !command -v cuspmv &> /dev/null
then
	export ENABLE_CUDA=1
else
	export ENABLE_CUDA=0
	echo "cuspmv not found! setting ENABLE_CUDA=$ENABLE_CUDA"
fi

geo=(`ls $BENCHMARK_DIR`)

workspace=`mktemp -d`

scalar_mf=lapl_matrix_free
vector_mf=linear_elasticity_matrix_free

mkdir -p results

today=`date +"%Y_%m_%d"`
csv_output=results/"$today"_matrix_free.csv

echo "rep,geo,op_type,ref,ptype,TTS,ndofs,nnz" > $csv_output

function bench_matrix_free_cuda()
{
	case_path=$1

	p1=$case_path/p1
	p2=$case_path/p2
	
	# Scalar problem
	lapl_matrix_free $p1/refined 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
	op_type=`grep "op: " $p1/matrix_scalar/meta.yaml | awk '{print $2}'`

	stats=`grep "mf:" $workspace/temp_log.txt | awk '{print $2, $3, $4}' | tr ' ' ','`
	echo "tet4,$g,$op_type,$r,scalar,$stats" >> $csv_output

	# Vector problem
	SFEM_USE_MACRO=0 $vector_mf $p2 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
	op_type=`grep "op: " $p1/matrix_vector/meta.yaml | awk '{print $2}'`

	stats=`grep "mf:" $workspace/temp_log.txt | awk '{print $2, $3, $4}' | tr ' ' ','`
	echo "tet10,$g,$op_type,$r,vector,$stats" >> $csv_output
}

for g in ${geo[@]}
do
	path="$BENCHMARK_DIR/$g"
	resolutions=(`ls $path`)

	for r in ${resolutions[@]}
	do
		case_path="$BENCHMARK_DIR/$g/$r"

		if [[ $ENABLE_CUDA ==  1 ]]
		then
			bench_matrix_free_cuda $case_path
		fi
	done
done	

rm -rf $workspace

cat $csv_output
