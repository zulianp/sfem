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

if [[ -z $ENABLE_CUDA ]]
then
	export ENABLE_CUDA=1
fi

if [[ $ENABLE_CUDA ==  1 ]]
then
	echo "CUDA is enabled!"
fi

# Limitation is GPU memory
max_tet4_size=1000000000
geo=(`ls $BENCHMARK_DIR`)

matrix_free_pattern="cf:"

workspace=`mktemp -d`

mkdir -p results
today=`date +"%Y_%m_%d"`
csv_output=results/"$today"_matrix_free.csv
# csv_output_extra=results/"$today"_matrix_free_extra.csv

echo "rep,geo,op_type,ref,ptype,TTS [s],throughput [GB/s],nelements,ndofs,nnz" > $csv_output

scalar_mf=lapl_matrix_free
vector_mf=linear_elasticity_matrix_free

function bench_matrix_free_cuda()
{
	local case_path=$1

	local p1=$case_path/p1
	local p2=$case_path/p2
	
	##############################################
	# Scalar problem
	##############################################

	mesh_size=`ls -la $p1/refined/i0.raw | awk '{print $5}'`

	if [[ $mesh_size -le $max_tet4_size ]]
	then
		lapl_matrix_free $p1/refined 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
		op_type=`grep "op: " $p1/matrix_scalar/meta.yaml | awk '{print $2}'`

		stats=`grep "$matrix_free_pattern" $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
		echo "tet4,$g,$op_type,$r,scalar,$stats" >> $csv_output
	fi

	lapl_matrix_free $p2 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
	op_type=`grep "op: " $p1/matrix_scalar/meta.yaml | awk '{print $2}'`

	stats=`grep "$matrix_free_pattern" $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
	echo "tet10,$g,$op_type,$r,scalar,$stats" >> $csv_output

	SFEM_USE_MACRO=1 lapl_matrix_free $p2 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
	op_type=`grep "op: " $p1/matrix_scalar/meta.yaml | awk '{print $2}'`

	stats=`grep "$matrix_free_pattern" $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
	echo "macrotet4,$g,$op_type,$r,scalar,$stats" >> $csv_output

	##############################################
	# Vector problem
	##############################################
	if [[ $mesh_size -le $max_tet4_size ]]
	then
		SFEM_USE_MACRO=0 $vector_mf $p1/refined 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
		op_type=`grep "op: " $p1/matrix_vector/meta.yaml | awk '{print $2}'`

		stats=`grep "$matrix_free_pattern" $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
		echo "tet4,$g,$op_type,$r,vector,$stats" >> $csv_output
	fi

	SFEM_USE_MACRO=0 $vector_mf $p2 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
	op_type=`grep "op: " $p1/matrix_vector/meta.yaml | awk '{print $2}'`

	stats=`grep "$matrix_free_pattern" $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
	echo "tet10,$g,$op_type,$r,vector,$stats" >> $csv_output

	SFEM_USE_MACRO=1 $vector_mf $p2 1 "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
	op_type=`grep "op: " $p1/matrix_vector/meta.yaml | awk '{print $2}'`

	stats=`grep "$matrix_free_pattern" $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
	echo "macrotet4,$g,$op_type,$r,vector,$stats" >> $csv_output
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
