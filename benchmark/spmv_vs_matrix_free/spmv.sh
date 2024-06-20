#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../workflows/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH

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

geo=(`ls $BENCHMARK_DIR`)

workspace=`mktemp -d`

mkdir -p results
today=`date +"%Y_%m_%d"`
csv_output=results/"$today"_crs.csv
spmv_pattern="spmv:"
spmv_pattern="cuspa:"

echo "rep,geo,op_type,ref,ptype,TTS [s],throughput [GB/s],nelements,ndofs,nnz" > $csv_output

function bench_spmv()
{
	exec=$1
	case_path=$2

	p1=$case_path/p1
	p2=$case_path/p2	

	##############################################
	# Scalar problem
	##############################################
	if [ -f "$p1/matrix_scalar/rowptr.raw" ]; then
		$exec 1 0 $p1/matrix_scalar "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
		op_type=`grep "op: " $p1/matrix_scalar/meta.yaml | awk '{print $2}'`

		stats=`grep $spmv_pattern $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
		echo "crs,$g,$op_type,$r,scalar,$stats" >> $csv_output
	fi

	##############################################
	# Vector problem
	##############################################

	if [ -f "$p1/matrix_vector/rowptr.raw" ]; then	
		$exec 1 0 $p1/matrix_vector "gen:ones" $workspace/test.raw > $workspace/temp_log.txt
		op_type=`grep "op: " $p1/matrix_vector/meta.yaml | awk '{print $2}'`

		stats=`grep $spmv_pattern $workspace/temp_log.txt | awk '{print $2, $3, $4, $5, $6}' | tr ' ' ','`
		echo "crs,$g,$op_type,$r,vector,$stats" >> $csv_output
	fi
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
			bench_spmv cuspmv $case_path
		else
			bench_spmv spmv $case_path
		fi
	done
done	

rm -rf $workspace

cat $csv_output
