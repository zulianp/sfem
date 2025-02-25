#!/usr/bin/env bash

set -e

export CC=gcc
$CC --version


export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

if  [[ -e stream.c ]]
then
	echo "stream.c available"
else
	# wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c
	# git clone https://github.com/jeffhammond/STREAM.git
	cd STREAM
fi

$CC   -mcmodel=large -Ofast -fno-math-errno -mcpu=apple-m1 -DNDEBUG -Xclang -fopenmp -L/opt/homebrew/lib/ -lomp 		\
      -DSTREAM_ARRAY_SIZE=87200000 -DNTIMES=400  	\
      -o stream_openmp.exe stream.c


OMP_DISPLAY_ENV=VERBOSE ./stream_openmp.exe
# if [[ -z arm-kernels ]]
# then
# 	git clone https://github.com/NVIDIA/arm-kernels.git
# fi

# cd arm-kernels
# make
# perf stat ./arithmetic/fp64_sve_pred_fmla.x
# cd ..
