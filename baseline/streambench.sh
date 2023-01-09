#!/usr/bin/env bash

git clone https://github.com/jeffhammond/STREAM.git stream

set -e

cd stream
gcc -mcmodel=large -DSTREAM_TYPE=double -mavx2 -DSTREAM_ARRAY_SIZE=560000000 -DNTIMES=10 -ffp-contract=fast -fopenmp -o stream_c.exe stream.c

export OMP_NUM_THREADS=20
export OMP_PROC_BIND=close

./stream_c.exe 
