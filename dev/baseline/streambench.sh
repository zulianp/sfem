#!/usr/bin/env bash

# git clone https://github.com/jeffhammond/STREAM.git stream

set -e

cd stream
cc -mcmodel=large -DSTREAM_TYPE=double -Xclang -fopenmp /opt/homebrew/lib/libomp.dylib -DSTREAM_ARRAY_SIZE=10000000 -DNTIMES=10 -ffp-contract=fast -o stream_c.exe stream.c

# export OMP_NUM_THREADS=20
# export OMP_PROC_BIND=close

./stream_c.exe 
