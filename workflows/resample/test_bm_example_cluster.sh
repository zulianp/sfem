#!/usr/bin/bash

cluster_sizes=(1 2 3 4  6 8 12 16 24 32 56 64 96 128 256)
output="cluster_size,tet_number,time_sec,tets_per_sec\n"

for size in "${cluster_sizes[@]}"; do
    echo "Running with SFEM_CLUSTER_SIZE=$size"
    export SFEM_CLUSTER_SIZE=$size
    line=$(./example.sh | grep '^<cluster_bench>' | sed 's/^<cluster_bench> //')
    output+="$line\n"
done


echo -e "\n\n"
echo -e "$output"


