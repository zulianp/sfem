#!/bin/bash

export SFEM_CLUSTER_SIZE_ADJOINT=256
export SFEM_TET_PER_BLOCK_ADJOINT=16
export SFEM_REPETITIONS_ADJOINT=10

export OUT_FILENAME_RAW=/capstor/scratch/cscs/sriva/prj/sfem_d/sfem/workflows/resample/test_field.raw


output_file="resuts_adjoint.csv"
rm -f "$output_file"


# Scrive l'intestazione nel file CSV
echo "cluster_size,tet_per_block,nelements,closk,Throughtput" > "$output_file"

cluster_size_list=(1 4 8 12 16 24 32 48 64 96 128 192 256)
tets_per_block_list=(4 6 8 12 16)

# cluster_size_list=(1 4 8)
# tets_per_block_list=(4 6)

for cluster_size in "${cluster_size_list[@]}"; do
    for tets_per_block in "${tets_per_block_list[@]}"; do

        export SFEM_CLUSTER_SIZE_ADJOINT=$cluster_size
        export SFEM_TET_PER_BLOCK_ADJOINT=$tets_per_block
        echo "Running with cluster size: $cluster_size, tets per block: $tets_per_block"

        # Esegue l'esempio, filtra la riga di benchmark e la aggiunge al file CSV
        line=$(./example.sh | grep "<cluster_bench>")
        data=$(echo "$line" | sed 's/<cluster_bench> *//')
        
        echo "$data" >> "$output_file"
        echo "$data"

    done
done