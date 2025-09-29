#!/bin/bash

export SFEM_CLUSTER_SIZE_ADJOINT=256
export SFEM_TETS_PER_BLOCK_ADJOINT=16
export SFEM_REPETITIONS_ADJOINT=10


output_file="resuts_adjoint.csv"

# Scrive l'intestazione nel file CSV
echo "cluster_size,tet_per_block,nelements,closk,Throughtput" > "$output_file"

cluster_size_list=(1 4 8 12 16 24 32 48 64 96 128 192 256)
tets_per_block_list=(4 6 8 12 16 24 32)

# cluster_size_list=(1 4 8)
# tets_per_block_list=(4 6)

for cluster_size in "${cluster_size_list[@]}"; do
    for tets_per_block in "${tets_per_block_list[@]}"; do
        export SFEM_CLUSTER_SIZE_ADJOINT=$cluster_size
        export SFEM_TETS_PER_BLOCK_ADJOINT=$tets_per_block
        # echo "Running with cluster size: $cluster_size, tets per block: $tets_per_block \n"

        # Esegue l'esempio, filtra la riga di benchmark e la aggiunge al file CSV
        ./example.sh | grep "<cluster_bench>" | while read -r line; do
            # Estrae i dati dopo <cluster_bench>
            data=$(echo "$line" | sed 's/<cluster_bench> *//')
            # Sostituisce gli spazi con virgole
            csv_data=$(echo "$data" | tr -s ' ' ',')
            # Scrive la riga completa nel file CSV
            echo "$data" >> "$output_file"
            echo "$data"
        done


    done
done