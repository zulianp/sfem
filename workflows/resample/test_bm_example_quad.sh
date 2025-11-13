#!/usr/bin/bash

# Use PRECISION environment variable if defined, otherwise default to float32
PRECISION=${PRECISION:-float32}

output="nelements, time(s), tet/s, hex_nodes/s, tet_nodes/s, n0, n1, n2, dx, dy, dz, origin0, origin1, origin2\n"

hex_sizes=(100 125 150 175 200 225 250 300 350 400 450 500 600 700 800 900 1000 1250)
hex_sizes=(100 125)

for size in "${hex_sizes[@]}"; do
    echo "Running with SFEM_HEX_SIZE=$size"
    export SFEM_HEX_SIZE=$size
    line=$(./example.sh | grep '^\s*<quad_bench>' | sed 's/^<quad_bench> //')
    first_number=$(echo "$line" | awk '{print $1}')
    output+="$line\n"
done # END for size in hex_sizes

echo -e "\n\n"
echo -e "$output"

# Save the output to a file with precision and first_number in the name
echo -e "$output" > "benchmark_results_quad_tetnr${first_number}_${PRECISION}.csv"