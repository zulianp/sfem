#!/usr/bin/bash

# Use PRECISION environment variable if defined, otherwise default to float32
# PRECISION=${PRECISION:-float32}

# output="nelements, time(s), tet/s, hex_nodes/s, tet_nodes/s, n0, n1, n2, dx, dy, dz, origin0, origin1, origin2\n"

hex_sizes=(100 125 150 175 200 225 250 300 350 400 450 500 600 700 800 900 1000 1250)
hex_sizes=(100 125)

export SFEM_REPETITIONS_QUAD_ADJOINT=10

for size in "${hex_sizes[@]}"; do
    echo "Running with SFEM_HEX_SIZE=$size"
    export SFEM_HEX_SIZE=$size
    
    # Store the output of example.sh in a string
    example_output=$(./example.sh)
    
    # Search the pattern in the stored string and extract the line
    line=$(echo "$example_output" | grep '^\s*<quad_bench>' | sed 's/^<quad_bench> //')
    precision_so=$(echo "$example_output" | grep '^\s*<sizeof_real_t>' | sed 's/^<sizeof_real_t> //')
    header_line=$(echo "$example_output" | grep '^\s*<quad_bench_head>' | sed 's/^<quad_bench_head> //')
    
    # Extract nelements only from the first line of the first run
    if [ -z "$first_number" ]; then
        first_number=$(echo "$line" | head -n 1 | awk '{print $1}')
        printf "First number (nelements) extracted: %s\n" "$first_number"
    fi

    # Append header line only once
    if [ -z "$output" ]; then
        output=$(echo "$header_line" | head -n 1)
        output+="\n"
    fi

    output+="$line\n"
done # END for size in hex_sizes

if [ "$precision_so" == "4" ]; then
    PRECISION="float32"
else
    PRECISION="float64"
fi

echo -e "\n\n"
echo -e "$output"

# Save the output to a file with precision and first_number in the name
echo -e "$output" > "benchmark_results_quad_tetnr${first_number}_${PRECISION}_rep.csv"
