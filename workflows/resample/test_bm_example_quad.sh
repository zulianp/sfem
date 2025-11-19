#!/usr/bin/bash

#########################################################################
#########################################################################
# run_benchmark_for_hex_sizes
#########################################################################
#########################################################################
run_benchmark_for_hex_sizes() {
    local -n hex_sizes_arr=$1
    local test_name=$2
    local output=""
    local first_number=""
    local precision_so=""
    local header_line=""
    
    for size in "${hex_sizes_arr[@]}"; do
        echo "Running with SFEM_HEX_SIZE=$size"
        export SFEM_HEX_SIZE=$size
        
        # Store the output of example.sh in a string
        local example_output=$(./example.sh)
        
        # Search the pattern in the stored string and extract the line
        local line=$(echo "$example_output" | grep '^\s*<quad_bench>' | sed 's/^<quad_bench> //')
        precision_so=$(echo "$example_output" | grep '^\s*<sizeof_real_t>' | sed 's/^<sizeof_real_t> //')
        header_line=$(echo "$example_output" | grep '^\s*<quad_bench_head>' | sed 's/^<quad_bench_head> //')
        
        # Extract nelements only from the first line of the first run
        if [ -z "$first_number" ]; then
            first_number=$(echo "$line" | head -n 1 | awk '{print $1}')
            printf "First number (nelements) extracted: %s\n" "$first_number"
        fi # END if [ -z "$first_number" ]

        # Append header line only once
        if [ -z "$output" ]; then
            output=$(echo "$header_line" | head -n 1)
            output+="\n"
        fi # END if [ -z "$output" ]

        output+="$line\n"
    done # END for size in hex_sizes_arr

    # Determine precision
    local PRECISION
    if [ "$precision_so" == "4" ]; then
        PRECISION="float32"
    else
        PRECISION="float64"
    fi # END if [ "$precision_so" == "4" ]

    # Display results
    echo -e "\n\n"
    echo -e "$output"

    # Save the output to a file with precision and first_number in the name
    local output_file="benchmark_results_${test_name}_tetnr${first_number}_${PRECISION}_rep.csv"
    echo -e "$output" > "$output_file"
    
    echo "Benchmark results saved to: $output_file"
    
    return 0
} # END Function: run_benchmark_for_hex_sizes

#########################################################################
# Main execution
#########################################################################

# Use PRECISION environment variable if defined, otherwise default to float32
# PRECISION=${PRECISION:-float32}

export test_name="quad"

hex_sizes=(100 125 150 175 200 225 250 300 350 400 450 500 600 700 800 900 1000 1250)
hex_sizes=(100 125)

input_data_dirs=(data2 data3 data4)
mesh_names=(torus2 torus3 torus4)

export SFEM_REPETITIONS_QUAD_ADJOINT=10

echo "Starting benchmark for test: $test_name"

# Run benchmark
# Check if array has any elements
if [ ${#input_data_dirs[@]} -eq 0 ]; then
    echo "Input data directories provided is empty. Running benchmark with current directory data."
    run_benchmark_for_hex_sizes hex_sizes "$test_name"
else
    i=0
    for data_dir in "${input_data_dirs[@]}"; do
        echo "Processing data directory: $data_dir"
        cp "$data_dir/"* ./

        export SFEM_TORUS_MESH=${mesh_names[$i]}
        ((i++))

        run_benchmark_for_hex_sizes hex_sizes "$test_name"
    done # END for data_dir in "${input_data_dirs[@]}"
fi # END if [ ${#input_data_dirs[@]} -gt 0 ]
