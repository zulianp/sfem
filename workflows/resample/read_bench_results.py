import numpy as np
import re
import os

def read_bench_results(bench_results_dir, output_file=None):
    """
    Read the benchmark results from the benchmark results directory.
    """
    bench_results = []
    for filename in os.listdir(bench_results_dir):
        if filename.endswith('.log'):
            with open(os.path.join(bench_results_dir, filename), 'r') as f:
                lines = f.read()
            
            # Extract the benchmark header
            header = re.search(r"^\s*<BenchH>\s*(.*)", lines, re.MULTILINE)
            if header:
                header = header.group(1)
                # print (header)
                
            # Extract the benchmark results
            results = re.search(r"^\s*<BenchR>\s*(.*)", lines, re.MULTILINE)
            if results:
                results = results.group(1)
                # print (results)
                bench_results.append(results)
                
    print (header)
    for result in bench_results:
        print (result)
        
    if output_file is None:
        output_file = os.path.join(bench_results_dir, "bench_results.csv")
        
    with open(output_file, "w") as file:
        file.write(header + "\n")
        for result in bench_results:
            file.write(result + "\n")
         
if __name__ == '__main__':
    
    # make a cnd line parser for the input directory and output file
    # read the benchmark results
    
    import argparse
    parser = argparse.ArgumentParser(description='Read the benchmark results from the benchmark results directory.')
    parser.add_argument('--ind', type=str, help='The directory containing the benchmark results.')
    parser.add_argument('--out', type=str, help='The output file to write the benchmark results to.')
    args = parser.parse_args()
    
    bench_results_dir = args.ind
    output_file = args.out
    
    read_bench_results(bench_results_dir, output_file)
            
            