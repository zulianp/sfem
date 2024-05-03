#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys, getopt

col = 'throughput [GB/s]'
real_str = "double"
elem_type 	 = "tet10"
elem_type_p1 = "tet4"
elem_type_macro_p1 = "macrotet4"
size_of_real = 8

def plot(spMV_df, MF_df, geo, op_type, gpu):
    SpMV_rep 		= "CRS"
    SpMV_is_geo 	= SpMV_df["geo"] == geo
    SpMV_is_op_type = SpMV_df["op_type"] == op_type
    SpMV_series 	= SpMV_df[SpMV_is_geo & SpMV_is_op_type]
    SpMV_x 			= SpMV_series['ndofs'].values

    # From Giga to Mega
    SpMV_y 			= 1000 * SpMV_series[col].values / size_of_real

    MF_is_geo 	  = MF_df["geo"] == geo
    MF_is_elem	  = MF_df["rep"] == elem_type
    MF_is_elem_p1 = MF_df["rep"] == elem_type_p1
    MF_is_elem_macro_p1 = MF_df["rep"] == elem_type_macro_p1

    MF_is_op_type = MF_df["op_type"] == op_type
    MF_series 	  = MF_df[MF_is_geo & MF_is_op_type & MF_is_elem]
    MF_series_p1  = MF_df[MF_is_geo & MF_is_op_type & MF_is_elem_p1]
    MF_series_macro_p1  = MF_df[MF_is_geo & MF_is_op_type & MF_is_elem_macro_p1]

    MF_nels_p1 = MF_series_p1['nelements'].values
    MF_nels    = MF_series['nelements'].values
    SpMV_nnz   = SpMV_series['nnz'].values

    MF_x 		  = MF_series['ndofs'].values
    # From Giga to Mega
    MF_y 		  = 1000 * MF_series[col].values / size_of_real

    MF_x_p1 	  = MF_series_p1['ndofs'].values
    # From Giga to Mega
    MF_y_p1 	  = 1000 * MF_series_p1[col].values / size_of_real

    MF_x_macro_p1 	  = MF_series_macro_p1['ndofs'].values
    # From Giga to Mega
    MF_y_macro_p1 	  = 1000 * MF_series_macro_p1[col].values / size_of_real

    plt.figure().clear()
    plt.loglog(SpMV_x,  SpMV_y,  marker='o', linestyle='-', label=f"SpMV (cuSPARSE: {SpMV_rep})")
    plt.loglog(MF_x,    MF_y,	 marker='x', linestyle='-', label=f"MF ({elem_type})")
    plt.loglog(MF_x_p1, MF_y_p1, marker='.', linestyle='-', label=f"MF ({elem_type_p1})")
    plt.loglog(MF_x_macro_p1, MF_y_macro_p1, marker='o', linestyle='-', label=f"MF ({elem_type_macro_p1})")

    plt.xlabel('Degrees of Freedom (DOF)')
    plt.ylabel(f"MDOF/s")
    plt.title(f"Operator: {op_type}, Mesh: {geo}, {gpu}")
    plt.legend()

    # Display the plot
    plt.grid(True)  # Optional: Add grid for better readability
    plt.grid(True, which='minor', linestyle='--')
    plt.tight_layout()
    # plt.show()

    plt.savefig(f'plot_{geo}_{op_type}.pdf')
    # plt.savefig(f'plot_{geo}_{op_type}.pgf')


    print('############################')
    print(f'Summary for {geo} {op_type}')
    print('############################')


    print('----------------------------')
    print('Memory overhead (lower bound)')
    print('----------------------------')

    mem_scale = 1e-6 #MB
    geo_elast_factor = (9+1)*2*mem_scale # Half-precision
    geo_lapl_factor = 6*2*mem_scale # Half-precision
    p1_idx_factor = (4 * 4*mem_scale)
    p2_idx_factor = (10 * 4*mem_scale)

    p1_elast_factor = p1_idx_factor + geo_elast_factor
    p2_elast_factor = p2_idx_factor + geo_elast_factor

    p1_lapl_factor = p1_idx_factor + geo_lapl_factor
    p2_lapl_factor = p2_idx_factor + geo_lapl_factor

    print(f'MF P1          (Elasticity) {round(np.min(MF_nels_p1)*p1_elast_factor, 3)}-{round(np.max(MF_nels_p1)*p1_elast_factor, 3)} MB')
    print(f'MF P2/Macro-P1 (Elasticity) {round(np.min(MF_nels)*p2_elast_factor, 3)}-{round(np.max(MF_nels)*p2_elast_factor, 3)} MB')

    geo_factor = 6*mem_scale # Half-precision
    print(f'MF P1          (Laplacian)  {round(np.min(MF_nels_p1)*p1_lapl_factor, 3)}-{round(np.max(MF_nels_p1)*p1_lapl_factor, 3)} MB')
    print(f'MF P2/Macro-P1 (Laplacian)  {round(np.min(MF_nels)*p2_lapl_factor, 3)}-{round(np.max(MF_nels)*p2_lapl_factor, 3)} MB')

    crs_factor = 8*2*mem_scale
    print(f'CRS                         {round(np.min(SpMV_nnz)*crs_factor, 3)}-{round(np.max(SpMV_nnz)*crs_factor, 3)} MB')

    print('----------------------------')
    print('Mesh')
    print('----------------------------')
    print(f'#elements P1 {np.min(MF_nels_p1)}-{np.max(MF_nels_p1)}')
    print(f'#elements    {np.min(MF_nels)}-{np.max(MF_nels)}')

    print('----------------------------')
    print('Matrix')
    print('----------------------------')
    print(f'DOFs (SpMV)  {np.min(MF_x)}-{np.max(MF_x)} ({np.max(SpMV_x)})')
    print(f'NNZ          {np.min(SpMV_nnz)}-{np.max(SpMV_nnz)}')

    print('----------------------------')
    print('Max throughput (MDOF/s)')
    print('----------------------------')
    print(f'SpMV (P1)    {round(np.max(SpMV_y), 1)}')
    print(f'tet10        {round(np.max(MF_y), 1)}')
    print(f'tet4         {round(np.max(MF_y_p1), 1)}')

    # if op_type == "Laplacian":
    print(f'macrotet4    {round(np.max(MF_y_macro_p1), 1)}')

    print('----------------------------')
    print('Speed-up (range)')
    print('----------------------------')
    speedup = MF_y / MF_y_p1
    print(f'tet4/tet10:     {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')
    speedup = MF_y[:len(SpMV_y)] / SpMV_y 
    print(f'SpMV/tet10:     {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')

    # if op_type == "Laplacian":
    speedup = MF_y_macro_p1 / MF_y_p1
    print(f'tet4/macrotet4: {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')
    speedup = MF_y_macro_p1[:len(SpMV_y)] / SpMV_y
    print(f'SpMV/macrotet4: {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')
    print('----------------------------')



if __name__ == '__main__':
    argv = sys.argv

    usage = f'usage: {argv[0]} <spmv.csv> <matrix_free.csv>'

    if len(argv) < 3:
        print("Error: Please provide the CSV file path as an argument.")
        print(usage)
        exit(1)

    gpu = "1 x P100 GPU"

    try:
        opts, args = getopt.getopt(
            argv[3:], "d:h",
            ["device=", "help"])

    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ("-d", "--device"):
            gpu = arg

    # Get the CSV file path from the first argument
    SpMV_data_path = sys.argv[1]
    MF_data_path   = sys.argv[2]

    # Read the CSV data into a DataFrame
    SpMV_df = pd.read_csv(SpMV_data_path)
    MF_df   = pd.read_csv(MF_data_path)

    plot(SpMV_df, MF_df, "cylinder", "LinearElasticity", gpu)
    plot(SpMV_df, MF_df, "sphere", "LinearElasticity", gpu)

    plot(SpMV_df, MF_df, "cylinder", "Laplacian", gpu)
    plot(SpMV_df, MF_df, "sphere", "Laplacian", gpu)

