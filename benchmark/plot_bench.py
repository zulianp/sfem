#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys

col = 'throughput [GB/s]'
real_str = "double"
geo = "sphere"
# geo = "cylinder"
# op_type = "Laplacian"
op_type = "LinearElasticity"
elem_type 	 = "tet10"
elem_type_p1 = "tet4"
elem_type_macro_p1 = "macrotet4"
size_of_real = 8
gpu = "1 x P100 GPU"

# Check for command line arguments
if len(sys.argv) < 3:
    print("Error: Please provide the CSV file path as an argument.")
    exit(1)

# Get the CSV file path from the first argument
SpMV_data_path = sys.argv[1]
MF_data_path   = sys.argv[2]

# Read the CSV data into a DataFrame
SpMV_df = pd.read_csv(SpMV_data_path)
MF_df   = pd.read_csv(MF_data_path)

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

plt.savefig('plot.pdf')
plt.savefig('plot.pgf')

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

if op_type == "Laplacian":
    print(f'macrotet4    {round(np.max(MF_y_macro_p1), 1)}')

print('----------------------------')
print('Speed-up (range)')
print('----------------------------')
speedup = MF_y / MF_y_p1
print(f'tet4/tet10:     {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')
speedup = MF_y[:len(SpMV_y)] / SpMV_y 
print(f'SpMV/tet10:     {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')

if op_type == "Laplacian":
    speedup = MF_y_macro_p1 / MF_y_p1
    print(f'tet4/macrotet4: {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')
    speedup = MF_y_macro_p1[:len(SpMV_y)] / SpMV_y
    print(f'SpMV/macrotet4: {round(np.min(speedup), 1)}-{round(np.max(speedup), 1)}')
print('----------------------------')
