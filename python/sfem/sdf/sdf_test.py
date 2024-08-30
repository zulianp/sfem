#!/usr/bin/env python3

import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor


output_path = '/home/sriva/git/sfem/workflows/resample/sdf.float32.raw'

print("sdf_test.py: ==========================================")

sdf_t = np.float32

D = 200
dims = (D, D, D)

print(f'sdf_test.py: Generating field of size {dims[0]} x {dims[1]} x {dims[2]}')

mn = -0.88
mx = 0.88
pmin = np.array([mn, mn, mn])
pmax = np.array([mx, mx, mx])

field = np.zeros(dims, dtype=sdf_t)

# nedt = edt.to_numpy().astype(sdf_t)

mc = np.float64(10)

def chess_board(x, y, z):
    vx = 1.0 if np.int32(np.abs(mc * x)) % 2 == 0 else -1.0
    vy = 1.0 if np.int32(np.abs(mc * y)) % 2 == 0 else -1.0
    vz = 1.0 if np.int32(np.abs(mc * z)) % 2 == 0 else -1.0
         
    return vx * vy * vz

start_clock = time.time()


################################################################
# def process_element(i, pmin, pmax, dims):
#     x = pmin[0] + i * (pmax[0] - pmin[0]) / (dims[0] - 1)
    
#     for j in range(dims[1]):
#         y = pmin[1] + j * (pmax[1] - pmin[1]) / (dims[1] - 1)
        
#         for k in range(dims[2]):
#             z = pmin[2] + k * (pmax[2] - pmin[2]) / (dims[2] - 1)
#             field[i, j, k] = chess_board(x, y, z)
            
#     print(f'Processed element {i}')

# with ThreadPoolExecutor(max_workers=18) as executor:
#     futures = []
#     for i in range(dims[0]):
#         futures.append(executor.submit(process_element, i, pmin, pmax, dims))
    
#     for future in futures:
#         future.result()
#         # field[i, j, k] = value
################################################################

xv = np.zeros(dims[1], dtype=sdf_t)
xv[:] = np.linspace(pmin[0], pmax[0], dims[0])

yv = np.zeros(dims[1], dtype=sdf_t)
yv[:] = np.linspace(pmin[1], pmax[1], dims[1])

zv = np.zeros(dims[2], dtype=sdf_t)
zv[:] = np.linspace(pmin[2], pmax[2], dims[2])

for i in range(dims[0]):
    x = pmin[0] + i * (pmax[0] - pmin[0]) / (dims[0] - 1)
    
    for j in range(dims[1]):
        y = pmin[1] + j * (pmax[1] - pmin[1]) / (dims[1] - 1)
                
        for k in range(dims[2]):    
            z = pmin[2] + k * (pmax[2] - pmin[2]) / (dims[2] - 1)
            # field[i, j, k] = 10.0 * np.exp( -( x**2 + y**2 + z**2 ) ) * (1.0 if np.sqrt(x*x + y*y + z*z) < 0.2 else -1.0)
            # field[i, j, k] = np.sin(4.0 * np.pi * x) + np.cos(4.0 * np.pi * y + 4.0 * np.pi * z)**2 
            field[i, j, k] = chess_board(x, y, z)
            
end_clock = time.time()
clock = end_clock - start_clock
print(f'sdf_test.py: Time taken to generate field: {clock} seconds')

np.reshape(field, ( dims[0]*dims[1]*dims[2], 1)).tofile(output_path)

header =    f'nx: {dims[0]}\n'
header +=   f'ny: {dims[1]}\n'
header +=   f'nz: {dims[2]}\n'
header +=   f'block_size: 1\n'
header +=   f'type: float\n'
header +=   f'ox: {pmin[0]}\n'
header +=   f'oy: {pmin[1]}\n'
header +=   f'oz: {pmin[2]}\n'
header +=   f'dx: {(pmax[0] - pmin[0])/(dims[0] - 1)}\n'
header +=   f'dy: {(pmax[1] - pmin[1])/(dims[1] - 1)}\n'
header +=   f'dz: {(pmax[2] - pmin[2])/(dims[2] - 1)}\n'
header +=   f'path: {os.path.abspath(output_path)}\n'

fname, fextension = os.path.splitext(output_path)
pdir = os.path.dirname(fname)

if pdir == "":
    pdir = "./"

fname = os.path.basename(fname)

print("sdf_test.py: ==========================================")
print("sdf_test.py: Writing metadata file to ", end="")
print(f'{pdir}/metadata_{fname}.yml')

with open(f'{pdir}/metadata_{fname}.yml', 'w') as f:
    f.write(header)