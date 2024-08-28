#!/usr/bin/env python3

import numpy as np
import os


output_path = '/home/sriva/git/sfem/workflows/resample/sdf.float32.raw'


sdf_t = np.float32

D = 200
dims = (D+40, D+10, D)

mn = -0.88
mx = 1.8
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

for i in range(dims[0]):
    for j in range(dims[1]):
        for k in range(dims[2]):
            x = pmin[0] + i * (pmax[0] - pmin[0]) / (dims[0] - 1)
            y = pmin[1] + j * (pmax[1] - pmin[1]) / (dims[1] - 1)
            z = pmin[2] + k * (pmax[2] - pmin[2]) / (dims[2] - 1)
            # field[i, j, k] = 10.0 * np.exp( -( x**2 + y**2 + z**2 ) ) * (1.0 if np.sqrt(x*x + y*y + z*z) < 0.2 else -1.0)
            field[i, j, k] = np.sin(4.0 * np.pi * x) + np.cos(4.0 * np.pi * y + 4.0 * np.pi * z)**2 
            # field[i, j, k] = chess_board(x, y, z)

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