#!/usr/bin/env python3

import cupy as cp
import os
import time

output_path = os.path.join(os.environ['HOME'], 'git/sfem/workflows/resample/sdf.float32.raw')

print("sdf_test_GPU.py: ==========================================")

sdf_t = cp.float32

D = 600
dims = (D, D, D)

print(f'sdf_test_GPU.py: Generating field of size {dims[0]} x {dims[1]} x {dims[2]}')

mn = -0.88
mx = 0.88
pmin = cp.array([mn, mn, mn])
pmax = cp.array([mx, mx, mx])

field = cp.zeros(dims, dtype=sdf_t)

mc = cp.float64(10)

def chess_board(x, y, z, mc):
    vx = cp.where(cp.int32(cp.abs(mc * x)) % 2 == 0, cp.ones_like(x), -cp.ones_like(x))
    vy = cp.where(cp.int32(cp.abs(mc * y)) % 2 == 0, cp.ones_like(y), -cp.ones_like(y))
    vz = cp.where(cp.int32(cp.abs(mc * z)) % 2 == 0, cp.ones_like(z), -cp.ones_like(z))
    
    return vx * vy * vz

start_clock = time.time()

xv = cp.zeros(dims[0], dtype=sdf_t)
xv[:] = cp.linspace(pmin[0], pmax[0], dims[0])

yv = cp.zeros(dims[1], dtype=sdf_t)
yv[:] = cp.linspace(pmin[1], pmax[1], dims[1])

zv = cp.zeros(dims[2], dtype=sdf_t)
zv[:] = cp.linspace(pmin[2], pmax[2], dims[2])

X, Y, Z = cp.meshgrid(xv, yv, zv)

field  = cp.sin(4.0 * cp.pi * X) + cp.cos(4.0 * cp.pi * Y + 4.0 * cp.pi * Z)**2
# field  = cp.sin(4.0 * cp.pi * X) + cp.cos(4.0 * cp.pi * Y) * cp.tanh(4.0 * cp.pi * (Z + X))
# field = chess_board(X, Y, Z, mc)

end_clock = time.time()
clock = end_clock - start_clock
print(f'sdf_test_GPU.py: Time taken to generate field: {clock} seconds')

cp.reshape(field, (dims[0]*dims[1]*dims[2], 1)).tofile(output_path)

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

print("sdf_test_GPU.py: ==========================================")
print("sdf_test_GPU.py: Writing metadata file to ", end="")
print(f'{pdir}/metadata_{fname}.yml')

with open(f'{pdir}/metadata_{fname}.yml', 'w') as f:
    f.write(header)