#!/usr/bin/env python3

import numpy as np
import os


output_path = '/home/sriva/git/sfem/workflows/resample/sdf.float32.raw'


sdf_t = np.float32

D = 100
dims = (D, D, D)

mn = -0.7
mx = 0.7
pmin = np.array([mn, mn, mn])
pmax = np.array([mx, mx, mx])

field = np.zeros(dims, dtype=sdf_t)

# nedt = edt.to_numpy().astype(sdf_t)

for i in range(D):
    for j in range(D):
        for k in range(D):
            field[i, j, k] = 1

np.reshape(field, (D*D*D, 1)).tofile(output_path)

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

with open(f'{pdir}/metadata_{fname}.yml', 'w') as f:
    f.write(header)