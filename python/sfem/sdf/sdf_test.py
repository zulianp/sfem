#!/usr/bin/env python3

import numpy as np
import os
import sys
import time
# from concurrent.futures import ThreadPoolExecutor


# output_path = os.path.join(
# os.environ['HOME'], 'git/sfem/workflows/resample/sdf.float32.raw')
# output_path = os.path.join(os.environ['SCRATCH'], 'prj/sfem/workflows/resample/sdf.float32.raw')
output_path = sys.argv[1]

print("sdf_test.py: ==========================================")
print(f'sdf_test.py: Writing field to {output_path}')

sdf_t = np.float32


# Default grid dimension
D = 300

# Check if a second argument exists and if it's a number to use as grid dimension
if len(sys.argv) > 2:
    try:
        grid_size = int(sys.argv[2])
        D = grid_size
        print(
            f'sdf_test.py: Using grid size D = {D} from command line argument')
    except ValueError:
        print(
            f'sdf_test.py: Warning - second argument is not a valid integer, using default D = {D}')

dims = (D, D, D)

print(
    f'sdf_test.py: Generating field of size {dims[0]} x {dims[1]} x {dims[2]}')
print(
    f'sdf_test.py: Size of field in memory: {np.prod(dims) * np.dtype(sdf_t).itemsize / 1024 / 1024 / 1024} GB')

mult_f = 1.0
mn = -0.88 * mult_f
mx = 0.88 * mult_f

pmin = np.array([mn, mn, mn])
pmax = np.array([mx, mx, mx])

field = np.zeros(dims, dtype=sdf_t)

# nedt = edt.to_numpy().astype(sdf_t)

mc = np.float64(10)


def chess_board(x, y, z, mc):
    vx = np.where(np.int32(np.abs(mc * x)) %
                  2 == 0, np.ones_like(x), -np.ones_like(x))
    vy = np.where(np.int32(np.abs(mc * y)) %
                  2 == 0, np.ones_like(y), -np.ones_like(y))
    vz = np.where(np.int32(np.abs(mc * z)) %
                  2 == 0, np.ones_like(z), -np.ones_like(z))

    return vx * vy * vz


start_clock = time.time()

xv = np.zeros(dims[0], dtype=sdf_t)
xv[:] = np.linspace(pmin[0], pmax[0], dims[0])

yv = np.zeros(dims[1], dtype=sdf_t)
yv[:] = np.linspace(pmin[1], pmax[1], dims[1])

zv = np.zeros(dims[2], dtype=sdf_t)
zv[:] = np.linspace(pmin[2], pmax[2], dims[2])

X, Y, Z = np.meshgrid(xv, yv, zv)

X = X / mult_f
Y = Y / mult_f
Z = Z / mult_f


def normalize_field(field):
    field_max = np.max(field)
    field_min = np.min(field)

    mean_max_min = (field_max + field_min) / 2.0
    field = field - mean_max_min

    field_max = np.max(field)
    field = field / field_max

    field_max = np.max(field)
    field_min = np.min(field)

    print(
        f'sdf_test.py: field_max = {field_max:.7e} field_min = {field_min:.7e}')
    return field

# 0: sin(4.0 * pi * X) + cos(4.0 * pi * Y + 4.0 * pi * Z)**2
# 1: sin(4.0 * pi * X) + cos(4.0 * pi * Y) * tanh(4.0 * pi * (Z + X))
# 2: chess_board(X, Y, Z, mc)
# 3: np.ones_like(X)


option = 5

########## Normalize field in an interval [-1, 1] ##########

normalize = True

if option == 0:
    print("sdf_test.py: Using field: sin(4.0 * pi * X) + cos(4.0 * pi * Y + 4.0 * pi * Z)**2")
    # field  = np.sin(4.0 * np.pi * X  ) + np.cos(4.0 * np.pi * Y + 4.0 * np.pi * Z)**2
    field = np.sin(4.0 * np.pi * (X)) + np.cos(4.0 * np.pi * (Y + Z))**2

elif option == 1:
    print("sdf_test.py: Using field: sin(4.0 * pi * X/mult_f) + cos(4.0 * pi * Y/mult_f) * tanh(4.0 * pi * (Z/mult_f + X/mult_f))")
    field = np.sin(4.0 * np.pi * (X)) + np.cos(4.0 * np.pi *
                                               (Y)) * np.tanh(4.0 * np.pi * (Z + X))

elif option == 2:
    print("sdf_test.py: Using field: chess_board(X, Y, Z, mc)")
    field = chess_board(X, Y, Z, mc)

elif option == 3:
    print("sdf_test.py: Using field: np.ones_like(X)")
    field = np.ones_like(X)

else:
    print("sdf_test.py: Zero field")
    field = np.zeros_like(X)
    normalize = False


if normalize and not option == 3:
    field = normalize_field(field)


end_clock = time.time()
clock = end_clock - start_clock
print(f'sdf_test.py: Time taken to generate field: {clock} seconds')

np.reshape(field, (dims[0]*dims[1]*dims[2], 1)).tofile(output_path)

dx_val = (pmax[0] - pmin[0])/(dims[0] - 1)
dy_val = (pmax[1] - pmin[1])/(dims[1] - 1)
dz_val = (pmax[2] - pmin[2])/(dims[2] - 1)

header = f'nx: {dims[0]}\n'
header += f'ny: {dims[1]}\n'
header += f'nz: {dims[2]}\n'
header += f'block_size: 1\n'
header += f'type: float\n'
header += f'ox: {pmin[0]}\n'
header += f'oy: {pmin[1]}\n'
header += f'oz: {pmin[2]}\n'
header += f'dx: {dx_val}\n'
header += f'dy: {dy_val}\n'
header += f'dz: {dz_val}\n'
header += f'path: {os.path.abspath(output_path)}\n'

print(f"sdf_test.py: nx: {dims[0]}")
print(f"sdf_test.py: ny: {dims[1]}")
print(f"sdf_test.py: nz: {dims[2]}")
print(f"sdf_test.py: block_size: 1")
print(f"sdf_test.py: type: float")
print(f"sdf_test.py: ox: {pmin[0]}")
print(f"sdf_test.py: oy: {pmin[1]}")
print(f"sdf_test.py: oz: {pmin[2]}")
print(f"sdf_test.py: dx: {dx_val}")
print(f"sdf_test.py: dy: {dy_val}")
print(f"sdf_test.py: dz: {dz_val}")
print(f"sdf_test.py: path: {os.path.abspath(output_path)}")

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
