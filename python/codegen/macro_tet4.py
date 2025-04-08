#!/usr/bin/env python3


def assign_matrix(name, mat):
    rows, cols = mat.shape

    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


from fe import FE
import sympy as sp
from sfem_codegen import *


def subJ(micro_ref):
    x0m = micro_ref[0]
    x1m = micro_ref[1]
    x2m = micro_ref[2]
    x3m = micro_ref[3]

    um = x1m - x0m
    vm = x2m - x0m
    wm = x3m - x0m

    Am = sp.Matrix(
        3, 3, [um[0], vm[0], wm[0], um[1], vm[1], wm[1], um[2], vm[2], wm[2]]
    )

    Aminv = inv3(Am)
    Rm = Am
    Rminv = inv3(Rm)

    FFFs = matrix_coeff("fff", 3, 3)
    FFFs[1, 0] = FFFs[0, 1]
    FFFs[2, 0] = FFFs[0, 2]
    FFFs[2, 1] = FFFs[1, 2]

    FFAms = Rm * FFFs * Rm.T
    c_code(assign_matrix("fffm", FFAms))

    # print(Rm)
    # c_code(assign_matrix("Rm", Rm))


# Reference element

x0 = vec3(0.0, 0.0, 0)
x1 = vec3(1.0, 0.0, 0)
x2 = vec3(0.0, 1.0, 0)
x3 = vec3(0.0, 0.0, 1)
x4 = (x0 + x1) / 2
x5 = (x1 + x2) / 2
x6 = (x0 + x2) / 2
x7 = (x0 + x3) / 2
x8 = (x1 + x3) / 2
x9 = (x2 + x3) / 2

# Physical element

############################
# Corner tests
############################

print("------------------")
print("A0 ... A3 (corners)")
subJ([x0, x4, x6, x7]),
# print("------------------")
# print("A1")
# subJ([x4, x1, x5, x8]),
# print("------------------")
# print("A2")
# subJ([x6, x5, x2, x9]),
# print("------------------")
# print("A3")
# subJ([x7, x8, x9, x3]),

############################
# Octahedron tets
############################

print("------------------")
print("A4")
subJ([x4, x5, x6, x8])

print("------------------")
print("A5")
subJ([x7, x4, x6, x8])

print("------------------")
print("A6")
subJ([x6, x5, x9, x8])

print("------------------")
print("A7")
subJ([x7, x6, x9, x8])

# FFFs = matrix_coeff("fff", 3, 3)
# FFFs[1, 0] = FFFs[0, 1]
# FFFs[2, 0] = FFFs[0, 2]
# FFFs[2, 1] = FFFs[1, 2]
