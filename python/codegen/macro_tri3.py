#!/usr/bin/env python3


def assign_matrix(name, mat):
	rows, cols = mat.shape

	expr = []
	for i in range(0, rows):
		for j in range(0, cols):
			var = sp.symbols(f'{name}[{i*cols + j}]')
			expr.append(ast.Assignment(var, mat[i, j]))
	return expr

from fe import FE
import sympy as sp
from sfem_codegen import *

def subJ(micro_ref):
	x0m = micro_ref[0]
	x1m = micro_ref[1]
	x2m = micro_ref[2]
	um = x1m - x0m
	vm = x2m - x0m
	Am = sp.Matrix(2, 2, [um[0], vm[0], um[1], vm[1]])
	Aminv = inv2(Am)
	Rm = Am
	Rminv = inv2(Rm)
	
	FFFs = matrix_coeff("fff", 2, 2)
	FFFs[1, 0] = FFFs[0, 1] 

	FFAms = Rm * FFFs * Rm.T
	c_code(assign_matrix("fffm", FFAms))
	c_code(assign_matrix("Rm", Rm))

# Reference element

x0 = vec2(0., 0.)
x1 = vec2(1., 0.)
x2 = vec2(0., 1.)
x3 = (x0 + x1)/2
x4 = (x1 + x2)/2
x5 = (x2 + x0)/2

# Physical element

px0 = vec2(1, 2)
px1 = vec2(5, 3)
px2 = vec2(1, 6)

print("------------------")
print("A0...A2")
subJ([x0, x3, x5])

# print("------------------")
# print("A1")
# subJ([x3, x1, x4])

# print("------------------")
# print("A2")
# subJ([x5, x4, x2])

print("------------------")
print("A3")
subJ([x3, x4, x5])

FFFs = matrix_coeff("fff", 2, 2)
FFFs[1, 0] = FFFs[0, 1] 
Rs = matrix_coeff("R", 2, 2)
Rs[0, 1] = 0
Rs[1, 0] = 0

FFFAs = Rs * FFFs * Rs.T

c_code(assign_matrix("fffm", FFFAs))
