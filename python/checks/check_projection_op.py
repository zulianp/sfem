#!/usr/bin/env python3

import sympy as sp
from sympy.physics.quantum import TensorProduct

def lumped(M):
	r, c = M.shape
	ret = sp.zeros(r, c)

	for i in range(0, r):
		for j in range(0, c):
			ret[i,i] += M[i,j]

	return ret

def sum_matrix(M):
	r, c = M.shape
	ret = 0
	for i in range(0, r):
		for j in range(0, c):
			ret += M[i,j]
	return ret

def print_matrix(A):
	r, c = A.shape
	for i in range(0, r):
		row = ""
		for j in range(0, c):
			row += f'{A[i,j]} '
		print(row)

x, y, z = sp.symbols('x y z')
f = [1 - x - y - z, x, y, z]
g = [0] * 4

for i in range(0, 4):
	gx = sp.Matrix(3, 1, [0, 0, 0])
	gx[0] = sp.diff(f[i], x)
	gx[1] = sp.diff(f[i], y)
	gx[2] = sp.diff(f[i], z)
	g[i] = gx

M_list = []

print("Grad")
print(g)

for i in range(0, 4):
	for j in range(0, 4):
		expr = sp.integrate(f[i] * f[j], (z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1))
		M_list.append(expr)

MASS = sp.Matrix(4, 4, M_list)
MASS = lumped(MASS)

print("MASS")
print_matrix(MASS)

MASS_inv = MASS.inv()
print("MASS_inv")
print_matrix(MASS_inv)

Id = sp.Matrix(3, 3, 
	[1, 0, 0, 
	 0, 1, 0, 
	 0, 0, 1])

tMASS_inv = TensorProduct(MASS_inv, Id)
print("tMASS_inv")
print_matrix(tMASS_inv)

DIV = sp.zeros(4, 12)

place_holder = sp.symbols('pg')

for i in range(0, 4):
	grad_op = g[i]
	for j in range(0, 4):
		integr = sp.integrate(f[j] * place_holder, (z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1))

		for d in range(0, 3):
			expr = integr.subs(place_holder, grad_op[d])
			DIV[i, j * 3 + d] = expr

GRAD = sp.zeros(12, 4)

# P1 shape function
for i in range(0, 4):
	integr = sp.integrate(f[i] * place_holder, (z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1))

	# P0 Gradient
	for d in range(0, 3):
		for j in range(0, 4):	
			grad_op = g[j][d]
			expr = integr.subs(place_holder, grad_op)
			GRAD[i * 3 + d, j] = expr


print("GRAD")
print_matrix(GRAD)

print("DIV")
print_matrix(DIV)

print("GT * M^-1 * G")
GTG = GRAD.T *  tMASS_inv * GRAD
print_matrix(GTG)

A_list = []
for i in range(0, 4):
	for j in range(0, 4):
		integr = (1./6) * (g[i][0] * g[j][0]+ g[i][1] * g[j][1] + g[i][2] * g[j][2])
		A_list.append(integr)

A = sp.Matrix(4, 4, A_list)

print("A")
print_matrix(A)

print("A-GT * M^-1 * G")
print_matrix(A - GTG)

# print("Actions")

# v = sp.Matrix(4, 1, [1, 2, 3, 4])

# print("A * v")
# print_matrix(A * v)

# print("GRAD.T * (tMASS_inv * (GRAD * v))")
# print_matrix(GRAD.T * ( tMASS_inv * (GRAD * v)))

# print("DIV * (tMASS_inv * (GRAD * v))")
# print_matrix(DIV * ( tMASS_inv * (GRAD * v)))

