#!/usr/bin/env python3


import sympy as sp

p = sp.Matrix(3, 1, [1, 1, 0])
p1 = sp.Matrix(3, 1, [0, 1, 0])

def modes(p):
	x = p[0] 
	y = p[1]
	z = p[2]

	vals = [
		1, 0, 0, -y, z, 0, 	
		0, 1, 0, x, 0, -z,  
		0, 0, 1, 0, -x, y
	]

	M = sp.Matrix(3, 6, vals)
	return M


M = modes(p)
# M += modes(p1)


print(M.T)
print(M.T * p1)


def modes2D(p):
	x = p[0] 
	y = p[1]
	vals = [
		1, 0, -y,
		0, 1,  x 
	]

	M = sp.Matrix(3, 6, vals)
	return M

