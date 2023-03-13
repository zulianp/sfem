#!/usr/bin/env python3

# ./shear.py

from sfem_codegen import *

dV = det3(A)

s = coeffs('shear', 6)

C = sp.Matrix(3, 3, [
	s[0], s[1], s[2],
	s[1], s[3], s[4],
	s[2], s[4], s[5]
]);

u = sp.Matrix(3, 1,[
	x1 - x0,
	y1 - y0,
	z1 - z0
])

v = sp.Matrix(3, 1, [
	x2 - x0,
	y2 - y0,
	z2 - z0
])

len_u = norm2(u)
len_v = norm2(v)

u = u / len_u
v = v / len_v

n = cross(u, v)
len_n = norm2(n)

n = n / len_n

wss = dot3(n, C * n) * n
wssmag = norm2(wss)

expr = []

assign = sp.symbols(f'wssmag[0]')
expr.append(ast.Assignment(assign, wssmag))

c_code(expr)
