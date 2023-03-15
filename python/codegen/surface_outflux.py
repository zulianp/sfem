#!/usr/bin/env python3

from sfem_codegen import *

S = sp.Matrix(3, 2, [
	 x1 - x0, x2 - x0,
	 y1 - y0, y2 - y0,
	 z1 - z0, z2 - z0
	])

StS = S.T * S
dS = sp.sqrt(det2(StS))

vx = coeffs('vx', 3)
vy = coeffs('vy', 3)
vz = coeffs('vz', 3)

nx, ny, nz = sp.symbols('nx ny nz')

rf = [1 - qx - qy, qx, qy]

tx = 0.
ty = 0.
tz = 0.

for i in range(0, 3):
	tx += rf[i] * vx[i]
	ty += rf[i] * vy[i]
	tz += rf[i] * vz[i]

# dot product
f = tx * nx + ty * ny + tz * nz

# dS outside since it is constant
integr = sp.integrate(f, (qy, 0, 1 - qx), (qx, 0, 1)) * dS

expr = []
res = sp.symbols(f'value[0]')
expr.append(ast.Assignment(res, sp.simplify(integr)))

c_code(expr)
