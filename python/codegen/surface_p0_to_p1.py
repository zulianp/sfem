#!/usr/bin/env python3

from sfem_codegen import *

S = sp.Matrix(3, 2, [x1 - x0, x2 - x0, y1 - y0, y2 - y0, z1 - z0, z2 - z0])

StS = S.T * S
dS = sp.sqrt(det2(StS))
rf = [1 - qx - qy, qx, qy]

u = coeffs("u_p0", 1)

expr = []
for i in range(0, 3):
    integr = sp.simplify(sp.integrate(rf[i], (qy, 0, 1 - qx), (qx, 0, 1)) * dS)

    lhs = sp.symbols(f"weight[{i}]")
    expr.append(ast.Assignment(lhs, integr))

    lhs = sp.symbols(f"u_p1[{i}]")
    expr.append(ast.Assignment(lhs, u[0] * integr))

c_code(expr)

print("-----------------------")
print("No-weights version")
print("-----------------------")

expr = []
for i in range(0, 3):
    integr = sp.simplify(sp.integrate(rf[i], (qy, 0, 1 - qx), (qx, 0, 1)) * dS)
    lhs = sp.symbols(f"u_p1[{i}]")
    expr.append(ast.Assignment(lhs, u[0] * integr))

c_code(expr)
