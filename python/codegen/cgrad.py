#!/usr/bin/env python3

from sfem_codegen import *

dV = det3(A)
u = coeffs("f", 4)
f = fun(qx, qy, qz)

gradf = sp.Matrix(3, 1, [0.0, 0.0, 0.0])

# sum_i u_i * grad(phi_i)
for i in range(0, 4):
    fi = f[i]
    dfdx = sp.diff(fi, qx)
    dfdy = sp.diff(fi, qy)
    dfdz = sp.diff(fi, qz)

    gradf[0] += dfdx * u[i]
    gradf[1] += dfdy * u[i]
    gradf[2] += dfdz * u[i]

c_log(gradf)

expr = []

dfdx = sp.symbols(f"dfdx[0]")
expr.append(ast.Assignment(dfdx, gradf[0]))

dfdy = sp.symbols(f"dfdy[0]")
expr.append(ast.Assignment(dfdy, gradf[1]))

dfdz = sp.symbols(f"dfdz[0]")
expr.append(ast.Assignment(dfdz, gradf[2]))

c_code(expr)
