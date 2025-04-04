#!/usr/bin/env python3

# ./div.py

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

cux = coeffs("ux", 4)
cuy = coeffs("uy", 4)
cuz = coeffs("uz", 4)

cu = [cux, cuy, cuz]
f = fun(qx, qy, qz)

# Evaluate u in the quadrature points
u = [0, 0, 0]
for d in range(0, 3):
    for i in range(0, 4):
        u[d] += cu[d][i] * rf[i]

# Symbolic integration
var_grad = sp.symbols("var_grad")
integrals = [0, 0, 0]
for d in range(0, 3):
    udotvar = u[d] * var_grad
    integr = (
        sp.integrate(udotvar, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) * dV
    )
    integrals[d] = integr

expr = []
for i in range(0, 4):
    fi = f[i]
    dfdx = sp.diff(fi, qx)
    dfdy = sp.diff(fi, qy)
    dfdz = sp.diff(fi, qz)
    g = [dfdx, dfdy, dfdz]

    integr = 0
    for d in range(0, 3):
        integr += integrals[d].subs(var_grad, g[d])

    var = sp.symbols(f"element_vector[{i}]")
    expr.append(ast.Assignment(var, integr))

c_code(expr)
