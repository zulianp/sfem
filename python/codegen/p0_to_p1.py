#!/usr/bin/env python3

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

u = coeffs("u_p0", 1)
expr = []

dual = [
    4 * rf[0] - rf[1] - rf[2] - rf[3],
    -rf[0] + 4 * rf[1] - rf[2] - rf[3],
    -rf[0] - rf[1] + 4 * rf[2] - rf[3],
    -rf[0] - rf[1] - rf[2] + 4 * rf[3],
]

# Should be equivalent
basis = rf
# basis = dual

for i in range(0, 4):
    integr = sp.simplify(
        sp.integrate(basis[i], (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) * dV
    )

    lhs = sp.symbols(f"weight[{i}]")
    expr.append(ast.Assignment(lhs, integr))

    lhs = sp.symbols(f"u_p1[{i}]")
    expr.append(ast.Assignment(lhs, u[0] * integr))

c_code(expr)

print("-----------------------")
print("No-weights version")
print("-----------------------")

expr = []
for i in range(0, 4):
    integr = sp.simplify(
        sp.integrate(basis[i], (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1)) * dV
    )
    lhs = sp.symbols(f"u_p1[{i}]")
    expr.append(ast.Assignment(lhs, u[0] * integr))

c_code(expr)
