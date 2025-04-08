#!/usr/bin/env python3

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

expr = []

for i in range(0, 4):

    acc = 0
    for j in range(0, 4):
        form = rf[i] * rf[j] * dV
        integr = sp.integrate(form, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))
        acc += integr

    form = sp.symbols(f"element_vector[{i}]")
    expr.append(ast.Assignment(form, sp.simplify(acc)))

c_code(expr)
