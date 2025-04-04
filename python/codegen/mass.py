#!/usr/bin/env python3

from sfem_codegen import *

rf = ref_fun(qx, qy, qz)
dV = det3(A)

expr = []
sumterms = 0
for i in range(0, 4):
    for j in range(0, 4):
        form = rf[i] * rf[j] * dV
        integr = sp.integrate(form, (qz, 0, 1 - qx - qy), (qy, 0, 1 - qx), (qx, 0, 1))

        bform = sp.symbols(f"element_matrix[{i*4+j}]")
        expr.append(ast.Assignment(bform, sp.simplify(integr)))
        sumterms += integr

c_code(expr)

print("Test:")

test = sumterms.subs(x0, 0)
test = test.subs(x1, 1)
test = test.subs(x2, 0)
test = test.subs(x3, 0)

test = test.subs(y0, 0)
test = test.subs(y1, 0)
test = test.subs(y2, 1)
test = test.subs(y3, 0)

test = test.subs(z0, 0)
test = test.subs(z1, 0)
test = test.subs(z2, 0)
test = test.subs(z3, 1)

print(f"{test} = 1/6")
