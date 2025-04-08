#!/usr/bin/env python3

from sfem_codegen import *

mu, lmbda = sp.symbols("mu lambda")

dV = det3(A) / 6
eps = symm_grad(qx, qy, qz)

# Preallocate (for preserving write order)
expr = [0] * (4 * 3 * 4 * 3)

for i in range(0, 4 * 3):
    for j in range(i, 4 * 3):
        # Bilinear form
        integr = (2 * mu * inner(eps[i], eps[j]) + lmbda * tr(eps[i]) * tr(eps[j])) * dV

        # Simplify expressions (switch comment on lines for reducing times)
        sintegr = sp.simplify(integr)
        # sintegr = integr

        # Store results in array
        bform1 = sp.symbols(f"element_matrix[{i*(4*3)+j}]")
        expr[i * 4 * 3 + j] = ast.Assignment(bform1, sintegr)

        # Take advantage of symmetry to reduce code-gen times
        if i != j:
            bform2 = sp.symbols(f"element_matrix[{i+(4*3)*j}]")
            expr[i + 4 * 3 * j] = ast.Assignment(bform2, sintegr)

c_code(expr)
