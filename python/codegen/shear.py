#!/usr/bin/env python3

# ./shear.py

from sfem_codegen import *

dV = det3(A)

ux = coeffs("ux", 4)
uy = coeffs("uy", 4)
uz = coeffs("uz", 4)

f = fun(qx, qy, qz)

gradf = sp.Matrix(3, 3, [0.0] * 9)

for i in range(0, 4):
    fi = f[i]
    dfdx = sp.diff(fi, qx)
    dfdy = sp.diff(fi, qy)
    dfdz = sp.diff(fi, qz)

    gradf[0, 0] += dfdx * ux[i]
    gradf[0, 1] += dfdy * ux[i]
    gradf[0, 2] += dfdz * ux[i]

    gradf[1, 0] += dfdx * uy[i]
    gradf[1, 1] += dfdy * uy[i]
    gradf[1, 2] += dfdz * uy[i]

    gradf[2, 0] += dfdx * uz[i]
    gradf[2, 1] += dfdy * uz[i]
    gradf[2, 2] += dfdz * uz[i]

shear = 0.5 * (gradf + gradf.T)

# c_log(gradf)

expr = []

shear0 = sp.symbols(f"shear[0]")
expr.append(ast.Assignment(shear0, shear[0, 0]))

shear1 = sp.symbols(f"shear[1]")
expr.append(ast.Assignment(shear1, shear[0, 1]))

shear2 = sp.symbols(f"shear[2]")
expr.append(ast.Assignment(shear2, shear[0, 2]))

shear3 = sp.symbols(f"shear[3]")
expr.append(ast.Assignment(shear3, shear[1, 1]))

shear4 = sp.symbols(f"shear[4]")
expr.append(ast.Assignment(shear4, shear[1, 2]))

shear5 = sp.symbols(f"shear[5]")
expr.append(ast.Assignment(shear5, shear[2, 2]))

c_code(expr)
