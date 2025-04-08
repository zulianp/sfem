#!/usr/bin/env python3

from sfem_codegen import *

c = coeffs("fff", 6)
FFF = sp.Matrix(3, 3, [c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5]])

L = sp.symbols("L")

J = sp.eye(3, 3)

u = J[:, 0]
v = J[:, 1]
w = J[:, 2]

J0 = J / L
J1 = sp.zeros(3, 3)
J2 = sp.zeros(3, 3)
J3 = sp.zeros(3, 3)
J4 = sp.zeros(3, 3)
J5 = sp.zeros(3, 3)

# Cat 1
J1[:, 0] = -u + w
J1[:, 1] = w
J1[:, 2] = -u + v + w
J1 = J1 / L

# Cat 2
J2[:, 0] = v
J2[:, 1] = -u + v + w
J2[:, 2] = w
J2 = J2 / L

# Cat 3
J3[:, 0] = -u + v
J3[:, 1] = -u + w
J3[:, 2] = -u + v + w
J3 = J3 / L

# Cat 4
J4[:, 0] = -v + w
J4[:, 1] = w
J4[:, 2] = -u + w
J4 = J4 / L

# Cat 5
J5[:, 0] = -u + v
J5[:, 1] = -u + v + w
J5[:, 2] = v
J5 = J5 / L

Js = [J0, J1, J2, J3, J4, J5]


def read_file(path):
    with open(path, "r") as f:
        tpl = f.read()
        return tpl
    assert False
    return ""


def assign_fff(name, mat):
    rows, cols = mat.shape

    expr = []
    idx = 0
    for i in range(0, rows):
        for j in range(i, cols):
            var = sp.symbols(f"{name}[{idx}]")
            expr.append(ast.Assignment(var, mat[i, j]))
            idx += 1
    return expr


def fff_code(name, FFF):
    funs = []
    tpl = read_file("tpl/macro_sub_jacobian_tpl.c")
    code = tpl.format(NUM=name, CODE=c_gen(assign_fff("sub_fff", FFF)))

    return code


C = 0
for Ji in Js:
    detJi = determinant(Ji)
    Ji_inv = inverse(Ji)

    FFFi = Ji_inv * FFF * Ji_inv.T * detJi

    print(fff_code(f"{C}", FFFi))
    C += 1
