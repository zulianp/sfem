#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt

# https://nalu-wind.readthedocs.io/en/latest/source/theory/discretizationApproach.html

# Quad4 nodes
p0 = sp.Matrix(2, 1, [x0, y0])
p1 = sp.Matrix(2, 1, [x1, y1])
p2 = sp.Matrix(2, 1, [x2, y2])
p3 = sp.Matrix(2, 1, [x3, y3])

# Integration points
qx = [0.5, 0.75, 0.5, 0.25]
qy = [0.25, 0.5, 0.75, 0.5]


def perp(e):
    return sp.Matrix(2, 1, [-e[1], e[0]])


def fun(x, y):
    return [(1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y]


def p(x, y):
    f = fun(x, y)
    px = f[0] * p0[0] + f[1] * p1[0] + f[2] * p2[0] + f[3] * p3[0]
    py = f[0] * p0[1] + f[1] * p1[1] + f[2] * p2[1] + f[3] * p3[1]
    return [px, py]


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


def cv_interp(v):
    vx = sp.zeros(4, 1)
    for i in range(0, 4):
        f = fun(qx[i], qy[i])
        for j in range(0, 4):
            vx[i] += f[j] * v[j]
    return vx


def cv_normals():
    b = (p0 + p1 + p2 + p3) / 4
    dn = []
    for i in range(0, 4):
        pi = p(qx[i], qy[i])
        ei = [pi[0] - b[0], pi[1] - b[1]]
        dn.append(2 * perp(ei))
    return dn


def advective_fluxes(vx, vy, dn):
    q = []
    for i in range(0, 4):
        qi = vx[i] * dn[i][0] + vy[i] * dn[i][1]
        q.append(qi)

    return q


def pw_max(a, b):
    return sp.Piecewise((b, a < b), (a, True))


def advection_op(q):
    A = sp.zeros(4, 4)

    # Node 0
    A[0, 0] = -pw_max(q[0], 0) - pw_max(-q[3], 0)
    A[0, 1] = pw_max(-q[0], 0)
    A[0, 2] = 0
    A[0, 3] = pw_max(q[3], 0)

    # Node 1
    A[1, 1] = -pw_max(-q[0], 0) - pw_max(q[1], 0)
    A[1, 0] = pw_max(q[0], 0)
    A[1, 2] = pw_max(-q[1], 0)
    A[1, 3] = 0

    # Node 2
    A[2, 2] = -pw_max(-q[1], 0) - pw_max(q[2], 0)
    A[2, 0] = 0
    A[2, 1] = pw_max(q[1], 0)
    A[2, 3] = pw_max(-q[2], 0)

    # Node 3
    A[3, 3] = -pw_max(q[3], 0) - pw_max(-q[2], 0)
    A[3, 0] = pw_max(-q[3], 0)
    A[3, 1] = 0
    A[3, 2] = pw_max(q[2], 0)

    return A


def ref_subs(expr):
    expr = expr.subs(x0, 0)
    expr = expr.subs(y0, 0)

    expr = expr.subs(x1, 1)
    expr = expr.subs(y1, 0)

    expr = expr.subs(x2, 1)
    expr = expr.subs(y2, 1)

    expr = expr.subs(x3, 0)
    expr = expr.subs(y3, 1)
    return expr


vx = coeffs("vx", 4)
vy = coeffs("vy", 4)

vcx = cv_interp(vx)
vcy = cv_interp(vy)

dn = cv_normals()
q = advective_fluxes(vcx, vcy, dn)

# for e in vcx:
# 	print(e)

# for e in dn:
# print(ref_subs(e))

# for e in q:
# 	print(e)

print("----------------------------")
print("Hessian")
print("----------------------------")

A = advection_op(q)
expr = assign_matrix("element_matrix", A)
c_code(expr)

print("----------------------------")
print("Apply")
print("----------------------------")

x = coeffs("x", 4)
y = A * x
expr = assign_matrix("element_vector", y)
c_code(expr)

# Check on ref element
# if False:
if True:
    for i in range(0, 4):

        line = ""

        for j in range(0, 4):
            su = ref_subs(A[i, j])

            for v in vx:
                su = su.subs(v, 0)

            for v in vy:
                su = su.subs(v, 1)

            line += f"{round(su, 1)} "

        print(line)
        print("\n")
