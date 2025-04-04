#!/usr/bin/env python3

import sympy as sp
from sfem_codegen import *
import sys
from sympy.integrals.quadrature import gauss_lobatto, gauss_legendre


def lagr(order, x, qx):
    N = order + 1
    l = [x * 0 + 1] * N

    for j in range(0, N):

        for m in range(0, N):
            if m == j:
                continue

            l[j] = l[j] * (x - qx[m]) / (qx[j] - qx[m])

    for i in range(0, len(l)):
        l[i] = sp.simplify(l[i])

    return sp.Matrix(len(l), 1, l)


def qrule(N):
    qx, qw = gauss_lobatto(N, 10)

    qx = np.array(qx)
    qw = np.array(qw)

    # Points
    for i in range(0, N):
        qx[i] = (qx[i] + 1) / 2

    sum_qw = 0
    for i in range(0, N):
        sum_qw += qw[i]
    return qx, qw


def plot_lagr(order):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 200)

    N = order + 1
    qx, qw = qrule(N)

    l = lagr(order, x, qx)
    plt.figure()
    for i in range(len(l)):
        plt.plot(x, l[i], label=f"f{i}")
        plt.scatter(qx, qx * 0.0 + 1)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_lagr(16)
    # exit()

    code = "template<typename scalar_t>\n"
    code += "int lagrange_GLL_eval(const int order, const int Q, const scalar_t *const SFEM_RESTRICT qx, scalar_t * const SFEM_RESTRICT S)\n"
    code += "{\n\n"
    code += "const int N = order + 1;\n"
    code += "switch(order) {\n"

    for order in [1, 2, 4, 8, 16]:
        N = order + 1

        qx, qw = qrule(N)

        x = sp.symbols("x")

        l = lagr(order, x, qx)
        expr = assign_matrix("Sq", l)
        code += f"case {order}:\n"
        code += "{\n"
        code += "for(int q = 0; q < Q; q++) {\n"
        code += "const scalar_t x = qx[q];\n"
        code += "scalar_t * const Sq = &S[q * N];"
        code += c_gen(expr)
        code += "}\n"
        code += "\nbreak;\n"
        code += "}\n"

    code += "default: return 1;\n"

    code += "}\n\n"
    code += "return 0;\n"
    code += "}\n"

    code += "template<typename scalar_t>\n"
    code += "int lagrange_GLL_diff_eval(const int order, const int Q, const scalar_t *const SFEM_RESTRICT qx, scalar_t * const SFEM_RESTRICT D)\n"
    code += "{\n\n"
    code += "const int N = order + 1;\n"
    code += "switch(order) {\n"

    for order in [1, 2, 4, 8, 16]:
        N = order + 1
        x = sp.symbols("x")
        h = sp.Rational(1, order)
        qx, qw = qrule(N)

        l = lagr(order, x, qx)
        for i in range(0, len(l)):
            l[i] = sp.diff(l[i], x)

        expr = assign_matrix("Dq", l)
        code += f"case {order}:\n"
        code += "{\n"
        code += "for(int q = 0; q < Q; q++) {\n"
        code += "const scalar_t x = qx[q];\n"
        code += "scalar_t * const Dq = &D[q * N];"
        code += c_gen(expr)
        code += "}\n"
        code += "\nbreak;\n"
        code += "}\n"

    code += "default: return 1;\n"

    code += "}\n\n"
    code += "return 0;\n"
    code += "}\n"

    print(code)
