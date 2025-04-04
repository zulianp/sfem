#!/usr/bin/env python3

import sympy as sp
from sfem_codegen import *


# Uniform mesh


def uniform_lagr(order, x, h):
    N = order + 1
    l = [x * 0 + 1] * N

    for j in range(0, N):

        for m in range(0, N):
            if m == j:
                continue

            l[j] = l[j] * (x - m * h) / (j * h - m * h)

    for i in range(0, len(l)):
        l[i] = sp.simplify(l[i])

    return sp.Matrix(len(l), 1, l)


def plot_uniform_lagr(order):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 200)
    h = sp.Rational(1, order)

    l = uniform_lagr(order, x, h)
    plt.figure()
    for i in range(len(l)):
        plt.plot(x, l[i], label=f"f{i}")
        plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_uniform_lagr(4)
    # exit()

    code = "template<typename scalar_t>\n"
    code += "int lagrange_eval(const int order, const int Q, const scalar_t *const SFEM_RESTRICT qx, scalar_t * const SFEM_RESTRICT S)\n"
    code += "{\n\n"
    code += "const int N = order + 1;\n"
    code += "switch(order) {\n"

    for order in [1, 2, 4, 8, 16]:
        N = order + 1
        x = sp.symbols("x")
        h = sp.Rational(1, order)
        l = uniform_lagr(order, x, h)
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
    code += "int lagrange_diff_eval(const int order, const int Q, const scalar_t *const SFEM_RESTRICT qx, scalar_t * const SFEM_RESTRICT D)\n"
    code += "{\n\n"
    code += "const int N = order + 1;\n"
    code += "switch(order) {\n"

    for order in [1, 2, 4, 8, 16]:
        N = order + 1
        x = sp.symbols("x")
        h = sp.Rational(1, order)
        l = uniform_lagr(order, x, h)
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
