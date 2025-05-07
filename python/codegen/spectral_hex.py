#!/usr/bin/env python3


# Sources
# https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=Yi60qASPO7tg
# https://www.sciencedirect.com/science/article/abs/pii/S0098300421002909?via%3Dihub

from sfem_codegen import *
from sympy import init_printing
from sympy import pprint

import pdb
init_printing()

# eqviz = True
eqviz = False

use_factorization = True
# use_factorization = False

optimizations = "basic"
# optimizations=None


def mprint(expr):
    if eqviz:
        pprint(expr)
    else:
        print(expr)


def QS_mat(name, Q, N):
    list_coeffs = []

    for i in range(0, Q):
        for j in range(0, N):
            if eqviz:
                ui = sp.symbols(f"{name}^{i}_{j}", real=True)
            else:
                ui = sp.symbols(f"{name}[{i*N + j}]", real=True)

            list_coeffs.append(ui)

    ret = sp.Matrix(Q, N, list_coeffs)
    return ret


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


def tensor_element_laplacian(N, D, FFF, qw, u):
    N2 = N * N
    N3 = N2 * N

    gx = sp.zeros(N3, 1)
    gy = sp.zeros(N3, 1)
    gz = sp.zeros(N3, 1)

    out = sp.zeros(N3, 1)

    use_intermediate = False
    # use_intermediate = True

    ret = {}

    for k in range(0, N):
        for j in range(0, N):
            for i in range(0, N):
                idx = k * N2 + j * N + i

                D0 = D[N * i:]
                D1 = D[N * j:]
                D2 = D[N * k:]

                u0 = u[k * N2 + j * N:]
                u1 = u[k * N2 + i:]
                u2 = u[j * N + i:]

                acc = [D0[0] * u0[0], D1[0] * u1[0], D2[0] * u2[0]]
                for n in range(1, N):
                    acc[0] += D0[n] * u0[n]
                    acc[1] += D1[n] * u1[n * N]
                    acc[2] += D2[n] * u2[n * N2]

                gxq = FFF[0] * acc[0] + FFF[1] * acc[1] + FFF[2] * acc[2]
                gyq = FFF[1] * acc[0] + FFF[3] * acc[1] + FFF[4] * acc[2]
                gzq = FFF[2] * acc[0] + FFF[4] * acc[1] + FFF[5] * acc[2]
                w = qw[i] * qw[j] * qw[k]

                gx[idx] = gxq * w
                gy[idx] = gyq * w
                gz[idx] = gzq * w


    if use_intermediate:
        ret["gx"] = gx
        ret["gy"] = gy
        ret["gz"] = gz 

        gx = coeffs("gx", N3)
        gy = coeffs("gy", N3)
        gz = coeffs("gz", N3)     

    for k in range(0, N):
        for j in range(0, N):
            for i in range(0, N):
                D0 = D[i:]
                D1 = D[j:]
                D2 = D[k:]

                g0 = gx[k * N2 + j * N:]
                g1 = gy[k * N2 + i:]
                g2 = gz[j * N + i:]

                acc = [D0[0] * g0[0], D1[0] * g1[0], D2[0] * g2[0]]
                for n in range(1, N):
                    nidx = n * N
                    acc[0] += D0[nidx] * g0[n]
                    acc[1] += D1[nidx] * g1[nidx]
                    acc[2] += D2[nidx] * g2[n * N2]

                out[k * N2 + j * N + i] += acc[0] + acc[1] + acc[2]

    ret["out"] = out

    for k, v in ret.items():
        print(f"// {k}")
        expr = assign_matrix(k, v)
        c_code(expr, optimizations)


def tensor_element_interpolate(S, u):
    Q, N = S.shape

    N2 = N * N
    N3 = N2 * N

    Q2 = Q * Q
    Q3 = Q2 * Q

    def kji(k, j, i):
        return k * N * N + j * N + i

    def kj(k, j):
        return k * N + j

    ret = {}

    S1xU = sp.zeros(Q, N2)
    for qi in range(0, Q):
        for k in range(0, N):
            for j in range(0, N):
                for i in range(0, N):
                    # Sum over dimension 0
                    S1xU[qi, kj(k, j)] += S[qi, i] * u[kji(k, j, i)]

    ret["S1xU"] = S1xU
    if not eqviz and use_factorization:
        S1xU = matrix_coeff("S1xU", Q, N2)

    # pprint(S1xU)
    S2xS1xU = sp.zeros(Q2, N)

    for qj in range(0, Q):
        for qi in range(0, Q):
            for k in range(0, N):
                for j in range(0, N):
                    # Sum over dimension 1
                    S2xS1xU[qj * Q + qi, k] += S[qj, j] * S1xU[qi, kj(k, j)]

    ret["S2xS1xU"] = S2xS1xU
    if not eqviz and use_factorization:
        S2xS1xU = matrix_coeff("S2xS1xU", Q2, N)

    S3xS2xS1xU = sp.zeros(Q3, 1)

    for qk in range(0, Q):
        for qj in range(0, Q):
            for qi in range(0, Q):
                for k in range(0, N):
                    # Sum over dimension 2
                    S3xS2xS1xU[qk * Q * Q + qj * Q + qi] += (
                        S[qk, k] * S2xS1xU[qj * Q + qi, k]
                    )

    ret["S3xS2xS1xU"] = S3xS2xS1xU

    if eqviz:
        for i in range(0, Q3):
            print(f"{i})")
            mprint(S3xS2xS1xU[i])
    elif not use_factorization:
        expr = assign_matrix("S3xS2xS1xU", S3xS2xS1xU)
        c_code(expr, optimizations)
    else:
        for k, v in ret.items():
            print(f"// {k}")
            expr = assign_matrix(k, v)
            c_code(expr, optimizations)

    return S3xS2xS1xU


def tensor_element_integrate(S, qw, q):
    Q, N = S.shape

    N2 = N * N
    N3 = N2 * N

    Q2 = Q * Q
    Q3 = Q2 * Q

    def kji(k, j, i):
        return k * N * N + j * N + i

    def kj(k, j):
        return k * N + j

    def q_kji(k, j, i):
        return k * Q * Q + j * Q + i

    def q_kj(k, j):
        return k * Q + j

    ret = {}

    S1TxQ = sp.zeros(N, Q2)
    for i in range(0, N):
        for qk in range(0, Q):
            for qj in range(0, Q):
                for qi in range(0, Q):
                    # Sum over dimension 0
                    S1TxQ[i, q_kj(qk, qj)] += S[qi, i] * q[q_kji(qk, qj, qi)] * qw[qi]

    ret["S1TxQ"] = S1TxQ
    if not eqviz and use_factorization:
        S1TxQ = matrix_coeff("S1TxQ", N, Q2)

    S2TxS1TxQ = sp.zeros(N2, Q)

    for j in range(0, N):
        for i in range(0, N):
            for qk in range(0, Q):
                for qj in range(0, Q):
                    # Sum over dimension 1
                    S2TxS1TxQ[kj(j, i), qk] += (
                        S[qj, j] * S1TxQ[i, q_kj(qk, qj)] * qw[qj]
                    )

    ret["S2TxS1TxQ"] = S2TxS1TxQ
    if not eqviz and use_factorization:
        S2TxS1TxQ = matrix_coeff("S2TxS1TxQ", N2, Q)

    S3TxS2TxS1TxQ = sp.zeros(N3, 1)

    for k in range(0, N):
        for j in range(0, N):
            for i in range(0, N):
                for qk in range(0, Q):
                    # Sum over dimension 2
                    S3TxS2TxS1TxQ[kji(k, j, i)] += (
                        S[qk, k] * S2TxS1TxQ[kj(j, i), qk] * qw[qk]
                    )

    ret["S3TxS2TxS1TxQ"] = S3TxS2TxS1TxQ

    if eqviz:
        for i in range(0, N3):
            print(f"{i})")
            mprint(S3TxS2TxS1TxQ[i])
    elif not use_factorization:
        expr = assign_matrix("S3TxS2TxS1TxQ", S3TxS2TxS1TxQ)
        c_code(expr, optimizations)
    else:
        for k, v in ret.items():
            print(f"// {k}")
            expr = assign_matrix(k, v)
            c_code(expr, optimizations)


# def tensor_element_laplacian(S, D, fff, qw, u):
# D0 = D x S x S
# D1 = S x D x S
# D2 = S x S x D
# G = [D0, D1, D2]

# L * u = ((G  * u) * fff)^T * G


N = 2
Q = 2

# 1D_fun x quad_points
S = QS_mat("S", Q, N)
u = coeffs("u", N * N * N)

q = tensor_element_interpolate(S, u)

print("\n//----------------------------\n")

qw = coeffs("qw", Q)

if not eqviz and use_factorization:
    q = coeffs("q", Q * Q * Q)

tensor_element_integrate(S, qw, q)

print("\n//----------------------------\n")
print("Laplacian\n")
print("\n//----------------------------\n")


def hex8_coeffs():
    u = coeffs("u", 2 * 2* 2)
    u2 = u[2]
    u3 = u[3]
    u6 = u[6]
    u7 = u[7]

    u[2] = u3
    u[3] = u2

    u[6] = u7
    u[7] = u6
    return u


tensor_element_laplacian(
    2,
    sp.Matrix(4, 1, [-1, 1, -1, 1]),
    matrix_coeff("fff", 3, 3),
    sp.Matrix(2, 1, [0.5, 0.5]),
    hex8_coeffs(),
)
