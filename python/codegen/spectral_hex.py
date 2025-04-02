#!/usr/bin/env python3


# Sources
# https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=Yi60qASPO7tg
# https://www.sciencedirect.com/science/article/abs/pii/S0098300421002909?via%3Dihub

from sfem_codegen import *
from sympy import init_printing
from sympy import pprint

init_printing() 

# eqviz = True
eqviz = False

use_factorization = True
# use_factorization = False

optimizations='basic'
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
                ui = sp.symbols(f'{name}^{i}_{j}', real=True)
            else:
                ui = sp.symbols(f'{name}[{i*N + j}]', real=True)

            list_coeffs.append(ui)

    ret = sp.Matrix(Q, N, list_coeffs)
    return ret

def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f'{name}[{i*cols + j}]')
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr

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
                    S3xS2xS1xU[qk * Q * Q + qj * Q + qi] += S[qk, k] * S2xS1xU[qj * Q + qi, k]

    ret["S3xS2xS1xU"] = S3xS2xS1xU

    if eqviz:
        for i in range(0, Q3):
            print(f"{i})")
            mprint(S3xS2xS1xU[i])
    elif not use_factorization:
        expr = assign_matrix("S3xS2xS1xU", S3xS2xS1xU)
        c_code(expr, optimizations)
    else:
        for k,v in ret.items():
            print(f'// {k}')
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
                        S2TxS1TxQ[kj(j, i), qk] += S[qj, j] * S1TxQ[i, q_kj(qk, qj)] * qw[qj]

    ret["S2TxS1TxQ"] = S2TxS1TxQ
    if not eqviz and use_factorization:
        S2TxS1TxQ = matrix_coeff("S2TxS1TxQ", N2, Q)

    S3TxS2TxS1TxQ = sp.zeros(N3, 1)

    for k in range(0, N):
        for j in range(0, N):
            for i in range(0, N):
                for qk in range(0, Q):
                    # Sum over dimension 2
                    S3TxS2TxS1TxQ[kji(k, j, i)] += S[qk, k] * S2TxS1TxQ[kj(j, i), qk] * qw[qk]

    ret["S3TxS2TxS1TxQ"] = S3TxS2TxS1TxQ

    if eqviz:
        for i in range(0, N3):
            print(f"{i})")
            mprint(S3TxS2TxS1TxQ[i])
    elif not use_factorization:
        expr = assign_matrix("S3TxS2TxS1TxQ", S3TxS2TxS1TxQ)
        c_code(expr, optimizations)
    else:
        for k,v in ret.items():
            print(f'// {k}')
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
S = QS_mat('S', Q, N)
u = coeffs('u', N * N * N)

q = tensor_element_interpolate(S, u)

print("\n//----------------------------\n")

qw = coeffs("qw", Q)

if not eqviz and use_factorization:
    q = coeffs('q', Q * Q * Q)

tensor_element_integrate(S, qw, q)

