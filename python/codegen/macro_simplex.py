#!/usr/bin/env python3

import numpy as np

# import pdb
from fe import FE
import sympy as sp
from sfem_codegen import *

dim = 2

if dim == 2:
    sub_simplices = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])

    edges = [
        [0, 1],
        [1, 2],
        [2, 0],
    ]

    unique = [0, 3]

    c = coeffs("fff", 3)
    FFFs = sp.Matrix(dim, dim, [c[0], c[1], c[1], c[2]])

    ref_volume = sp.Rational(1, 2)

else:
    sub_simplices = [
        [0, 4, 6, 7],
        [4, 1, 5, 8],
        [6, 5, 2, 9],
        [7, 8, 9, 3],
        [4, 5, 6, 8],
        [7, 4, 6, 8],
        [6, 5, 9, 8],
        [7, 6, 9, 8],
    ]

    edges = [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]]

    unique = [0, 4, 5, 6, 7]

    c = coeffs("fff", 6)
    FFFs = sp.Matrix(dim, dim, [c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5]])

    ref_volume = sp.Rational(1, 6)

adjugate = strided_matrix_coeff("adjugate", dim, dim, "stride")

detFFF = determinant(FFFs)
print("---------------- detFFF ----------------")
c_code(detFFF)
print("----------------------------------------")


def read_file(path):
    with open(path, "r") as f:
        tpl = f.read()
        return tpl
    assert False
    return ""


def str_to_file(path, mystr):
    with open(path, "w") as f:
        f.write(mystr)
        f.close()


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


def assign_adjugate(name, mat):
    rows, cols = mat.shape

    expr = []
    idx = 0
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{idx}]")
            expr.append(ast.Assignment(var, mat[i, j]))
            idx += 1
    return expr


def sub_fff_generic(micro_ref, FFFs):
    Am = sp.zeros(dim, dim)

    for d1 in range(0, dim):
        for d2 in range(0, dim):
            Am[d2, d1] = micro_ref[d1 + 1][d2] - micro_ref[0][d2]

    detAm = determinant(Am)
    Aminv = inverse(Am)

    # inv(J * A) = (inv(A) * inv(J))
    # inv(J * A)^T = (inv(J)^T * inv(A)^T)
    # <(inv(J)^T * inv(A)^T) gi, (inv(J)^T * inv(A)^T) gj>
    # <(inv(J)^T * inv(A)*T)^T * inv(J)^T * inv(A)^T gi, gj>
    # <inv(A) * inv(J) * inv(J)^T * inv(A)^T gi, gj>
    # <inv(A) * FFF * inv(A)^T gi, gj>

    FFAms = Aminv * FFFs * Aminv.T * detAm

    # print("------------------")
    # print(Aminv)
    # print(detAm)
    # print(FFAms)
    return FFAms


def sub_adj_generic(micro_ref, adj):
    Am = sp.zeros(dim, dim)

    for d1 in range(0, dim):
        for d2 in range(0, dim):
            Am[d2, d1] = micro_ref[d1 + 1][d2] - micro_ref[0][d2]

    detAm = determinant(Am)
    Aminv = inverse(Am)

    return Aminv * adj, detAm


def subJ(micro_ref):
    for d1 in range(0, dim):
        for d2 in range(d1 + 1, dim):
            FFFs[d2, d1] = FFFs[d1, d2]

    return sub_fff_generic(micro_ref, FFFs)


def sub_adjugate(micro_ref):
    return sub_adj_generic(micro_ref, adjugate)


def fff_code_basic(FFF):
    funs = []
    tpl = read_file("tpl/macro_sub_jacobian_tpl.c")
    code0to2 = tpl.format(NUM="0to2", CODE=c_gen(assign_fff("sub_fff", FFF[0])))

    code3 = tpl.format(NUM="3", CODE=c_gen(assign_fff("sub_fff", FFF[3])))

    funs = [code0to2, code3]
    return funs


def fff_code(name, FFF):
    funs = []
    tpl = read_file("tpl/macro_sub_jacobian_tpl.c")
    code = tpl.format(NUM=name, CODE=c_gen(assign_fff("sub_fff", FFF)))

    return code


def adjugate_code(name, adj):
    funs = []
    tpl = read_file("tpl/macro_sub_adjugate_tpl.c")
    code = tpl.format(NUM=name, CODE=c_gen(assign_adjugate("sub_adjugate", adj)))

    return code


class MacroSimplex:
    def __init__(self):
        points = []
        points.append(sp.zeros(dim, 1))
        for d1 in range(0, dim):
            p = sp.zeros(dim, 1)
            p[d1] = 1
            points.append(p)

        for e in edges:
            p = (points[e[0]] + points[e[1]]) / 2
            points.append(p)

        self.points = np.array(points)

    def adjugate_level_n(self, n_levels):
        levels = [[]] * n_levels
        levels_det = [[]] * n_levels
        x = self.points

        for ss in sub_simplices:
            refpattern = x[ss]
            adj, adj_det = sub_adjugate(refpattern)
            levels[0].append(adj)
            levels_det[0].append(adj_det)

        for l in range(1, n_levels):
            n_ffs = len(levels[l - 1])
            levels[l] = [0] * (n_ffs * 2)

            for i in range(0, n_ffs):
                adj = levels[l - 1][i]
                for ss in sub_simplices:
                    refpattern = x[ss]
                    sub_adj, sub_det = sub_adj_generic(refpattern, adj)
                    levels[l].append(sub_adj)
                    levels_det[l].append(sub_det)
        return levels, levels_det

    def fff_level_n(self, n_levels):
        levels = [[]] * n_levels
        x = self.points

        for ss in sub_simplices:
            refpattern = x[ss]
            fff = subJ(refpattern)
            levels[0].append(fff)

        for l in range(1, n_levels):
            n_ffs = len(levels[l - 1])
            levels[l] = [0] * (n_ffs * 2)

            for i in range(0, n_ffs):
                fff = levels[l - 1][i]
                for ss in sub_simplices:
                    refpattern = x[ss]
                    sub_fff = sub_fff_generic(refpattern, fff)
                    levels[l].append(sub_fff)

        self.levels = levels
        return levels

    def plot_test(self):
        import meshio

        points = np.array(self.points, dtype=np.float32)
        cells = []

        if dim == 3:
            cells.append(("tetra", np.array(sub_simplices, np.int32)))
            mesh = meshio.Mesh(points, cells)
            mesh.write("macro_simplex.vtk")

        x = self.points[:, 0].flatten()
        y = self.points[:, 1].flatten()

        if dim == 3:
            z = self.points[:, 2].flatten()
            J = sp.Matrix(
                dim,
                dim,
                [
                    x[1] - x[0],
                    x[2] - x[0],
                    x[3] - x[0],
                    y[1] - y[0],
                    y[2] - y[0],
                    y[3] - y[0],
                    z[1] - z[0],
                    z[2] - z[0],
                    z[3] - z[0],
                ],
            )
        elif dim == 2:
            J = sp.Matrix(
                dim, dim, [x[1] - x[0], x[2] - x[0], y[1] - y[0], y[2] - y[0]]
            )
        else:
            assert False

        dJ = determinant(J)
        Jinv = inverse(J)

        FFF = (Jinv * Jinv.T) * dJ * ref_volume

        for i in range(0, len(sub_simplices)):
            ss = sub_simplices[i]

            xi = x[ss]
            yi = y[ss]

            if dim == 3:
                zi = z[ss]

                Ji = sp.Matrix(
                    dim,
                    dim,
                    [
                        xi[1] - xi[0],
                        xi[2] - xi[0],
                        xi[3] - xi[0],
                        yi[1] - yi[0],
                        yi[2] - yi[0],
                        yi[3] - yi[0],
                        zi[1] - zi[0],
                        zi[2] - zi[0],
                        zi[3] - zi[0],
                    ],
                )

            elif dim == 2:
                Ji = sp.Matrix(
                    dim,
                    dim,
                    [xi[1] - xi[0], xi[2] - xi[0], yi[1] - yi[0], yi[2] - yi[0]],
                )
            else:
                assert False

            dJi = determinant(Ji)
            Jiinv = inverse(Ji)

            FFFi = (Jiinv * Jiinv.T) * dJi * ref_volume
            FFFj = self.levels[0][i]

            for i in range(0, dim):
                for j in range(0, dim):
                    for d1 in range(0, dim):
                        for d2 in range(0, dim):
                            FFFj[i, j] = FFFj[i, j].subs(FFFs[d1, d2], FFF[d1, d2])

            diff = 0
            for d1 in range(0, dim):
                for d2 in range(0, dim):
                    diff += FFFi[d1, d2] - FFFj[d1, d2]

            assert diff == 0


nl = 1
ms = MacroSimplex()
fffl = ms.fff_level_n(nl)
ms.plot_test()

for l in range(0, nl):
    num = 0
    print(f"level {l} #fff {len(fffl[l])}")
    for i in range(0, len(fffl[l])):
        f = fffl[l][i]
        print(fff_code(f"{l}_{num}", f))
        num += 1


print("------------------------------------")
print("ADJUGATE")
print("------------------------------------")

fffl, fffl_det = MacroSimplex().adjugate_level_n(nl)

for l in range(0, nl):
    num = 0
    print(f"level {l} #adjugate {len(fffl[l])}")
    # for i in unique:
    for i in range(0, len(fffl[l])):
        f = fffl[l][i]
        f_det = fffl_det[l][i]
        if nl > 1:
            print(adjugate_code(f"{l}_{num}", f))
            print(c_code(f_det))

        else:
            print(adjugate_code(f"{num}", f))
            print(c_code(f_det))
        num += 1
