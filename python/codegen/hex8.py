#!/usr/bin/env python3

from fe import FE
from sfem_codegen import *
from weighted_fe import *
from quad4 import QuadShell4


class Hex8(FE):
    def __init__(self, isoparam=False):
        super().__init__()
        self.isoparam = isoparam

    def is_isoparametric(self):
        return self.isoparam

    def reference_measure(self):
        return 1

    def subparam_n_nodes(self):
        return 2

    def barycenter(self):
        return vec3(sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2))

    def coords_sub_parametric(self):
        xyz = self.coords()
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        return [[x[0], x[6]], [y[0], y[6]], [z[0], z[6]]]

    def coords(self):
        return [coeffs("x", 8), coeffs("y", 8), coeffs("z", 8)]

    def trace_element(self):
        return QuadShell4()

    def sides(self):
        return [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [3, 2, 1, 0],
            [4, 5, 6, 7],
        ]

    def map_to_side(self, side_idx, qp):
        assert len(qp) == 2
        x = qp[0]
        y = qp[1]
        match side_idx:
            case 0:
                return sp.Matrix(3, 1, [x, 0, y])
            case 1:
                return sp.Matrix(3, 1, [1, x, y])
            case 2:
                return sp.Matrix(3, 1, [1 - x, 1, y])
            case 3:
                return sp.Matrix(3, 1, [0, 1 - x, y])
            case 4:
                return sp.Matrix(3, 1, [x, 1 - y, 0])
            case 5:
                return sp.Matrix(3, 1, [x, y, 1])
            case _:
                assert False

    def select_coords(self, selection):
        xyz = self.coords()
        return [
            [xyz[0][j] for j in selection],
            [xyz[1][j] for j in selection],
            [xyz[2][j] for j in selection],
        ]

    def side_coords(self, side_idx):
        sides = self.sides()
        f = sides[side_idx]
        return self.select_coords(f)

    def fun(self, p):
        x = p[0]
        y = p[1]
        z = p[2]

        f = sp.zeros(8, 1)

        xm = 1 - x
        ym = 1 - y
        zm = 1 - z

        f[0] = xm * ym * zm  # (0, 0, 0)
        f[1] = x * ym * zm  # (1, 0, 0)
        f[2] = x * y * zm  # (1, 1, 0)
        f[3] = xm * y * zm  # (0, 1, 0)
        f[4] = xm * ym * z  # (0, 0, 1)
        f[5] = x * ym * z  # (1, 0, 1)
        f[6] = x * y * z  # (1, 1, 1)
        f[7] = xm * y * z  # (0, 1, 1)

        return f

    def n_sides(self):
        return 6

    def n_nodes(self):
        return 8

    def manifold_dim(self):
        return 3

    def spatial_dim(self):
        return 3

    def integrate(self, q, expr):
        return sp.integrate(expr, (q[2], 0, 1), (q[1], 0, 1), (q[0], 0, 1))

    def jacobian(self, q):
        return self.isoparametric_jacobian(q)

    def jacobian_inverse(self, q):
        return inverse(self.isoparametric_jacobian(q))

    def transform(self, q):
        f = self.fun(q)
        xyz = self.coords()

        pp = sp.zeros(3, 1)

        for i in range(0, 8):
            for d in range(0, 3):
                pp[d] += xyz[d][i] * f[i]

        return pp

    def inverse_transform(self, p):
        assert False
        return 0

    def jacobian_determinant(self, q):
        return det3(self.jacobian(q))

    def measure(self, q):
        return self.jacobian_determinant(q)

    def to_stencil(self, M):
        G = sp.zeros(27, 27)
        ld = [1, 3, 9]
        ii = [-1] * 8

        left = 0
        right = 2
        for k in range(left, right):
            for j in range(left, right):
                for i in range(left, right):

                    ii[0] = (i + 0) * ld[0] + (j + 0) * ld[1] + (k + 0) * ld[2]
                    ii[1] = (i + 1) * ld[0] + (j + 0) * ld[1] + (k + 0) * ld[2]
                    ii[2] = (i + 1) * ld[0] + (j + 1) * ld[1] + (k + 0) * ld[2]
                    ii[3] = (i + 0) * ld[0] + (j + 1) * ld[1] + (k + 0) * ld[2]
                    ii[4] = (i + 0) * ld[0] + (j + 0) * ld[1] + (k + 1) * ld[2]
                    ii[5] = (i + 1) * ld[0] + (j + 0) * ld[1] + (k + 1) * ld[2]
                    ii[6] = (i + 1) * ld[0] + (j + 1) * ld[1] + (k + 1) * ld[2]
                    ii[7] = (i + 0) * ld[0] + (j + 1) * ld[1] + (k + 1) * ld[2]

                    for l in range(0, 8):
                        for s in range(0, 8):
                            ll = ii[l]
                            ss = ii[s]
                            G[ll, ss] += M[l, s]

        ii_center = 1 * ld[0] + 1 * ld[1] + 1 * ld[2]
        stencil = G[ii_center, :]

        ret = {
            # "stencil111": stencil
        }

        xi, yi, zi = sp.symbols("xi yi zi")
        xc, yc, zc = sp.symbols("xc yc zc")
        allc = [xc, yc, zc]

        for k in range(left, right + 1):
            for j in range(left, right + 1):
                for i in range(left, right + 1):
                    idx = (i + 0) * ld[0] + (j + 0) * ld[1] + (k + 0) * ld[2]

                    si = max(i - 1, 0)
                    sj = max(j - 1, 0)
                    sk = max(k - 1, 0)

                    ei = min(i + 2, 3)
                    ej = min(j + 2, 3)
                    ek = min(k + 2, 3)

                    key = f"stencil{i}{j}{k}"

                    # print("------------------")
                    # print(f"{i}, {j}, {k})")

                    sacc = [i, j, k]
                    ssize = [ei - si, ej - sj, ek - sk]
                    sinoffset = [si, sj, sk]
                    soutoffset = [i - si, j - sj, k - sk]
                    sextent = [1, 1, 1]

                    for l in range(0, 3):
                        if ssize[l] > 2:
                            sextent[l] = allc[l] - 2

                        if sacc[l] == 2:
                            sinoffset[l] = allc[l] - sextent[l] - 1
                            soutoffset[l] = allc[l] - sextent[l]

                    # print(f'acc = {sacc[0]}, {sacc[1]}, {sacc[2]}')
                    # print(f'size = {ssize[0]}, {ssize[1]}, {ssize[2]}')
                    # print(f'begin = {sinoffset[0]}, {sinoffset[1]}, {sinoffset[2]}')
                    # print(f'sextent = {sextent[0]}, {sextent[1]}, {sextent[2]}')

                    ss = []

                    for lk in range(sk, ek):
                        for lj in range(sj, ej):
                            for li in range(si, ei):
                                lidx = (
                                    (li + 0) * ld[0]
                                    + (lj + 0) * ld[1]
                                    + (lk + 0) * ld[2]
                                )

                                # print(f'({xi + li - i}, {yi + lj - j}, {zi + lk - k})', end=' ')
                                # print(G[idx, lidx], end=' ')
                                ss.append(G[idx, lidx])
                            # print(' ')
                        # print('\n')
                    # print("------------------")

                    loop = {
                        "acc": sacc,
                        "size": ssize,
                        "inoffset": sinoffset,
                        "outoffset": soutoffset,
                        "extent": sextent,
                        "stencil": sp.Matrix(len(ss), 1, ss),
                    }

                    ret[key] = loop

        return ret

    # def to_stencil(self, M):
    # 	G = sp.zeros(27, 27)
    # 	ld = [1, 3, 9]
    # 	ii = [-1] * 8

    # 	left = 0
    # 	right = 2
    # 	for k in range(left, right):
    # 		for j in range(left, right):
    # 			for i in range(left, right):

    # 				ii[0] = (i + 0) * ld[0] + (j + 0) * ld[1] + (k + 0) * ld[2]
    # 				ii[1] = (i + 1) * ld[0] + (j + 0) * ld[1] + (k + 0) * ld[2]
    # 				ii[2] = (i + 1) * ld[0] + (j + 1) * ld[1] + (k + 0) * ld[2]
    # 				ii[3] = (i + 0) * ld[0] + (j + 1) * ld[1] + (k + 0) * ld[2]
    # 				ii[4] = (i + 0) * ld[0] + (j + 0) * ld[1] + (k + 1) * ld[2]
    # 				ii[5] = (i + 1) * ld[0] + (j + 0) * ld[1] + (k + 1) * ld[2]
    # 				ii[6] = (i + 1) * ld[0] + (j + 1) * ld[1] + (k + 1) * ld[2]
    # 				ii[7] = (i + 0) * ld[0] + (j + 1) * ld[1] + (k + 1) * ld[2]

    # 				for l in range(0, 8):
    # 					for s in range(0, 8):
    # 						ll = ii[l]
    # 						ss = ii[s]
    # 						G[ll, ss] += M[l, s]

    # 	ii_center = 1 * ld[0] + 1 * ld[1] + 1 * ld[2]
    # 	stencil = G[ii_center, :]
    # 	return {
    # 	"stencil111": stencil
    # 	}

    def to_masked_stencil(self, M):
        xi, yi, zi, level = sp.symbols("xi yi zi level", integer=True)

        G = sp.zeros(27, 27)
        ld = [1, 3, 9]
        ii = [-1] * 8

        mask = [True] * 8

        left = 0
        right = 2
        for k in range(left, right):
            for j in range(left, right):
                for i in range(left, right):

                    ii[0] = (i + 0) * ld[0] + (j + 0) * ld[1] + (k + 0) * ld[2]
                    ii[1] = (i + 1) * ld[0] + (j + 0) * ld[1] + (k + 0) * ld[2]
                    ii[2] = (i + 1) * ld[0] + (j + 1) * ld[1] + (k + 0) * ld[2]
                    ii[3] = (i + 0) * ld[0] + (j + 1) * ld[1] + (k + 0) * ld[2]
                    ii[4] = (i + 0) * ld[0] + (j + 0) * ld[1] + (k + 1) * ld[2]
                    ii[5] = (i + 1) * ld[0] + (j + 0) * ld[1] + (k + 1) * ld[2]
                    ii[6] = (i + 1) * ld[0] + (j + 1) * ld[1] + (k + 1) * ld[2]
                    ii[7] = (i + 0) * ld[0] + (j + 1) * ld[1] + (k + 1) * ld[2]

                    # NEIGH STENCIL
                    km1 = k - 1
                    kp1 = k + 1
                    jm1 = j - 1
                    jp1 = j + 1
                    im1 = i - 1
                    ip1 = i + 1

                    lb = [xi + im1, yi + jm1, zi + km1]
                    rb = [xi + ip1, yi + jp1, zi + kp1]

                    mask[0] = (lb[0] >= 0) & (lb[1] >= 0) & (lb[2] >= 0)
                    mask[1] = (lb[0] < level) & (lb[1] >= 0) & (lb[2] >= 0)
                    mask[2] = (lb[0] < level) & (rb[1] < level) & (lb[2] >= 0)
                    mask[3] = (lb[0] >= 0) & (rb[1] < level) & (lb[2] >= 0)

                    mask[4] = (lb[0] >= 0) & (lb[1] >= 0) & (rb[2] < level)
                    mask[5] = (rb[0] < level) & (lb[1] >= 0) & (rb[2] < level)
                    mask[6] = (rb[0] < level) & (rb[1] < level) & (rb[2] < level)
                    mask[7] = (lb[0] >= 0) & (rb[1] < level) & (rb[2] < level)

                    for l in range(0, 8):
                        for s in range(0, 8):
                            ll = ii[l]
                            ss = ii[s]

                            nonz = sp.simplify(mask[l] & mask[s])
                            G[ll, ss] += M[l, s] * sp.Piecewise((1, nonz), (0, True))

        ii_center = 1 * ld[0] + 1 * ld[1] + 1 * ld[2]
        stencil = G[ii_center, :]
        return stencil


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


def points_from_sub_ref_hex8():
    x0, x1, y0, y1, z0, z1 = sp.symbols("px0 px1 py0 py1 pz0 pz1")
    hex8 = Hex8()

    qs = [
        [x0, x1, x1, x0, x0, x1, x1, x0],
        [y0, y0, y1, y1, y0, y0, y1, y1],
        [z0, z0, z0, z0, z1, z1, z1, z1],
    ]

    x = sp.zeros(8, 1)
    y = sp.zeros(8, 1)
    z = sp.zeros(8, 1)

    for k in range(0, 8):
        q = [qs[0][k], qs[1][k], qs[2][k]]
        f = hex8.fun(q)
        xyz = hex8.coords()

        for i in range(0, 8):
            fi = sp.simplify(f[i])
            x[k] += xyz[0][i] * fi
            y[k] += xyz[1][i] * fi
            z[k] += xyz[2][i] * fi

    expr = assign_matrix("lx", x)
    expr.extend(assign_matrix("ly", y))
    expr.extend(assign_matrix("lz", z))

    c_code(expr)


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


def sub_fff():
    c = coeffs("fff", 6)
    FFF = sp.Matrix(3, 3, [c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5]])

    h = sp.symbols("h")

    A = sp.Matrix(3, 3, [h, 0, 0, 0, h, 0, 0, 0, h])

    Aminv = inv3(A)
    detAm = determinant(A)

    sub_FFF = Aminv * FFF * Aminv.T * detAm

    expr = assign_fff("sub_fff", sub_FFF)
    c_code(expr)


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


def sub_adj():
    c = coeffs("adjugate", 9)
    adj = sp.Matrix(3, 3, c)
    detJ = sp.symbols("determinant")

    h = sp.symbols("h")

    A = sp.Matrix(3, 3, [h, 0, 0, 0, h, 0, 0, 0, h])

    detAm = determinant(A)
    Aadj = adjugate(A)

    sub_adj = Aadj * adj
    expr = assign_matrix("sub_adjugate", sub_adj)
    # expr.extend([ast.Assignment('determinant', detAm)])

    c_code(expr)

    c_code([ast.Assignment(sp.symbols("sub_determinant[0]"), detAm * detJ)])


def check_op():
    hex8 = Hex8()

    q = vec3(qx, qy, qz)
    g = hex8.grad(q)

    A = sp.zeros(8, 8)
    for i in range(0, 8):
        for j in range(0, 8):
            A[i, j] = hex8.integrate(q, inner(g[i], g[j]))
        print(A[i, :])


def gen_grads_SoA():
    g = Hex8().grad(vec3(qx, qy, qz))

    for d in range(0, 3):
        expr = []
        for i in range(0, len(g)):
            expr.append(ast.Assignment(sp.symbols(f"val[{i}]"), g[i][d]))

        print(f"// grad({d})")
        c_code(expr)


def gen_grads_AoS():
    g = Hex8().grad(vec3(qx, qy, qz))

    expr = []
    for i in range(0, len(g)):
        for d in range(0, 3):
            expr.append(ast.Assignment(sp.symbols(f"val[{i*3+d}]"), g[i][d]))

    print(f"// grads")
    c_code(expr)


def gen_grads_AoS_separate():
    g = Hex8().grad(vec3(qx, qy, qz))

    for i in range(0, len(g)):
        expr = []
        for d in range(0, 3):
            expr.append(ast.Assignment(sp.symbols(f"val[{d}]"), g[i][d]))
        print(f"case {i}: {{")
        c_code(expr)
        print(f"return; }}")


if __name__ == "__main__":
    # Hex8().generate_qp_based_code()
    # points_from_sub_ref_hex8()
    # sub_fff()
    # check_op()
    # sub_adj()

    # gen_grads_SoA()
    # gen_grads_AoS()
    gen_grads_AoS_separate()
