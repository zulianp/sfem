#!/usr/bin/env python3

from fe import FE
from sfem_codegen import *
from weighted_fe import *


class Tet10(FE):
    def __init__(self):
        super().__init__()

    def reference_measure(self):
        return sp.Rational(1, 6)

    def barycenter(self):
        return vec3(sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))

    def subparam_n_nodes(self):
        return 4

    def coords_sub_parametric(self):
        return [[x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3]]

    def coords(self):
        return [coeffs("x", 10), coeffs("y", 10), coeffs("z", 10)]

    def name(self):
        return "Tet10"

    def f0(self, x, y, z):
        l0 = 1 - x - y - z
        return (2 * l0 - 1) * l0

    def f1(self, x, y, z):
        l1 = x
        return (2 * l1 - 1) * l1

    def f2(self, x, y, z):
        l2 = y
        return (2 * l2 - 1) * l2

    def f3(self, x, y, z):
        l3 = z
        return (2 * l3 - 1) * l3

    def f4(self, x, y, z):
        l0 = 1 - x - y - z
        l1 = x
        return 4 * l0 * l1

    def f5(self, x, y, z):
        l1 = x
        l2 = y
        return 4 * l1 * l2

    def f6(self, x, y, z):
        l0 = 1 - x - y - z
        l2 = y
        return 4 * l0 * l2

    def f7(self, x, y, z):
        l0 = 1 - x - y - z
        l3 = z
        return 4 * l0 * l3

    def f8(self, x, y, z):
        l1 = x
        l3 = z
        return 4 * l1 * l3

    def f9(self, x, y, z):
        l2 = y
        l3 = z
        return 4 * l2 * l3

    def fun(self, p):
        x = p[0]
        y = p[1]
        z = p[2]

        return [
            self.f0(x, y, z),
            self.f1(x, y, z),
            self.f2(x, y, z),
            self.f3(x, y, z),
            self.f4(x, y, z),
            self.f5(x, y, z),
            self.f6(x, y, z),
            self.f7(x, y, z),
            self.f8(x, y, z),
            self.f9(x, y, z),
        ]

    def n_nodes(self):
        return 10

    def manifold_dim(self):
        return 3

    def spatial_dim(self):
        return 3

    def integrate(self, q, expr):
        return sp.integrate(
            expr, (q[2], 0, 1 - q[0] - q[1]), (q[1], 0, 1 - q[0]), (q[0], 0, 1)
        )

    def jacobian(self, q):
        return A

    def jacobian_inverse(self, q):
        return Ainv

    def transform(self, q):
        return self.jacobian(q) * q + sp.Matrix(3, 1, [x0, y0, z0])

    def inverse_transform(self, p):
        diff = p - sp.Matrix(3, 1, [x0, y0, z0])
        return self.jacobian_inverse(p) * diff

    def jacobian_determinant(self, q):
        return det3(A)

    def measure(self, q):
        return det3(A) / 6


#######################################
# Dual basis trafo and weights
#######################################

# ls_tet10_trafo_ = [1, 0, 0, 0, 0.2, 0,   0.2, 0.2, 0,   0,   0, 1, 0, 0, 0.2, 0.2, 0, 0,   0.2, 0,
#               0, 0, 1, 0, 0,   0.2, 0.2, 0,   0,   0.2, 0, 0, 0, 1, 0,   0,   0, 0.2, 0.2, 0.2,
#               0, 0, 0, 0, 0.6, 0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0.6, 0, 0,   0,   0,
#               0, 0, 0, 0, 0,   0,   0.6, 0,   0,   0,   0, 0, 0, 0, 0,   0,   0, 0.6, 0,   0,
#               0, 0, 0, 0, 0,   0,   0,   0,   0.6, 0,   0, 0, 0, 0, 0,   0,   0, 0,   0,   0.6]

ls_tet10_trafo_ = [
    1,
    0,
    0,
    0,
    sp.Rational(1, 5),
    0,
    sp.Rational(1, 5),
    sp.Rational(1, 5),
    0,
    0,
    0,
    1,
    0,
    0,
    sp.Rational(1, 5),
    sp.Rational(1, 5),
    0,
    0,
    sp.Rational(1, 5),
    0,
    0,
    0,
    1,
    0,
    0,
    sp.Rational(1, 5),
    sp.Rational(1, 5),
    0,
    0,
    sp.Rational(1, 5),
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    sp.Rational(1, 5),
    sp.Rational(1, 5),
    sp.Rational(1, 5),
    0,
    0,
    0,
    0,
    sp.Rational(3, 5),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    sp.Rational(3, 5),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    sp.Rational(3, 5),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    sp.Rational(3, 5),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    sp.Rational(3, 5),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    sp.Rational(3, 5),
]


r15div8 = sp.Rational(15, 8)
r7div40 = sp.Rational(7, 40)
r7div10 = sp.Rational(7, 10)
r39div10 = sp.Rational(39, 10)
r6div5 = sp.Rational(6, 5)
r111div20 = sp.Rational(111, 20)

ls_tet10_weights_ = [
    7,
    r7div10,
    r7div10,
    r7div10,
    r7div40,
    r7div10,
    r7div40,
    r7div40,
    r7div10,
    r7div10,
    r7div10,
    7,
    r7div10,
    r7div10,
    r7div40,
    r7div40,
    r7div10,
    r7div10,
    r7div40,
    r7div10,
    r7div10,
    r7div10,
    7,
    r7div10,
    r7div10,
    r7div40,
    r7div40,
    r7div10,
    r7div10,
    r7div40,
    r7div10,
    r7div10,
    r7div10,
    7,
    r7div10,
    r7div10,
    r7div10,
    r7div40,
    r7div40,
    r7div40,
    -r39div10,
    -r39div10,
    r6div5,
    r6div5,
    r111div20,
    -r15div8,
    -r15div8,
    -r15div8,
    -r15div8,
    r6div5,
    r6div5,
    -r39div10,
    -r39div10,
    r6div5,
    -r15div8,
    r111div20,
    -r15div8,
    r6div5,
    -r15div8,
    -r15div8,
    -r39div10,
    r6div5,
    -r39div10,
    r6div5,
    -r15div8,
    -r15div8,
    r111div20,
    -r15div8,
    r6div5,
    -r15div8,
    -r39div10,
    r6div5,
    r6div5,
    -r39div10,
    -r15div8,
    r6div5,
    -r15div8,
    r111div20,
    -r15div8,
    -r15div8,
    r6div5,
    -r39div10,
    r6div5,
    -r39div10,
    -r15div8,
    -r15div8,
    r6div5,
    -r15div8,
    r111div20,
    -r15div8,
    r6div5,
    r6div5,
    -r39div10,
    -r39div10,
    r6div5,
    -r15div8,
    -r15div8,
    -r15div8,
    -r15div8,
    r111div20,
]

tet10_trafo = sp.Matrix(10, 10, ls_tet10_trafo_)
tet10_weights = sp.Matrix(10, 10, ls_tet10_weights_)


# for i in range(0, 10):
# 	s = 0
# 	for j in range(0, 10):
# 		s += tet10_trafo[i, j]
# 	c_log(s)

# for i in range(0, 10):
# 	s = 0
# 	for j in range(0, 10):
# 		s += tet10_weights[i, j]
# 	c_log(s)

# c_log(tet10_trafo - tet10_trafo.T)


def tet10_basis_transform(x):
    r, c = tet10_trafo.shape

    ret = sp.Matrix(r, 1, [0] * r)

    # Use transposed here
    trafo = tet10_trafo.T

    for i in range(0, r):
        for j in range(0, c):
            ret[i] = ret[i] + trafo[i, j] * x[j]
    return ret


def tet10_basis_transform_expr():
    tx = tet10_basis_transform(coeffs("x", 10))
    ttx = coeffs("values", 10)

    expr = []

    for i in range(0, 10):
        print(tx[i])
        e = ast.Assignment(ttx[i], tx[i])
        expr.append(e)

    return expr


# Factory functions
def TransformedTet10():
    return WeightedFE(Tet10(), tet10_trafo, "Transformed")


def DualTet10():
    return WeightedFE(Tet10(), tet10_weights, "Dual")


# if __name__ == '__main__':
# Tet10().generate_c_code()
# Tet10().generate_qp_based_code()
# fe = Tet10()
# q = fe.quadrature_point()
# f = fe.f0(0.51, 0.0, 0.0)
# print(f)
# DualTet10().generate_qp_based_code();
