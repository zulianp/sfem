#!/usr/bin/env python3

from bubble import *


class MiniBase(FE):
    def __init__(self, p1, bubble):
        super().__init__()
        self.p1 = p1
        self.bubble = bubble

    def reference_measure(self):
        return self.p1.reference_measure()

    def coords_sub_parametric(self):
        return self.p1.coords_sub_parametric()

    def n_nodes(self):
        return self.bubble.n_nodes() + self.p1.n_nodes()

    def f0(self, x, y, z):
        return 27 * (1.0 - x - y) * x * y

    def bubble_dof_idx(self):
        return 0

    def fun(self, p):
        ret = self.bubble.fun(q)
        p1 = self.p1.fun(p)

        for f in p1:
            ret.append(f)

        return ret

    def manifold_dim(self):
        return self.p1.manifold_dim()

    def spatial_dim(self):
        return self.p1.spatial_dim()

    def integrate(self, q, expr):
        return self.p1.integrate(q, expr)

    def jacobian(self, q):
        return self.p1.jacobian(q)

    def jacobian_inverse(self, q):
        return self.p1.jacobian_inverse(q)

    def jacobian_determinant(self, q):
        return self.p1.jacobian_determinant(q)

    def measure(self, q):
        return self.p1.measure(q)


class Mini2D(MiniBase):
    def __init__(self):
        super().__init__(Tri3(), Bubble2D())

    def name(self):
        return "Mini2D"


class Mini3D(MiniBase):
    def __init__(self):
        super().__init__(Tet4(), Bubble3D())

    def name(self):
        return "Mini3D"


if __name__ == "__main__":
    Mini2D().generate_c_code()
    Mini3D().generate_c_code()
