from fe import FE
import sympy as sp
from sfem_codegen import *


class SymbolicFE2D(FE):
    def __init__(self):
        super().__init__()

        self.A_ = self.symbol_jacobian()
        self.Ainv_ = self.symbol_jacobian_inverse()

    def is_symbolic(self):
        return True

    def coords_sub_parametric(self):
        return []

    def name(self):
        return "FE2D"

    def fun(self, p):
        f = sp.symbols("shape_fun")
        return [f]

    def trial_fun(self, p, ncomp=1):
        f = sp.symbols("trial_fun")
        return self.tensorize([f], ncomp)

    def test_fun(self, p, ncomp=1):
        f = sp.symbols("test_fun")
        return self.tensorize([f], ncomp)

    def grad(self, p):
        g0, g1 = sp.symbols("shape_grad[0] shape_grad[1]")

        g = [0] * 1
        g[0] = sp.Matrix(2, 1, [g0, g1])
        return g

    def trial_grad(self, p, ncomp=1):
        g0, g1 = sp.symbols("trial_grad[0] trial_grad[1]")

        g = [0] * 1
        g[0] = sp.Matrix(2, 1, [g0, g1])
        return self.grad_tensorize(g, ncomp)

    def test_grad(self, p, ncomp=1):
        g0, g1 = sp.symbols("test_grad[0] test_grad[1]")

        g = [0] * 1
        g[0] = sp.Matrix(2, 1, [g0, g1])
        return self.grad_tensorize(g, ncomp)

    def n_nodes(self):
        return 1

    def manifold_dim(self):
        return 2

    def spatial_dim(self):
        return 2

    def integrate(self, q, expr):
        ref_vol = sp.symbols("qw")
        return expr * ref_vol

    def jacobian(self, q):
        return self.A_

    def jacobian_inverse(self, q):
        return self.Ainv_

    def jacobian_determinant(self, q):
        return sp.symbols("det_jac")

    def measure(self, q):
        return sp.symbols("measure")


from fe import FE
import sympy as sp
from sfem_codegen import *


class SymbolicFE3D(FE):
    def __init__(self):
        super().__init__()

        self.A_ = self.symbol_jacobian()
        self.Ainv_ = self.symbol_jacobian_inverse()

    def is_symbolic(self):
        return True

    def coords_sub_parametric(self):
        return []

    def name(self):
        return "FE3D"

    def fun(self, p):
        f = sp.symbols("shape_fun")
        return [f]

    def trial_fun(self, p, ncomp=1):
        f = sp.symbols("trial_fun")
        return self.tensorize([f], ncomp)

    def test_fun(self, p, ncomp=1):
        f = sp.symbols("test_fun")
        return self.tensorize([f], ncomp)

    def grad(self, p):
        g0, g1, g3 = sp.symbols("shape_grad[0] shape_grad[1] shape_grad[2]")

        g = [0] * 1
        g[0] = sp.Matrix(3, 1, [g0, g1, g3])
        return g

    def trial_grad(self, p, ncomp=1):
        g0, g1, g3 = sp.symbols("trial_grad[0] trial_grad[1] trial_grad[2]")

        g = [0] * 1
        g[0] = sp.Matrix(3, 1, [g0, g1, g3])
        return self.grad_tensorize(g, ncomp)

    def test_grad(self, p, ncomp=1):
        g0, g1, g3 = sp.symbols("test_grad[0] test_grad[1] test_grad[2]")

        g = [0] * 1
        g[0] = sp.Matrix(3, 1, [g0, g1, g3])
        return self.grad_tensorize(g, ncomp)

    def n_nodes(self):
        return 1

    def manifold_dim(self):
        return 3

    def spatial_dim(self):
        return 3

    def integrate(self, q, expr):
        ref_vol = sp.symbols("qw")
        return expr * ref_vol

    def jacobian(self, q):
        return self.A_

    def jacobian_inverse(self, q):
        return self.Ainv_

    def jacobian_determinant(self, q):
        return sp.symbols("det_jac")

    def measure(self, q):
        return sp.symbols("measure")

    def reference_measure(self):
        return sp.symbols("reference_measure")
