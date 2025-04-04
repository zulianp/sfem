#!/usr/bin/env python3

from edge2 import *
from quad4 import *
from tri3 import *

import sys


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


# From: M. Juntunen and R. Stenberg, Nitsche’s method for general boundary conditions
# $\frac{\partial u}{\partial n} = 1/\eps ( u_0 - u) + g on \Gamma$
class NitscheOp:
    def __init__(self, fe, symbolic_integration=True):
        self.fe = fe
        self.symbolic_integration = symbolic_integration

        # Ref element dims
        dims = fe.manifold_dim()

        if dims == 1:
            q = [qx]
        elif dims == 2:
            q = [qx, qy]
        else:
            q = [qx, qy, qz]

        # Quadrature point
        q = sp.Matrix(dims, 1, q)

        # Quadrature weight
        qw = sp.symbols("qw")

        # Params
        gamma, eps, hE = sp.symbols("gamma eps hE")

        u = coeffs("u", fe.n_nodes())
        u0, g0 = sp.symbols("u0 g0")
        # grad = fe.physical_grad(q)
        ref_grad = fe.grad(q)
        fun = fe.fun(q)
        normal = fe.symbol_normal()
        J_inv = fe.symbol_jacobian_inverse_as_adjugate()
        det_J = fe.symbol_jacobian_determinant()

        self.q = q
        self.qw = qw
        # self.grad = grad
        self.ref_grad = ref_grad
        self.normal = normal
        self.fun = fun
        self.gamma = gamma
        self.hE = hE
        self.eps = eps
        self.u = u

        self.u0 = u0
        self.g0 = g0
        self.J_inv = J_inv
        self.det_J = det_J

    # def hessian(self):
    # def hessian_diag(self):

    def adjugate_and_determinant(self):
        adj = self.fe.adj(self.q)
        det = self.fe.jacobian_determinant(self.q)

        expr = []
        expr.extend(assign_matrix("jacobian_adjugate", adj))
        expr.append(ast.Assignment(sp.symbols("jacobian_determinant"), det))
        return expr

    def gradient(self):
        fe = self.fe
        q = self.q
        qw = self.qw
        # grad = self.grad
        ref_grad = self.ref_grad
        normal = self.normal
        fun = self.fun
        gamma = self.gamma
        eps = self.eps
        hE = self.hE
        u = self.u
        u0 = self.u0
        g0 = self.g0
        J_inv = self.J_inv
        det_J = self.det_J

        if not self.symbolic_integration:
            assert False

        alpha0 = gamma * hE / (eps + gamma * hE)
        alpha1 = 1 / (eps + gamma * hE)
        alpha2 = eps * gamma * hE / (eps + gamma * hE)
        alpha3 = eps / (eps + gamma * hE)

        expr = []
        for i in range(0, fe.n_nodes()):
            g = J_inv.T * ref_grad[i]
            normal_deriv_v = inner(g, normal)

            integr0 = fe.integrate(q, u0 * normal_deriv_v)
            integr1 = fe.integrate(q, u0 * fun[i])
            integr2 = fe.integrate(q, g0 * normal_deriv_v)
            integr3 = fe.integrate(q, g0 * fun[i])

            integr = (
                -alpha0 * integr0
                + alpha1 * integr1
                - alpha2 * integr2
                + alpha3 * integr3
            ) * det_J
            # integr = sp.simplify(integr)

            form = sp.symbols(f"element_vector[{i}*stride]")

            expr.append(ast.Assignment(form, integr))

        return expr

    def apply(self):
        fe = self.fe
        q = self.q
        qw = self.qw
        ref_grad = self.ref_grad
        normal = self.normal
        fun = self.fun
        gamma = self.gamma
        eps = self.eps
        hE = self.hE
        u = self.u
        J_inv = self.J_inv
        det_J = self.det_J

        if not self.symbolic_integration:
            assert False

        alpha0 = gamma * hE / (eps + gamma * hE)
        alpha1 = 1 / (eps + gamma * hE)
        alpha2 = eps * gamma * hE / (eps + gamma * hE)

        uh = 0
        graduh = sp.zeros(fe.manifold_dim(), 1)
        for i in range(0, fe.n_nodes()):
            uh += fun[i] * u[i]
            graduh += ref_grad[i] * u[i]

        graduh = J_inv.T * graduh
        normal_deriv_u = inner(graduh, normal)

        expr = []
        for i in range(0, fe.n_nodes()):
            g = J_inv.T * ref_grad[i]
            normal_deriv_v = inner(g, normal)

            integr0 = fe.integrate(q, (normal_deriv_u * fun[i])) + fe.integrate(
                q, (uh * normal_deriv_v)
            )
            integr1 = fe.integrate(q, uh * fun[i])
            integr2 = fe.integrate(q, normal_deriv_u * normal_deriv_v)

            integr = (-alpha0 * integr0 + alpha1 * integr1 - alpha2 * integr2) * det_J
            # integr = sp.simplify(integr)

            form = sp.symbols(f"element_vector[{i}*stride]")

            expr.append(ast.Assignment(form, integr))

        return expr

    # def value(self):
    # 	fe = self.fe
    # 	q = self.q

    # 	if self.symbolic_integration:
    # 		trial_operand = self.FFF_symbolic * self.ref_grad_interp
    # 	else:
    # 		trial_operand = self.trial_operand_symbolic  # NOTE that quadrature weight is included here

    # 	integr = 0
    # 	for d in range(0, fe.manifold_dim()):
    # 		if self.symbolic_integration:
    # 			gsquared = fe.integrate(q, (trial_operand[d] * self.ref_grad_interp[d])) / 2
    # 		else:
    # 			gsquared = trial_operand[d] * self.ref_grad_interp[d] / 2
    # 		integr += gsquared

    # 	integr = sp.simplify(integr)

    # 	form = sp.symbols(f'element_scalar[0]')

    # 	if self.symbolic_integration:
    # 		return [ast.Assignment(form, integr)]
    # 	else:
    # 		return [ast.AddAugmentedAssignment(form, integr)]


def main():

    fes = {
        # "TRI6": Tri6(),
        "TRISHELL3": TriShell3(),
        # "TET4": Tet4(),
        # "TET10": Tet10(),
        # "TET20": Tet20(),
        # "AAQUAD4": AxisAlignedQuad4()
        "EDGESHELL2": EdgeShell2(),
    }

    if len(sys.argv) >= 2:
        fe = fes[sys.argv[1]]
    else:
        print("Fallback with EdgeShell2")
        fe = EdgeShell2()

    symbolic_integration = True
    if len(sys.argv) >= 3:
        symbolic_integration = int(sys.argv[2])

    op = NitscheOp(fe, symbolic_integration)

    # print('---------------------------------------------------')
    # print("fff")
    # print('---------------------------------------------------')

    # c_code(op.fff())

    print("---------------------------------------------------")
    print("adjugate_and_determinant")
    print("---------------------------------------------------")

    c_code(op.adjugate_and_determinant())

    # if not symbolic_integration:
    # 	print('---------------------------------------------------')
    # 	print("trial_operand")
    # 	print('---------------------------------------------------')
    # 	c_code(op.trial_operand_expr())

    # print('---------------------------------------------------')
    # print("hessian")
    # print('---------------------------------------------------')

    # c_code(op.hessian())

    print("---------------------------------------------------")
    print("apply")
    print("---------------------------------------------------")

    c_code(op.apply())

    print("---------------------------------------------------")
    print("gradient")
    print("---------------------------------------------------")

    c_code(op.gradient())

    # print('---------------------------------------------------')
    # print("hessian_diag")
    # print('---------------------------------------------------')

    # c_code(op.hessian_diag())

    # print('---------------------------------------------------')
    # print("Value")
    # print('---------------------------------------------------')

    # c_code(op.value())


if __name__ == "__main__":
    main()
