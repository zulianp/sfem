#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *
from symbolic_fe import *

# from sympy.utilities.codegen import (InputArgument, OutputArgument,
# InOutArgument, Routine)
# https://stackoverflow.com/questions/25309580/simplify-expression-generated-with-sympy-codegen

from time import perf_counter


def simplify(expr):
    return expr
    # return sp.simplify(expr)


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


class HyperElasticity:
    def __init__(self, fe, model):
        fe.use_adjugate = True

        F = model.deformation_gradient_symb()
        Psi = model.energy()
        dims = model.dims
        q = q_point(dims)

        # FE gradients
        trial_grad = matrix_coeff("trial_grad", dims, dims)
        test_grad = matrix_coeff("test_grad", dims, dims)
        inc_grad_symb = matrix_coeff("inc_grad", dims, dims)
        disp_grad_symb = matrix_coeff("disp_grad", dims, dims)

        displacement = coeffs_SoA("u", dims, fe.n_nodes())
        increment = coeffs_SoA("h", dims, fe.n_nodes())
        value = coeffs_SoA("v", dims, fe.n_nodes())

        ref_grad = fe.tgrad(q)
        jac_inv = fe.symbol_jacobian_inverse_as_adjugate()

        disp_grad = sp.zeros(dims, dims)
        for i in range(0, fe.n_nodes() * fe.manifold_dim()):
            disp_grad += displacement[i] * ref_grad[i]
        disp_grad = disp_grad * jac_inv

        inc_grad = sp.zeros(dims, dims)
        for i in range(0, fe.n_nodes() * fe.manifold_dim()):
            inc_grad += increment[i] * ref_grad[i]
        inc_grad = inc_grad * jac_inv

        vec_grad = sp.zeros(dims, dims)
        for i in range(0, fe.n_nodes() * fe.manifold_dim()):
            vec_grad += value[i] * ref_grad[i]
        vec_grad = vec_grad * jac_inv

        # First Piola
        P = sp.zeros(dims, dims)
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                P[d1, d2] = simplify(sp.diff(Psi, F[d1, d2]))

        # let us switch test with trial (exploit symmetry)
        gj = inner(P, trial_grad)

        # Stress linearization
        lin_stress = sp.zeros(dims, dims)
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                lin_stress[d1, d2] = simplify(sp.diff(gj, F[d1, d2]))

        ######################################
        # Store common quantities
        ######################################
        self.fe = fe
        self.q = q
        self.jac_inv = jac_inv
        self.dV = (
            fe.reference_measure()
            * fe.symbol_jacobian_determinant()
            * fe.quadrature_weight()
        )

        self.model = model

        self.trial_grad = trial_grad
        self.test_grad = test_grad
        self.inc_grad_symb = inc_grad_symb
        self.disp_grad_symb = disp_grad_symb
        self.F = F

        self.dims = dims
        self.Psi = Psi
        self.P = P
        self.lin_stress = lin_stress
        self.lin_stress_symb = matrix_coeff("lin_stress", dims, dims)
        self.loperand_symb = matrix_coeff("loperand", dims, dims)

        self.disp_grad = disp_grad
        self.inc_grad = inc_grad
        self.vec_grad = vec_grad
        self.ref_grad = ref_grad
        self.displacement = displacement

    def check_symmetries(self):
        dims = self.dims
        P = self.P
        lin_stress = self.lin_stress
        T = lin_stress

        T_t = T.T
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                diff = T_t[d1, d2] - T[d1, d2]
                diff = sp.simplify(diff)
                print(f"{d1}, {d2}) {diff}")

    # Reference expressions
    def gradient_expected(self):
        expr = []

        dims = self.fe.spatial_dim()
        test_grad = self.fe.physical_tgrad(self.q)
        disp_grad = self.disp_grad
        F = disp_grad + sp.eye(dims, dims)

        output = coeffs_SoA("out", dims, self.fe.n_nodes())

        for i in range(0, self.fe.spatial_dim() * self.fe.n_nodes()):
            gi = inner(self.P, test_grad[i])
            gi = subsmat(gi, self.F, F) * (
                self.fe.symbol_jacobian_determinant() * self.fe.reference_measure()
            )
            expr.append(ast.Assignment(output[i], gi))

        c_code(expr)

    def gradient_check(self):
        expr = []
        dims = self.fe.spatial_dim()
        jac_inv = self.fe.symbol_jacobian_inverse_as_adjugate()
        P = self.P

        rtest_grad = self.fe.tgrad(self.q)
        ptest_grad = self.fe.physical_tgrad(self.q)

        disp_grad = self.disp_grad

        F = disp_grad + sp.eye(dims, dims)
        output = coeffs_SoA("out", dims, self.fe.n_nodes())

        for i in range(0, self.fe.spatial_dim() * self.fe.n_nodes()):
            gi_expected = sp.simplify(inner(P, ptest_grad[i]))
            gi_actual = sp.simplify(inner(P * jac_inv.T, rtest_grad[i]))
            diff = sp.simplify(gi_expected - gi_actual)
            print(diff)
            assert diff == 0

    def gradient(self):
        P = self.P
        jac_inv = self.fe.symbol_jacobian_inverse_as_adjugate()
        PxJinv_t = (
            P
            * jac_inv.T
            * (self.fe.symbol_jacobian_determinant() * self.fe.reference_measure())
        )

        dims = self.fe.manifold_dim()
        for i in range(0, dims):
            for j in range(0, dims):
                PxJinv_t[i, j] = simplify(PxJinv_t[i, j])

        PxJinv_t_sym = matrix_coeff("PxJinv_t", dims, dims)
        lform = []
        output = coeffs_SoA("out", dims, self.fe.n_nodes())
        for i in range(0, self.fe.n_nodes() * dims):
            lform.append(
                ast.Assignment(output[i], inner(PxJinv_t_sym, self.ref_grad[i]))
            )

        ret = {
            "disp_grad": assign_matrix("disp_grad", self.disp_grad),
            "F": assign_matrix("F", self.disp_grad_symb + sp.eye(dims, dims)),
            "PxJinv_t": assign_matrix("PxJinv_t", PxJinv_t),
            "lform": lform,
        }

        return ret

    def hessian(self):
        # lin_stress = self.lin_stress
        trial_grad = self.trial_grad
        test_grad = self.test_grad
        inc_grad_symb = self.inc_grad_symb
        jac_inv = self.jac_inv
        dV = self.dV
        dims = self.fe.spatial_dim()
        ref_grad = self.ref_grad
        P = self.P
        F = self.F

        lin_stress = []
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                block = sp.zeros(dims, dims)

                for d3 in range(0, dims):
                    for d4 in range(0, dims):
                        block[d3, d4] = simplify(sp.diff(P[d1, d2], F[d3, d4]))

                # Sandwich to include Jac
                JBJT = jac_inv * block * jac_inv.T

                spblock = sp.zeros(dims, 1)
                # for d3 in range(0, dims):
                for d4 in range(0, dims):
                    spblock[d4] = simplify(JBJT[d2, d4])

                lin_stress.extend(assign_matrix(f"block_{d1}{d2}", spblock))

        # loperand = lin_stress * (jac_inv.T * dV)

        # ref_trial_grad = matrix_coeff('ref_trial_grad', dims, dims)
        # RT = ref_trial_grad * jac_inv

        # expr = loperand.copy()
        # for d1 in range(0, dims):
        # 	for d2 in range(0, dims):
        # 		expr[d1, d2] = subsmat(expr[d1, d2], trial_grad, RT)

        c_code(lin_stress)

    def hessian_apply(self):
        lin_stress = self.lin_stress
        lin_stress_symb = self.lin_stress_symb
        trial_grad = self.trial_grad
        test_grad = self.test_grad
        inc_grad_symb = self.inc_grad_symb
        jac_inv = self.jac_inv
        dV = self.dV
        dims = self.fe.spatial_dim()
        ref_grad = self.ref_grad

        loperand = lin_stress_symb * (jac_inv.T * dV)

        expr = loperand.copy()
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                expr[d1, d2] = subsmat(expr[d1, d2], trial_grad, inc_grad_symb)

        lform = []

        # if True:
        if False:
            for d1 in range(0, dims):
                val = 0
                for d2 in range(0, dims):
                    val += self.loperand_symb[d1, d2] * test_grad[0, d2]
                lform.append(
                    ast.AddAugmentedAssignment(sp.symbols(f"lform[{d1}]"), val)
                )
        else:
            for i in range(0, dims * self.fe.n_nodes()):
                lform.append(
                    ast.AddAugmentedAssignment(
                        sp.symbols(f"lform[{i}]"),
                        inner(self.loperand_symb, ref_grad[i]),
                    )
                )

        buffers = []
        buffers.extend(assign_matrix("vec_grad", self.vec_grad))
        ret = {
            "Gradient": buffers,
            "F": assign_matrix("F", self.disp_grad_symb + sp.eye(dims, dims)),
            "lin_stress": assign_matrix("lin_stress", lin_stress),
            "loperand": assign_matrix("loperand", expr),
            "lform": lform,
        }

        return ret


class HyperElasticModel:
    def __init__(self, dims):
        self.dims = dims
        self.J_symb = sp.symbols("J")
        self.trC_symb = sp.symbols("trC")
        self.F_symb = matrix_coeff("F", dims, dims)
        self.F_inv_t_symb = matrix_coeff("F_inv_t", dims, dims)
        self.F_inv_symb = matrix_coeff("F_inv", dims, dims)
        self.C_symb = matrix_coeff("C", dims, dims)

        self.J = determinant(self.F_symb)
        self.F_inv = inverse(self.F_symb)
        self.F_inv_t = self.F_inv.T
        self.C = self.F_symb.T * self.F_symb
        self.trC = sp.trace(self.C)


class NeoHookeanOgden(HyperElasticModel):
    def __init__(self, dims):
        super().__init__(dims)
        self.name = "neohookean_ogden"
        mu, lmbda = sp.symbols("mu lambda")
        self.params = [(mu, 1.0), (lmbda, 1.0)]

        trC = self.trC
        J = self.J

        self.fun = (
            mu / 2 * (trC - dims) - mu * sp.log(J) + (lmbda / 2) * (sp.log(J)) ** 2
        )

    def deformation_gradient_symb(self):
        return self.F_symb

    def energy(self):
        return self.fun


def main():
    start = perf_counter()

    # fe = Tet20()
    # fe = Tet10()
    fe = Tet4()
    model = NeoHookeanOgden(fe.spatial_dim())
    op = HyperElasticity(fe, model)

    # op.gradient_check()

    # op.gradient_expected()
    # if False:
    # 	op.check_symmetries()
    # else:

    # print("GRADIENT")
    # op_gradient = op.gradient()
    # for k, v in op_gradient.items():
    # 	print('// -------------------------------')
    # 	print(f'// {k}')
    # 	print('// -------------------------------')
    # 	c_code(v)

    # print("HESSIAN")
    # op_hessian_apply = op.hessian_apply()
    # for k, v in op_hessian_apply.items():
    # 	print('-------------------------------')
    # 	print(f'{k}')
    # 	print('-------------------------------')
    # 	c_code(v)
    # print('-------------------------------')

    op.hessian()

    stop = perf_counter()
    console.print(f"Overall: {stop - start} seconds")


if __name__ == "__main__":
    main()
