#!/usr/bin/env python3

from sfem_codegen import *
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *
from hex8 import *
from aahex8 import *

import sys

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


def assign_add_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.AddAugmentedAssignment(var, mat[i, j]))
    return expr


class GPULinearElasticityOp:
    SoA_IO = True
    # use_taylor = True

    def __init__(self, fe):
        dims = fe.manifold_dim()

        q_temp = [qx, qy, qz]
        q = sp.Matrix(dims, 1, q_temp[0:dims])

        self.init_opt(fe, q)

    def init_opt(self, fe, q):
        fe.use_adjugate = True
        dims = fe.manifold_dim()
        q = sp.Matrix(dims, 1, q)
        self.q = q
        shape_grad = fe.physical_tgrad(q)
        shape_grad_ref = fe.tgrad(q)
        # jac_inv = fe.symbol_jacobian_inverse()
        jac_inv = fe.symbol_jacobian_inverse_as_adjugate()

        if self.SoA_IO:
            disp = coeffs_SoA("u", dims, fe.n_nodes())
        else:
            disp = coeffs("u", dims * fe.n_nodes())

        self.jac = fe.jacobian(q)
        rows = fe.n_nodes() * dims
        cols = rows

        ###################################################################
        # Material law
        ###################################################################
        self.disp_grad_name = "disp_grad"

        mu, lmbda = sp.symbols("mu lambda", real=True)
        disp_grad = sp.Matrix(dims, dims, coeffs(self.disp_grad_name, dims * dims))

        self.eval_disp_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            self.eval_disp_grad += disp[i] * shape_grad_ref[i]
        self.eval_disp_grad = self.eval_disp_grad * jac_inv

        test_disp_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            test_disp_grad += disp[i] * shape_grad[i]

        for i in range(0, dims):
            for j in range(0, dims):
                self.eval_disp_grad[i, j] = sp.simplify(self.eval_disp_grad[i, j])

                # Check
                diff = sp.simplify(test_disp_grad[i, j] - self.eval_disp_grad[i, j])
                assert diff == 0

        # strain energy function
        epsu = (disp_grad + disp_grad.T) / 2
        e = mu * inner(epsu, epsu) + (lmbda / 2) * (tr(epsu) * tr(epsu))

        dV = (
            fe.reference_measure()
            * fe.symbol_jacobian_determinant()
            * fe.quadrature_weight()
        )
        # Objective
        self.eval_value = e * dV
        self.eval_value = simplify(self.eval_value)

        # Gradient
        P = sp.Matrix(dims, dims, [0] * (dims * dims))
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                P[d1, d2] = sp.diff(e, disp_grad[d1, d2])

        eval_grad = sp.zeros(rows, 1)
        self.eval_grad = sp.zeros(rows, 1)

        # Reference measure scaled here for reducing register usage
        self.P = P
        P_sym = matrix_coeff("P", dims, dims)

        for i in range(0, fe.n_nodes() * dims):
            self.eval_grad[i] = inner(P_sym, shape_grad[i]) * dV
            self.eval_grad[i] = simplify(self.eval_grad[i])
            eval_grad[i] = inner(P, shape_grad[i])

            # test_g = inner(P * jac_inv.T , shape_grad_ref[i])
            # diff = sp.simplify(eval_grad[i] - test_g)

            # print(P)
            # print(diff)
            # assert diff == 0

        # Trying to optimize the bilinear form
        P_tXJinv_t_sym = matrix_coeff("P_tXJinv_t", dims, dims)
        self.P_tXJinv_t_sym = P_tXJinv_t_sym

        self.eval_grad_opt = sp.zeros(rows, 1)
        eval_grad_opt = sp.zeros(rows, 1)
        for i in range(0, fe.n_nodes() * dims):
            self.eval_grad_opt[i] = inner(P_tXJinv_t_sym, shape_grad_ref[i])
            self.eval_grad_opt[i] = simplify(self.eval_grad_opt[i])

            # Evaluation for computing hessian
            eval_grad_opt[i] = simplify(inner(P * jac_inv.T * dV, shape_grad_ref[i]))

        self.eval_hessian = sp.zeros(rows, cols)
        self.lin_stress = []

        #

        for j in range(0, rows):
            dde = sp.zeros(dims, dims)
            for d1 in range(0, dims):
                for d2 in range(0, dims):
                    dde[d1, d2] = simplify(sp.diff(eval_grad_opt[j], disp_grad[d1, d2]))

            # self.lin_stress.append(dde)
            self.lin_stress.append(dde * jac_inv.T)
            lin_stress_sym = matrix_coeff(f"lin_stress{j}", dims, dims)

            for i in range(0, cols):
                test = inner(dde, shape_grad[i])
                actual = simplify(inner(dde * jac_inv.T, shape_grad_ref[i]))
                self.eval_hessian[i, j] = actual

        ###################################################################
        # Integrate and substitute
        ###################################################################
        ###################################################################

        self.e = e
        self.de = P
        self.mu = mu
        self.lmbda = lmbda
        self.disp = disp

        ###################################################################

        self.fe = fe
        self.increment = coeffs("u", fe.n_nodes() * dims)

        ###################################################################

    def hessian_check(self):
        H = self.integr_hessian
        rows, cols = H.shape

        A = sp.Matrix(rows, cols, [0] * (rows * cols))
        for i in range(0, rows):
            for j in range(0, cols):
                integr = H[i, j]

                coord = 0.5
                integr = integr.subs(self.mu, 2)
                integr = integr.subs(self.lmbda, 2)

                # coord = 1.
                # integr = integr.subs(self.mu, sp.Rational(1, 2))
                # integr = integr.subs(self.lmbda, 1)
                # integr = integr.subs(self.mu, 1)
                # integr = integr.subs(self.lmbda, 0)

                integr = integr.subs(x0, 0)
                integr = integr.subs(y0, 0)

                integr = integr.subs(x1, coord)
                integr = integr.subs(y1, 0)

                if rows == 8:
                    integr = integr.subs(x2, coord)
                    integr = integr.subs(y2, coord)

                    integr = integr.subs(x3, 0)
                    integr = integr.subs(y3, coord)
                else:
                    integr = integr.subs(x2, 0)
                    integr = integr.subs(y2, coord)

                A[i, j] = integr

        S = A.T - A
        row_sum = sp.Matrix(rows, 1, [0] * rows)
        for i in range(0, rows):
            for j in range(0, cols):
                row_sum[i] += A[i, j]

        # for i in range(0, rows):
        # 	c_log("%.3g" % A[i,i])

        for i in range(0, rows):
            line = ""
            for j in range(0, rows):
                line += "%.5g " % A[i, j]
            c_log(line)

        c_log(S)
        c_log(row_sum)

    def displacement_gradient(self):
        expr = assign_matrix(self.disp_grad_name, self.eval_disp_grad)
        return expr

    def first_piola(self):
        P = self.P
        expr = assign_matrix("P", P)
        return expr

    def cauchy_stress(self):
        dims = self.fe.manifold_dim()
        # It is linear so it is ok to  `F * P.T / J = Id * P / 1`
        P = self.P
        if dims == 2:
            CauchyStress = sp.Matrix(3, 1, [P[0, 0], P[0, 1], P[1, 1]])
        elif dims == 3:
            CauchyStress = sp.Matrix(
                6, 1, [P[0, 0], P[0, 1], P[0, 2], P[1, 1], P[1, 2], P[2, 2]]
            )
        return assign_matrix("cauchy_stress", CauchyStress)

    def strain(self):
        fe = self.fe
        disp = self.disp
        dims = fe.manifold_dim()
        q = fe.quadrature_point()
        shape_grad = fe.physical_tgrad(q)
        eval_strain = sp.zeros(dims, dims)

        for i in range(0, dims * fe.n_nodes()):
            eval_strain += disp[i] * (shape_grad[i] + shape_grad[i].T) / 2

        if dims == 2:
            ret = sp.Matrix(
                3, 1, [eval_strain[0, 0], eval_strain[0, 1], eval_strain[1, 1]]
            )
        elif dims == 3:
            ret = sp.Matrix(
                6,
                1,
                [
                    eval_strain[0, 0],
                    eval_strain[0, 1],
                    eval_strain[0, 2],
                    eval_strain[1, 1],
                    eval_strain[1, 2],
                    eval_strain[2, 2],
                ],
            )

        return assign_matrix("strain", ret)

    def l2_project(self):
        v = sp.symbols("v")
        fe = self.fe
        q = fe.quadrature_point()
        f = fe.fun(q)
        dV = self.fe.symbol_jacobian_determinant() * self.fe.reference_measure()
        nfuns = len(f)

        vals = sp.zeros(nfuns, 1)
        for i in range(0, nfuns):
            integr = f[i] * v * dV
            vals[i] = integr
        return assign_add_matrix("values", vals)

    def loperand(self):
        P = self.P

        jac_inv = self.fe.symbol_jacobian_inverse_as_adjugate()
        P_tXJinv_t = (
            P
            * jac_inv.T
            * (self.fe.symbol_jacobian_determinant() * self.fe.reference_measure())
        )

        dims = self.fe.manifold_dim()
        for i in range(0, dims):
            for j in range(0, dims):
                P_tXJinv_t[i, j] = simplify(P_tXJinv_t[i, j])

        expr = assign_matrix("P_tXJinv_t", P_tXJinv_t)
        return expr

    # def hessian_apply(self):
    # 	fe = self.fe
    # 	q = self.q
    # 	dims = fe.manifold_dim()
    # 	disp_grad_name = 'disp_grad'
    # 	mu, lmbda = sp.symbols('mu lambda', real=True)
    # 	disp_grad = sp.Matrix(dims, dims, coeffs(disp_grad_name, dims * dims))
    # 	trial_symb = sp.Matrix(dims, dims, coeffs("grad_trial", dims * dims))
    # 	test_symb = sp.Matrix(dims, dims, coeffs("grad_test", dims * dims))
    # 	loperand_symb = sp.Matrix(dims, dims, coeffs("loperand", dims * dims))
    # 	inc = coeffs("inc", dims*dims)

    # 	jac_inv = fe.symbol_jacobian_inverse_as_adjugate()

    # 	# Evaluate physical coordinates displacement gradient
    # 	shape_grad_ref = fe.tgrad(q)
    # 	eval_inc_grad = sp.zeros(dims, dims)
    # 	for i in range(0, dims * fe.n_nodes()):
    # 		eval_inc_grad += inc[i] * shape_grad_ref[i]
    # 	eval_inc_grad = eval_inc_grad * jac_inv

    # 	# Strain energy function
    # 	epsu = (disp_grad + disp_grad.T) / 2
    # 	Psi = mu * inner(epsu, epsu) + (lmbda/2) * (tr(epsu) * tr(epsu))

    # 	P = sp.Matrix(dims, dims, [0]*(dims*dims))
    # 	for d1 in range(0, dims):
    # 		for d2 in range(0, dims):
    # 			P[d1, d2] = sp.diff(Psi, disp_grad[d1, d2])

    # 	# Directional derivative of strain energy function (use trial instead of test, exploit symmetry )
    # 	dPsi = inner(P.T, trial_symb)

    # 	linearized_stress = sp.Matrix(dims, dims, [0]*(dims*dims))
    # 	for d1 in range(0, dims):
    # 		for d2 in range(0, dims):
    # 			linearized_stress[d1, d2] = sp.simplify(sp.diff(dPsi, disp_grad[d1, d2]))

    # 	dV = fe.reference_measure() * fe.symbol_jacobian_determinant() * fe.quadrature_weight()

    # 	loperand = linearized_stress.T * (jac_inv.T * dV)

    # 	expr = subsmat(expr, test_symb, test_repl)

    # 	bform = inner(loperand_symb, test_symb)

    def hessian_less_registers(self):
        fe = self.fe
        dims = fe.manifold_dim()
        disp_grad_name = "disp_grad"
        mu, lmbda = sp.symbols("mu lambda", real=True)
        disp_grad = sp.Matrix(dims, dims, coeffs(disp_grad_name, dims * dims))
        trial_symb = sp.Matrix(dims, dims, coeffs("grad_trial", dims * dims))
        test_symb = sp.Matrix(dims, dims, coeffs("grad_test", dims * dims))
        loperand_symb = sp.Matrix(dims, dims, coeffs("loperand", dims * dims))

        # Strain energy function
        epsu = (disp_grad + disp_grad.T) / 2
        Psi = mu * inner(epsu, epsu) + (lmbda / 2) * (tr(epsu) * tr(epsu))
        jac_inv = fe.symbol_jacobian_inverse_as_adjugate()

        P = sp.Matrix(dims, dims, [0] * (dims * dims))
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                P[d1, d2] = sp.diff(Psi, disp_grad[d1, d2])

        # Directional derivative of strain energy function
        dPsi = inner(P * jac_inv.T, test_symb)

        linearized_stress = sp.Matrix(dims, dims, [0] * (dims * dims))
        for d1 in range(0, dims):
            for d2 in range(0, dims):
                linearized_stress[d1, d2] = sp.simplify(
                    sp.diff(dPsi, disp_grad[d1, d2])
                )

        dV = (
            fe.reference_measure()
            * fe.symbol_jacobian_determinant()
            * fe.quadrature_weight()
        )

        # print(linearized_stress - linearized_stress)

        loperand = linearized_stress * (jac_inv.T * dV)
        bform = inner(loperand_symb, trial_symb)

        if False:
            loperands = []
            for d1 in range(0, dims):
                expr = loperand
                test_repl = sp.zeros(dims, dims)

                for d2 in range(0, dims):
                    test_repl[d1, d2] = sp.symbols(f"gtest[{d2}]")

                expr = subsmat(expr, test_symb, test_repl)
                loperands.extend(assign_matrix(f"loperand{d1}", expr))

            return {"loperands": loperands}

        else:
            return {
                "loperand": assign_matrix("loperand", loperand),
                "bform": [ast.Assignment(sp.symbols("bform"), sp.simplify(bform))],
            }

    def hessian(self):
        H = self.eval_hessian
        rows, cols = H.shape

        expr = []
        for i in range(0, rows):
            for j in range(0, cols):
                var = sp.symbols(f"element_matrix[{i*cols + j}*stride]")
                expr.append(ast.Assignment(var, H[i, j]))

        return expr

    def hessian_blocks(self):
        H = self.eval_hessian
        rows, cols = H.shape
        fe = self.fe

        n = fe.n_nodes()
        dims = fe.spatial_dim()

        blocks = []

        for d1 in range(0, dims):
            for d2 in range(0, dims):
                expr = []
                for i in range(0, n):
                    for j in range(0, n):
                        var = sp.symbols(f"element_matrix[{i*n + j}]")
                        expr.append(
                            ast.AddAugmentedAssignment(var, H[d1 * n + i, d2 * n + j])
                        )
                blocks.append((f"block_{d1}_{d2}", expr))

        return blocks

    def hessian_blocks_tpl(self):
        tpl = """
template<typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void  cu_hex8_linear_elasticity_matrix_{BLOCK_NAME}(
const scalar_t mu,
const scalar_t lambda,
const scalar_t *const SFEM_RESTRICT adjugate,
const scalar_t jacobian_determinant,
const scalar_t qx,
const scalar_t qy,
const scalar_t qz,
const scalar_t qw,
accumulator_t *const SFEM_RESTRICT
element_matrix) 
{{
	{CODE}
}}
"""

    def hessian_diag(self):
        H = self.eval_hessian
        rows, cols = H.shape

        expr = []

        if self.SoA_IO:
            assert self.fe.SoA

            coords = ["x", "y", "z"]
            for d in range(0, self.fe.spatial_dim()):
                name = f"diag{coords[d]}"
                for i in range(0, self.fe.n_nodes()):
                    idx = d * self.fe.n_nodes() + i
                    Hii = H[idx, idx]

                    var = sp.symbols(f"{name}[{i}]")
                    expr.append(ast.Assignment(var, Hii))
        else:
            for i in range(0, rows):
                var = sp.symbols(f"diag[{i}*stride]")
                expr.append(ast.Assignment(var, (H[i, i])))

        return expr

    def linearized_stress(self, i):
        s = self.lin_stress[i]

        rows, cols = s.shape

        for i in range(0, rows):
            for j in range(0, cols):
                s[i, j] = simplify(s[i, j])

        expr = assign_matrix("lin_stress", s)
        return expr

    def jacobian(self):
        expr = []
        expr.extend(assign_matrix("jacobian", self.jac))
        return expr

    def geometry(self):
        expr = []
        # J = self.jac
        J = matrix_coeff("jacobian", self.fe.spatial_dim(), self.fe.manifold_dim())

        # expr = assign_matrix('jacobian', J)

        if self.fe.use_adjugate:
            adj = adjugate(J)
            expr.extend(assign_matrix("adjugate", adj))
        else:
            jac_inv = inverse(J)
            expr.extend(assign_matrix("jacobian_inverse", jac_inv))

        J_det = determinant(J)
        expr.append(ast.AddAssignment(sp.symbols("jacobian_determinant"), J_det))
        return expr

    def gradient(self):
        g = self.eval_grad
        rows, cols = g.shape

        expr = []
        for i in range(0, rows):
            var = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.Assignment(var, g[i]))

        return expr

    def gradient_opt(self):
        g = self.eval_grad_opt
        rows, cols = g.shape

        expr = []
        if self.SoA_IO:
            assert self.fe.SoA

            coords = ["x", "y", "z"]
            for d in range(0, self.fe.spatial_dim()):
                name = f"out{coords[d]}"
                for i in range(0, self.fe.n_nodes()):
                    idx = d * self.fe.n_nodes() + i

                    var = sp.symbols(f"{name}[{i}]")
                    expr.append(ast.Assignment(var, g[idx]))

        else:
            for i in range(0, rows):
                var = sp.symbols(f"element_vector[{i}*stride]")
                expr.append(ast.Assignment(var, g[i]))

        return expr

    def value(self):
        form = sp.symbols(f"element_scalar[0]")
        return [ast.Assignment(form, self.eval_value)]

    def apply(self):
        H = self.integr_hessian
        rows, cols = H.shape
        inc = self.increment

        expr = []
        for i in range(0, rows):
            val = 0
            for j in range(0, cols):
                val += H[i, j] * inc[j]

            var = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.Assignment(var, val))
        return expr


def main():
    start = perf_counter()

    fes = {
        "TRI6": Tri6(),
        "TRI3": Tri3(),
        "TET4": Tet4(),
        "TET10": Tet10(),
        "TET20": Tet20(),
        "HEX8": Hex8(),
        "AAHEX8": AAHex8(),
        "AAQUAD4": AxisAlignedQuad4(),
    }

    if len(sys.argv) >= 2:
        fe = fes[sys.argv[1]]
    else:
        print("Fallback with TET10")
        fe = Tet10()

    op = GPULinearElasticityOp(fe)
    # op.hessian_check()

    # tpl = op.hessian_blocks_tpl()
    # blocks = op.hessian_blocks()
    # for k,v in blocks:
    # 	c_log("//--------------------------")
    # 	c_log(f"// hessian {k}")
    # 	c_log("//--------------------------")
    # 	code = c_gen(v)
    # 	c_log(tpl.format(BLOCK_NAME=k, CODE=code))

    # if False:
    # 	c_log("//--------------------------")
    # 	c_log("// New hessian")
    # 	c_log("//--------------------------")

    # 	kv = op.hessian_less_registers()
    # 	for k, v in kv.items():

    # 		print("---------------------")
    # 		print(f"{k}")
    # 		print("---------------------")
    # 		c_code(v)

    # 	# c_log("//--------------------------")
    # 	# c_log("// New hessian apply")
    # 	# c_log("//--------------------------")

    # 	# kv = op.hessian_apply()
    # 	# for k, v in kv.items():

    # 	# 	print("---------------------")
    # 	# 	print(f"{k}")
    # 	# 	print("---------------------")
    # 	# 	c_code(v)

    # else:
    # 	c_log("//--------------------------")
    # 	c_log("// geometry")
    # 	c_log("//--------------------------")

    # 	c_code(op.jacobian())
    # 	c_code(op.geometry())

    c_log("//--------------------------")
    c_log("// CauchyStress")
    c_log("//--------------------------")
    c_code(op.cauchy_stress())

    c_log("//--------------------------")
    c_log("// Strain")
    c_log("//--------------------------")
    c_code(op.strain())

    c_log("//--------------------------")
    c_log("// L2-project")
    c_log("//--------------------------")
    c_code(op.l2_project())

    c_log("//--------------------------")
    c_log("// displacement_gradient")
    c_log("//--------------------------")
    c_code(op.displacement_gradient())

    c_log("//--------------------------")
    c_log("// Piola")
    c_log("//--------------------------")
    c_code(op.first_piola())

    # c_log("//--------------------------")
    # c_log("// gradient")
    # c_log("//--------------------------")
    # c_code(op.gradient())

    c_log("//--------------------------")
    c_log("// loperand")
    c_log("//--------------------------")
    c_code(op.loperand())

    c_log("//--------------------------")
    c_log("// gradient_opt")
    c_log("//--------------------------")
    c_code(op.gradient_opt())

    # 	# c_log("//--------------------------")
    # 	# c_log("// value")
    # 	# c_log("//--------------------------")
    # 	# c_code(op.value())

    # c_log("--------------------------")
    # c_log("hessian")
    # c_log("--------------------------")
    # c_code(op.hessian())

    # 	c_log("--------------------------")
    # 	c_log("hessian_diag")
    # 	c_log("--------------------------")
    # 	c_code(op.hessian_diag())

    # 	# c_log("--------------------------")
    # 	# c_log("lin stress")
    # 	# c_log("--------------------------")

    # 	# print(fe.n_nodes() * fe.spatial_dim())

    # 	# for i in range(0, fe.n_nodes() * fe.spatial_dim()):
    # 	# 	c_code(op.linearized_stress(i))

    # 	# c_log("--------------------------")
    # 	# c_log("apply")
    # 	# c_log("--------------------------")
    # 	# c_code(op.apply())

    stop = perf_counter()
    console.print(f"// Overall: {stop - start} seconds")


if __name__ == "__main__":
    main()
