#!/usr/bin/env python3

from sfem_codegen import *
from tet4 import *
from tet10 import *
from tri3 import *
from tri6 import *
from mini import *
from fe_material import *

import pdb

# simplify_expr = False
simplify_expr = False
subs_jacobian = True

# 1) temptative momentum step
# - Implicit Euler
# `<u, v> + dt * nu * <grad(u), grad(v)> = <u_old, v> - dt * <(u_old . div) * u_old, v>`
# - Explicit Euler
# `<u, v> = <u_old, v> - dt * ( <(u_old . div) * u_old, v> + nu * <grad(u), grad(v)> )`
# 2) Potential eqaution
# `<grad(p), grad(q)> = - 1/dt * <div(u), q>`
# 3) Projection/Correction
# `<u_new, v> = <u, v> - dt * <grad(p), v>`


class NavierStokesOp:
    def __init__(self, fe_vel, fe_pressure):
        self.fe_vel = fe_vel
        self.fe_pressure = fe_pressure

        if fe_vel.spatial_dim() == 2:
            qp = [qx, qy]
        else:
            assert fe_vel.spatial_dim() == 3
            qp = [qx, qy, qz]

        qp = sp.Matrix(fe_vel.spatial_dim(), 1, qp)
        self.qp = qp

        grad_vel = fe_vel.physical_tgrad(qp)
        fun_vel = fe_vel.tfun(qp)
        fun_pressure = fe_pressure.fun(qp)
        grad_pressure = fe_pressure.physical_grad(qp)

        n_vel = len(grad_vel)
        n_pressure = len(fun_pressure)

        nu, rho, dt = sp.symbols("nu rho dt")
        self.params = [nu, rho, dt]

        self.form2_diffusion = sp.zeros(n_vel, n_vel)
        self.form2_mass = sp.zeros(n_vel, n_vel)
        self.form2_laplacian = sp.zeros(n_pressure, n_pressure)

        integrate_implicit = False

        if integrate_implicit:
            print("Integrating bilinear forms")
            for i in range(0, n_vel):
                for j in range(0, n_vel):
                    # dt * nu * < grad(u), grad(v) >
                    integr = (
                        dt
                        * nu
                        * fe_vel.integrate(qp, inner(grad_vel[i], grad_vel[j]))
                        * fe_vel.symbol_jacobian_determinant()
                    )
                    self.form2_diffusion[i, j] = integr

            for i in range(0, n_vel):
                for j in range(0, n_vel):
                    integr = (
                        fe_vel.integrate(qp, inner(fun_vel[i], fun_vel[j]))
                        * fe_vel.symbol_jacobian_determinant()
                    )
                    self.form2_mass[i, j] = integr

            # for i in range(0,  n_vel):
            # 	for j in range(0,  n_pressure):
            # 		integr = -(1/dt)*fe_pressure.integrate(qp, tr(grad_vel[i]) * fun_pressure[j]) * fe_pressure.symbol_jacobian_determinant()
            # 		self.form2_divergence[i, j] = integr

            for i in range(0, n_pressure):
                for j in range(0, n_pressure):
                    integr = (
                        fe_pressure.integrate(
                            qp, inner(grad_pressure[i], grad_pressure[j])
                        )
                        * fe_pressure.symbol_jacobian_determinant()
                    )
                    self.form2_laplacian[i, j] = integr

        u = coeffs("u", n_vel)
        uh = u[0] * fun_vel[0]
        for i in range(1, n_vel):
            uh += u[i] * fun_vel[i]

        grad_uh = u[0] * grad_vel[0]
        for i in range(1, n_vel):
            grad_uh += u[i] * grad_vel[i]

        div_uh = tr(grad_uh)

        p = coeffs("p", n_pressure)
        grad_ph = p[0] * grad_pressure[0]
        for i in range(1, n_pressure):
            grad_ph += p[i] * grad_pressure[i]

        ph = p[0] * fun_pressure[0]
        for i in range(1, n_pressure):
            ph += p[i] * fun_pressure[i]

        #########################################################
        # CONVECTION
        conv = sp.zeros(fe_vel.spatial_dim(), 1)

        for d in range(0, fe_vel.spatial_dim()):
            val = 0
            for d1 in range(0, fe_vel.spatial_dim()):
                val += uh[d1] * grad_uh[d, d1]
            conv[d] = val
        #########################################################

        self.form1_diffusion = sp.zeros(n_vel, 1)
        self.form1_convection = sp.zeros(n_vel, 1)
        self.form1_divergence = sp.zeros(n_pressure, 1)
        self.form1_correction = sp.zeros(n_vel, 1)
        self.form1_utm1 = sp.zeros(n_vel, 1)

        #########################################################

        print("Integrating linear forms")

        print("Diffusion")
        for i in range(0, n_vel):
            integr = 0
            print(f"{i+1}/{n_vel})")
            for d1 in range(0, fe_vel.spatial_dim()):
                for d2 in range(0, fe_vel.spatial_dim()):
                    print(
                        f"\t{d1 * fe_vel.spatial_dim() + d2 + 1}/{fe_vel.spatial_dim()*fe_vel.spatial_dim()}"
                    )
                    integr += (
                        -dt
                        * nu
                        * fe_vel.integrate(qp, grad_uh[d1, d2] * grad_vel[i][d1, d2])
                        * fe_vel.jacobian_determinant(qp)
                    )

            # integr = -dt * nu * fe_vel.integrate(qp, inner(grad_uh, grad_vel[i])) * fe_vel.jacobian_determinant(qp)
            self.form1_diffusion[i] = integr

        # if False:
        if True:
            print("Convection")
            for i in range(0, n_vel):
                integr = 0
                print(f"{i+1}/{n_vel})")
                for d in range(0, fe_vel.spatial_dim()):
                    print(f"\t{d+1}/{fe_vel.spatial_dim()}")
                    integr += (
                        -dt
                        * fe_vel.integrate(qp, conv[d] * fun_vel[i][d])
                        * fe_vel.jacobian_determinant(qp)
                    )
                self.form1_convection[i] = integr

        # print("Mass")
        # for i in range(0, n_vel):
        # 	integr = 0
        # 	print(f"{i+1}/{n_vel})")
        # 	for d in range(0, fe_vel.spatial_dim()):
        # 		print(f"\t{d+1}/{fe_vel.spatial_dim()}")
        # 		integr += fe_vel.integrate(qp, uh[d] * fun_vel[i][d]) * fe_vel.jacobian_determinant(qp)
        # 	self.form1_utm1[i] = integr

        print("Div")
        for i in range(0, n_pressure):
            print(f"{i+1}/{n_pressure})")
            integr = (
                -(rho / dt)
                * fe_pressure.integrate(qp, div_uh * fun_pressure[i])
                * fe_pressure.jacobian_determinant(qp)
            )
            self.form1_divergence[i] = integr

        print("Projection")
        for i in range(0, n_vel):
            integr = 0
            print(f"{i+1}/{n_vel})")
            for d in range(0, fe_vel.spatial_dim()):
                print(f"\t{d+1}/{fe_vel.spatial_dim()}")
                integr += (
                    -(dt / rho) * fe_vel.integrate(qp, grad_ph[d] * fun_vel[i][d])
                ) * fe_vel.jacobian_determinant(qp)
            self.form1_correction[i] = integr

        if integrate_implicit:
            print("------------------------------")
            print("LHS (diffusion, temptative momentum)")
            print("------------------------------")

            tent_mom = self.form2_mass + self.form2_diffusion
            c_code(self.assign_matrix(tent_mom))

            # print('------------------------------')
            # print('LHS (diffusion, temptative momentum) X')
            # print('------------------------------')

            # tent_mom_x = tent_mom[0:fe_vel.n_nodes(),0:fe_vel.n_nodes()]
            # c_code(self.assign_matrix(tent_mom_x))

            # print('------------------------------')
            # print('LHS (diffusion, temptative momentum) Y')
            # print('------------------------------')

            # tent_mom_y = tent_mom[fe_vel.n_nodes():2*fe_vel.n_nodes(),fe_vel.n_nodes():2*fe_vel.n_nodes()]
            # c_code(self.assign_matrix(tent_mom_y))

            print("------------------------------")
            print("LHS (potential equation)")
            print("------------------------------")
            c_code(self.assign_matrix(self.form2_laplacian))

            print("------------------------------")
            print("RHS (temptative momentum)")
            print("------------------------------")
            # c_code(self.assign_vector(self.form1_utm1 + self.form1_convection))
            # print('Mom')
            # c_code(self.assign_vector(self.form1_utm1))

            print("Conv")
            c_code(self.in_place_add_vector(self.form1_convection))

        print("------------------------------")
        print("RHS (potential equation)")
        print("------------------------------")
        c_code(self.assign_vector(self.form1_divergence))

        print("------------------------------")
        print("RHS (correction equation)")
        print("------------------------------")
        c_code(self.assign_vector(self.form1_correction))

        print("------------------------------")
        print("RHS (explicit momentum)")
        print("------------------------------")
        # c_code(self.assign_vector(self.form1_utm1 + self.form1_diffusion + self.form1_convection))

        # print('Mom')
        # c_code(self.assign_vector(self.form1_utm1))
        print("Diff")
        c_code(self.in_place_add_vector(self.form1_diffusion))
        print("Conv")
        c_code(self.in_place_add_vector(self.form1_convection))

    # def apply(self):
    # 	H = self.hessian
    # 	rows, cols = H.shape
    # 	x = self.increment

    # 	Hx = H * x
    # 	return self.assign_vector(Hx)

    # def gradient(self):
    # 	return self.apply()

    def assign_tensor(self, name, a):
        return self.op_tensor(name, a, ast.Assignment)

    def in_place_add(self, name, a):
        return self.op_tensor(name, a, ast.AddAugmentedAssignment)

    def in_place_add_vector(self, a):
        return self.op_tensor("element_vector", a, ast.AddAugmentedAssignment)

    def in_place_add_matrix(self, name, a):
        return self.op_tensor("element_matrix", a, ast.AddAugmentedAssignment)

    def op_tensor(self, name, a, op):
        fe = self.fe_vel
        qp = self.qp
        rows, cols = a.shape

        expr = []
        for i in range(0, rows):
            for j in range(0, cols):
                var = sp.symbols(f"{name}[{i*cols + j}]")
                value = a[i, j]

                if subs_jacobian:
                    value = subsmat(
                        value, fe.symbol_jacobian_inverse(), fe.jacobian_inverse(qp)
                    )
                    value = value.subs(
                        fe.symbol_jacobian_determinant(), fe.jacobian_determinant(qp)
                    )

                if simplify_expr:
                    value = sp.simplify(value)

                expr.append(op(var, value))
        return expr

    def assign_matrix(self, a):
        return self.assign_tensor("element_matrix", a)

    def assign_vector(self, a):
        return self.assign_tensor("element_vector", a)


def main():
    # op = NavierStokesOp(Mini2D())
    # op = NavierStokesOp(Tri6(), Tri3())
    op = NavierStokesOp(Tet10(), Tet4())


if __name__ == "__main__":
    main()
