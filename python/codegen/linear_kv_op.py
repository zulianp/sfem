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
from symbolic_fe import *

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


class LinearKVOp:
    SoA_IO = True

    def __init__(self, fe):
        dims = fe.manifold_dim()

        q = fe.quadrature_point()

        self.init_opt(fe, q)

    def init_opt(self, fe, q):
        fe.use_adjugate = True
        dV = fe.symbol_jacobian_determinant()
        dims = fe.manifold_dim()
        shape_grad = fe.physical_tgrad(q)
        shape_grad_ref = fe.tgrad(q)
        shape_vec = fe.tfun(q)
        jac_inv = fe.symbol_jacobian_inverse_as_adjugate()
        e_jac_inv = fe.jacobian_inverse(q)

        if self.SoA_IO:
            disp = coeffs_SoA("u", dims, fe.n_nodes())
            velo = coeffs_SoA("v", dims, fe.n_nodes())
            acce = coeffs_SoA("a", dims, fe.n_nodes())
        else:
            disp = coeffs("u", dims * fe.n_nodes())
            velo = coeffs("v", dims * fe.n_nodes())
            acce = coeffs("a", dims * fe.n_nodes())

        self.jac = fe.jacobian(q)
        rows = fe.n_nodes() * dims
        cols = rows

        ###################################################################
        # Material law
        ###################################################################
        self.disp_grad_name = "disp_grad"
        self.velo_grad_name = "velo_grad"
        self.acce_vec_name = "acce_vec"

        rho, eta, k, K, dt, gamma, beta = sp.symbols("rho eta k K dt gamma beta", real=True)
        disp_grad = sp.Matrix(dims, dims, coeffs(self.disp_grad_name, dims * dims))
        velo_grad = sp.Matrix(dims, dims, coeffs(self.velo_grad_name, dims * dims))
        acce_vec = sp.Matrix(dims, 1, coeffs(self.acce_vec_name, dims))

        # Calculate displacement gradient
        self.eval_disp_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            self.eval_disp_grad += disp[i] * shape_grad_ref[i]
        self.eval_disp_grad = self.eval_disp_grad * jac_inv

        # Calculate velocity gradient
        self.eval_velo_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            self.eval_velo_grad += velo[i] * shape_grad_ref[i]
        self.eval_velo_grad = self.eval_velo_grad * jac_inv

        # # Calculate acceleration vector
        self.eval_acce_vec = sp.zeros(dims, 1)
        for i in range(0, dims * fe.n_nodes()):
            self.eval_acce_vec += acce[i] * shape_vec[i]

        # Test displacement gradient (physical space)
        test_disp_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            test_disp_grad += disp[i] * shape_grad[i]

        # Test velocity gradient (physical space)
        test_velo_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            test_velo_grad += velo[i] * shape_grad[i]

        # Verify transformations
        for i in range(0, dims):
            for j in range(0, dims):
                self.eval_disp_grad[i, j] = simplify(self.eval_disp_grad[i, j])
                self.eval_velo_grad[i, j] = simplify(self.eval_velo_grad[i, j])

                # Check displacement gradient
                # diff_disp = sp.simplify(test_disp_grad[i, j] - self.eval_disp_grad[i, j])
                # assert diff_disp == 0
                
                # Check velocity gradient
                # diff_velo = sp.simplify(test_velo_grad[i, j] - self.eval_velo_grad[i, j])
                # assert diff_velo == 0

        # Strain, damping and stress tensors
        I_matrix = sp.eye(dims)
        epsu = (disp_grad + disp_grad.T) / 2
        epsv = (velo_grad + velo_grad.T) / 2
        
        # Material constitutive relations
        damping = eta * (epsv - (1/dims) * tr(epsv) * I_matrix)
        stiffness = k * epsu + (K - k/dims) * tr(epsu) * I_matrix
        inertia = rho * acce_vec


        # Compute weak forms for gradient vectors
        eval_inertia = sp.zeros(rows, 1)
        eval_damping = sp.zeros(rows, 1)
        eval_stiffness = sp.zeros(rows, 1)

        for i in range(0, fe.n_nodes() * dims):
            eval_damping[i] = inner(damping, shape_grad[i]) * dV
            eval_damping[i] = simplify(eval_damping[i])
            
            eval_stiffness[i] = inner(stiffness, shape_grad[i]) * dV
            eval_stiffness[i] = simplify(eval_stiffness[i])
            
            eval_inertia[i] = inner(inertia, shape_vec[i]) * dV
            eval_inertia[i] = simplify(eval_inertia[i])




        # Compute matrices
        eval_M_matrix = sp.zeros(rows, rows)
        eval_C_matrix = sp.zeros(rows, cols)
        eval_K_matrix = sp.zeros(rows, cols)

        for j in range(0, rows):
            eps_j = (shape_grad[j] + shape_grad[j].T) / 2
            
            for i in range(0, cols):
                eps_i = (shape_grad[i] + shape_grad[i].T) / 2
                
                # Damping matrix components
                integrand_C = eta * (inner(eps_j, eps_i) - (1/dims) * tr(eps_j) * tr(eps_i))
                eval_C_matrix[i, j] = integrand_C 
                eval_C_matrix[i, j] = simplify(eval_C_matrix[i, j])
                
                # Stiffness matrix components
                integrand_K = (K - (1/dims) * k) * tr(eps_j) * tr(eps_i) + k * inner(eps_j, eps_i)
                eval_K_matrix[i, j] = integrand_K 
                eval_K_matrix[i, j] = simplify(eval_K_matrix[i, j])

                # # Mass matrix components
                dim_i = i // fe.n_nodes()
                node_i = i % fe.n_nodes()
                dim_j = j // fe.n_nodes()
                node_j = j % fe.n_nodes()

                if dim_i == dim_j:
                    eval_M_matrix[i, j] = rho * shape_vec[node_i+dim_i*fe.n_nodes()][dim_i,0] * shape_vec[node_j+dim_j*fe.n_nodes()][dim_j,0] 
                    eval_M_matrix[i, j] = simplify(eval_M_matrix[i, j])
                else:
                    eval_M_matrix[i, j] = 0



        # Prepare LHS matrix for Newmark method
        # eval_lhs_matrix = eval_M_matrix/(beta*dt*dt) + eval_C_matrix*gamma/(beta*dt) + eval_K_matrix
        # eval_gradient = eval_stiffness + eval_damping + eval_inertia

        eval_lhs_matrix = eval_M_matrix/(beta*dt*dt) + eval_C_matrix*gamma/(beta*dt) + eval_K_matrix
        eval_gradient = eval_inertia + eval_stiffness + eval_damping 

        ###################################################################
        # Gradient
        ###################################################################

        full_eval = False



        for i in range(0, dims):
            for j in range(0, dims):
                integr = fe.integrate(q, self.eval_disp_grad[i, j])
                if full_eval:
                    integr = subsmat(integr, jac_inv, e_jac_inv)
                self.eval_disp_grad[i, j] = integr

        for i in range(0, dims):
            for j in range(0, dims):
                integr = fe.integrate(q, self.eval_velo_grad[i, j])
                if full_eval:
                    integr = subsmat(integr, jac_inv, e_jac_inv)
                self.eval_velo_grad[i, j] = integr

        for i in range(0, dims):
                integr = fe.integrate(q, self.eval_acce_vec[i])
                if full_eval:
                    integr = subsmat(integr, jac_inv, e_jac_inv)
                self.eval_acce_vec[i] = integr

        for i in range(0, rows):
            for j in range(0, cols):
                integr = fe.integrate(q, eval_lhs_matrix[i, j])
                if full_eval:
                    integr = subsmat(integr, disp_grad, self.eval_disp_grad)
                    integr = subsmat(integr, jac_inv, e_jac_inv)
                integr = integr * dV
                eval_lhs_matrix[i, j] = integr

        for i in range(0, rows):
            integr = fe.integrate(q, eval_gradient[i])
            if full_eval:
                    integr = subsmat(integr, disp_grad, self.eval_disp_grad)
                    integr = subsmat(integr, jac_inv, e_jac_inv)
            integr = integr 
            eval_gradient[i] = integr


        ###################################################################


        # Store results
        self.rho = rho
        self.eta = eta
        self.k = k
        self.K = K
        self.dt = dt
        self.gamma = gamma
        self.beta = beta

        self.eval_damping = eval_damping
        self.eval_stiffness = eval_stiffness
        self.eval_inertia = eval_inertia
        self.eval_C_matrix = eval_C_matrix
        self.eval_K_matrix = eval_K_matrix
        self.eval_M_matrix = eval_M_matrix
        self.eval_lhs_matrix = eval_lhs_matrix
        self.eval_gradient = eval_gradient 

        self.fe = fe
        if self.SoA_IO:
            self.increment = coeffs_SoA("increment", dims, fe.n_nodes())
        else:
            self.increment = coeffs("increment", dims * fe.n_nodes())


    def displacement_gradient(self):
        expr = assign_matrix(self.disp_grad_name, self.eval_disp_grad)
        return expr

    def velocity_gradient(self):
        expr = assign_matrix(self.velo_grad_name, self.eval_velo_grad)
        return expr
        
    def acceleration_vector(self):
        expr = assign_matrix(self.acce_vec_name, self.eval_acce_vec)
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

    def jacobian(self):
        expr = []
        expr.extend(assign_matrix("jacobian", self.jac))
        return expr


    
    def lhs_matrix(self):
        lhs = self.eval_lhs_matrix
        rows, cols = lhs.shape

        expr = []
        for i in range(0, rows):
            for j in range(0, cols):
                var = sp.symbols(f"element_matrix[{i*cols + j}*stride]")
                expr.append(ast.Assignment(var, lhs[i, j]))

        return expr
    
    def lhs_sym(self):
        lhs = self.eval_lhs_matrix
        rows, cols = lhs.shape

        expr = []
        idx = 0
        for i in range(0, rows):
            for j in range(0, cols):
                if j > i:
                    continue
                var = sp.symbols(f"element_matrix[{idx}*stride]")
                expr.append(ast.Assignment(var, lhs[i, j]))
                idx += 1

        return expr

    
    def gradient_M(self):
        g = self.eval_inertia
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

        
    def apply_lhs(self):  
        lhs = self.eval_lhs_matrix
        rows, cols = lhs.shape
        increment = self.increment

        expr = []
        if self.SoA_IO:
            assert self.fe.SoA

            coords = ["x", "y", "z"]
            for d in range(0, self.fe.spatial_dim()):
                name = f"out{coords[d]}"
                for i in range(0, self.fe.n_nodes()):
                    idx = d * self.fe.n_nodes() + i
                    L = 0
                    for j in range(0, cols):
                        L += lhs[idx, j] * increment[j]

                    var = sp.symbols(f"{name}[{i}]")
                    expr.append(ast.Assignment(var, L))
        else:
            for i in range(0, rows):
                L = 0
                for j in range(0, cols):
                    L += lhs[i, j] * increment[j]

                var = sp.symbols(f"element_vector[{i}*stride]")
                expr.append(ast.Assignment(var, L))
                
        return expr
    

    def gradient(self):
        g = self.eval_gradient
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

    def hessian_diag(self):
        H = self.eval_lhs_matrix
        rows, cols = H.shape

        expr = []
        for i in range(0, rows):
            var = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.Assignment(var, H[i, i]))

        return expr

    def hessian_sym(self):
        H = self.eval_lhs_matrix
        rows, cols = H.shape

        expr = []
        idx = 0
        for i in range(0, rows):
            for j in range(0, cols):
                if j > i:
                    continue
                var = sp.symbols(f"element_matrix[{idx}*stride]")
                expr.append(ast.Assignment(var, H[i, j]))
                idx += 1

        return expr


def main():
    start = perf_counter()

    # fe = Hex8(False, False)
    fe = SymbolicFE3D()
    # fe = Tri3()

    op = LinearKVOp(fe)

    # c_log("//--------------------------")
    # c_log("// geometry")
    # c_log("//--------------------------")
    # c_code(op.jacobian())
    # c_code(op.geometry())

    # c_log("//--------------------------")
    # c_log("// displacement_gradient")
    # c_log("//--------------------------")
    # c_code(op.displacement_gradient())

    # c_log("//--------------------------")
    # c_log("// velocity_gradient")
    # c_log("//--------------------------")
    # c_code(op.velocity_gradient())

    # c_log("//--------------------------")
    # c_log("// acceleration_vector")
    # c_log("//--------------------------")
    # c_code(op.acceleration_vector())

    # c_log("//--------------------------")
    # c_log("// C_matrix")
    # c_log("//--------------------------")
    # c_code(op.C_matrix())

    # c_log("//--------------------------")
    # c_log("// K_matrix")
    # c_log("//--------------------------")
    # c_code(op.K_matrix())

    # c_log("//--------------------------")
    # c_log("// M_matrix")
    # c_log("//--------------------------")
    # c_code(op.M_matrix())

    # c_log("//--------------------------")
    # c_log("// C_sym")
    # c_log("//--------------------------")
    # c_code(op.C_sym())

    # c_log("//--------------------------")
    # c_log("// K_sym")
    # c_log("//--------------------------")
    # c_code(op.K_sym())

    # c_log("//--------------------------")
    # c_log("// M_sym")
    # c_log("//--------------------------")
    # c_code(op.M_sym())

    # c_log("//--------------------------")
    # c_log("// gradient_C")
    # c_log("//--------------------------")
    # c_code(op.gradient_C())

    # c_log("//--------------------------")
    # c_log("// apply_C")
    # c_log("//--------------------------")
    # c_code(op.apply_C())

    # c_log("//--------------------------")
    # c_log("// gradient_K")
    # c_log("//--------------------------")
    # c_code(op.gradient_K())
      
    # c_log("//--------------------------")
    # c_log("// apply_K")
    # c_log("//--------------------------")
    # c_code(op.apply_K())

    # c_log("//--------------------------")
    # c_log("// gradient_M")
    # c_log("//--------------------------")
    # c_code(op.gradient_M())

    # c_log("//--------------------------")
    # c_log("// apply_M")
    # c_log("//--------------------------")
    # c_code(op.apply_M())


    # c_log("//--------------------------")
    # c_log("// lhs_matrix")
    # c_log("//--------------------------")
    # c_code(op.lhs_matrix())

    # c_log("//--------------------------")
    # c_log("// lhs_sym")
    # c_log("//--------------------------")
    # c_code(op.lhs_sym())

    # c_log("//--------------------------")
    # c_log("// apply_lhs")
    # c_log("//--------------------------")
    # c_code(op.apply_lhs())

    # c_log("//--------------------------")
    # c_log("// gradient")
    # c_log("//--------------------------")
    # c_code(op.gradient())


    # c_log("//--------------------------")
    # c_log("// hessian_diag")
    # c_log("//--------------------------")
    # c_code(op.hessian_diag())


    c_log("//--------------------------")
    c_log("// hessian_sym")
    c_log("//--------------------------")
    c_code(op.hessian_sym())

    stop = perf_counter()
    console.print(f"// Overall: {stop - start} seconds")


if __name__ == "__main__":
    main()
