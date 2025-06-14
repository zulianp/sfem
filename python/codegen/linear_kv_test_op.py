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
from mass_op import *

from edge2 import *

from fields import *

import sys
from time import perf_counter



class LinearKVOp:
	
    def __init__(self, fe):
        dims = fe.manifold_dim()                        # 2 or 3
        q = fe.quadrature_point()

        shape_grad = fe.physical_tgrad(q)               # size = [dims*dims, 1]
        e_jac_inv = fe.jacobian_inverse(q)
        shape_vec = fe.tfun(q)

        if fe.use_adjugate:
            dV = fe.symbol_jacobian_determinant()
        else:
            dV = fe.jacobian_determinant(q)
        s_jac_inv = fe.symbol_jacobian_inverse()
        
        rows = fe.n_nodes() * dims                       
        cols = rows

        disp = coeffs('u', rows)         # dims * (number of points)
        velo = coeffs('v', rows)
        acce = coeffs('a', rows)

        ###################################################################
        # Material law
        ###################################################################
        c_log("Material law")

        rho, eta, k, K, = sp.symbols('rho eta k K', real=True) # Todo beta and gamma
        s_disp_grad = sp.Matrix(dims, dims, coeffs('disp_grad', dims * dims))        # size = [dims, dims]
        s_velo_grad = sp.Matrix(dims, dims, coeffs('velo_grad', dims * dims))
        s_acce_vec = sp.Matrix(dims, 1, coeffs('acce_vec', dims))
        
        epsu = (s_disp_grad + s_disp_grad.T) / 2                                     # size = [dims, dims]
        epsv = (s_velo_grad + s_velo_grad.T) / 2
        # epsa = (s_acce_grad + s_acce_grad.T) / 2
        
        e_disp_grad = sp.Matrix(dims, dims, [0] * dims * dims)                       # size = [dims, dims]        
        e_velo_grad = sp.Matrix(dims, dims, [0] * dims * dims)
        e_acce_vec = sp.Matrix(dims, 1, [0] * dims)

        for i in range(0, rows):
            e_disp_grad += disp[i] * shape_grad[i]
            e_velo_grad += velo[i] * shape_grad[i]
            e_acce_vec += acce[i] * shape_vec[i]
            # e_acce_grad += acce[i] * shape_grad[i]

        ###################################################################
        # Weak form and M,C,K matrix
        ###################################################################

        I_matrix = sp.eye(dims)

        damping = eta*(epsv - 1/3 * tr(epsv) * I_matrix)
        stiffness = k * epsu + (K - k/3) * tr(epsu) * I_matrix
        inertia = rho * s_acce_vec
        ## todo 1.modify 1/2 to 1/3 for 3d version  2 add M_matrix

        eval_inertia = sp.Matrix(rows, 1, [0] * rows)
        eval_damping = sp.Matrix(rows, 1, [0] * rows)
        eval_stiffness = sp.Matrix(rows, 1, [0] * rows)

        # for example, it's the same as computing sigma:physical_grad_Phi in linear elasticity case
        for i in range(0, fe.n_nodes() * dims):
            eval_damping[i] = inner(damping, shape_grad[i])
            eval_stiffness[i] = inner(stiffness, shape_grad[i])
            eval_inertia[i] = inner(inertia, shape_vec[i])
        
        eval_M_matrix = sp.Matrix(rows, rows, [0] * (rows * rows))
        eval_C_matrix = sp.Matrix(rows, cols, [0] * (rows * cols))
        eval_K_matrix = sp.Matrix(rows, cols, [0] * (rows * cols))

        for j in range(0, rows):
            eps_j = (shape_grad[j] + shape_grad[j].T) / 2
            
            for i in range(0, cols):
                eps_i = (shape_grad[i] + shape_grad[i].T) / 2
                
                integrand_C = eta * (inner(eps_j, eps_i) - 1/3 * tr(eps_j) * tr(eps_i))
                integrand_K = (K - 1/3 * k) * tr(eps_j) * tr(eps_i) + k * inner(eps_j, eps_i)
                
                eval_C_matrix[i, j] = integrand_C
                eval_K_matrix[i, j] = integrand_K

                dim_i = i // fe.n_nodes()
                node_i = i % fe.n_nodes()
                dim_j = j // fe.n_nodes()
                node_j = j % fe.n_nodes()

                if dim_i == dim_j:
                    eval_M_matrix[i, j] = rho * shape_vec[node_i][dim_i] * shape_vec[node_j][dim_j]
                else:
                    eval_M_matrix[i, j] = 0

        ###################################################################
        # Integrate and substitute
        ###################################################################
        c_log("Integrate")
        
        full_eval = False

        integr_damping = sp.Matrix(rows, 1, [0] * rows)
        integr_C_matrix = sp.Matrix(rows, cols, [0] * (rows * cols))

        integr_stiffness = sp.Matrix(rows, 1, [0] * rows)
        integr_K_matrix = sp.Matrix(rows, cols, [0] * (rows * cols))

        integr_inertia = sp.Matrix(rows, 1, [0] * rows)
        integr_M_matrix = sp.Matrix(rows, cols, [0] * (rows * cols))


        for i in range(0, rows):
            integr_C = fe.integrate(q, eval_damping[i])
            integr_K = fe.integrate(q, eval_stiffness[i])
            integr_M = fe.integrate(q, eval_inertia[i])
            if full_eval:
                integr_C = subsmat(integr_C, s_velo_grad, e_velo_grad)
                integr_C = subsmat(integr_C, s_jac_inv, e_jac_inv)
                integr_K = subsmat(integr_K, s_disp_grad, e_disp_grad)
                integr_K = subsmat(integr_K, s_jac_inv, e_jac_inv)
                integr_M = subsmat(integr_M, s_acce_vec, e_acce_vec)
                integr_M = subsmat(integr_M, s_jac_inv, e_jac_inv)
                
            integr_C = integr_C * dV
            integr_K = integr_K * dV
            integr_M = integr_M * dV
            integr_damping[i] = integr_C
            integr_stiffness[i] = integr_K
            integr_inertia[i] = integr_M


        for i in range(0, rows):
            for j in range(0, cols):
                integr_C = fe.integrate(q, eval_C_matrix[i, j])
                integr_K = fe.integrate(q, eval_K_matrix[i, j])
                integr_M = fe.integrate(q, eval_M_matrix[i, j])
                
                # if full_eval:
                #     integr_C = subsmat(integr_C, s_velo_grad, e_velo_grad)
                #     integr_C = subsmat(integr_C, s_jac_inv, e_jac_inv)
                #     integr_K = subsmat(integr_K, s_disp_grad, e_disp_grad)
                #     integr_K = subsmat(integr_K, s_jac_inv, e_jac_inv)
                #     integr_M = subsmat(integr_M, s_acce_vec, e_acce_vec)
                #     integr_M = subsmat(integr_M, s_jac_inv, e_jac_inv)
                # don't know whether it's necessary to do this

                integr_C = integr_C * dV
                integr_K = integr_K * dV
                integr_M = integr_M * dV
                integr_C_matrix[i, j] = integr_C
                integr_K_matrix[i, j] = integr_K
                integr_M_matrix[i, j] = integr_M

        
        ###################################################################
        # Store results
        ###################################################################
        self.rho = rho
        self.eta = eta
        self.k = k
        self.K = K

        self.eval_damping = eval_damping
        self.eval_stiffness = eval_stiffness
        self.eval_inertia = eval_inertia
        self.eval_C_matrix = eval_C_matrix
        self.eval_K_matrix = eval_K_matrix
        self.eval_M_matrix = eval_M_matrix

        self.integr_damping = integr_damping
        self.integr_stiffness = integr_stiffness
        self.integr_inertia = integr_inertia

        self.integr_C_matrix = integr_C_matrix  
        self.integr_K_matrix = integr_K_matrix  
        self.integr_M_matrix = integr_M_matrix

        self.fe = fe
        # self.increment = coeffs('increment', fe.n_nodes() * dims)
        self.u = coeffs('u', fe.n_nodes() * dims)
        self.v = coeffs('v', fe.n_nodes() * dims)
        self.a = coeffs('a', fe.n_nodes() * dims)

        ##################################################################################
    
    def C_matrix(self):
        C = self.integr_C_matrix
        rows, cols = C.shape

        expr = []
        for i in range(0, rows):
            for j in range(0, cols):
                var = sp.symbols(f'element_matrix[{i*cols + j}*stride]')
                expr.append(ast.Assignment(var, C[i, j]))

        return expr
    
    def K_matrix(self):
        K = self.integr_K_matrix
        rows, cols = K.shape

        expr = []
        for i in range(0, rows):
            for j in range(0, cols):
                var = sp.symbols(f'element_matrix[{i*cols + j}*stride]')
                expr.append(ast.Assignment(var, K[i, j]))

        return expr
    

    def C_sym(self):
        C = self.integr_C_matrix
        rows, cols = C.shape

        expr = []
        idx = 0
        for i in range(0, rows):
            for j in range(0, cols):
                if j > i:
                    continue
                var = sp.symbols(f'element_matrix[{idx}*stride]')
                expr.append(ast.Assignment(var, C[i, j]))
                idx += 1

        return expr
    
    def K_sym(self):
        K = self.integr_K_matrix
        rows, cols = K.shape

        expr = []
        idx = 0
        for i in range(0, rows):
            for j in range(0, cols):
                if j > i:
                    continue
                var = sp.symbols(f'element_matrix[{idx}*stride]')
                expr.append(ast.Assignment(var, K[i, j]))
                idx += 1

        return expr
    

    def gradient_C(self):
        g = self.integr_damping
        rows, cols = g.shape

        expr = []
        for i in range(0, rows):
            var = sp.symbols(f'element_vector[{i}*stride]')
            expr.append(ast.Assignment(var, g[i]))

        return expr


    def apply_C(self):
        C = self.integr_C_matrix
        rows, cols = C.shape
        v = self.v

        expr = []
        for i in range(0, rows):
            Cv = 0
            for j in range(0, cols):
                Cv += C[i, j] * v[j]

            var = sp.symbols(f'element_vector[{i}*stride]')
            expr.append(ast.Assignment(var, Cv))
        return expr
    

    def gradient_K(self):
        g = self.integr_stiffness
        rows, cols = g.shape

        expr = []
        for i in range(0, rows):
            var = sp.symbols(f'element_vector[{i}*stride]')
            expr.append(ast.Assignment(var, g[i]))

        return expr
    
    def gradient_M(self):
        g = self.integr_inertia
        rows, cols = g.shape

        expr = []
        for i in range(0, rows):
            var = sp.symbols(f'element_vector[{i}*stride]')
            expr.append(ast.Assignment(var, g[i]))

        return expr


    def apply_K(self):
        K = self.integr_K_matrix
        rows, cols = K.shape
        u = self.u

        expr = []
        for i in range(0, rows):
            Ku = 0
            for j in range(0, cols):
                Ku += K[i, j] * u[j]

            var = sp.symbols(f'element_vector[{i}*stride]')
            expr.append(ast.Assignment(var, Ku))

        return expr
    
    def apply_M(self):
        M = self.integr_M_matrix
        rows, cols = M.shape
        a = self.a

        expr = []
        for i in range(0, rows):
            Ma = 0
            for j in range(0, cols):
                Ma += M[i, j] * a[j]

            var = sp.symbols(f'element_vector[{i}*stride]')
            expr.append(ast.Assignment(var, Ma))

        return expr

    def M_matrix(self):
        M = self.integr_M_matrix
        rows, cols = M.shape

        expr = []
        for i in range(0, rows):
            for j in range(0, cols):
                var = sp.symbols(f'element_matrix[{i*cols + j}*stride]')
                expr.append(ast.Assignment(var, M[i, j]))

        return expr
    
    def M_sym(self):
        M = self.integr_M_matrix
        rows, cols = M.shape

        expr = []
        idx = 0
        for i in range(0, rows):
            for j in range(0, cols):
                if j > i:
                    continue
                var = sp.symbols(f'element_matrix[{idx}*stride]')
                expr.append(ast.Assignment(var, M[i, j]))
                idx += 1

        return expr

def main():
	start = perf_counter()

	fe = Hex8()

	fe.use_adjugate = True
	
	op = LinearKVOp(fe)

	c_log("--------------------------")
	c_log("C_matrix")	
	c_log("--------------------------")
	c_code(op.C_matrix())

	c_log("--------------------------")
	c_log("K_matrix")	
	c_log("--------------------------")
	c_code(op.K_matrix())

	c_log("--------------------------")
	c_log("M_matrix")	
	c_log("--------------------------")
	c_code(op.M_matrix())

	c_log("--------------------------")
	c_log("C_sym")	
	c_log("--------------------------")
	c_code(op.C_sym())

	c_log("--------------------------")
	c_log("K_sym")	
	c_log("--------------------------")
	c_code(op.K_sym())

	c_log("--------------------------")
	c_log("M_sym")	
	c_log("--------------------------")
	c_code(op.M_sym())

	c_log("--------------------------")
	c_log("gradient_C")	
	c_log("--------------------------")
	c_code(op.gradient_C())

	c_log("--------------------------")
	c_log("apply_C")	
	c_log("--------------------------")
	c_code(op.apply_C())

	c_log("--------------------------")
	c_log("gradient_K")	
	c_log("--------------------------")
	c_code(op.gradient_K())
      
	c_log("--------------------------")
	c_log("apply_K")	
	c_log("--------------------------")
	c_code(op.apply_K())

	c_log("--------------------------")
	c_log("gradient_M")	
	c_log("--------------------------")
	c_code(op.gradient_M())

	c_log("--------------------------")
	c_log("apply_M")	
	c_log("--------------------------")
	c_code(op.apply_M())

	stop = perf_counter()
	console.print(f'Overall: {stop - start} seconds')


if __name__ == '__main__':
	main()