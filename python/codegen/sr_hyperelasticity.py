#!/usr/bin/env python3

from ctypes import Array, Array
from sfem_codegen import *

from sympy.parsing.sympy_parser import parse_expr

from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *
from hex8 import *
from aahex8 import *

from sympy import Array
import numpy as np

# https://github.com/tzakharko/m4-sme-exploration

# from tpl_hyperelasticity import TPLHyperelasticity

 # INSERT_YOUR_CODE
def detect_constitutive_tensor_symmetries(tensor):
    """
    Detects and prints all index symmetries of a 4th order constitutive tensor.
    Prints which (i,j,k,l) index permutations leave the tensor unchanged.
    """
    import itertools

    dim = tensor.shape[0]
    index_permutations = [
        # Major symmetry: (i,j,k,l) == (k,l,i,j)
        ("major", lambda i,j,k,l: (k,l,i,j)),
        # Minor symmetry 1: (i,j,k,l) == (j,i,k,l)
        ("minor_ij", lambda i,j,k,l: (j,i,k,l)),
        # Minor symmetry 2: (i,j,k,l) == (i,j,l,k)
        ("minor_kl", lambda i,j,k,l: (i,j,l,k)),
        # Full symmetry: (i,j,k,l) == (j,i,l,k)
        ("full", lambda i,j,k,l: (j,i,l,k)),
    ]
    symmetries_found = set()
    for name, perm in index_permutations:
        symmetric = True
        for i, j, k, l in itertools.product(range(dim), repeat=4):
            orig = tensor[i, j, k, l]
            permuted = tensor[perm(i,j,k,l)]
            if not sp.simplify(orig - permuted) == 0:
                symmetric = False
                break
        if symmetric:
            print(f"Constitutive tensor is symmetric under {name} symmetry: (i,j,k,l) <-> {perm(0,1,2,3)}")
            symmetries_found.add(name)
    if not symmetries_found:
        print("No standard symmetries detected in constitutive tensor.")
    else:
        print("Detected symmetries:", symmetries_found)
    # Print all (i,j,k,l) patterns that are symmetric
    # print("Symmetric index patterns:")
    # for name, perm in index_permutations:
    #     for i, j, k, l in itertools.product(range(dim), repeat=4):
    #         if sp.simplify(tensor[i, j, k, l] - tensor[perm(i,j,k,l)]) == 0:
    #             print(f"({i},{j},{k},{l}) == {perm(i,j,k,l)} under {name} symmetry")

def dbg():
    import pdb; pdb.set_trace()

def simplify(expr):
    # return expr
    return sp.simplify(expr)

def create_matrix_symbol(name, rows, cols):
    return matrix_coeff(name, rows, cols)


def create_tensor4_symbol(name, size0, size1, size2, size3):
    terms = []
    for i in range(0, size0):
        for j in range(0, size1):
            for k in range(0, size2):
                for l in range(0, size3):
                    terms.append(sp.symbols(f"{name}[{i*size1*size2*size3 + j*size2*size3 + k*size3 + l}]"))
    return Array(terms, shape=(size0, size1, size2, size3))

def simplify_matrix(mat):
    rows, cols = mat.shape
    return sp.Matrix(rows, cols, [simplify(mat[i, j]) for i in range(0, rows) for j in range(0, cols)])

class SRHyperelasticity:
    @staticmethod
    def create_from_string_F(fe, str_expr: str):
        fun = parse_expr(str_expr) 
        return SRHyperelasticity(fe, fun)

    def __init__(self, fe, fun): 
        self.fe = fe
        self.expression_table = {}
        self.__init_fun_of_F(fun)

    def __init_fun_of_F(self, fun):
        self.__init_symbols(fun)

        F = self.F_symb
        C = F.T * F
       
        I1, I2, J = sp.symbols("I1 I2 J", real=True)
        invariants = [I1, I2, J]
        symbol_names = list(self.fun.free_symbols)
        all_symbols = []
        for s in symbol_names + invariants:
            if str(s) not in [str(sym) for sym in all_symbols]:
                all_symbols.append(s)
                sname = str(s)
                if sname == "I1":
                    I1 = s
                elif sname == "I2":
                    I2 = s
                elif sname == "J":
                    J = s

        self.fun = self.fun.subs({
            I1: sp.trace(C), 
            I2: sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2)), 
            J: sp.det(F)})

    def __init_symbols(self, fun):
        dims = self.fe.manifold_dim()
        self.fun = simplify(fun)

        self.F_symb = self.__create_matrix_symbol("F")
        self.S_lin_symb = self.__create_tensor4_symbol("S_lin")
        self.disp_symb = coeffs_SoA("disp", dims, self.fe.n_nodes())
        self.inc_symb = coeffs_SoA("inc", dims, self.fe.n_nodes())
        self.inc_grad_symb = self.__create_matrix_symbol("inc_grad")
        self.S_ikmn_symb = self.__create_tensor4_symbol("S_ikmn")
        self.SdotH_km_symb = self.__create_matrix_symbol("SdotH_km")
        self.gradx_symb = coeffs("gradx", self.fe.n_nodes())
        self.grady_symb = coeffs("grady", self.fe.n_nodes())
        self.gradz_symb = coeffs("gradz", self.fe.n_nodes())

    def __create_matrix_symbol(self, name):
        dim = self.fe.spatial_dim()
        return create_matrix_symbol(name, dim, dim)

    def __create_zero_matrix(self):
        return sp.zeros(self.fe.spatial_dim(), self.fe.spatial_dim())

    def __create_tensor4_symbol(self, name):
        dim = self.fe.spatial_dim()
        return create_tensor4_symbol(name, dim, dim, dim, dim)

    def __compute_dV(self):
        dV = self.fe.jacobian_determinant(self.fe.quadrature_point()) * (self.fe.reference_measure() *  self.fe.quadrature_weight())
        self.expression_table["dV"] = dV
        return dV

    def __compute_jacobian_adjugate(self):
        Jadj = simplify_matrix(self.fe.jacobian(self.fe.quadrature_point()) / self.fe.jacobian_determinant(self.fe.quadrature_point()))
        self.expression_table["Jadj"] = Jadj
        return Jadj

    def __compute_Jinv(self):
        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()
        self.expression_table["Jinv"] = Jinv
        return Jinv

    def __compute_disp_grad(self):
        dims = self.fe.manifold_dim()
        disp_grad = self.__create_zero_matrix()
        ref_grad = self.fe.tgrad(self.fe.quadrature_point())
        for i in range(0, self.fe.n_nodes() * dims):
            disp_grad += ref_grad[i] * self.disp_symb[i]

        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()

        # Physical Gradient
        self.expression_table["disp_grad"] = disp_grad * Jinv
        return disp_grad

    def __compute_inc_grad(self):
        dims = self.fe.manifold_dim()
        inc_grad = self.__create_zero_matrix()
        ref_grad = self.fe.tgrad(self.fe.quadrature_point())
        for i in range(0, self.fe.n_nodes() * dims):
            inc_grad += ref_grad[i] * self.inc_symb[i]

        # Reference Gradient
        self.expression_table["inc_grad"] = inc_grad
        return inc_grad

    def __compute_F(self):
        dims = self.fe.manifold_dim()
        disp_grad = self.expression_table["disp_grad"]
        F = disp_grad + sp.eye(dims, dims)
        self.expression_table["F"] = F
        return F

    def __compute_piola_stress(self):
        F = self.F_symb
        P = self.__create_zero_matrix()
        for i in range(0, P.shape[0]):
            for j in range(0, P.shape[1]):
                P[i, j] = sp.diff(self.fun, F[i, j])
        P = simplify_matrix(P)
        self.expression_table["P"] = P
        return P

    def __compute_linearized_stress(self):
        P = self.expression_table["P"]
        F = self.F_symb
        dim = self.fe.spatial_dim()
        terms = []
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    for l in range(0, dim):
                        terms.append(sp.diff(P[i, j], F[k, l]))
        S_lin = Array(terms, shape=(dim, dim, dim, dim))
        self.expression_table["S_lin"] = S_lin
        return S_lin
        
    def __compute_metric_tensor(self):
        P = self.__compute_piola_stress()
        F = self.F_symb
        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()
        dim = self.fe.spatial_dim()     
        S_lin = self.S_lin_symb

        terms = []
        for i in range(0, dim):
            for k in range(0, dim):
                for m in range(0, dim):
                    for n in range(0, dim):
                        S_ikmn = 0
                        for j in range(0, dim):
                            reduce_l = 0
                            for l in range(0, dim):
                                reduce_l += S_lin[i, j, k, l] * Jinv[m, l] 
                            S_ikmn += reduce_l * Jinv[n, j]
                        
                        terms.append(S_ikmn)

        dV = self.fe.symbol_jacobian_determinant() * (self.fe.reference_measure() *  self.fe.quadrature_weight())
        for i in range(0, dim**4):
            terms[i] *= dV

        S_ikmn = Array(terms, shape=(dim, dim, dim, dim))
        self.expression_table["S_ikmn"] = S_ikmn
        return S_ikmn  

    def __compute_SdotH_km(self):
        S_ikmn = self.S_ikmn_symb
        inc_grad = self.inc_grad_symb
        dim = self.fe.spatial_dim()
        SdotH_km = self.__create_zero_matrix()
        for k in range(0, dim):
            for m in range(0, dim):
                for i in range(0, dim):
                    for n in range(0, dim):
                        SdotH_km[k, m] += S_ikmn[i, k, m, n] * inc_grad[i, n]
        
        self.expression_table["SdotH_km"] = SdotH_km
        return SdotH_km

    def __compute_apply(self):
        SdotH_km = self.__create_matrix_symbol("SdotH_km")
        nnodes = self.fe.n_nodes()
        eoutx = sp.Matrix(nnodes, 1, [0] * nnodes)
        eouty = sp.Matrix(nnodes, 1, [0] * nnodes)
        eoutz = sp.Matrix(nnodes, 1, [0] * nnodes)
        g = self.fe.grad(self.fe.quadrature_point())

        for node in range(0, nnodes):
            eoutx[node] += SdotH_km[0, 0] * g[node][0] + SdotH_km[0, 1] * g[node][1] + SdotH_km[0, 2] * g[node][2]
            eouty[node] += SdotH_km[1, 0] * g[node][0] + SdotH_km[1, 1] * g[node][1] + SdotH_km[1, 2] * g[node][2]
            eoutz[node] += SdotH_km[2, 0] * g[node][0] + SdotH_km[2, 1] * g[node][1] + SdotH_km[2, 2] * g[node][2]

        self.expression_table["eoutx"] = eoutx
        self.expression_table["eouty"] = eouty
        self.expression_table["eoutz"] = eoutz
        

    def partial_assembly(self):
        self.__compute_dV()
        self.__compute_jacobian_adjugate()
        self.__compute_Jinv()
        self.__compute_disp_grad()
        self.__compute_inc_grad()
        self.__compute_F()
        self.__compute_piola_stress()
        self.__compute_linearized_stress()
        self.__compute_metric_tensor()
        self.__compute_SdotH_km()
        self.__compute_apply()
