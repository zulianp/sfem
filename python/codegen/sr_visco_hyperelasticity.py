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

import canon

# https://github.com/tzakharko/m4-sme-exploration
# https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid


def compress_tensor4(tensor, tensor_name):
    unique_values = {}
    index_map = {}
    compressed_idx = 0
    
    for i in range(nfun):
        for m in range(dim):
            for p in range(nfun):
                for n in range(dim):
                    value = tensor[i, m, p, n]
                    value_str = str(sp.simplify(value))
                    
                    if value_str not in unique_values:
                        unique_values[value_str] = {
                            'value': value,
                            'compressed_idx': compressed_idx,
                            'indices': []
                        }
                        compressed_idx += 1
                    
                    index_map[(i, m, p, n)] = unique_values[value_str]['compressed_idx']
                    unique_values[value_str]['indices'].append((i, m, p, n))
    
    compressed_tensor = []
    for value_str, data in unique_values.items():
        compressed_tensor.append(data['value'])

    compressed_tensor_names = sp.symbols(f"{tensor_name}_compressed[0:{len(unique_values)}]")
    # print(compressed_tensor_names)

    terms = []
    terms_test = []
    for _, v in index_map.items():
        terms.append(compressed_tensor_names[v])
        terms_test.append(v)

    compressed_tensor_symb = Array(terms, shape=(nfun, dim, nfun, dim))
    compressed_test = Array(terms_test, shape=(nfun, dim, nfun, dim))

    for i in range(nfun):
        for m in range(dim):
            for p in range(nfun):
                for n in range(dim):
                    assert tensor[i,m,p,n] == compressed_tensor[compressed_test[i, m, p, n]]

    return compressed_tensor_symb, compressed_tensor_names, compressed_tensor


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
        ("minor_il", lambda i,j,k,l: (l,j,k,i)),
        ("minor_jk", lambda i,j,k,l: (i,k,j,l)),
        # Full symmetry: (i,j,k,l) == (j,i,l,k)
        ("full", lambda i,j,k,l: (j,i,l,k)),
    ]

    names = ['i', 'j', 'k', 'l']
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
            print(f"Constitutive tensor is symmetric under {name} symmetry: (i,j,k,l) <-> {perm(*names)}")
            symmetries_found.add(name)
    if not symmetries_found:
        print("No standard symmetries detected in constitutive tensor.")
    else:
        print("Detected symmetries:", symmetries_found)

def dbg():
    import pdb; pdb.set_trace()

def simplify(expr):
    # return sp.simplify(expr)
    return expr

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

class SRViscoHyperelasticity:
    @staticmethod
    def create_from_string(fe, name, str_expr, num_prony_terms=0, include_geometric_stiffness=True):
        if isinstance(str_expr, list) or isinstance(str_expr, tuple):
            assert len(str_expr) == 2
            fun_vol = parse_expr(str_expr[0])
            fun_dev = parse_expr(str_expr[1])
        else:
            # If only one string is provided, we assume it is the TOTAL energy
            # But for optimal performance, one should provide them separately
            fun_vol = 0
            fun_dev = parse_expr(str_expr)
            
        return SRViscoHyperelasticity(fe, name, fun_vol, fun_dev, unimodular=False, num_prony_terms=num_prony_terms, include_geometric_stiffness=include_geometric_stiffness)

    @staticmethod
    def create_from_string_unimodular(fe, name, str_expr, num_prony_terms=0, include_geometric_stiffness=True):
        if isinstance(str_expr, list) or isinstance(str_expr, tuple):
            assert len(str_expr) == 2
            fun_vol = parse_expr(str_expr[0])
            fun_dev = parse_expr(str_expr[1])
        else:
            fun_vol = 0
            fun_dev = parse_expr(str_expr)

        return SRViscoHyperelasticity(fe, name, fun_vol, fun_dev, unimodular=True, num_prony_terms=num_prony_terms, include_geometric_stiffness=include_geometric_stiffness)

    def __init__(self, fe, name, fun_vol, fun_dev, unimodular=False, num_prony_terms=0, include_geometric_stiffness=True): 
        self.fe = fe
        self.name = name
        self.expression_table = {}
        self.params = []
        self.num_prony_terms = num_prony_terms
        self.include_geometric_stiffness = include_geometric_stiffness

        if unimodular:
            self.__init_fun_unimodular(fun_vol, fun_dev)
        else:
            self.__init_fun(fun_vol, fun_dev)

        self.__init_visco_params()

    def __init_visco_params(self):
        if self.num_prony_terms <= 0:
            return
            
        self.dt_symb = sp.symbols("dt", real=True)
        self.params.append(self.dt_symb)
        
        self.prony_coeffs = []
        for i in range(self.num_prony_terms):
            # g_i, tau_i
            gi = sp.symbols(f"g{i+1}", real=True)
            taui = sp.symbols(f"tau{i+1}", real=True)
            self.prony_coeffs.append((gi, taui))
            self.params.append(gi)
            self.params.append(taui)
        
        # Symbolic gamma for flexible Prony term version
        self.gamma_symb = sp.symbols("gamma", real=True)

    def __init_fun_unimodular(self, fun_vol, fun_dev):
        # Merge for symbol detection
        fun_total = fun_vol + fun_dev
        self.__init_symbols(fun_total)
        
        self.fun_vol_invariants = fun_vol
        self.fun_dev_invariants = fun_dev
        self.fun_invariants = fun_total # For compatibility or total energy if needed

        F = self.F_symb
        det_F = sp.det(F)
        B = F.T * F
        I1 = sp.trace(B)
        I2 = sp.Rational(1, 2) * (sp.trace(B)**2 - sp.trace(B**2))

        I1b, I2b, J = sp.symbols("I1b I2b J", real=True)
        invariants = [I1b, I2b, J]
        symbol_names = list(fun_total.free_symbols)
        all_symbols = []

        for s in symbol_names + invariants:
            if str(s) not in [str(sym) for sym in all_symbols]:
                all_symbols.append(s)
                sname = str(s)
                if sname == "I1b":
                    I1b = s
                elif sname == "I2b":
                    I2b = s
                elif sname == "J":
                    J = s
        
        # Save invariant symbols for later reconstruction
        self.invariants_map = {
            "I1b": I1b,
            "I2b": I2b,
            "J": J,
            "type": "unimodular"
        }

        # We don't substitute into self.fun yet, we keep components
        # But we need self.fun for backward compatibility or total energy
        self.fun = fun_total.subs({
            I1b: det_F**sp.Rational(-2, 3) * I1, 
            I2b: det_F**sp.Rational(-4, 3) * I2, 
            J: det_F})

        reserved_syms = [I1b, I2b, J]
        params = []
        for s in symbol_names:
            if str(s) not in [str(sym) for sym in reserved_syms]:
                params.append(s)
        # print(params)

        self.params = params

    def __init_fun(self, fun_vol, fun_dev):
        # Merge for symbol detection
        fun_total = fun_vol + fun_dev
        self.__init_symbols(fun_total)
        
        self.fun_vol_invariants = fun_vol
        self.fun_dev_invariants = fun_dev
        self.fun_invariants = fun_total

        F = self.F_symb
        C = F.T * F
       
        I1, I2, J = sp.symbols("I1 I2 J", real=True)
        invariants = [I1, I2, J]
        symbol_names = list(fun_total.free_symbols)
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

        # Save invariant symbols for later reconstruction
        self.invariants_map = {
            "I1": I1,
            "I2": I2,
            "J": J,
            "type": "standard"
        }

        self.fun = fun_total.subs({
            I1: sp.trace(C), 
            I2: sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2)), 
            J: sp.det(F)})

        reserved_syms = [I1, I2, J]
        params = []
        for s in symbol_names:
            if str(s) not in [str(sym) for sym in reserved_syms]:
                params.append(s)
        self.params = params

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
        self.S_ikmn_canonical_symb = self.__create_tensor4_symbol_canonical("S_ikmn_canonical")
        self.S_ikmn_packed_symb = self.__create_tensor4_symbol_packed("S_ikmn_packed")
    
    def __create_tensor4_symbol_canonical(self, name):
        dim = self.fe.spatial_dim()
        N = (dim**2+1)*(dim**2)/2
        can = [sp.symbols(f"{name}[{i}]") for i in range(int(N))]
        can_map, _, _ = canon.build_canonical_map(dim)
        canon_reconstruct = canon.reconstruct_full(can, can_map, dim, as_sympy=True)
        return canon_reconstruct

    def __create_tensor4_symbol_packed(self, name):
        dim = self.fe.spatial_dim()
        N = (dim**2+1)*(dim**2)/2
        packed = [sp.symbols(f"{name}[{i}]") for i in range(int(N))]
        return packed

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
        if "disp_grad" in self.expression_table:
            return self.expression_table["disp_grad"]

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
        if "inc_grad" in self.expression_table:
            return self.expression_table["inc_grad"]

        dims = self.fe.manifold_dim()
        inc_grad = self.__create_zero_matrix()
        ref_grad = self.fe.tgrad(self.fe.quadrature_point())
        for i in range(0, self.fe.n_nodes() * dims):
            inc_grad += ref_grad[i] * self.inc_symb[i]

        # Reference Gradient
        self.expression_table["inc_grad"] = inc_grad
        return inc_grad
        

    def __compute_F(self):
        if "F" in self.expression_table:
            return self.expression_table["F"]

        dims = self.fe.manifold_dim()
        disp_grad = self.expression_table["disp_grad"]
        F = disp_grad + sp.eye(dims, dims)
        self.expression_table["F"] = F
        return F

    def __sub_matrix(self, expr, mat_sym, mat_val):
        rows, cols = mat_sym.shape
        for r in range(rows):
            for c in range(cols):
                expr = expr.subs(mat_sym[r, c], mat_val[r, c])
        return expr

    def __compute_piola_stress(self):
        if "P" in self.expression_table:
            return self.expression_table["P"]

        # 1. Compute S_elastic symbolically (2nd PK Stress)
        # We define S_elastic = 2 * dW/dC
        
        C = self.__create_matrix_symbol("C") 
        
        # Reconstruct W in terms of C
        W_C = self.fun_invariants
        
        if self.invariants_map["type"] == "standard":
            I1 = sp.trace(C)
            I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
            # J = sqrt(det(C)). J>0
            J = sp.sqrt(sp.det(C))
            
            W_C = W_C.subs({
                self.invariants_map["I1"]: I1,
                self.invariants_map["I2"]: I2,
                self.invariants_map["J"]: J
            })
            
        elif self.invariants_map["type"] == "unimodular":
            J = sp.sqrt(sp.det(C))
            I1 = sp.trace(C)
            I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
            
            I1b = J**sp.Rational(-2, 3) * I1
            I2b = J**sp.Rational(-4, 3) * I2
            
            W_C = W_C.subs({
                self.invariants_map["I1b"]: I1b,
                self.invariants_map["I2b"]: I2b,
                self.invariants_map["J"]: J
            })
        
        # S = 2 * dW/dC
        def get_W_C(expr_invariants):
            if expr_invariants == 0:
                return 0
            
            if self.invariants_map["type"] == "standard":
                I1 = sp.trace(C)
                I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
                J = sp.sqrt(sp.det(C))
                
                return expr_invariants.subs({
                    self.invariants_map["I1"]: I1,
                    self.invariants_map["I2"]: I2,
                    self.invariants_map["J"]: J
                })
                
            elif self.invariants_map["type"] == "unimodular":
                J = sp.sqrt(sp.det(C))
                I1 = sp.trace(C)
                I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
                
                I1b = J**sp.Rational(-2, 3) * I1
                I2b = J**sp.Rational(-4, 3) * I2
                
                return expr_invariants.subs({
                    self.invariants_map["I1b"]: I1b,
                    self.invariants_map["I2b"]: I2b,
                    self.invariants_map["J"]: J
                })
            return 0

        W_vol_C = get_W_C(self.fun_vol_invariants)
        W_dev_C = get_W_C(self.fun_dev_invariants)
        
        # Compute stresses S = 2 * dW/dC
        def compute_S(W_expr):
            if W_expr == 0: return sp.zeros(3, 3)
            S = sp.zeros(3, 3)
            for i in range(3):
                for j in range(3):
                    S[i, j] = 2 * sp.diff(W_expr, C[i, j])
            return S

        S_vol_C = compute_S(W_vol_C)
        S_dev_C = compute_S(W_dev_C)

        # Substitute C -> F.T*F
        F = self.F_symb
        C_expr = F.T * F
        
        def sub_C(S_matrix):
            S_out = sp.zeros(3, 3)
            for i in range(3):
                for j in range(3):
                    S_out[i, j] = self.__sub_matrix(S_matrix[i, j], C, C_expr)
            return simplify_matrix(S_out)

        S_vol = sub_C(S_vol_C)
        S_dev = sub_C(S_dev_C)

        if self.num_prony_terms == 0:
            # Pure elastic
            S_total = S_vol + S_dev
            P = F * S_total
            P = simplify_matrix(P)
            self.expression_table["P"] = P
            return P

        # 2. Viscoelasticity
        dt = self.dt_symb
        
        # Calculate algorithmic modulus coefficient gamma
        # gamma = g_inf + sum(beta_i)
        sum_gi = sum(g for g, tau in self.prony_coeffs)
        g_inf = 1 - sum_gi
        
        gamma = g_inf
        betas = []
        alphas = []
        
        for i in range(self.num_prony_terms):
            gi, taui = self.prony_coeffs[i]
            alpha = sp.exp(-dt / taui)
            x = dt / taui
            # beta = gi * (1 - exp(-x)) / x
            beta = gi * (1 - alpha) / x
            
            alphas.append(alpha)
            betas.append(beta)
            gamma += beta
            
        # Construct Algorithmic Energy W_algo
        # W_algo = W_vol + gamma * W_dev
        # This is much cleaner for differentiation
        W_algo_C = W_vol_C + gamma * W_dev_C
        
        # S_algo = 2 * d(W_algo)/dC
        S_algo_C = compute_S(W_algo_C)
        S_algo = sub_C(S_algo_C)
        
        # Calculate History Stress S_hist
        # S_hist = sum( alpha_i * H_i^n - beta_i * S_dev^n )
        
        idx_map = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
        hist_ptr = 0
        
        # Read S_dev_n
        S_dev_n = sp.zeros(3, 3)
        for r, c in idx_map:
            val = sp.symbols(f"history[{hist_ptr}]", real=True)
            S_dev_n[r, c] = val
            S_dev_n[c, r] = val
            hist_ptr += 1
            
        S_hist = sp.zeros(3, 3)
        
        for i in range(self.num_prony_terms):
            alpha = alphas[i]
            beta = betas[i]
            
            H_old = sp.zeros(3, 3)
            for r, c in idx_map:
                val = sp.symbols(f"history[{hist_ptr}]", real=True)
                H_old[r, c] = val
                H_old[c, r] = val
                hist_ptr += 1
            
            # Contribution to S_hist
            S_hist += alpha * H_old - beta * S_dev_n
            
        # Total Stress
        S_total = S_algo + S_hist
        
        # Compute P_total
        P = F * S_total
        P = simplify_matrix(P)
        
        # Also store P_algo (elastic only, without S_hist) for loop-based hessian
        P_algo = F * S_algo
        P_algo = simplify_matrix(P_algo)
        
        self.expression_table["P"] = P
        self.expression_table["P_algo"] = P_algo  # For elastic-only hessian
        self.expression_table["S"] = S_total
        self.expression_table["S_algo"] = S_algo  # Algorithmic stress (without history)
        
        return P

    def __compute_piola_stress_flexible(self):
        """Compute Piola stress using symbolic gamma (for flexible Prony terms).
        
        This version uses a single symbol 'gamma' instead of expanding all Prony terms.
        The gamma value should be computed at runtime:
            gamma = g_inf + sum(beta_i) where beta_i = g_i * (1 - exp(-dt/tau_i)) / (dt/tau_i)
        
        This allows the generated hessian to work with ANY number of Prony terms.
        """
        if "P_flexible" in self.expression_table:
            return self.expression_table["P_flexible"]

        F = self.F_symb
        
        # Use SYMBOLIC C matrix for differentiation, then substitute C -> F.T * F
        C = self.__create_matrix_symbol("C")
        
        inv_map = self.invariants_map
        inv_type = inv_map.get("type", "standard")
        
        # Build W in terms of symbolic C
        def get_W_C(expr_invariants):
            if expr_invariants == 0:
                return 0
            
            if inv_type == "standard":
                I1 = sp.trace(C)
                I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
                J = sp.sqrt(sp.det(C))
                
                return expr_invariants.subs({
                    inv_map["I1"]: I1,
                    inv_map["I2"]: I2,
                    inv_map["J"]: J
                })
                
            elif inv_type == "unimodular":
                J = sp.sqrt(sp.det(C))
                I1 = sp.trace(C)
                I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
                
                I1b = J**sp.Rational(-2, 3) * I1
                I2b = J**sp.Rational(-4, 3) * I2
                
                return expr_invariants.subs({
                    inv_map["I1b"]: I1b,
                    inv_map["I2b"]: I2b,
                    inv_map["J"]: J
                })
            return 0
        
        W_vol_C = get_W_C(self.fun_vol_invariants)
        W_dev_C = get_W_C(self.fun_dev_invariants)
        
        # Compute S = 2 * dW/dC in terms of symbolic C
        def compute_S(W_expr):
            if W_expr == 0:
                return sp.zeros(3, 3)
            S = sp.zeros(3, 3)
            for i in range(3):
                for j in range(3):
                    S[i, j] = 2 * sp.diff(W_expr, C[i, j])
            return S
        
        S_vol_C = compute_S(W_vol_C)
        S_dev_C = compute_S(W_dev_C)
        
        # Substitute C -> F.T * F
        C_expr = F.T * F
        
        def sub_C(S_matrix):
            S_out = sp.zeros(3, 3)
            for i in range(3):
                for j in range(3):
                    S_out[i, j] = self.__sub_matrix(S_matrix[i, j], C, C_expr)
            return simplify_matrix(S_out)
        
        S_vol = sub_C(S_vol_C)
        S_dev = sub_C(S_dev_C)
        
        # Use symbolic gamma instead of expanding Prony series!
        gamma = self.gamma_symb
        
        # S_algo = S_vol + gamma * S_dev
        S_algo_flexible = S_vol + gamma * S_dev
        S_algo_flexible = simplify_matrix(S_algo_flexible)
        
        # P = F * S (without history, history added at runtime via loops)
        P_flexible = F * S_algo_flexible
        P_flexible = simplify_matrix(P_flexible)
        
        self.expression_table["P_flexible"] = P_flexible
        self.expression_table["S_algo_flexible"] = S_algo_flexible
        self.expression_table["S_vol"] = S_vol
        self.expression_table["S_dev"] = S_dev
        
        return P_flexible

    def __compute_linearized_stress_flexible(self):
        """Compute linearized stress using symbolic gamma.
        
        dP_flexible/dF where P_flexible = F * (S_vol + gamma * S_dev)
        gamma is a symbol, not expanded.
        """
        if "S_lin_flexible" in self.expression_table:
            return self.expression_table["S_lin_flexible"]

        P_flexible = self.__compute_piola_stress_flexible()
        F = self.F_symb
        dim = self.fe.spatial_dim()
        
        terms = []
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    for l in range(0, dim):
                        terms.append(sp.diff(P_flexible[i, j], F[k, l]))
        S_lin_flexible = Array(terms, shape=(dim, dim, dim, dim))
        self.expression_table["S_lin_flexible"] = S_lin_flexible
        return S_lin_flexible

    def __compute_metric_tensor_flexible(self):
        """Compute metric tensor using symbolic gamma."""
        if "S_ikmn_flexible" in self.expression_table:
            return self.expression_table["S_ikmn_flexible"]

        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()
        dim = self.fe.spatial_dim()
        S_lin = self.__compute_linearized_stress_flexible()

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

        dV = self.fe.symbol_jacobian_determinant() * (self.fe.reference_measure() * self.fe.quadrature_weight())
        for i in range(0, dim**4):
            terms[i] *= dV

        S_ikmn_flexible = Array(terms, shape=(dim, dim, dim, dim))
        self.expression_table["S_ikmn_flexible"] = S_ikmn_flexible
        return S_ikmn_flexible

    def __compute_hessian_flexible(self):
        """Compute hessian using symbolic gamma."""
        if "hessian_flexible" in self.expression_table:
            return self.expression_table["hessian_flexible"]

        S_ikmn_flexible = self.__compute_metric_tensor_flexible()
        refgrad = self.fe.tgrad(self.fe.quadrature_point())
        
        dim = self.fe.spatial_dim()
        nfun = self.fe.n_nodes()
        H_flexible = sp.zeros(dim*nfun, dim*nfun)

        for test in range(0, nfun * dim):
            for trial in range(0, nfun * dim):
                for k in range(0, dim): 
                    for m in range(0, dim):
                        for i in range(0, dim):
                            for n in range(0, dim):
                                H_flexible[test, trial] += S_ikmn_flexible[i, k, m, n] * refgrad[trial][i, n] * refgrad[test][k, m]

        self.expression_table["hessian_flexible"] = H_flexible
        return H_flexible

    def __compute_linearized_stress(self):
        # TODO: This tensor has symmetries and it can be compressed
        # S_lin_symb, S_lin_names, S_lin_vals = compress_tensor4("S_lin", S_lin)
        if "S_lin" in self.expression_table:
            return self.expression_table["S_lin"]

        P = self.expression_table["P"]
        F = self.F_symb
        dim = self.fe.spatial_dim()
        
        # Automatic differentiation
        # SymPy includes ALL dependencies on F, including geometric stiffness from history:
        # d(F * S_history)/dF = I x S_history (approx)
        
        # If we want to exclude it (inconsistent tangent), we must manually construct P without history
        # But for now, we trust the consistent tangent provided by SymPy
        
        terms = []
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    for l in range(0, dim):
                        terms.append(sp.diff(P[i, j], F[k, l]))
        S_lin = Array(terms, shape=(dim, dim, dim, dim))
        self.expression_table["S_lin"] = S_lin
        return S_lin

    def __compute_linearized_stress_algo(self):
        """Compute linearized stress for algorithmic part only (without S_hist).
        
        This allows loop-based computation of history contributions at runtime,
        enabling flexible number of Prony terms.
        
        S_lin_algo = dP_algo/dF where P_algo = F * S_algo
        """
        if "S_lin_algo" in self.expression_table:
            return self.expression_table["S_lin_algo"]

        if "P_algo" not in self.expression_table:
            # For pure elastic case, P_algo = P
            P_algo = self.expression_table.get("P")
        else:
            P_algo = self.expression_table["P_algo"]
            
        F = self.F_symb
        dim = self.fe.spatial_dim()
        
        terms = []
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    for l in range(0, dim):
                        terms.append(sp.diff(P_algo[i, j], F[k, l]))
        S_lin_algo = Array(terms, shape=(dim, dim, dim, dim))
        self.expression_table["S_lin_algo"] = S_lin_algo
        return S_lin_algo
        
    def __compute_metric_tensor(self):
        if "S_ikmn" in self.expression_table:
            return self.expression_table["S_ikmn"]

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
        if "SdotH_km" in self.expression_table:
            return self.expression_table["SdotH_km"] 

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

    def __compute_metric_tensor_canonical(self):
        if "S_ikmn_canonical" in self.expression_table:
            return self.expression_table["S_ikmn_canonical"]

        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()
        dim = self.fe.spatial_dim()     
        S_lin = self.expression_table["S_lin"]

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

        dim = self.fe.spatial_dim()
        S_ikmn_canonical, canon_list = canon.pack_tensor(S_ikmn, dim)
        self.expression_table["S_ikmn_canonical"] = S_ikmn_canonical
        return S_ikmn_canonical

    def __compute_SdotH_km_canonical(self):
        if "SdotH_km_canonical" in self.expression_table:
            return self.expression_table["SdotH_km_canonical"]

        S_ikmn_canonical = self.S_ikmn_canonical_symb
        dim = self.fe.spatial_dim()
        inc_grad = self.inc_grad_symb
        dim = self.fe.spatial_dim()
        SdotH_km_canonical = self.__create_zero_matrix()
        for k in range(0, dim):
            for m in range(0, dim):
                for i in range(0, dim):
                    for n in range(0, dim):
                        SdotH_km_canonical[k, m] += S_ikmn_canonical[i, k, m, n] * inc_grad[i, n]
        
        self.expression_table["SdotH_km_canonical"] = SdotH_km_canonical
        return SdotH_km_canonical

    def __compute_apply(self):
        if "eoutx" in self.expression_table:
            return

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

    def __compute_apply_canonical(self):
        SdotH_km_canonical = self.expression_table["SdotH_km_canonical"]
        nnodes = self.fe.n_nodes()
        eoutx = sp.Matrix(nnodes, 1, [0] * nnodes)
        eouty = sp.Matrix(nnodes, 1, [0] * nnodes)
        eoutz = sp.Matrix(nnodes, 1, [0] * nnodes)
        g = self.fe.grad(self.fe.quadrature_point())

        for node in range(0, nnodes):
            eoutx[node] += SdotH_km_canonical[0, 0] * g[node][0] + SdotH_km_canonical[0, 1] * g[node][1] + SdotH_km_canonical[0, 2] * g[node][2]
            eouty[node] += SdotH_km_canonical[1, 0] * g[node][0] + SdotH_km_canonical[1, 1] * g[node][1] + SdotH_km_canonical[1, 2] * g[node][2]
            eoutz[node] += SdotH_km_canonical[2, 0] * g[node][0] + SdotH_km_canonical[2, 1] * g[node][1] + SdotH_km_canonical[2, 2] * g[node][2]

        self.expression_table["eoutx"] = eoutx
        self.expression_table["eouty"] = eouty
        self.expression_table["eoutz"] = eoutz
        return SdotH_km_canonical

    def __compute_constant_grad_tp(self):
        if "Wimpn" in self.expression_table:
            return self.expression_table["Wimpn"]

        dims = self.fe.spatial_dim()
        nfun = self.fe.n_nodes()
        g = self.fe.grad(self.fe.quadrature_point())

        terms = []
        for i in range(0, nfun):
            for m in range(0, dims):
                for p in range(0, nfun):
                    for n in range(0, dims):
                        expr = g[i][m] * g[p][n] / self.fe.reference_measure()
                        integr = self.fe.integrate(self.fe.quadrature_point(), expr)
                        terms.append(integr)

        Wimpn = Array(terms, shape=(nfun, dims, nfun, dims))
        self.expression_table["Wimpn"] = Wimpn
        return Wimpn

    def __compute_loperand(self):
        if "loperand" in self.expression_table:
            return self.expression_table["loperand"] 

        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()
        P = self.expression_table["P"]
        dV = self.fe.symbol_jacobian_determinant() * (self.fe.reference_measure() *  self.fe.quadrature_weight())

        loperand = P * Jinv.T * dV
        self.expression_table["loperand"] = loperand
        return loperand

    def __params_to_args(self):
        params = self.params
        lines = []

        for p in params:
            lines.append(f'    const {real_t}                      {str(p)},\n')

        return "".join(lines)

    def __history_to_args(self):
        if self.num_prony_terms <= 0:
            return ""
        return f'    const {real_t} *const SFEM_RESTRICT history,\n'

    def emit_objective(self):
        self.__compute_dV()
        self.__compute_Jinv()
        self.__compute_disp_grad()
        self.__compute_F()
        
        dV = self.fe.symbol_jacobian_determinant() * (self.fe.reference_measure() *  self.fe.quadrature_weight())
        fun = self.fun * dV


        fe = self.fe
        dim = fe.spatial_dim()

        sig_objective = (
            f'static SFEM_INLINE void {fe.name().lower()}_{self.name}_objective(\n'
            f'    const {real_t} *const SFEM_RESTRICT adjugate,\n'
            f'    const {real_t}                      jacobian_determinant,\n'
            f'    const {real_t}                      qx,\n'
            f'    const {real_t}                      qy,\n'
            f'    const {real_t}                      qz,\n'
            f'    const {real_t}                      qw,\n'
            f'{self.__params_to_args()}'
            f'{self.__history_to_args()}'
            f'    const {real_t} *const SFEM_RESTRICT dispx,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispy,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispz,\n'
            f'    {real_t} *const SFEM_RESTRICT       v)'
        )

    
        body_objective = (
            f'{{\n'
            f'{real_t} F[{dim*dim}];\n'
            f'{fe.name().lower()}_F(adjugate, jacobian_determinant, qx, qy, qz, dispx, dispy, dispz, F);\n'
            f'{c_gen(ast.AddAugmentedAssignment(sp.symbols("v[0]"), fun))}\n'
            f'}}\n'
        )

        print(sig_objective + body_objective)
        
        
    def emit_gradient(self):
        self.__compute_dV()
        self.__compute_jacobian_adjugate()
        self.__compute_Jinv()
        self.__compute_disp_grad()
        self.__compute_F()
        self.__compute_piola_stress()
        self.__compute_loperand()

        fe = self.fe

        grad_S = (
            f'static SFEM_INLINE void {fe.name().lower()}_{self.name}_grad(\n'
            f'    const {real_t} *const SFEM_RESTRICT adjugate,\n'
            f'    const {real_t}                      jacobian_determinant,\n'
            f'    const {real_t}                      qx,\n'
            f'    const {real_t}                      qy,\n'
            f'    const {real_t}                      qz,\n'
            f'    const {real_t}                      qw,\n'
            f'{self.__params_to_args()}'
            f'{self.__history_to_args()}'
            f'    const {real_t} *const SFEM_RESTRICT dispx,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispy,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispz,\n'
            f'    {real_t} *const SFEM_RESTRICT       gx,\n'
            f'    {real_t} *const SFEM_RESTRICT       gy,\n'
            f'    {real_t} *const SFEM_RESTRICT       gz)'
            f'\n'
        )
        
        loperand = self.expression_table["loperand"]

        refgrad = self.fe.tgrad(self.fe.quadrature_point())
        grads = []

        for d in range(0, fe.spatial_dim()):
            gradd = []
            for i in range(0, fe.n_nodes()):
                gradd.append(inner(loperand, refgrad[d * fe.n_nodes() + i]))
            grads.append(gradd)
        
        expr = []
        syms = ["x", "y", "z"]
        for d in range(0, fe.spatial_dim()):
            for i in range(0, fe.n_nodes()):
                expr.append(ast.AddAugmentedAssignment(sp.symbols(f"g{syms[d]}[{i}]"), grads[d][i]))

        F_actual = c_gen(assign_matrix("F", self.expression_table["F"]))

        grad_body = (
            f'{{\n'
            f'{real_t} F[{fe.spatial_dim()*fe.spatial_dim()}];\n'
            f'{{'
            f'{F_actual}\n'
            f'}}\n'
            f'{c_gen(expr)}'
            f'\n}}'
        )

        print(grad_S + grad_body)


    def __subs_tensor4(self, expr, syms, vals):
        s0, s1, s2, s3 = syms.shape

        assert s0 == vals.shape[0]
        assert s1 == vals.shape[1]
        assert s2 == vals.shape[2]
        assert s3 == vals.shape[3]

        for i0 in range(0, s0):
            for i1 in range(0, s1):
                for i2 in range(0, s2):
                    for i3 in range(0, s3):
                        expr = expr.subs(syms[i0, i1, i2, i3], vals[i0, i1, i2, i3])
        return expr

    def __compute_hessian(self):
        # Lazy
        if "hessian" in self.expression_table:
            return self.expression_table["hessian"]

        S_ikmn = self.expression_table["S_ikmn"]
        refgrad = self.fe.tgrad(self.fe.quadrature_point())
        
        dim = self.fe.spatial_dim()
        nfun = self.fe.n_nodes()
        H = sp.zeros(dim*nfun, dim*nfun)

        for test in range(0, nfun * dim):
            for trial in range(0, nfun * dim):
                for k in range(0, dim): 
                    for m in range(0, dim):
                        for i in range(0, dim):
                            for n in range(0, dim):
                                 H[test, trial] += S_ikmn[i, k, m, n] * refgrad[trial][i, n] * refgrad[test][k, m]
        
        H_diag = sp.zeros(dim*nfun, 1)
        for test in range(0, nfun * dim):
            H_diag[test] = H[test, test]

        self.expression_table["hessian"] = H
        self.expression_table["hessian_diag"] = H_diag
        return H

    def __assign_tensor4(self, name, tensor):
        s0, s1, s2, s3 = tensor.shape

        expr = []
        idx = 0
        for i0 in range(0, s0):
            for i1 in range(0, s1):
                for i2 in range(0, s2):
                    for i3 in range(0, s3):
                        var = sp.symbols(f'{name}[{idx}]')
                        ass = ast.Assignment(var, tensor[i0, i1, i2, i3])
                        expr.append(ass)
                        idx += 1
        return expr


    def emit_history_update(self):
        if self.num_prony_terms == 0:
            print("// No history update needed for pure elasticity")
            return

        # Prepare necessary quantities
        self.__compute_dV()
        self.__compute_disp_grad()
        self.__compute_F()
        
        # Re-compute Elastic Deviatoric Stress S_dev (Pure Elastic)
        # We need to reconstruct the calculation locally or extract it
        # But since __compute_piola_stress is complex, we just re-calculate S_dev here
        
        C = self.__create_matrix_symbol("C") 
        
        def get_W_C(expr_invariants):
            if expr_invariants == 0: return 0
            
            if self.invariants_map["type"] == "standard":
                I1 = sp.trace(C)
                I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
                J = sp.sqrt(sp.det(C))
                return expr_invariants.subs({
                    self.invariants_map["I1"]: I1,
                    self.invariants_map["I2"]: I2,
                    self.invariants_map["J"]: J
                })
            elif self.invariants_map["type"] == "unimodular":
                J = sp.sqrt(sp.det(C))
                I1 = sp.trace(C)
                I2 = sp.Rational(1, 2) * (sp.trace(C)**2 - sp.trace(C**2))
                I1b = J**sp.Rational(-2, 3) * I1
                I2b = J**sp.Rational(-4, 3) * I2
                return expr_invariants.subs({
                    self.invariants_map["I1b"]: I1b,
                    self.invariants_map["I2b"]: I2b,
                    self.invariants_map["J"]: J
                })
            return 0

        W_dev_C = get_W_C(self.fun_dev_invariants)
        
        # S_dev_C = 2 * d(W_dev)/dC
        S_dev_C = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                S_dev_C[i, j] = 2 * sp.diff(W_dev_C, C[i, j])
        
        # Substitute C -> F.T*F
        F = self.F_symb
        C_expr = F.T * F
        
        S_dev_next = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                S_dev_next[i, j] = self.__sub_matrix(S_dev_C[i, j], C, C_expr)
        
        # We don't need to simplify fully here, CSE will handle it
        
        # Parameters
        dt = self.dt_symb
        
        # Update Logic
        # We need to read old values, compute new values, and assign them back
        # Layout: [S_dev_n (6), H_1_n (6), H_2_n (6), ...]
        
        idx_map = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)] # Voigt notation for symmetry
        
        hist_ptr_read = 0
        hist_ptr_write = 0
        
        assignments = []
        
        # 1. Read S_dev_n (old elastic stress)
        S_dev_n = sp.zeros(3, 3)
        for r, c in idx_map:
            # history is input pointer (const)
            val = sp.symbols(f"history[{hist_ptr_read}]", real=True)
            S_dev_n[r, c] = val
            S_dev_n[c, r] = val
            hist_ptr_read += 1
            
        # 2. Write S_dev_next (current elastic stress becomes new history)
        # new_history is output pointer (mutable)
        for r, c in idx_map:
            # Store S_dev_next into new_history[0...5]
            lhs = sp.symbols(f"new_history[{hist_ptr_write}]")
            rhs = S_dev_next[r, c]
            assignments.append(ast.Assignment(lhs, rhs))
            hist_ptr_write += 1
            
        # 3. Update and Write H_i
        for i in range(self.num_prony_terms):
            gi, taui = self.prony_coeffs[i]
            
            # Coefficients
            x = dt / taui
            alpha = sp.exp(-x)
            beta = gi * (1 - alpha) / x
            
            for r, c in idx_map:
                # Read H_old
                h_old = sp.symbols(f"history[{hist_ptr_read}]", real=True)
                hist_ptr_read += 1
                
                # Calculate H_new
                # H_new = alpha * H_old + beta * (S_dev_next - S_dev_n)
                h_new_val = alpha * h_old + beta * (S_dev_next[r, c] - S_dev_n[r, c])
                
                # Write H_new
                lhs = sp.symbols(f"new_history[{hist_ptr_write}]")
                assignments.append(ast.Assignment(lhs, h_new_val))
                hist_ptr_write += 1

        # Generate C Function
        fe = self.fe
        dim = fe.spatial_dim()
        
        signature = (
            f'static SFEM_INLINE void {fe.name().lower()}_{self.name}_update_history(\n'
            f'    const {real_t}                      jacobian_determinant,\n'
            f'    const {real_t}                      qx,\n'
            f'    const {real_t}                      qy,\n'
            f'    const {real_t}                      qz,\n'
            f'    const {real_t}                      qw,\n'
            f'{self.__params_to_args()}'
            f'    const {real_t} *const SFEM_RESTRICT history,\n'     # Read-only old history
            f'    {real_t} *const SFEM_RESTRICT       new_history,\n' # Write-only new history
            f'    const {real_t} *const SFEM_RESTRICT dispx,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispy,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispz)'
            f'\n'
        )
        
        F_actual = c_gen(assign_matrix("F", self.expression_table["F"]))
        update_code = c_gen(assignments)
        
        body = (
            f'{{\n'
            f'{real_t} F[{dim**2}];\n'
            f'{{\n'
            f'{F_actual}'
            f'}}\n\n'
            f'{update_code}\n'
            f'}}\n'
        )
        
        print(signature + body)

    def emit_hessian(self):
        self.__compute_dV()
        self.__compute_jacobian_adjugate()
        self.__compute_Jinv()
        self.__compute_disp_grad()
        self.__compute_F()
        self.__compute_piola_stress()
        self.__compute_linearized_stress()
        self.__compute_metric_tensor()
        self.__compute_hessian()

        H = self.expression_table["hessian"]
        
        fe = self.fe
        dim = fe.spatial_dim()

        signature = (
            f'static SFEM_INLINE void {fe.name().lower()}_{self.name}_hessian(\n'
            f'    const {real_t} *const SFEM_RESTRICT adjugate,\n'
            f'    const {real_t}                      jacobian_determinant,\n'
            f'    const {real_t}                      qx,\n'
            f'    const {real_t}                      qy,\n'
            f'    const {real_t}                      qz,\n'
            f'    const {real_t}                      qw,\n'
            f'{self.__params_to_args()}'
            f'{self.__history_to_args()}'
            f'    const {real_t} *const SFEM_RESTRICT dispx,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispy,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispz,\n'
            f'    {real_t} *const SFEM_RESTRICT       H)'
            f'\n'
        )

        F_actual = c_gen(assign_matrix("F", self.expression_table["F"]))
        S_actual = c_gen(self.__assign_tensor4("S_lin", self.expression_table["S_lin"]))
        combined_code = c_gen(add_assign_matrix("H", H))

        body = (
            f'{{\n'
            f'{real_t} F[{dim**2}];\n'
            f'{{\n'
            f'{F_actual}'
            f'}}\n\n'
            f'{real_t} S_lin[{dim**4}];\n'
            f'{{\n'
            f'{S_actual}\n'
            f'}}\n'
            f'{combined_code}\n'
            f'}}\n'
        )

        print(signature + body)

    def __compute_metric_tensor_algo(self):
        """Compute metric tensor for algorithmic part only."""
        if "S_ikmn_algo" in self.expression_table:
            return self.expression_table["S_ikmn_algo"]

        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()
        dim = self.fe.spatial_dim()     
        S_lin_algo = self.expression_table["S_lin_algo"]

        terms = []
        for i in range(0, dim):
            for k in range(0, dim):
                for m in range(0, dim):
                    for n in range(0, dim):
                        S_ikmn = 0
                        for j in range(0, dim):
                            reduce_l = 0
                            for l in range(0, dim):
                                reduce_l += S_lin_algo[i, j, k, l] * Jinv[m, l] 
                            S_ikmn += reduce_l * Jinv[n, j]
                        
                        terms.append(S_ikmn)

        dV = self.fe.symbol_jacobian_determinant() * (self.fe.reference_measure() *  self.fe.quadrature_weight())
        for i in range(0, dim**4):
            terms[i] *= dV

        S_ikmn_algo = Array(terms, shape=(dim, dim, dim, dim))
        self.expression_table["S_ikmn_algo"] = S_ikmn_algo
        return S_ikmn_algo

    def __compute_hessian_algo(self):
        """Compute element hessian for algorithmic part only."""
        if "hessian_algo" in self.expression_table:
            return self.expression_table["hessian_algo"]

        S_ikmn_algo = self.expression_table["S_ikmn_algo"]
        refgrad = self.fe.tgrad(self.fe.quadrature_point())
        
        dim = self.fe.spatial_dim()
        nfun = self.fe.n_nodes()
        H_algo = sp.zeros(dim*nfun, dim*nfun)

        for test in range(0, nfun * dim):
            for trial in range(0, nfun * dim):
                for k in range(0, dim): 
                    for m in range(0, dim):
                        for i in range(0, dim):
                            for n in range(0, dim):
                                 H_algo[test, trial] += S_ikmn_algo[i, k, m, n] * refgrad[trial][i, n] * refgrad[test][k, m]
        
        self.expression_table["hessian_algo"] = H_algo
        return H_algo

    def emit_hessian_algo(self):
        """Emit hessian for algorithmic part only (elastic + gamma*deviatoric, no S_hist).
        
        This version does NOT include history contributions in the generated code.
        History contributions (geometric stiffness I x S_hist) should be added
        at runtime using loops, enabling flexible number of Prony terms.
        """
        self.__compute_dV()
        self.__compute_jacobian_adjugate()
        self.__compute_Jinv()
        self.__compute_disp_grad()
        self.__compute_F()
        self.__compute_piola_stress()
        self.__compute_linearized_stress_algo()
        self.__compute_metric_tensor_algo()
        self.__compute_hessian_algo()

        H_algo = self.expression_table["hessian_algo"]
        
        fe = self.fe
        dim = fe.spatial_dim()

        # Note: this version does NOT take history parameter - it's handled at runtime
        signature = (
            f'static SFEM_INLINE void {fe.name().lower()}_{self.name}_hessian_algo(\n'
            f'    const {real_t} *const SFEM_RESTRICT adjugate,\n'
            f'    const {real_t}                      jacobian_determinant,\n'
            f'    const {real_t}                      qx,\n'
            f'    const {real_t}                      qy,\n'
            f'    const {real_t}                      qz,\n'
            f'    const {real_t}                      qw,\n'
            f'{self.__params_to_args()}'
            f'    const {real_t} *const SFEM_RESTRICT dispx,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispy,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispz,\n'
            f'    {real_t} *const SFEM_RESTRICT       H)'
            f'\n'
        )

        F_actual = c_gen(assign_matrix("F", self.expression_table["F"]))
        S_actual = c_gen(self.__assign_tensor4("S_lin", self.expression_table["S_lin_algo"]))
        combined_code = c_gen(add_assign_matrix("H", H_algo))

        body = (
            f'{{\n'
            f'// Algorithmic hessian only (no history contributions)\n'
            f'// History geometric stiffness (I x S_hist) should be added at runtime\n'
            f'{real_t} F[{dim**2}];\n'
            f'{{\n'
            f'{F_actual}'
            f'}}\n\n'
            f'{real_t} S_lin[{dim**4}];\n'
            f'{{\n'
            f'{S_actual}\n'
            f'}}\n'
            f'{combined_code}\n'
            f'}}\n'
        )

        print(signature + body)

    def emit_hessian_flexible(self):
        """Emit S_lin (linearized stress tensor) using symbolic gamma.
        
        This version generates ONLY S_lin[81] - the 4th order tensor.
        The caller assembles the Hessian using loops in C code.
        
        This is much faster to generate and produces shorter code.
        
        Usage in C:
            1. Compute gamma = g_inf + sum(g_i * (1 - exp(-dt/tau_i)) / (dt/tau_i))
            2. Call this function to get S_lin[81]
            3. Assemble Hessian with loops: H[test,trial] += S_lin[i,k,m,n] * grad[trial][i,n] * grad[test][k,m]
            4. Add history geometric stiffness (I x S_hist) via loops
        """
        self.__compute_dV()
        self.__compute_jacobian_adjugate()
        self.__compute_Jinv()
        self.__compute_disp_grad()
        self.__compute_F()
        self.__compute_piola_stress_flexible()
        self.__compute_linearized_stress_flexible()
        self.__compute_metric_tensor_flexible()
        # Skip __compute_hessian_flexible() - we only need S_lin!
        
        fe = self.fe
        dim = fe.spatial_dim()
        
        S_ikmn_flexible = self.expression_table["S_ikmn_flexible"]

        # Use only basic params (K, C10, C01) + gamma, no Prony terms!
        signature = (
            f'static SFEM_INLINE void {fe.name().lower()}_{self.name}_S_lin_flexible(\n'
            f'    const {real_t} *const SFEM_RESTRICT adjugate,\n'
            f'    const {real_t}                      jacobian_determinant,\n'
            f'    const {real_t}                      qx,\n'
            f'    const {real_t}                      qy,\n'
            f'    const {real_t}                      qz,\n'
            f'    const {real_t}                      qw,\n'
            f'    const {real_t}                      K,\n'
            f'    const {real_t}                      C10,\n'
            f'    const {real_t}                      C01,\n'
            f'    const {real_t}                      gamma,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispx,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispy,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispz,\n'
            f'    {real_t} *const SFEM_RESTRICT       S_lin)'
            f'\n'
        )

        F_actual = c_gen(assign_matrix("F", self.expression_table["F"]))
        S_actual = c_gen(self.__assign_tensor4("S_lin", S_ikmn_flexible))

        body = (
            f'{{\n'
            f'// S_lin (metric tensor) with symbolic gamma\n'
            f'// Caller should:\n'
            f'//   1. Compute gamma = g_inf + sum(g_i * (1 - exp(-dt/tau_i)) / (dt/tau_i))\n'
            f'//   2. Assemble Hessian: H[test,trial] += S_lin[idx] * grad[trial] * grad[test]\n'
            f'//   3. Add history geometric stiffness (I x S_hist) via loops\n'
            f'{real_t} F[{dim**2}];\n'
            f'{{\n'
            f'{F_actual}'
            f'}}\n\n'
            f'{{\n'
            f'{S_actual}\n'
            f'}}\n'
            f'}}\n'
        )

        print(signature + body)

    def emit_hessian_diag(self):
        self.__compute_dV()
        self.__compute_jacobian_adjugate()
        self.__compute_Jinv()
        self.__compute_disp_grad()
        self.__compute_F()
        self.__compute_piola_stress()
        self.__compute_linearized_stress()
        self.__compute_metric_tensor()
        self.__compute_hessian()

        fe = self.fe
        dim = fe.spatial_dim()
        nfun = fe.n_nodes()

        H_diag = self.expression_table["hessian_diag"]

        sub_S_lin = True
        if sub_S_lin:
            for test in range(0, nfun * dim):
                print(f"// Substituting {test+1}/{nfun * dim}...", end="")
                H_diag[test] = self.__subs_tensor4(H_diag[test], self.S_lin_symb, self.expression_table["S_lin"])
                print("DONE")
        
        
        signature = (
            f'static SFEM_INLINE void {fe.name().lower()}_{self.name}_hessian_diag(\n'
            f'    const {real_t} *const SFEM_RESTRICT adjugate,\n'
            f'    const {real_t}                      jacobian_determinant,\n'
            f'    const {real_t}                      qx,\n'
            f'    const {real_t}                      qy,\n'
            f'    const {real_t}                      qz,\n'
            f'    const {real_t}                      qw,\n'
            f'{self.__params_to_args()}'
            f'{self.__history_to_args()}'
            f'    const {real_t} *const SFEM_RESTRICT dispx,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispy,\n'
            f'    const {real_t} *const SFEM_RESTRICT dispz,\n'
            f'    {real_t} *const SFEM_RESTRICT       H_diag)'
            f'\n'
        )

        F_actual = c_gen(assign_matrix("F", self.expression_table["F"]))
        S_code = ""
        if not sub_S_lin:
            S_actual = c_gen(self.__assign_tensor4("S_lin", self.expression_table["S_lin"]))
            S_code = f'{real_t} S_lin[{dim**4}];\n'
            f'{{\n'
            f'{S_actual}\n'
            f'}}\n'

        body = (
            f'{{\n'
            f'{real_t} F[{dim**2}];\n'
            f'{{\n'
            f'{F_actual}'
            f'}}\n\n'
            f'{S_code}'
            f'{c_gen(add_assign_matrix("H_diag", H_diag))}\n'
            f'}}\n'
        )

        print(signature + body)
        

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

        self.__compute_metric_tensor_canonical()
        self.__compute_SdotH_km_canonical()
        self.__compute_apply_canonical()

        self.__compute_constant_grad_tp()

    def check_metric_tensor_symmetries(self):
        self.partial_assembly()
        

        dim = self.fe.spatial_dim()
        S_lin = self.expression_table["S_lin"]
        Jinv = self.fe.symbol_jacobian_inverse_as_adjugate()


        # detect_constitutive_tensor_symmetries(S_lin)

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

        S_ikmn = sp.MutableDenseNDimArray(terms, shape=(dim, dim, dim, dim))
        # detect_constitutive_tensor_symmetries(S_ikmn)

        vec, canon_list = canon.pack_tensor(S_ikmn)      # length 45 in 3D
        S_reconstructed = canon.reconstruct_full(vec, canon_list, dim, as_sympy=True)
        S_diff = S_reconstructed - S_ikmn

        for i in range(0, dim):
            for k in range(0, dim):
                for m in range(0, dim):
                    for n in range(0, dim):
                        S_diff[i, k, m, n] = sp.simplify(S_diff[i, k, m, n])
                        print(f"{i}, {k}, {m}, {n}: {S_diff[i, k, m, n]}")


if __name__ == "__main__":
    fe = Hex8()
    # fe = Tet4()
    # fe = Tet10()
    
    # Mooney-Rivlin Model with Viscoelasticity
    # 1. Volumetric part (Penalty for incompressibility)
    w_vol = "K / 2 * (J - 1)**2"
    
    # 2. Deviatoric part (Mooney-Rivlin)
    # Using I1b (I1_bar) and I2b (I2_bar) for isochoric invariants
    w_dev = "C10 * (I1b - 3) + C01 * (I2b - 3)"
    
    # num_prony_terms for unrolled version (hardcoded history indices)
    # For loop-based version, use emit_hessian_algo() which doesn't depend on num_prony_terms
    num_prony = 10  # Must match what C code expects for unrolled version
    
    op = SRViscoHyperelasticity.create_from_string_unimodular(
        fe, 
        "mooney_rivlin", 
        [w_vol, w_dev], 
        num_prony_terms=num_prony
    )
    
    # op.check_metric_tensor_symmetries()
    
    # op.emit_objective()
    # op.emit_gradient()
    # print("// -------------------------------------------------")
    # op.emit_history_update()
    
    # print("// ============= UNROLLED VERSION (fixed {} Prony terms) =============".format(num_prony))
    # op.emit_hessian()
    # op.emit_hessian_diag()
    
    print("// ============= FLEXIBLE VERSION (gamma as parameter, works with ANY Prony terms) =============")
    op.emit_hessian_flexible()
