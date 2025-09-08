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
    return expr
    # return sp.simplify(expr)

def create_matrix_symbol(name, rows, cols):
    # return sp.Matrix(rows, cols, [sp.symbols(f"{name}[{i*cols + j}]") for i in range(0, rows) for j in range(0, cols)])
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


def assign_matrix(var, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            expr.append(ast.Assignment(var[i, j], mat[i, j]))
    return expr


def assign_tensor4(var, mat):
    size0, size1, size2, size3 = mat.shape
    expr = []
    for i in range(0, size0):
        for j in range(0, size1):
            for k in range(0, size2):
                for l in range(0, size3):
                    expr.append(ast.Assignment(var[i, j, k, l], mat[i, j, k, l]))
    return expr

def assign_add_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.AddAugmentedAssignment(var, mat[i, j]))
    return expr


class SRHyperelasticity:
    @staticmethod
    def create_from_string(fe, str_expr: str):
        # Allow users to pass expressions in terms of invariants I1b, I2b, J
        
        fun = parse_expr(str_expr)    
        return SRHyperelasticity(fe, fun) #, TPLHyperelasticity(fe))

    def init_symbols(self, fun):
        self.fun = simplify(fun)
        I1b, I2b, J = sp.symbols("I1b I2b J", real=True)
        invariants = [I1b, I2b, J]
        symbol_names = list(fun.free_symbols)

        all_symbols = []
        for s in symbol_names + invariants:
            if str(s) not in [str(sym) for sym in all_symbols]:
                all_symbols.append(s)
                match str(s):
                    case "I1b":
                        self.I1b_symb = s
                    case "I2b":
                        self.I2b_symb = s
                    case "J":
                        self.J_symb = s
        
        self.disp_grad_symb = create_matrix_symbol("disp_grad", self.fe.spatial_dim(), self.fe.spatial_dim())
        self.F_symb = create_matrix_symbol("F", self.fe.spatial_dim(), self.fe.spatial_dim())
        self.jac_inv_symb = create_matrix_symbol("jac_inv", self.fe.manifold_dim(), self.fe.spatial_dim())
        self.Pinv_xJinv_t_symb = create_matrix_symbol("Pinv_xJinv_t", self.fe.manifold_dim(), self.fe.manifold_dim())
        self.trial_grad_symb = create_matrix_symbol("trial_grad", self.fe.manifold_dim(), self.fe.manifold_dim())
        self.inc_grad_symb = create_matrix_symbol("inc_grad", self.fe.manifold_dim(), self.fe.manifold_dim())
        self.safe_log = lambda x: sp.log(sp.Max(1e-8, x))
        self.safe_sqrt = lambda x: sp.sqrt(sp.Max(1e-8, x))
        self.constitutive_tensor_symb = create_tensor4_symbol("constitutive_tensor", self.fe.spatial_dim(), self.fe.spatial_dim(), self.fe.spatial_dim(), self.fe.spatial_dim())
        self.S_iklm_symb = create_tensor4_symbol("S_iklm", self.fe.spatial_dim(), self.fe.spatial_dim(), self.fe.spatial_dim(), self.fe.spatial_dim())

    def __init__(self, fe, fun): #  tpl: TPLHyperelasticity):
        self.fe = fe
        self.init_symbols(fun)

        # Expressions to be evaluated for code generation
        self.expression_table = {}
        self.__build_sym(fe)
        
    # ---------------- Directional derivatives (separated from FEM) ----------------
    def __compute_piola_stress(self):
        F = self.F_symb
        F_inv_T = F.T.inv()
        J = determinant(F)
        B = F * F.T
        Jm23 = J ** (-sp.Rational(2, 3))
        Bbar = Jm23 * B
        I1b = sp.trace(Bbar)

        I2b = (I1b * I1b - sp.trace(Bbar * Bbar)) / 2

        # Differentiate strain energy with respect to invariants
        dW_dI1b = sp.diff(self.fun, self.I1b_symb)
        dW_dI2b = sp.diff(self.fun, self.I2b_symb)
        dW_dJ = sp.diff(self.fun, self.J_symb)

        # Isochoric stress contributions from chain rule
        P_iso_I1b = Jm23 * (2*F - sp.Rational(2,3) * I1b * F_inv_T)
        P_iso_I2b = Jm23 * (2*I1b*F - 2*F*B - sp.Rational(2,3) * I2b * F_inv_T)
        
        # Volumetric stress contribution
        P_vol = J * F_inv_T
        
        # Apply chain rule: P = dW/dI1b * dI1b/dF + dW/dI2b * dI2b/dF + dW/dJ * dJ/dF
        P = dW_dI1b * P_iso_I1b + dW_dI2b * P_iso_I2b + dW_dJ * P_vol

        for i in range(0, P.shape[0]):
            for j in range(0, P.shape[1]):
                P[i, j] = P[i, j].subs({self.I1b_symb: I1b, self.I2b_symb: I2b, self.J_symb: J})

        return simplify_matrix(P)

    def __build_sym(self, fe):
        fe = self.fe

        # First Piola-Kirchhoff stress
        P = self.__compute_piola_stress()
        dV = fe.reference_measure() * fe.symbol_jacobian_determinant() * fe.quadrature_weight()        
        P_tXJinv_t = P * self.jac_inv_symb.T * dV
        jac_inv = self.fe.symbol_jacobian_inverse_as_adjugate()

        self.expression_table["P"] = P
        self.expression_table["P_tXJinv_t"] = P_tXJinv_t
        self.expression_table["dV"] = dV
        
        F_dot_h = simplify(inner(P, self.trial_grad_symb))

        lin_stress = sp.zeros(self.fe.spatial_dim(), self.fe.spatial_dim())
        for i in range(0, self.fe.spatial_dim()):
            for j in range(0, self.fe.spatial_dim()):
                lin_stress[i, j] = sp.diff(F_dot_h, self.F_symb[i, j])

        dim = self.fe.spatial_dim()
        terms = []
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    for l in range(0, dim):
                        print(f"computing {i}, {j}, {k}, {l})")
                        terms.append(sp.diff(P[i, j], self.F_symb[k, l]))
        
        print("Done computing terms")
        constitutive_tensor = Array(terms, shape=(dim, dim, dim, dim))
        print("Created constitutive tensor")

        # detect_constitutive_tensor_symmetries(constitutive_tensor)
        # print("Done detecting symmetries")


        terms = []
        for i in range(0, dim):
            for k in range(0, dim):
                for l in range(0, dim):
                    for m in range(0, dim):
                        acc = 0
                        for j in range(0, dim):
                            acc += constitutive_tensor[i, j, k, l] * jac_inv[m, j]
                        terms.append(acc)

        S_iklm = Array(terms, shape=(dim, dim, dim, dim))

   
        # print(lin_stress)
        self.expression_table["lin_stress"] = lin_stress
        self.expression_table["lin_stressXJinv_t"] = lin_stress * self.jac_inv_symb.T * dV
        self.expression_table["S_iklm"] = S_iklm

        # Build FEM symbols
        dims = self.fe.manifold_dim()
        q = self.fe.quadrature_point()
        gref = self.fe.tgrad(q, ncomp=dims)
        
        
        u = coeffs_SoA("u", dims, self.fe.n_nodes())
        disp_grad = sp.zeros(dims, dims)
        for i in range(0, dims * self.fe.n_nodes()):
            disp_grad += u[i] * gref[i]
        disp_grad = disp_grad * jac_inv

        inc_grad = sp.zeros(dims, dims)
        for i in range(0, dims * self.fe.n_nodes()):
            inc_grad += u[i] * gref[i]
        inc_grad = inc_grad * jac_inv
        
        F = sp.eye(dims) + self.disp_grad_symb
        
        self.expression_table["disp_grad"] = disp_grad
        self.expression_table["inc_grad"] = inc_grad
        self.expression_table["F"] = F
        self.expression_table["jac_inv"] = jac_inv
        self.expression_table["constitutive_tensor"] = constitutive_tensor

    def def_grad(self):
        fe = self.fe
        dims = self.fe.manifold_dim()
        F = self.expression_table["F"]

        for i in range(0, dims):
            for j in range(0, dims):
                F[i, j] = subsmat(F[i, j], self.disp_grad_symb, self.expression_table["disp_grad"])

        return F

    def kernel_value(self):
        fe = self.fe
        expr = []
        form = sp.symbols("element_scalar[0]")
        expr.append(ast.AddAugmentedAssignment(form, s['W'] * s['dV']))
        return expr

    def partial_assembly(self):
        S_iklm = self.expression_table["S_iklm"]
        # c_code(assign_tensor4(self.S_iklm_symb, S_iklm))
        return S_iklm

    # def kernel_apply(self):
    #     fe = self.fe
    #     dims = self.fe.manifold_dim()
    #     grad = fe.tgrad(q, ncomp=dims)
        
    #     expr = []
    #     F = self.def_grad()
    #     expr.extend(assign_matrix(self.F_symb, F))

    #     print(f"T F[{dims*dims}];")
    #     print("{", end="")
    #     c_code(expr)
    #     print("}\n\n")

    #     print(f"T trial_grad[{dims*dims}];")
    #     print("{", end="")
    #     c_code(assign_matrix(self.trial_grad_symb, self.expression_table["inc_grad"]))
    #     print("}\n\n")

    #     lin_stress_x_jinv_t = self.expression_table["lin_stressXJinv_t"]

    #     lin_stress_symb = matrix_coeff("lin_stressXJinv_t", dims, dims)

    #     print(f"T *lin_stressXJinv_t = F;")
    #     print("{", end="")
    #     c_code(assign_matrix(lin_stress_symb, lin_stress_x_jinv_t))
    #     print("}\n\n")

    #     expr = []
    #     for i in range(0, dims * fe.n_nodes()):
    #         lform = sp.symbols(f"element_vector[{i}*stride]")
    #         expr.append(ast.Assignment(lform, inner(lin_stress_symb, grad[i])))

    #     print("{", end="")
    #     c_code(expr)
    #     print("}\n\n")
    #     return expr

    def kernel_apply(self):
        fe = self.fe
        dims = self.fe.manifold_dim()
        grad = fe.tgrad(q, ncomp=dims)
        
        expr = []
        F = self.def_grad()
        expr.extend(assign_matrix(self.F_symb, F))

        print(f"T F[{dims*dims}];")
        print("{", end="")
        c_code(expr)
        print("}\n\n")

        print(f"T inc_grad[{dims*dims}];")
        print("{", end="")
        c_code(assign_matrix(self.inc_grad_symb, self.expression_table["inc_grad"]))
        print("}\n\n")

        partial_assembly = self.partial_assembly()
        S_iklm_symb = self.S_iklm_symb

        print(f"T S_iklm[{dims*dims*dims*dims}];")
        c_code(assign_tensor4(S_iklm_symb, partial_assembly))
        print("}\n\n")

        Hkl = self.inc_grad_symb

        Lim = sp.zeros(dims, dims)
        for i in range(0, dims):
            for k in range(0, dims):
                for l in range(0, dims):
                    for m in range(0, dims):
                        Lim[i, m] += S_iklm_symb[i, k, l, m] * Hkl[k, l]

        expr = []
        for i in range(0, dims * fe.n_nodes()):
            lform = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.Assignment(lform, inner(Lim, grad[i])))

        print("{", end="")
        c_code(expr)
        print("}\n\n")
        return expr
        

    def kernel_gradient(self):
        fe = self.fe
        dims = self.fe.manifold_dim()
        grad = fe.tgrad(q, ncomp=dims)
        
        # Generate the actual chain rule expressions directly
        expr = []

        F = self.def_grad()
        expr.extend(assign_matrix(self.F_symb, F))

        print(f"T F[{dims*dims}];")
        print("{", end="")
        c_code(expr)
        print("}\n\n")

        print(f"T *Pinv_xJinv_t = F;")
        print("{", end="")
        c_code(assign_matrix(self.Pinv_xJinv_t_symb, self.expression_table["P_tXJinv_t"]))
        print("}\n\n")
   
        expr = []
        for i in range(0, dims * fe.n_nodes()):
            lform = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.Assignment(lform, inner( self.Pinv_xJinv_t_symb, grad[i])))
        
        print("{", end="")
        c_code(expr)
        print("}\n\n")
        return expr

    def kernel_hessian(self):
        """Emit code for the full element Hessian (matrix) using P_tXJinv_t idiom.

        H[i,j] = inner( (dP(F):H_j) * Jinv^T * dV, gref_i )
        where H_j is the physical gradient for trial dof j derived from reference gradients.
        """
        fe = self.fe
        dims = s['dims']
        ndofs = dims * fe.n_nodes()

        F = s['F']
        P = s['P']
        jac_inv = s['jac_inv']
        dV = s['dV']

        expr = []

        # Loop over trial dofs j and test dofs i
        for j in range(0, ndofs):
            # Physical gradient for trial j: Htrial = gref[j] * jac_inv
            Htrial = s['gref'][j] * jac_inv

            # Linearized stress dP = (∂P/∂F):Htrial
            dP = sp.Matrix(dims, dims, [0] * (dims * dims))
            for a in range(dims):
                for b in range(dims):
                    acc = 0
                    for i1 in range(dims):
                        for i2 in range(dims):
                            acc += sp.diff(P[a, b], F[i1, i2]) * Htrial[i1, i2]
                    dP[a, b] = simplify(acc)

            Lop = dP * jac_inv.T * dV

            for i in range(0, ndofs):
                var = sp.symbols(f"element_matrix[{i*ndofs + j}*stride]")
                expr.append(ast.Assignment(var, inner(Lop, s['gref'][i])))

        return expr

    def emit_all(self, out_dir: str = None, opname: str = "hyperelasticity"):
        fe = self.fe
        # k_value = self.kernel_value()
        # k_grad = self.kernel_gradient()
        # k_apply = self.kernel_apply()
        k_apply = self.kernel_apply()
        # k_partial_assembly = self.partial_assembly()
        
        # self.tpl.emit_all(opname=opname, kernels={'gradient': k_grad, 'apply': k_apply, 'value': k_value}, out_dir=out_dir)

def main():
    strain_energy_function = "(mu/2)*(I1b - 3) + (lmbda/2)*(log(J))**2"
    # strain_energy_function = "J * (I2b + (I1b * 0.8341331275382947))"
    name = "neohookean"
    fe = Tet4()
    # fe = Tri3()
    # fe = Hex8()
    op = SRHyperelasticity.create_from_string(fe, strain_energy_function)
    op.emit_all(opname=name)


if __name__ == "__main__":
    main()
