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
                sname = str(s)
                if sname == "I1b":
                    self.I1b_symb = s
                elif sname == "I2b":
                    self.I2b_symb = s
                elif sname == "J":
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
        # Use explicit inverse to ensure correct differentiation through F^{-T}
        F_inv_T = inverse(F).T
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
        return S_iklm

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
        print("{", end="")
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
        dV = self.expression_table["dV"]
        for i in range(0, dims * fe.n_nodes()):
            lform = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.Assignment(lform, inner(Lim, grad[i]) * dV))

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
        dims = self.fe.manifold_dim()
        ndofs = dims * fe.n_nodes()

        # Use internally managed symbols and expressions
        # Differentiate P w.r.t. the fundamental F symbols, then substitute F = I + disp_grad
        P = self.expression_table["P"]
        jac_inv = self.expression_table["jac_inv"]
        dV = self.expression_table["dV"]

        # Reference gradients and physical mapping
        q = self.fe.quadrature_point()
        gref = fe.tgrad(q, ncomp=dims)

        expr = []

        # Loop over trial dofs j and test dofs i
        for j in range(0, ndofs):
            # Physical gradient for trial dof j
            Htrial = gref[j] * jac_inv

            # Linearized stress dP = (∂P/∂F):Htrial
            dP = sp.Matrix(dims, dims, [0] * (dims * dims))
            for a in range(dims):
                for b in range(dims):
                    acc = 0
                    for i1 in range(dims):
                        for i2 in range(dims):
                            # Differentiate w.r.t. the fundamental F symbols
                            acc += sp.diff(P[a, b], self.F_symb[i1, i2]) * Htrial[i1, i2]
                    # Substitute F = I + disp_grad symbols to evaluate at current state
                    dP[a, b] = simplify(subsmat(acc, self.F_symb, self.expression_table["F"]))

            # Map to reference test side and scale by measure
            Lop = dP * jac_inv.T * dV

            for i in range(0, ndofs):
                var = sp.symbols(f"element_matrix[{i*ndofs + j}*stride]")
                expr.append(ast.Assignment(var, inner(Lop, gref[i])))

        return expr

    def emit_all(self, out_dir: str = None, opname: str = "hyperelasticity"):
        fe = self.fe
        # k_value = self.kernel_value()
        # k_grad = self.kernel_gradient()
        # k_apply = self.kernel_apply()
        k_apply = self.kernel_apply()
        # k_partial_assembly = self.partial_assembly()
        
        # self.tpl.emit_all(opname=opname, kernels={'gradient': k_grad, 'apply': k_apply, 'value': k_value}, out_dir=out_dir)

def verify_kernel_apply_against_hessian(op: SRHyperelasticity, ntests: int = 2, seed: int = 0):
    """Numerically verify that matrix-free kernel_apply matches full Hessian action.

    Compares (S_iklm : H) : T * dV vs assembled H @ h for random increments on a Tet4.
    """
    np.random.seed(seed)

    fe = op.fe
    assert isinstance(fe, Tet4)

    dims = fe.manifold_dim()
    ndofs = dims * fe.n_nodes()

    # Symbols to substitute
    mu, lmbda = sp.symbols("mu lmbda")
    qw = fe.quadrature_weight()
    q = fe.quadrature_point()
    gref = fe.tgrad(q, ncomp=dims)

    # Random invertible J
    J = np.random.randn(dims, dims)
    while abs(np.linalg.det(J)) < 1e-2:
        J = np.random.randn(dims, dims)
    detJ = float(np.linalg.det(J))
    Jinv = np.linalg.inv(J)
    Adj = detJ * Jinv  # adjugate

    # Substitutions for geometry and parameters
    subs = {}
    subs[op.fe.symbol_jacobian_determinant()] = detJ
    subs[qw] = 1.0
    # Assign adjugate symbols
    adj = fe.symbol_adjugate()
    for i in range(dims):
        for j in range(dims):
            subs[adj[i, j]] = Adj[i, j]
    # Material params
    subs[mu] = 2.0
    subs[lmbda] = 5.0

    # Evaluate reference gradients numerically (Tet4 are constant)
    # Evaluate each tensorized gradient G (dims x dims) numerically
    gref_num = []
    for G in gref:
        Gnum = np.zeros((dims, dims), dtype=float)
        for r in range(dims):
            for c in range(dims):
                Gnum[r, c] = float(sp.N(G[r, c].subs(subs)))
        gref_num.append(Gnum)

    # Evaluate S_iklm (partial assembly) at F = I
    S = op.partial_assembly()
    Fsym = op.F_symb
    I = sp.eye(dims)
    subs_F = {Fsym[i, j]: (1.0 if i == j else 0.0) for i in range(dims) for j in range(dims)}
    subs_all = {**subs, **subs_F}

    # Numeric S_iklm tensor
    S_num = np.zeros((dims, dims, dims, dims), dtype=float)
    for i in range(dims):
        for k in range(dims):
            for l in range(dims):
                for m in range(dims):
                    S_num[i, k, l, m] = float(sp.N(S[i, k, l, m].subs(subs_all)))

    dV = float(sp.N(op.expression_table["dV"].subs(subs)))

    for _ in range(ntests):
        # Random increment vector h and corresponding H = sum_j h_j * (gref[j] * Jinv)
        h = np.random.randn(ndofs)
        H = np.zeros((dims, dims), dtype=float)
        for j in range(ndofs):
            H += h[j] * (gref_num[j] @ Jinv)

        # Matrix-free apply: r_i = inner((S:H), gref[i]) * dV
        r_apply = np.zeros((ndofs,), dtype=float)
        for i in range(ndofs):
            comp_i = i // fe.n_nodes()
            # Compute Lim[comp_i, :] only
            Lim_row = np.zeros((dims,), dtype=float)
            for k in range(dims):
                for l in range(dims):
                    Lim_row += S_num[comp_i, k, l, :] * H[k, l]
            # inner(Lim, gref[i]) reduces to dot(Lim_row, row of gref[i])
            r_apply[i] = float(np.dot(Lim_row, gref_num[i][comp_i, :]) * dV)

        # Full Hessian assemble times h
        # Build constitutive tensor C = dP/dF at F = I
        C = op.expression_table["constitutive_tensor"]
        C_num = np.zeros((dims, dims, dims, dims), dtype=float)
        for a in range(dims):
            for b in range(dims):
                for k in range(dims):
                    for l in range(dims):
                        C_num[a, b, k, l] = float(sp.N(C[a, b, k, l].subs(subs_all)))

        # Assemble Hmat * h via dP contraction
        r_full = np.zeros((ndofs,), dtype=float)
        for j in range(ndofs):
            Htrial = gref_num[j] @ Jinv
            dP = np.zeros((dims, dims), dtype=float)
            for a in range(dims):
                for b in range(dims):
                    dP[a, b] = (C_num[a, b] * Htrial).sum()
            Lop = dP @ Jinv.T * dV
            for i in range(ndofs):
                comp_i = i // fe.n_nodes()
                r_full[i] += np.dot(Lop[comp_i, :], gref_num[i][comp_i, :]) * h[j]

        err = np.linalg.norm(r_apply - r_full) / max(1.0, np.linalg.norm(r_full))
        print(f"apply vs full error: {err:.3e}")
        assert err < 1e-8, f"Mismatch between kernel_apply and full Hessian action: {err}"

def main():
    strain_energy_function = "(mu/2)*(I1b - 3) + (lmbda/2)*(log(J))**2"
    # strain_energy_function = "J * (I2b + (I1b * 0.8341331275382947))"
    name = "neohookean"
    fe = Tet4()
    # fe = Tri3()
    # fe = Hex8()
    op = SRHyperelasticity.create_from_string(fe, strain_energy_function)
    op.emit_all(opname=name)
    # Numerical consistency check: matrix-free vs full Hessian
    verify_kernel_apply_against_hessian(op, ntests=2, seed=42)


if __name__ == "__main__":
    main()
