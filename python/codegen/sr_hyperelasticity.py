#!/usr/bin/env python3

try:
    from sfem_codegen import *
except Exception:
    # Minimal shims to allow file generation without SymPy at runtime
    import sympy as sp  # will fail if sympy missing; we guard uses
    def simplify(expr):
        return expr
from quad4 import *
from tri3 import *
from tri6 import *
from tet4 import *
from tet10 import *
from tet20 import *
from hex8 import *
from aahex8 import *

import sys
import os

from time import perf_counter
from tpl_hyperelasticity import TPLHyperelasticity


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


class SRHyperelasticity:
    @staticmethod
    def create_from_string(fe, str_expr: str):
        # Allow users to pass expressions in terms of invariants I1b, I2b, J
        I1b, I2b, J = sp.symbols("I1b I2b J", real=True)
        mu, lam = sp.symbols("mu lam", real=True)
        # Replace Python keyword 'lambda' with 'lam' if present
        expr_src = str_expr.replace("lambda", "lam")
        local_ns = {"I1b": I1b, "I2b": I2b, "J": J, "mu": mu, "lam": lam, "log": sp.log}
        fun = sp.parse_expr(expr_src, local_dict=local_ns)
        return SRHyperelasticity(fe, fun, TPLHyperelasticity(fe))

    def init_templates(self, tpl: TPLHyperelasticity = None):
        self.tpl = tpl if tpl is not None else TPLHyperelasticity(self.fe)
      

    def init_symbols(self):
        # Invariants
        self.I1b = sp.symbols("I1b", real=True)
        self.I2b = sp.symbols("I2b", real=True)
        self.J = sp.symbols("J", real=True)
        self.safe_log = lambda x: sp.log(sp.Max(1e-8, x))
        self.safe_sqrt = lambda x: sp.sqrt(sp.Max(1e-8, x))

    def __init__(self, fe, fun, tpl: TPLHyperelasticity = None):
        self.init_symbols()
        self.fe = fe
        self.init_templates(tpl)
        self.fun = simplify(fun)

    def value(self):
        dfdI1b = sp.diff(self.fun, self.I1b)
        dfdI2b = sp.diff(self.fun, self.I2b)
        dfdJ = sp.diff(self.fun, self.J)
        return dfdI1b, dfdI2b, dfdJ


    def apply(self):
        pass

    # ---------------- Symbolic core (P_tXJinv_t idiom) ----------------
    def _build_sym(self, fe):
        dims = fe.manifold_dim()
        q = fe.quadrature_point()

        fe.use_adjugate = True
        gref = fe.tgrad(q, ncomp=dims)
        jac_inv = fe.symbol_jacobian_inverse_as_adjugate()

        # Build displacement gradient in physical coordinates
        u = coeffs_SoA("u", dims, fe.n_nodes())
        disp_grad = sp.zeros(dims, dims)
        for i in range(0, dims * fe.n_nodes()):
            disp_grad += u[i] * gref[i]
        disp_grad = disp_grad * jac_inv

        # Define a symbolic F_s to differentiate W(F_s) w.r.t. F_s entries
        F_s = sp.Matrix(dims, dims, matrix_coeff("Fsym", dims, dims))
        J_s = determinant(F_s)
        B_s = F_s * F_s.T
        Jm23_s = J_s ** (-sp.Rational(2, 3))
        Bbar_s = Jm23_s * B_s
        I1b_s = sp.trace(Bbar_s)
        I2b_s = (I1b_s * I1b_s - sp.trace(Bbar_s * Bbar_s)) / 2

        W_s = self.fun.subs({self.I1b: I1b_s, self.I2b: I2b_s, self.J: J_s})
        P_s = sp.Matrix(dims, dims, [0] * (dims * dims))
        for i in range(dims):
            for j in range(dims):
                P_s[i, j] = sp.diff(W_s, F_s[i, j])

        # Substitute F_s with actual F = I + disp_grad
        F = sp.eye(dims) + disp_grad
        subs_map = {}
        for i in range(dims):
            for j in range(dims):
                subs_map[F_s[i, j]] = F[i, j]
        P = P_s.xreplace(subs_map)

        # Rename adjugate[...] -> jacobian_adjugate[...] to match runtime variable names
        adj_map = {}
        for i in range(dims):
            for j in range(dims):
                adj_map[sp.symbols(f"adjugate[{i*dims + j}]")] = sp.symbols(f"jacobian_adjugate[{i*dims + j}]")
        P = P.xreplace(adj_map)

        dV = fe.reference_measure() * fe.symbol_jacobian_determinant() * fe.quadrature_weight()

        W_eval = W_s.xreplace(subs_map).xreplace(adj_map)
        P_tXJinv_t = P * jac_inv.T * dV

        return {
            'dims': dims,
            'gref': gref,
            'jac_inv': jac_inv,
            'disp_grad': disp_grad,
            'F': F,
            'W': W_eval,
            'P': P,
            'Ps': P_s,
            'Fs': F_s,
            'subs_map': subs_map,
            'P_tXJinv_t': P_tXJinv_t,
            'dV': dV,
        }

    def kernel_value(self, fe):
        s = self._build_sym(fe)
        expr = []
        form = sp.symbols("element_scalar[0]")
        expr.append(ast.AddAugmentedAssignment(form, s['W'] * s['dV']))
        return c_gen(expr)

    def kernel_gradient(self, fe):
        s = self._build_sym(fe)
        dims = s['dims']
        expr = []
        for i in range(0, dims * fe.n_nodes()):
            lform = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.AddAugmentedAssignment(lform, inner(s['P_tXJinv_t'], s['gref'][i])))
        return c_gen(expr)

    def kernel_apply(self, fe):
        s = self._build_sym(fe)
        dims = s['dims']
        jac_inv = s['jac_inv']
        dV = s['dV']

        # Build linearized stress using symbolic Ps and Fs, then substitute Fs->F
        Ps = s['Ps']
        Fs = s['Fs']
        H = sp.Matrix(dims, dims, coeffs("grad_trial", dims * dims))
        dPs = sp.Matrix(dims, dims, [0] * (dims * dims))
        for a in range(dims):
            for b in range(dims):
                acc = 0
                for i in range(dims):
                    for j in range(dims):
                        acc += sp.diff(Ps[a, b], Fs[i, j]) * H[i, j]
                dPs[a, b] = simplify(acc)

        # Substitute Fs entries with actual F entries
        adj_map = {}
        for i in range(dims):
            for j in range(dims):
                adj_map[sp.symbols(f"adjugate[{i*dims + j}]")] = sp.symbols(f"jacobian_adjugate[{i*dims + j}]")
        dP = dPs.xreplace(s['subs_map']).xreplace(adj_map)

        Lop = dP * jac_inv.T * dV

        expr = []
        for i in range(0, dims * fe.n_nodes()):
            lform = sp.symbols(f"element_vector[{i}*stride]")
            expr.append(ast.AddAugmentedAssignment(lform, inner(Lop, s['gref'][i])))
        return c_gen(expr)

    def kernel_hessian(self, fe):
        """Emit code for the full element Hessian (matrix) using P_tXJinv_t idiom.

        H[i,j] = inner( (dP(F):H_j) * Jinv^T * dV, gref_i )
        where H_j is the physical gradient for trial dof j derived from reference gradients.
        """
        s = self._build_sym(fe)
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

        return c_gen(expr)

    def emit_all(self, out_dir: str = None, opname: str = "hyperelasticity"):
        """Emit gradient/apply/value for the given FE using the injected backend.

        This hides any template/rendering details from the generator.
        """
        fe = self.fe
        k_value = self.kernel_value(fe)
        k_grad = self.kernel_gradient(fe)
        k_apply = self.kernel_apply(fe)
        self.tpl.emit_all(opname=opname, kernels={'gradient': k_grad, 'apply': k_apply, 'value': k_value}, out_dir=out_dir)

def main():
    fes = {
        "HEX8": Hex8(),
        "TET4": Tet4()
    }

    if len(sys.argv) < 2:
        print("Usage: python3 sr_hyperelasticity.py \"<strain_energy_function>\" [EType]")
        print("Example: python3 sr_hyperelasticity.py \"J * (I2b + (I1b * 0.8341331275382947))\" HEX8")
        exit(1)

    strain_energy_function = sys.argv[1]
    
    fe = Hex8()
    if len(sys.argv) >= 3:
        et = sys.argv[2].upper()
        if et in fes:
            fe = fes[et]
        else:
            print(f"[WARN] Unknown EType '{et}', defaulting to HEX8")

    # Demo shortcut: build standard Neo-Hookean Ogden from invariants
    if strain_energy_function.upper() == "DEMO_NEOHOOKEAN_OGDEN":
        # Section 2.1 style: isochoric + volumetric split
        # Wiso(I1b) = mu/2 (I1b - 3),  Wvol(J) = lambda/2 (log J)^2
        strain_energy_function = "(mu/2)*(I1b - 3) + (lambda/2)*(log(J))**2"

    op = SRHyperelasticity.create_from_string(fe, strain_energy_function)
    op.emit_all(opname="hyperelasticity")

def demo_neohookean_ogden(fe_name: str = "TET4", mu_val: float = 1.0, lambda_val: float = 1.0):
    """Demo: generate kernels for Neo-Hookean Ogden using invariants (Section 2.1 split).

    W(I1b, J) = mu/2 (I1b - 3) + lambda/2 (log J)^2
    """
    fes = {
        "HEX8": Hex8(),
        "TET4": Tet4()
    }
    fe = fes.get(fe_name.upper(), Tet4())
    W = f"(mu/2)*(I1b - 3) + (lambda/2)*(log(J))**2"
    op = SRHyperelasticity.create_from_string(fe, W)
    # Substitute constants if needed; emit symbolic code independent of specific numeric values
    op.emit_all(opname="neohookean")


if __name__ == "__main__":
    main()
