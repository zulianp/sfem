#!/usr/bin/env python3

"""
Partial-assembly generator for hyperelasticity: builds precontracted S_ikqm with dV.

This module is meant to complement sr_hyperelasticity.py by focusing only on
the partial-assembly data needed for a fast matrix-free apply:

  - S_iklm = sum_j C[i,j,k,l] * Jinv[m,j] * dV
  - S_ikqm = sum_l S_iklm * Jinv[q,l] = sum_{j,l} C[i,j,k,l] Jinv[m,j] Jinv[q,l] * dV

With S_ikqm in memory and the reference trial gradient R = sum_j h_j * gref[j],
the apply is simply: Lim[i,m] = sum_{k,q} S_ikqm[i,k,q,m] * R[k,q], and lform[i] = inner(Lim, gref[i]).
"""

import sympy as sp
import sympy.codegen.ast as ast

from sfem_codegen import c_code, c_gen, real_t, matrix_coeff
from sympy import Array
from tet4 import Tet4
from sr_hyperelasticity import SRHyperelasticity
from sympy import Array


class PAKernelGenerator:
    def __init__(self, op: SRHyperelasticity):
        self.op = op
        self.fe = op.fe
        self.dims = self.fe.manifold_dim()

        self.constitutive_tensor = op.expression_table["constitutive_tensor"]
        self.jac_inv = op.expression_table["jac_inv"]
        self.dV = op.expression_table["dV"]

        self.S_iklm = None
        self.S_ikqm = None

    def build_S_tensors(self):
        """Build S_iklm (with dV) and S_ikqm (with dV) from the constitutive tensor.

        S_iklm[i,k,l,m] = sum_j C[i,j,k,l] * Jinv[m,j] * dV
        S_ikqm[i,k,q,m] = sum_l S_iklm[i,k,l,m] * Jinv[q,l]
        """
        dim = self.dims
        C = self.constitutive_tensor
        Jinv = self.jac_inv
        dV = self.dV

        # S_iklm with dV included
        terms = []
        for i in range(dim):
            for k in range(dim):
                for l in range(dim):
                    for m in range(dim):
                        acc = 0
                        for j in range(dim):
                            acc += C[i, j, k, l] * Jinv[m, j]
                        terms.append(acc * dV)
        self.S_iklm = Array(terms, shape=(dim, dim, dim, dim))

        # S_ikqm precontracted also on trial-side mapping
        terms_q = []
        for i in range(dim):
            for k in range(dim):
                for q in range(dim):
                    for m in range(dim):
                        acc = 0
                        for l in range(dim):
                            acc += self.S_iklm[i, k, l, m] * Jinv[q, l]
                        terms_q.append(acc)
        self.S_ikqm = Array(terms_q, shape=(dim, dim, dim, dim))

        return self.S_iklm, self.S_ikqm

    def emit_S_ikqm_assignments(self, name: str = "S_ikqm"):
        """Emit C code assigning the precontracted tensor into a flat array name[]."""
        assert self.S_ikqm is not None, "Call build_S_tensors() first"
        dim = self.dims
        S_symb = self._create_tensor4_symbol(name, dim, dim, dim, dim)

        print(f"T {name}[{dim*dim*dim*dim}];")
        print("{", end="")
        c_code(self._assign_tensor4(S_symb, self.S_ikqm))
        print("}\n\n")

    @staticmethod
    def _create_tensor4_symbol(name, size0, size1, size2, size3):
        terms = []
        for i in range(size0):
            for j in range(size1):
                for k in range(size2):
                    for l in range(size3):
                        idx = i*size1*size2*size3 + j*size2*size3 + k*size3 + l
                        terms.append(sp.symbols(f"{name}[{idx}]"))
        return Array(terms, shape=(size0, size1, size2, size3))

    @staticmethod
    def _assign_tensor4(var, mat):
        size0, size1, size2, size3 = mat.shape
        expr = []
        for i in range(size0):
            for j in range(size1):
                for k in range(size2):
                    for l in range(size3):
                        expr.append(ast.Assignment(var[i, j, k, l], mat[i, j, k, l]))
        return expr

    @staticmethod
    def _assign_matrix(name: str, mat):
        rows, cols = mat.shape
        var = matrix_coeff(name, rows, cols)
        expr = []
        for i in range(rows):
            for j in range(cols):
                expr.append(ast.Assignment(var[i, j], mat[i, j]))
        return expr

    def emit_header(self, out_path: str,
                    func_name: str = None,
                    guard: str = None,
                    tensor_name: str = "S_ikqm"):
        """Write an inline header computing S_ikqm in-place (dV included).

        Signature:
          static SFEM_INLINE void <func_name>(
              const scalar_t *const SFEM_RESTRICT adjugate,
              const scalar_t                      jacobian_determinant,
              const scalar_t *const SFEM_RESTRICT F,
              const scalar_t                      mu,
              const scalar_t                      lmbda,
              const scalar_t                      qw,
              scalar_t *const SFEM_RESTRICT       S_ikqm)
        """
        assert self.S_ikqm is not None, "Call build_S_tensors() first"
        dim = self.dims

        if func_name is None:
            func_name = f"tet4_{tensor_name}"
        if guard is None:
            GUARD = f"TET4_PARTIAL_ASSEMBLY_{tensor_name.upper()}_INLINE_H"
        else:
            GUARD = guard

        S_symb = self._create_tensor4_symbol(tensor_name, dim, dim, dim, dim)
        body_S = c_gen(self._assign_tensor4(S_symb, self.S_ikqm))

        # Helper: F from (adjugate, detJ, ux,uy,uz)
        Fmat = self.op.def_grad()
        body_F = c_gen(self._assign_matrix("F", Fmat))

        # Helper: reference increment gradient (no Jinv)
        inc_ref = self.op.expression_table["inc_ref"]
        body_R = c_gen(self._assign_matrix("inc_grad", inc_ref))

        sigF = (
            "static SFEM_INLINE void tet4_F(\n"
            f"    const {real_t} *const SFEM_RESTRICT adjugate,\n"
            f"    const {real_t}                      jacobian_determinant,\n"
            f"    const {real_t} *const SFEM_RESTRICT ux,\n"
            f"    const {real_t} *const SFEM_RESTRICT uy,\n"
            f"    const {real_t} *const SFEM_RESTRICT uz,\n"
            f"    {real_t} *const SFEM_RESTRICT       F) "
            "{\n"
        )

        sigR = (
            "static SFEM_INLINE void tet4_ref_inc_grad(\n"
            f"    const {real_t} *const SFEM_RESTRICT ux,\n"
            f"    const {real_t} *const SFEM_RESTRICT uy,\n"
            f"    const {real_t} *const SFEM_RESTRICT uz,\n"
            f"    {real_t} *const SFEM_RESTRICT       inc_grad) "
            "{\n"
        )

        sigS = (
            f"static SFEM_INLINE void {func_name}(\n"
            f"    const {real_t} *const SFEM_RESTRICT adjugate,\n"
            f"    const {real_t}                      jacobian_determinant,\n"
            f"    const {real_t} *const SFEM_RESTRICT F,\n"
            f"    const {real_t}                      mu,\n"
            f"    const {real_t}                      lmbda,\n"
            f"    const {real_t}                      qw,\n"
            f"    {real_t} *const SFEM_RESTRICT       {tensor_name}) "
            "{\n"
        )

        content = []
        content.append(f"#ifndef {GUARD}")
        content.append(f"#define {GUARD}")
        content.append("")
        content.append(sigF)
        content.append(body_F)
        content.append("}\n")
        content.append("")

        content.append(sigR)
        content.append(body_R)
        content.append("}\n")
        content.append("")

        content.append(sigS)
        content.append(body_S)
        content.append("}\n")
        content.append("")

        # Emit apply kernel using SoA gradients (gx, gy, gz)
        # Embed reference shape function gradients via FE.grad at the reference point
        q = self.fe.quadrature_point()
        g = self.fe.grad(q)
        nfun = self.fe.n_nodes()
        grad_assign = []
        # Declarations
        grad_decl = "scalar_t grad_x[{}];\nscalar_t grad_y[{}];\nscalar_t grad_z[{}];\n".format(nfun, nfun, nfun)
        for node in range(nfun):
            grad_assign.append(ast.Assignment(sp.symbols(f"grad_x[{node}]"), g[node][0]))
            grad_assign.append(ast.Assignment(sp.symbols(f"grad_y[{node}]"), g[node][1]))
            grad_assign.append(ast.Assignment(sp.symbols(f"grad_z[{node}]"), g[node][2]))
        grad_code = c_gen(grad_assign)

        apply_fun = f"""
static SFEM_INLINE void tet4_apply_S_ikqm(
    const scalar_t *const SFEM_RESTRICT S_ikqm,    // 3x3x3x3, includes dV
    const scalar_t *const SFEM_RESTRICT inc_grad,  // 3x3 reference trial gradient R
    scalar_t *const SFEM_RESTRICT       element_outx,
    scalar_t *const SFEM_RESTRICT       element_outy,
    scalar_t *const SFEM_RESTRICT       element_outz)
{{
    #define D 3
    #define IDX(i,k,q,m) ((((i) * D + (k)) * D + (q)) * D + (m))

    {grad_decl}
    {grad_code}

    // M[i][m] = sum{{k,q}} S_ikqm[i,k,q,m] * inc_grad[k,q]
    scalar_t M[D][D];
    for (int i = 0; i < D; ++i) {{
        for (int m = 0; m < D; ++m) {{
            scalar_t acc = 0;
            for (int k = 0; k < D; ++k) {{
                for (int q = 0; q < D; ++q) {{
                    acc += S_ikqm[IDX(i, k, q, m)] * inc_grad[k * D + q];
                }}
            }}
            M[i][m] = acc;
        }}
    }}

    // Write SoA outputs: x,y,z components into separate arrays
    for (int node = 0; node < {nfun}; ++node) {{
        const scalar_t gx = grad_x[node];
        const scalar_t gy = grad_y[node];
        const scalar_t gz = grad_z[node];

        const scalar_t valx = M[0][0] * gx + M[0][1] * gy + M[0][2] * gz;
        const scalar_t valy = M[1][0] * gx + M[1][1] * gy + M[1][2] * gz;
        const scalar_t valz = M[2][0] * gx + M[2][1] * gy + M[2][2] * gz;

        element_outx[node] = valx;
        element_outy[node] = valy;
        element_outz[node] = valz;
    }}

    #undef IDX
    #undef D
}}
"""

        content.append(apply_fun)
        content.append("")
        content.append(f"#endif /* {GUARD} */\n")

        with open(out_path, "w") as f:
            f.write("\n".join(content))
        print(f"Wrote {out_path}")


def demo_emit_for_tet4_neohookean():
    strain_energy_function = "(mu/2)*(I1b - 3) + (lmbda/2)*(log(J))**2"
    fe = Tet4()
    op = SRHyperelasticity.create_from_string(fe, strain_energy_function)

    gen = PAKernelGenerator(op)
    gen.build_S_tensors()
    gen.emit_S_ikqm_assignments("S_ikqm")
    gen.emit_header("operators/tet4/tet4_partial_assembly_pa_inline.h",
                    func_name="tet4_S_ikqm",
                    guard="TET4_PARTIAL_ASSEMBLY_S_IKQM_INLINE_H",
                    tensor_name="S_ikqm")


if __name__ == "__main__":
    demo_emit_for_tet4_neohookean()
