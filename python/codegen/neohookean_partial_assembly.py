#!/usr/bin/env python3


import sympy as sp
import sympy.codegen.ast as ast

from sfem_codegen import c_code, c_gen, real_t, matrix_coeff
from sympy import Array
from tet4 import Tet4
from sr_hyperelasticity import SRHyperelasticity
from sympy import Array


class PAKernelGenerator:
    def __init__(self, op: SRHyperelasticity):
        op.partial_assembly()
        self.fe = op.fe
        self.expression_table = op.expression_table
        
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

    def emit_header(self, out_path, guard):
        dim = self.fe.spatial_dim()
        tensor_name = "S_ikmn"
        S_lin_name = "S_lin"
        S_lin = self.expression_table[S_lin_name]
        S_ikmn = self.expression_table[tensor_name]
        Fmat = self.expression_table["F"]
        inc_grad = self.expression_table["inc_grad"]
      
        S_lin_symb = self._create_tensor4_symbol(S_lin_name, dim, dim, dim, dim)
        body_S_lin = c_gen(self._assign_tensor4(S_lin_symb, S_lin))

        S_symb = self._create_tensor4_symbol(tensor_name, dim, dim, dim, dim)
        body_S = c_gen(self._assign_tensor4(S_symb, S_ikmn))

        body_F = c_gen(self._assign_matrix("F", Fmat))
        body_R = c_gen(self._assign_matrix("inc_grad", inc_grad))

        sigF = (
            "static SFEM_INLINE void tet4_F(\n"
            f"    const {real_t} *const SFEM_RESTRICT adjugate,\n"
            f"    const {real_t}                      jacobian_determinant,\n"
            f"    const {real_t} *const SFEM_RESTRICT dispx,\n"
            f"    const {real_t} *const SFEM_RESTRICT dispy,\n"
            f"    const {real_t} *const SFEM_RESTRICT dispz,\n"
            f"    {real_t} *const SFEM_RESTRICT       F) "
            "{\n"
        )

        sigR = (
            "static SFEM_INLINE void tet4_ref_inc_grad(\n"
            f"    const {real_t} *const SFEM_RESTRICT incx,\n"
            f"    const {real_t} *const SFEM_RESTRICT incy,\n"
            f"    const {real_t} *const SFEM_RESTRICT incz,\n"
            f"    {real_t} *const SFEM_RESTRICT       inc_grad) "
            "{\n"
        )

        sigS = (
            f"static SFEM_INLINE void tet4_{tensor_name}(\n"
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
        content.append(f"#ifndef {guard}")
        content.append(f"#define {guard}")
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
        content.append(f"   {real_t} {S_lin_name}[{dim**4}]; // Check if used in SSA mode")
        content.append("    {")
        content.append(body_S_lin)
        content.append("    }")

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
        # grad_decl = "scalar_t gradx[{}];\nscalar_t grady[{}];\nscalar_t gradz[{}];\n".format(nfun, nfun, nfun)
        # for node in range(nfun):
        #     grad_assign.append(ast.Assignment(sp.symbols(f"gradx[{node}]"), g[node][0]))
        #     grad_assign.append(ast.Assignment(sp.symbols(f"grady[{node}]"), g[node][1]))
        #     grad_assign.append(ast.Assignment(sp.symbols(f"gradz[{node}]"), g[node][2]))
        # grad_code = c_gen(grad_assign)


        expr = []
        expr.extend(self._assign_matrix("SdotH_km", self.expression_table["SdotH_km"]))
        expr.extend(self._assign_matrix("eoutx", self.expression_table["eoutx"]))
        expr.extend(self._assign_matrix("eouty", self.expression_table["eouty"]))
        expr.extend(self._assign_matrix("eoutz", self.expression_table["eoutz"]))
        body_apply = c_gen(expr)

        apply_fun = f"""
static SFEM_INLINE void tet4_apply_{tensor_name}(
    const scalar_t *const SFEM_RESTRICT S_ikmn,    // 3x3x3x3, includes dV
    const scalar_t *const SFEM_RESTRICT inc_grad,  // 3x3 reference trial gradient R
    scalar_t *const SFEM_RESTRICT       eoutx,
    scalar_t *const SFEM_RESTRICT       eouty,
    scalar_t *const SFEM_RESTRICT       eoutz)
{{
    scalar_t SdotH_km[{dim*dim}];
    {body_apply}
}}
"""

        content.append(apply_fun)
        content.append("")
        content.append(f"#endif /* {guard} */\n")

        with open(out_path, "w") as f:
            f.write("\n".join(content))
        print(f"Wrote {out_path}")


def demo_emit_for_tet4_neohookean():
    strain_energy_function = "mu / 2 * (I1 - 3) - mu * log(J) + (lmbda/2) * log(J)**2"
    fe = Tet4()
    op = SRHyperelasticity.create_from_string_F(fe, strain_energy_function)

    gen = PAKernelGenerator(op)
    gen.emit_header("operators/tet4/tet4_partial_assembly_neohookean_inline.h",
                    guard="SFEM_TET4_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H",)


if __name__ == "__main__":
    demo_emit_for_tet4_neohookean()
