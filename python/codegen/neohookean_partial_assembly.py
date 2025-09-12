#!/usr/bin/env python3

import os

import sympy as sp
import sympy.codegen.ast as ast

from sfem_codegen import c_code, c_gen, real_t, matrix_coeff
from sympy import Array
from tet4 import Tet4
from hex8 import Hex8
from tri3 import Tri3
from tet10 import Tet10
from sr_hyperelasticity import SRHyperelasticity
from sympy import Array


# This is for non HEX elements (use sum factorization for HEX)
build_Z_pikmn_body="""
for(int qp = 0; qp < nqp; qp++) {{
    scalar_t Hkm[9];
    scalar_t gx[NFUN];
    scalar_t gy[NFUN];
    scalar_t gz[NFUN];
    const scalar_t qx = q_qx[qp];
    const scalar_t qy = q_qy[qp];
    const scalar_t qz = q_qz[qp];

    // Compute grads and Hkm at qp
    {COMPUTE_G_AND_H}
  
    # Tensor product of 3 * NFUN with D*D
    for(int pi = 0; pi < NFUN; pi++) {{
        scalar_t * const SFEM_RESTRICT Z_kmnx = &Z_pikmn[pi * (D*D*D)];
        scalar_t * const SFEM_RESTRICT Z_kmny = &Z_kmnx[1];
        scalar_t * const SFEM_RESTRICT Z_kmnz = &Z_kmnx[2];
        for(int km = 0; km < (D*D); km++) {{
            Z_kmnx[km * 3] += gx[pi] * Hkm[km];
            Z_kmny[km * 3] += gy[pi] * Hkm[km];
            Z_kmnz[km * 3] += gz[pi] * Hkm[km];
        }}
    }}
}}
"""

# Expanded S (Can this be optimized?)
# S = (i, k, m, n) == (k, i, n, m)
SdotZ_body="""
for(int p = 0; p < n_fun; p++) {
    scalar_t acc[D] = {0};
    const scalar_t * const SFEM_RESTRICT Zpi = &Z_pikmn[Z_IDX(p, 0, 0, 0, 0)];
    for(int i = 0; i < D; i++) {
        const scalar_t * const SFEM_RESTRICT Si = &S[S_IDX(i, 0, 0, 0)];
        for(int k = 0; k < D * D * D; k++) {
            acc[i] += Si[iter] * Zpi[iter];
        }
    }

    outx[p] += acc[0];
    outy[p] += acc[1];
    outz[p] += acc[2];
}
"""


class PAKernelGenerator:
    def __init__(self, name, op: SRHyperelasticity, metric_tensor_only=False):
        self.name = name
        self.metric_tensor_only = metric_tensor_only
        op.partial_assembly()
        self.fe = op.fe
        self.expression_table = op.expression_table
        self.use_canonical = True
        
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

    def __create_tensor4_symbol_packed(self, name):
        dim = self.fe.spatial_dim()
        N = (dim**2+1)*(dim**2)/2
        packed = [sp.symbols(f"{name}[{i}]") for i in range(int(N))]
        return packed

    def __assign_tensor4_packed(self, var, mat):
        N = len(mat)

        assert len(var) == N, "Variable and matrix must have the same length"
        expr = []
        for i in range(N):
            expr.append(ast.Assignment(var[i], mat[i]))
        return expr

    def emit_header(self, out_path, guard):
        dim = self.fe.spatial_dim()
        elem_type_lc = self.fe.name().lower()
        tensor_name = "S_ikmn"
        S_lin_name = "S_lin"
        S_lin = self.expression_table[S_lin_name]
        S_ikmn = self.expression_table[tensor_name]
        Fmat = self.expression_table["F"]
        inc_grad = self.expression_table["inc_grad"]
      
        S_lin_symb = self._create_tensor4_symbol(S_lin_name, dim, dim, dim, dim)
        S_symb = self._create_tensor4_symbol(tensor_name, dim, dim, dim, dim)
        body_S = c_gen(self._assign_tensor4(S_symb, S_ikmn))

        body_F = c_gen(self._assign_matrix("F", Fmat))
        body_R = c_gen(self._assign_matrix("inc_grad", inc_grad))

        is_constant = True
        params_q = "" 
        if not isinstance(self.fe, Tet4) and not isinstance(self.fe, Tri3):
            is_constant = False
            params_q =  f"  const {real_t}            qx,\n" 
            params_q += f"  const {real_t}            qy,\n" if dim > 2 else ""
            params_q += f"  const {real_t}            qz,\n" if dim > 2 else ""

        sigF = (
            f"static SFEM_INLINE void {elem_type_lc}_F(\n"
            f"    const {real_t} *const SFEM_RESTRICT adjugate,\n"
            f"    const {real_t}                      jacobian_determinant,\n"
            f"{params_q}"
            f"    const {real_t} *const SFEM_RESTRICT dispx,\n"
            f"    const {real_t} *const SFEM_RESTRICT dispy,\n"
            f"    const {real_t} *const SFEM_RESTRICT dispz,\n"
            f"    {real_t} *const SFEM_RESTRICT       F) "
            "{\n"
        )

        sigR = (
            f"static SFEM_INLINE void {elem_type_lc}_ref_inc_grad(\n"
            f"{params_q}"
            f"    const {real_t} *const SFEM_RESTRICT incx,\n"
            f"    const {real_t} *const SFEM_RESTRICT incy,\n"
            f"    const {real_t} *const SFEM_RESTRICT incz,\n"
            f"    {real_t} *const SFEM_RESTRICT       inc_grad) "
            "{\n"
        )

        sigS = (
            f"static SFEM_INLINE void {elem_type_lc}_{tensor_name}_{self.name}(\n"
            f"    const {real_t} *const SFEM_RESTRICT adjugate,\n"
            f"    const {real_t}                      jacobian_determinant,\n"
            f"{params_q}"
            f"    const {real_t} *const SFEM_RESTRICT F,\n"
            f"    const {real_t}                      mu,\n"
            f"    const {real_t}                      lmbda,\n"
            f"    const {real_t}                      qw,\n"
            f"    {real_t} *const SFEM_RESTRICT       {tensor_name + "_canonical" if self.use_canonical else ""}) "
            "{\n"
        )

        content = []
        content.append(f"#ifndef {guard}")
        content.append(f"#define {guard}")
        if not self.metric_tensor_only:
            content.append("")
            content.append(sigF)
            content.append(body_F)
            content.append("}\n")
            content.append("")

            content.append(sigR)
            content.append(body_R)
            content.append("}\n")
            content.append("")

        if self.use_canonical:
            S_canonical_name = "S_ikmn_canonical"
            S_canonical = self.expression_table[S_canonical_name]
            S_canonical_symb = self.__create_tensor4_symbol_packed(S_canonical_name)
            body_S_canonical = c_gen(self.__assign_tensor4_packed(S_canonical_symb, S_canonical))
            content.append(f"#define {elem_type_lc.upper()}_{tensor_name.upper()}_SIZE {len(S_canonical)}")
            content.append(sigS)
            content.append(body_S_canonical)
            content.append("}\n")
            content.append("")
        else:
            body_S_lin = c_gen(self._assign_tensor4(S_lin_symb, S_lin))
            content.append(sigS)
            content.append(f"   {real_t} {S_lin_name}[{dim**4}]; // Check if used in SSA mode")
            content.append("    {")
            content.append(body_S_lin)
            content.append("    }")

            content.append(body_S)
            content.append("}\n")
            content.append("")


        if not self.metric_tensor_only:
            expr = []
            if not self.use_canonical:
                SdotH_km_decl = f"scalar_t SdotH_km[{dim*dim}];"
                expr.extend(self._assign_matrix("SdotH_km", self.expression_table["SdotH_km"]))
            else:
                SdotH_km_decl = ""

            expr.extend(self._assign_matrix("eoutx", self.expression_table["eoutx"]))
            expr.extend(self._assign_matrix("eouty", self.expression_table["eouty"]))
            expr.extend(self._assign_matrix("eoutz", self.expression_table["eoutz"]))
            body_apply = c_gen(expr)

            apply_fun = f"""
static SFEM_INLINE void {elem_type_lc}_apply_{tensor_name}(
    {params_q}const scalar_t *const SFEM_RESTRICT S_ikmn{f"_canonical" if self.use_canonical else ""},    // 3x3x3x3, includes dV
    const scalar_t *const SFEM_RESTRICT inc_grad,  // 3x3 reference trial gradient R
    scalar_t *const SFEM_RESTRICT       eoutx,
    scalar_t *const SFEM_RESTRICT       eouty,
    scalar_t *const SFEM_RESTRICT       eoutz)
{{
    {SdotH_km_decl}
    {body_apply}
}}
"""

            content.append(apply_fun)
        content.append("")
        content.append(f"#endif /* {guard} */\n")

        with open(out_path, "w") as f:
            f.write("\n".join(content))
        print(f"Wrote {out_path}")


def neohookean(fe):
    name = "neohookean"
    strain_energy_function = "mu / 2 * (I1 - 3) - mu * log(J) + (lmbda/2) * log(J)**2"

    op = SRHyperelasticity.create_from_string(fe, strain_energy_function)

    elem_type_lc = fe.name().lower()
    elem_type_uc = fe.name().upper()
    gen = PAKernelGenerator(name, op)

    output_dir = f"operators/{elem_type_lc}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    gen.emit_header(f"{output_dir}/{elem_type_lc}_partial_assembly_{name}_inline.h",
                    guard=f"SFEM_{elem_type_uc}_PARTIAL_ASSEMBLY_{name.upper()}_INLINE_H",)

def compressible_mooney_rivlin(fe):
    name = "compressible_mooney_rivlin"
    strain_energy_function = "C01 * (I2b - 3) + C10 * (I1b - 3) + 1/D1 * (J - 1)**2"
    op = SRHyperelasticity.create_from_string_unimodular(fe, strain_energy_function)
    
    elem_type_lc = fe.name().lower()
    elem_type_uc = fe.name().upper()
    gen = PAKernelGenerator(name, op)

    output_dir = f"operators/{elem_type_lc}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    gen.emit_header(f"{output_dir}/{elem_type_lc}_partial_assembly_{name}_inline.h",
                    guard=f"SFEM_{elem_type_uc}_PARTIAL_ASSEMBLY_{name.upper()}_INLINE_H",)



if __name__ == "__main__":
    # fe = Tet4()
    fe = Hex8()
    # fe = Tet10()
    neohookean(fe)
    compressible_mooney_rivlin(fe)
