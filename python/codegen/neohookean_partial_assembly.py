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



class PAKernelGenerator:
    def __init__(self, name, op: SRHyperelasticity, metric_tensor_only=False):
        self.name = name
        self.metric_tensor_only = metric_tensor_only
        op.partial_assembly()
        self.fe = op.fe
        self.expression_table = op.expression_table
        self.use_canonical = True
        self.S_ikmn_canonical_symb = op.S_ikmn_canonical_symb
        
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

    def detect_tensor_symmetries(self, tensor, tensor_name, nfun, dim):
        """
        Detect all symmetries of a 4th-order tensor with indices (i, m, p, n).
        
        Args:
            tensor: The 4th-order tensor to analyze
            tensor_name: String name for the tensor (for output)
            nfun: Number of function indices (i, p)
            dim: Number of spatial dimensions (m, n)
        """
        print(f"\n=== Symmetry Analysis for {tensor_name} ===")
        
        # Define all possible symmetry operations for a 4th-order tensor
        # Each symmetry is defined as a permutation of indices
        symmetries = {
            "Major symmetry (i,m,p,n) = (p,n,i,m)": (1, 2, 0, 3),  # (i,m,p,n) -> (p,n,i,m)
            "Minor symmetry in (m,n)": (0, 3, 2, 1),  # (i,m,p,n) -> (i,n,p,m)
            "Minor symmetry in (i,p)": (2, 1, 0, 3),  # (i,m,p,n) -> (p,m,i,n)
            "Full minor symmetry (i,m,p,n) = (i,n,p,m) = (p,m,i,n) = (p,n,i,m)": (1, 2, 0, 3),  # Combined
            "Transpose (i,m,p,n) = (p,m,i,n)": (2, 1, 0, 3),  # (i,m,p,n) -> (p,m,i,n)
            "Swap first pair (i,m,p,n) = (m,i,n,p)": (1, 0, 3, 2),  # (i,m,p,n) -> (m,i,n,p)
            "Swap second pair (i,m,p,n) = (i,p,m,n)": (0, 2, 1, 3),  # (i,m,p,n) -> (i,p,m,n)
            "Swap last pair (i,m,p,n) = (i,m,n,p)": (0, 1, 3, 2),  # (i,m,p,n) -> (i,m,n,p)
            "Reverse all (i,m,p,n) = (n,p,m,i)": (3, 2, 1, 0),  # (i,m,p,n) -> (n,p,m,i)
            "Reverse first two (i,m,p,n) = (m,i,p,n)": (1, 0, 2, 3),  # (i,m,p,n) -> (m,i,p,n)
            "Reverse last two (i,m,p,n) = (i,m,n,p)": (0, 1, 3, 2),  # (i,m,p,n) -> (i,m,n,p)
        }
        
        # Check each symmetry
        symmetry_results = {}
        
        for sym_name, permutation in symmetries.items():
            violations = []
            max_violations = 10  # Limit output to avoid spam
            
            for i in range(nfun):
                for m in range(dim):
                    for p in range(nfun):
                        for n in range(dim):
                            # Get original indices
                            orig_indices = (i, m, p, n)
                            
                            # Apply permutation
                            perm_indices = tuple(orig_indices[permutation[i]] for i in range(4))
                            
                            # Check if indices are valid
                            if (perm_indices[0] < 0 or perm_indices[0] >= nfun or
                                perm_indices[1] < 0 or perm_indices[1] >= dim or
                                perm_indices[2] < 0 or perm_indices[2] >= nfun or
                                perm_indices[3] < 0 or perm_indices[3] >= dim):
                                continue
                            
                            # Check if symmetry holds
                            diff = sp.simplify(tensor[orig_indices] - tensor[perm_indices])
                            if diff != 0:
                                if len(violations) < max_violations:
                                    violations.append({
                                        'original': orig_indices,
                                        'permuted': perm_indices,
                                        'difference': diff
                                    })
                                elif len(violations) == max_violations:
                                    violations.append("... (more violations exist)")
                                    break
                        if len(violations) >= max_violations:
                            break
                    if len(violations) >= max_violations:
                        break
                if len(violations) >= max_violations:
                    break
            
            symmetry_results[sym_name] = {
                'holds': len(violations) == 0,
                'violations': violations
            }
        
        # Print results
        for sym_name, result in symmetry_results.items():
            if result['holds']:
                print(f"✓ {sym_name}: HOLDS")
            else:
                print(f"✗ {sym_name}: VIOLATED")
                for i, violation in enumerate(result['violations']):
                    if isinstance(violation, dict):
                        print(f"    {violation['original']} ≠ {violation['permuted']}: {violation['difference']}")
                    else:
                        print(f"    {violation}")
        
        # Find unique entries (for compression analysis)
        print(f"\n=== Compression Analysis for {tensor_name} ===")
        self.analyze_tensor_compression(tensor, tensor_name, nfun, dim)
        
        print("=" * 50)

    def analyze_tensor_compression(self, tensor, tensor_name, nfun, dim):
        """
        Analyze how much the tensor can be compressed by exploiting symmetries.
        """
        # Count unique entries by exploiting major and minor symmetries
        unique_entries = set()
        total_entries = nfun * dim * nfun * dim
        
        for i in range(nfun):
            for m in range(dim):
                for p in range(nfun):
                    for n in range(dim):
                        # Create canonical form by sorting pairs
                        # Major symmetry: (i,m,p,n) = (p,n,i,m)
                        # Minor symmetry: (i,m,p,n) = (i,n,p,m) = (p,m,i,n) = (p,n,i,m)
                        
                        # Find the lexicographically smallest equivalent entry
                        candidates = [
                            (i, m, p, n),
                            (p, n, i, m),  # Major symmetry
                            (i, n, p, m),  # Minor symmetry
                            (p, m, i, n),  # Minor symmetry
                        ]
                        
                        # Sort each pair to get canonical form
                        canonical_candidates = []
                        for (a, b, c, d) in candidates:
                            # Sort first pair and second pair
                            pair1 = tuple(sorted([a, c]))
                            pair2 = tuple(sorted([b, d]))
                            canonical_candidates.append((pair1[0], pair2[0], pair1[1], pair2[1]))
                        
                        # Take the lexicographically smallest
                        canonical = min(canonical_candidates)
                        unique_entries.add(canonical)
        
        unique_count = len(unique_entries)
        compression_ratio = unique_count / total_entries
        savings = (total_entries - unique_count) / total_entries * 100
        
        print(f"Total entries: {total_entries}")
        print(f"Unique entries: {unique_count}")
        print(f"Compression ratio: {compression_ratio:.4f}")
        print(f"Memory savings: {savings:.2f}%")
        
        # Show some examples of unique entries
        # print(f"\nFirst 10 unique entries (canonical form):")
        # for i, entry in enumerate(sorted(unique_entries)[:10]):
        #     print(f"  {entry}")
        # if len(unique_entries) > 10:
        #     print(f"  ... and {len(unique_entries) - 10} more")

        for i, entry in enumerate(sorted(unique_entries)):
            print(f"  {entry}")

    def compress_tensor_unique_entries(self, tensor, tensor_name, nfun, dim):
        """
        Compress a 4th-order tensor by identifying unique entries only.
        No symmetry considerations - just pure unique value compression.
        
        Args:
            tensor: The 4th-order tensor to compress
            tensor_name: String name for the tensor
            nfun: Number of function indices (i, p)
            dim: Number of spatial dimensions (m, n)
            
        Returns:
            dict: Contains compressed data and mapping information
        """
        print(f"\n=== Unique Entry Compression for {tensor_name} ===")
        
        # Collect all unique values and their indices
        unique_values = {}
        index_map = {}
        compressed_idx = 0
        
        # First pass: collect all unique values
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
                        
                        # Map this index to the compressed index
                        index_map[(i, m, p, n)] = unique_values[value_str]['compressed_idx']
                        unique_values[value_str]['indices'].append((i, m, p, n))
        
        # Create compressed tensor array
        compressed_tensor = []
        for value_str, data in unique_values.items():
            compressed_tensor.append(data['value'])

        compressed_tensor_symb = sp.symbols(f"{tensor_name}_compressed[0:{len(unique_values)}]")
        print(compressed_tensor_symb)

        terms = []
        terms_test = []
        for _, v in index_map.items():
            terms.append(compressed_tensor_symb[v])
            terms_test.append(v)

        Wimpn_compressed = Array(terms, shape=(nfun, dim, nfun, dim))
        Wimpn_compressed_test = Array(terms_test, shape=(nfun, dim, nfun, dim))

        for i in range(nfun):
            for m in range(dim):
                for p in range(nfun):
                    for n in range(dim):
                        assert tensor[i,m,p,n] == compressed_tensor[Wimpn_compressed_test[i, m, p, n]]
        
        # Print compression statistics
        total_entries = nfun * dim * nfun * dim
        unique_count = len(unique_values)
        compression_ratio = unique_count / total_entries
        savings = (total_entries - unique_count) / total_entries * 100
        
        print(f"Total entries: {total_entries}")
        print(f"Unique entries: {unique_count}")
        print(f"Compression ratio: {compression_ratio:.4f}")
        print(f"Memory savings: {savings:.2f}%")
        
        # Show some examples of unique entries
        print(f"\nFirst 10 unique entries:")
        for i, (value_str, data) in enumerate(list(unique_values.items())[:10]):
            print(f"  [{data['compressed_idx']}] = {data['value']} (appears in {len(data['indices'])} positions)")

        if len(unique_values) > 10:
            print(f"  ... and {len(unique_values) - 10} more unique entries")
        
        return {
            'unique_values': unique_values,
            'index_map': index_map,
            'compressed_tensor': sp.Matrix(len(compressed_tensor), 1, compressed_tensor),
            'compression_ratio': compression_ratio,
            'memory_savings': savings,
            tensor_name : Wimpn_compressed
            # 'c_code': compressed_c_code
        }


    def emit_header(self, out_path, guard):
        dim = self.fe.spatial_dim()
        nfun = self.fe.n_nodes()
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

        # Wimpnq_symb = self._create_tensor4_symbol("Wimpnq", nfun, dim, nfun, dim)
        # body_Wimpnq = c_gen(self._assign_tensor4(Wimpnq_symb, self.expression_table["Wimpnq"]))
        # print(body_Wimpnq)

        compression_result = self.compress_tensor_unique_entries(self.expression_table["Wimpn"], "Wimpn", nfun, dim)
        Wimpn = compression_result["Wimpn"]

        syms = ['x', 'y', 'z']
        hki_terms = []
        for i in range(dim):
            hki_terms.append(sp.symbols(f"inc{syms[i]}[0:{nfun}]"))
        hki = Array(hki_terms, shape=(dim, nfun))


        Zpkmn_terms = []
        for p in range(nfun):
            for k in range(dim):
                for m in range(dim):
                    for n in range(dim):
                        acc = 0
                        for i in range(nfun):
                            acc += hki[k, i] * Wimpn[i, m, p, n]
                        Zpkmn_terms.append(acc)
        Zpkmn = Array(Zpkmn_terms, shape=(nfun, dim, dim, dim))


        # Partial generation    
        Zpkmn_symb = self._create_tensor4_symbol("Zpkmn", nfun, dim, dim, dim)

        Zpkmn_S = (
            f"static SFEM_INLINE void {elem_type_lc}_Zpkmn(\n"
            f"    const {real_t} *const SFEM_RESTRICT Wimpn_compressed,\n"
            f"    const {real_t} *const SFEM_RESTRICT incx,\n"
            f"    const {real_t} *const SFEM_RESTRICT incy,\n"
            f"    const {real_t} *const SFEM_RESTRICT incz,\n"
            f"    {real_t} *const SFEM_RESTRICT Zpkmn)\n"
        )
        Zpkmn_body = (
            f"{{\n"
            f"{c_gen(self._assign_tensor4(Zpkmn_symb, Zpkmn))}\n"
            f"}}\n"
        )
        
        Zpkmn_fun = Zpkmn_S + Zpkmn_body

        partial_out_terms = sp.zeros(dim, nfun)
        for i in range(dim):
            for p in range(nfun):
                acc = 0
                for k in range(dim):
                    for m in range(dim):
                        for n in range(dim):
                            acc += self.S_ikmn_canonical_symb[i,k,m,n] * Zpkmn_symb[p, k, m, n]

                partial_out_terms[i, p] = acc

        
        sig_SdotZ = (
            f"static SFEM_INLINE void {elem_type_lc}_SdotZ(\n"
            f"    const {real_t} *const SFEM_RESTRICT S_ikmn_canonical,\n"
            f"    const {real_t} *const SFEM_RESTRICT Zpkmn,\n"
            f"    {real_t} *const SFEM_RESTRICT       outx,\n"
            f"    {real_t} *const SFEM_RESTRICT       outy,\n"
            f"    {real_t} *const SFEM_RESTRICT       outz) "
            "\n"
        )

        out_symb = sp.zeros(dim, nfun)
        for i in range(dim):
            for j in range(nfun):
                out_symb[i, j] = sp.symbols(f"out{syms[i]}[{j}]")

        expr = []
        for i in range(dim):
            for j in range(nfun):
                expr.append(ast.Assignment(out_symb[i, j], partial_out_terms[i, j]))
        body_SdotZ = f'{{\n{c_gen(expr)} }}\n'
        SdotZ_fun = sig_SdotZ + body_SdotZ


        sig_expand_S = (
            f"static SFEM_INLINE void {elem_type_lc}_expand_S(\n"
            f"    const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,\n"
            f"    {real_t} *const SFEM_RESTRICT S_ikmn)\n"
        )

        body_expand_S = (
            f"{{\n"
            f"{c_gen(self._assign_tensor4(self._create_tensor4_symbol("S_ikmn", dim, dim, dim, dim), self.S_ikmn_canonical_symb))}\n"
            f"}}\n"
        )

        expand_S_fun = sig_expand_S + body_expand_S

        SdotZ_expanded_fun=( 
f"static SFEM_INLINE void {elem_type_lc}_SdotZ_expanded(\n"
            f"    const {real_t} *const SFEM_RESTRICT S_ikmn,\n"
            f"    const {real_t} *const SFEM_RESTRICT Zpkmn,\n"
            f"    {real_t} *const SFEM_RESTRICT       outx,\n"
            f"    {real_t} *const SFEM_RESTRICT       outy,\n"
            f"    {real_t} *const SFEM_RESTRICT       outz)\n") + f"""
{{
    static const int pstride = {dim} * {dim} * {dim};
    static const int ksize = {dim} * {dim} * {dim};
    
    for(int p = 0; p < {nfun}; p++) {{
        scalar_t acc[{dim}] = {{0}};
        const scalar_t * const SFEM_RESTRICT Zkmn = &Zpkmn[p * pstride];
        for(int i = 0; i < {dim}; i++) {{
            const scalar_t * const SFEM_RESTRICT Skmn = &S_ikmn[i * ksize];
            for(int k = 0; k < ksize; k++) {{
                acc[i] += Skmn[k] * Zkmn[k];
            }}
        }}

        outx[p] = acc[0];
        outy[p] = acc[1];
        outz[p] = acc[2];
        }}
}}
"""

        # Full generation
        out_terms = sp.zeros(dim, nfun)
        for i in range(dim):
            for p in range(nfun):
                acc = 0
                for k in range(dim):
                    for m in range(dim):
                        for n in range(dim):
                            acc += self.S_ikmn_canonical_symb[i,k,m,n] * Zpkmn[p, k, m, n]

                out_terms[i, p] = acc

        sig_Wimpn_compressed = (
            f"static SFEM_INLINE void {elem_type_lc}_Wimpn_compressed(\n"
            f"      {real_t} *const SFEM_RESTRICT Wimpn_compressed)\n"
        )

        body_Wimpn_compressed = (
            f"{{"
            f"{c_gen(self._assign_matrix("Wimpn_compressed", compression_result["compressed_tensor"]))}"
            f"}}\n"
        )

        Wimpn_compressed_fun = sig_Wimpn_compressed + body_Wimpn_compressed

        sig_S_ikmn_x_H_km_x_G_np = (
            f"static SFEM_INLINE void {elem_type_lc}_SdotHdotG(\n"
            f"    const {real_t} *const SFEM_RESTRICT S_ikmn_canonical,\n"
            f"    const {real_t} *const SFEM_RESTRICT Wimpn_compressed,\n"
            f"    const {real_t} *const SFEM_RESTRICT incx,\n"
            f"    const {real_t} *const SFEM_RESTRICT incy,\n"
            f"    const {real_t} *const SFEM_RESTRICT incz,\n"
            f"    {real_t} *const SFEM_RESTRICT       outx,\n"
            f"    {real_t} *const SFEM_RESTRICT       outy,\n"
            f"    {real_t} *const SFEM_RESTRICT       outz) "
            "\n"
        )

        out_symb = sp.zeros(dim, nfun)
        for i in range(dim):
            for j in range(nfun):
                out_symb[i, j] = sp.symbols(f"out{syms[i]}[{j}]")

        expr = []
        for i in range(dim):
            for j in range(nfun):
                expr.append(ast.Assignment(out_symb[i, j], out_terms[i, j]))
        body_S_ikmn_x_H_km_x_G_np = f'{{\n{c_gen(expr)} }}\n'

        S_ikmn_x_H_km_x_G_np_fun = sig_S_ikmn_x_H_km_x_G_np + body_S_ikmn_x_H_km_x_G_np

        sig_Hessian_from_S_ikmn = (
            f'static SFEM_INLINE void {elem_type_lc}_{self.name}_hessian_from_S_ikmn(\n'
            f'    const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,\n'
            f'    const {real_t} *const SFEM_RESTRICT Wimpn_compressed,\n'
            f'    {real_t} *const SFEM_RESTRICT       H)'
            f'\n'
        )


        H = sp.zeros(nfun * dim, nfun * dim)

        for p in range(0, nfun):
            for q in range(0, nfun):
                for i in range(0, dim):
                    for k in range(0, dim):
                        acc = 0
                        for m in range(0, dim):
                            for n in range(0, dim):
                                acc += self.S_ikmn_canonical_symb[i, k, m, n] * Wimpn[q, m, p, n]
                        H[i * nfun + p, k * nfun + q] = acc
        
        combined_code = c_gen(self._assign_matrix("H", H))

        body_Hessian_from_S_ikmn = (
            f'{{\n'
            f'{combined_code}\n'
            f'}}\n'
        )

        Hessian_from_S_ikmn_fun = sig_Hessian_from_S_ikmn + body_Hessian_from_S_ikmn


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
        content.append("\n#include \"sfem_macros.h\"\n")
        if not self.metric_tensor_only:
            content.append("")
            content.append(sigF)
            content.append(body_F)
            content.append("}\n")
            content.append("")

            content.append(Wimpn_compressed_fun)
            content.append(S_ikmn_x_H_km_x_G_np_fun)
            content.append("")

            content.append(sigR)
            content.append(body_R)
            content.append("}\n")
            content.append("")

            content.append(Zpkmn_fun)
            content.append(SdotZ_fun)
            content.append(expand_S_fun)
            content.append(SdotZ_expanded_fun)
            content.append(Hessian_from_S_ikmn_fun)

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


def neohookean_ogden(fe):
    name = "neohookean_ogden"
    strain_energy_function = "mu / 2 * (I1 - 3) - mu * log(J) + (lmbda/2) * log(J)**2"

    op = SRHyperelasticity.create_from_string(fe, name, strain_energy_function)

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
    op = SRHyperelasticity.create_from_string_unimodular(fe, name, strain_energy_function)
    
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
    neohookean_ogden(fe)
    # compressible_mooney_rivlin(fe)
