#!/usr/bin/env python3
# https://i10git.cs.fau.de/hyteg/hog/-/blob/main/hog/operator_generation/optimizer.py?ref_type=heads

import pystencils as ps
import pystencils.astnodes as ast
import pystencils.backends.cuda_backend as cuda_backend
import pystencils.gpu.indexing as gpu_indexing
from pystencils.astnodes import (
    Block,
    LoopOverCoordinate,
    KernelFunction,
    SympyAssignment,
)
from pystencils.typing import BasicType, PointerType, TypedSymbol

# Example: Loop over a quadrature rule integrating a user-defined function using Pystencils
# This generates pure C code using Pystencils' native loop constructs

import sympy as sp
from sympy.integrals.quadrature import gauss_legendre
import pystencils.transformations as tr

def generate_quadrature_kernels(
    expr,
    x_sym,
    y_sym,
    n_qp,
    name="quadrature_integrate",
    block_size=(4, 4),
    cuda_block_size=128,
):
    real_t = BasicType("double")
    qp_x_ptr = TypedSymbol("qp_x", PointerType(real_t))
    qp_y_ptr = qp_x_ptr
    # qp_y_ptr = TypedSymbol("qp_y", PointerType(real_t))
    qp_w_ptr = TypedSymbol("qp_w", PointerType(real_t))
    result_ptr = TypedSymbol("result", PointerType(real_t))

    batch_count = TypedSymbol("n_batches", BasicType("int32"))

    qp_x = sp.IndexedBase(qp_x_ptr, shape=(n_qp,))
    qp_y = sp.IndexedBase(qp_y_ptr, shape=(n_qp,))
    qp_w = sp.IndexedBase(qp_w_ptr, shape=(n_qp,))
    result = sp.IndexedBase(result_ptr, shape=(batch_count,))

    b_counter = TypedSymbol("b", BasicType("int32"))
    i_counter = TypedSymbol("i", BasicType("int32"))
    j_counter = TypedSymbol("j", BasicType("int32"))

    expr_eval = expr.subs({x_sym: qp_x[i_counter], y_sym: qp_y[j_counter]})
    expr_eval = expr_eval * (qp_w[i_counter] * qp_w[j_counter])

    indexed_terms = list(expr_eval.atoms(sp.Indexed))
    precompute = []
    for term in indexed_terms:
        if term.base in (qp_x, qp_y):
            precompute.append(term * term)

    replacements, reduced = sp.cse(
        precompute + [expr_eval],
        symbols=sp.numbered_symbols("t"),
    )
    expr_reduced = reduced[-1]

    typed_map = {sym: TypedSymbol(str(sym), real_t) for sym, _ in replacements}
    replacements_typed = [(typed_map[sym], rhs.xreplace(typed_map)) for sym, rhs in replacements]
    expr_typed = expr_reduced.xreplace(typed_map)

    def _depends_on(idx_symbol, rhs_expr):
        for term in rhs_expr.atoms(sp.Indexed):
            if idx_symbol in term.indices:
                return True
        return False

    precompute_i = []
    precompute_j = []
    for lhs, rhs in replacements_typed:
        if _depends_on(i_counter, rhs):
            precompute_i.append(SympyAssignment(lhs, rhs, is_const=True))
        elif _depends_on(j_counter, rhs):
            precompute_j.append(SympyAssignment(lhs, rhs, is_const=True))
        else:
            precompute_i.append(SympyAssignment(lhs, rhs, is_const=True))
    accumulate_stmt = SympyAssignment(
        result[b_counter],
        result[b_counter] + expr_typed,
        is_const=False,
    )

    inner_loop = LoopOverCoordinate(
        Block(precompute_j + [accumulate_stmt]),
        0,
        0,
        n_qp,
        custom_loop_ctr=j_counter,
    )
    quadrature_loop = LoopOverCoordinate(
        Block(precompute_i + [inner_loop]),
        1,
        0,
        n_qp,
        custom_loop_ctr=i_counter,
    )

    init_stmt = SympyAssignment(result[b_counter], 0, is_const=False)
    batch_loop = LoopOverCoordinate(
        Block([init_stmt, quadrature_loop]),
        2,
        0,
        batch_count,
        custom_loop_ctr=b_counter,
    )

    kernel = KernelFunction(
        batch_loop,
        target=ps.Target.CPU,
        backend=ps.Backend.C,
        compile_function=None,
        ghost_layers=None,
        function_name=name,
    )

    cpu_block_size = (min(block_size[0], n_qp), min(block_size[1], n_qp), 0)
    tr.loop_blocking(kernel, block_size=cpu_block_size)
    tr.move_constants_before_loop(kernel)
    tr.cleanup_blocks(kernel)

    thread_id = gpu_indexing.THREAD_IDX[0]
    block_id = gpu_indexing.BLOCK_IDX[0]
    block_dim = gpu_indexing.BLOCK_DIM[0]
    b_expr = block_id * block_dim + thread_id

    # ---------------------------

    cuda_init = SympyAssignment(result[b_expr], 0, is_const=False)
    cuda_accumulate = SympyAssignment(
        result[b_expr],
        result[b_expr] + expr_typed,
        is_const=False,
    )

    cuda_inner_loop = LoopOverCoordinate(
        Block(precompute_j + [cuda_accumulate]),
        0,
        0,
        n_qp,
        custom_loop_ctr=j_counter,
    )
    cuda_outer_loop = LoopOverCoordinate(
        Block(precompute_i + [cuda_inner_loop]),
        1,
        0,
        n_qp,
        custom_loop_ctr=i_counter,
    )

    cuda_body = Block(
        [
            ast.Conditional(
                sp.Lt(b_expr, batch_count),
                Block([cuda_init, cuda_outer_loop]),
                Block([]),
            )
        ]
    )

    cuda_kernel = KernelFunction(
        cuda_body,
        target=ps.Target.GPU,
        backend=ps.Backend.CUDA,
        compile_function=None,
        ghost_layers=None,
        function_name=f"{name}_cuda",
    )
    cuda_kernel.global_variables = {thread_id, block_id, block_dim}
    cuda_kernel.indexing = gpu_indexing.BlockIndexing(
        iteration_space=(slice(0, batch_count),),
        data_layout=(0,),
        block_size=(cuda_block_size, 1, 1),
    )

    return ps.get_code_str(kernel), cuda_backend.generate_cuda(cuda_kernel)


# Generate a Gauss-Legendre quadrature rule for [0,1]
order = 13
x, w = gauss_legendre(order, 10)
n_qp = len(w)

# Transform from [-1,1] to [0,1] and normalize weights
x_transformed = [(xi + 1) / 2 for xi in x]
w_normalized = [wi / 2 for wi in w]  # Account for domain transformation [0,1]

# User-defined function to integrate: f(x, y) = x^2 + y^2 + x*y + 1
x_sym = sp.Symbol("x", real=True)
y_sym = sp.Symbol("y", real=True)
user_func = x_sym**2 + y_sym**2 + x_sym * y_sym + 1

c_code, cuda_code = generate_quadrature_kernels(
    user_func,
    x_sym,
    y_sym,
    n_qp,
)

print("=" * 70)
print(f"Generated C code for quadrature integration: {n_qp}x{n_qp}")
print("=" * 70)
print(c_code)
print("=" * 70)

print("=" * 70)
print(f"Generated CUDA code for batch quadrature: {n_qp}x{n_qp}")
print("=" * 70)
print(cuda_code)
print("=" * 70)

# The generated C code expects the following arrays to be provided:
# static const scalar_t qp_x[n_qp] = {...};
# static const scalar_t qp_y[n_qp] = {...};
# static const scalar_t qp_w[n_qp] = {...};