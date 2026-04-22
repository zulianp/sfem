#!/usr/bin/env python3
# https://i10git.cs.fau.de/hyteg/hog/-/blob/main/hog/operator_generation/optimizer.py?ref_type=heads

import pystencils as ps
import pystencils.astnodes as ast
import pystencils.backends.cuda_backend as cuda_backend
import pystencils.gpu.indexing as gpu_indexing
from pystencils.cpu.vectorization import vectorize
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
    # Stage 1: vectorizable integrand kernel with create_kernel (CPU + CUDA)
    integrand, qp_x2d, qp_y2d, qp_w2d = ps.fields(
        "integrand, qp_x2d, qp_y2d, qp_w2d: [2D]"
    )
    integrand_expr = expr.subs({x_sym: qp_x2d[0, 0], y_sym: qp_y2d[0, 0]})
    integrand_assign = ps.Assignment(
        integrand[0, 0],
        integrand_expr * qp_w2d[0, 0],
    )

    cpu_config = ps.CreateKernelConfig(
        target=ps.Target.CPU,
        backend=ps.Backend.C,
        cpu_vectorize_info={"assume_aligned": True},
        cpu_blocking=block_size,
    )
    integrand_cpu_kernel = ps.create_kernel(integrand_assign, config=cpu_config)

    gpu_config = ps.CreateKernelConfig(
        target=ps.Target.GPU,
        backend=ps.Backend.CUDA,
    )
    integrand_cuda_kernel = ps.create_kernel(integrand_assign, config=gpu_config)

    # Stage 2: reduction kernel (batch sum over integrand field)
    real_t = BasicType("double")
    integrand_ptr = TypedSymbol("integrand", PointerType(real_t))
    result_ptr = TypedSymbol("result", PointerType(real_t))
    batch_count = TypedSymbol("n_batches", BasicType("int32"))

    integrand_arr = sp.IndexedBase(integrand_ptr, shape=(n_qp, n_qp))
    result = sp.IndexedBase(result_ptr, shape=(batch_count,))

    b_counter = TypedSymbol("b", BasicType("int32"))
    i_counter = TypedSymbol("i", BasicType("int32"))
    j_counter = TypedSymbol("j", BasicType("int32"))

    accumulate = SympyAssignment(
        result[b_counter],
        result[b_counter] + integrand_arr[i_counter, j_counter],
        is_const=False,
    )

    inner_loop = LoopOverCoordinate(
        Block([accumulate]),
        0,
        0,
        n_qp,
        custom_loop_ctr=j_counter,
    )
    outer_loop = LoopOverCoordinate(
        Block([inner_loop]),
        1,
        0,
        n_qp,
        custom_loop_ctr=i_counter,
    )
    init_stmt = SympyAssignment(result[b_counter], 0, is_const=False)
    batch_loop = LoopOverCoordinate(
        Block([init_stmt, outer_loop]),
        2,
        0,
        batch_count,
        custom_loop_ctr=b_counter,
    )

    reduction_kernel = KernelFunction(
        Block([batch_loop]),
        target=ps.Target.CPU,
        backend=ps.Backend.C,
        compile_function=None,
        ghost_layers=None,
        function_name=f"{name}_reduce",
    )
    cpu_block_size = (min(block_size[0], n_qp), min(block_size[1], n_qp), 0)
    tr.loop_blocking(reduction_kernel, block_size=cpu_block_size)

    cuda_thread_id = gpu_indexing.THREAD_IDX[0]
    cuda_block_id = gpu_indexing.BLOCK_IDX[0]
    cuda_block_dim = gpu_indexing.BLOCK_DIM[0]
    b_expr = cuda_block_id * cuda_block_dim + cuda_thread_id

    cuda_init = SympyAssignment(result[b_expr], 0, is_const=False)
    cuda_accum = SympyAssignment(
        result[b_expr],
        result[b_expr] + integrand_arr[i_counter, j_counter],
        is_const=False,
    )
    cuda_inner = LoopOverCoordinate(
        Block([cuda_accum]),
        0,
        0,
        n_qp,
        custom_loop_ctr=j_counter,
    )
    cuda_outer = LoopOverCoordinate(
        Block([cuda_inner]),
        1,
        0,
        n_qp,
        custom_loop_ctr=i_counter,
    )
    cuda_body = Block(
        [
            ast.Conditional(
                sp.Lt(b_expr, batch_count),
                Block([cuda_init, cuda_outer]),
                None,
            )
        ]
    )
    reduction_cuda_kernel = KernelFunction(
        cuda_body,
        target=ps.Target.GPU,
        backend=ps.Backend.CUDA,
        compile_function=None,
        ghost_layers=None,
        function_name=f"{name}_reduce_cuda",
    )
    reduction_cuda_kernel.global_variables = {cuda_thread_id, cuda_block_id, cuda_block_dim}
    reduction_cuda_kernel.indexing = gpu_indexing.BlockIndexing(
        iteration_space=(slice(0, batch_count),),
        data_layout=(0,),
        block_size=(cuda_block_size, 1, 1),
    )

    return (
        ps.get_code_str(integrand_cpu_kernel),
        ps.get_code_str(reduction_kernel),
        cuda_backend.generate_cuda(integrand_cuda_kernel),
        cuda_backend.generate_cuda(reduction_cuda_kernel),
    )


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

c_integrand, c_reduce, cuda_integrand, cuda_reduce = generate_quadrature_kernels(
    user_func,
    x_sym,
    y_sym,
    n_qp,
)

print("=" * 70)
print(f"Generated C code for quadrature integration: {n_qp}x{n_qp}")
print("=" * 70)
print(c_integrand)
print("=" * 70)

print("=" * 70)
print(f"Generated C reduction code for batch quadrature: {n_qp}x{n_qp}")
print("=" * 70)
print(c_reduce)
print("=" * 70)

print("=" * 70)
print(f"Generated CUDA integrand code: {n_qp}x{n_qp}")
print("=" * 70)
print(cuda_integrand)
print("=" * 70)

print("=" * 70)
print(f"Generated CUDA reduction code: {n_qp}x{n_qp}")
print("=" * 70)
print(cuda_reduce)
print("=" * 70)
# The generated C code expects the following arrays to be provided:
# static const scalar_t qp_x[n_qp] = {...};
# static const scalar_t qp_y[n_qp] = {...};
# static const scalar_t qp_w[n_qp] = {...};