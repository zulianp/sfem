# Elemental Assembly API

This note explains how to call the generated header-only NeoHookean dense local Hessian API:

```cpp
#include "sfem_GeneratedNeoHookeanOgden_element_api.hpp"

sfem::codegen::neohookean_ogden_hessian_2d_element_soa<real_t, VECTOR_SIZE>(
        element_type, nelements, coords, lmbda, mu, u_local, H_local);
```

The function evaluates dense element matrices only. It does not gather from global arrays, does not use connectivity, and does not scatter into CRS/BSR. The caller gathers local element data before the call and scatters or stores the dense matrices after the call.

## Layout

Inputs and outputs are SoA arrays over a batch of elements.

```text
DIM = 2
N_SHAPE = nodes per element
NDOFS = DIM * N_SHAPE

local dof = shape * DIM + component
coords[dof][lane]
u_local[dof][lane]
H_local[row * NDOFS + col][lane]
```

`NDOFS` is the active number of local degrees of freedom for the current element. `MAX_NDOFS` is only a compile-time capacity used by fixed scratch arrays; it must be greater than or equal to `NDOFS`, but it is not used in element indexing formulas.

For example, in 2D:

```text
dof 0 = shape 0, x
dof 1 = shape 0, y
dof 2 = shape 1, x
dof 3 = shape 1, y
...
```

`H_local` stores a full dense `NDOFS x NDOFS` matrix in row-major local-dof order, with one vector lane per element in the batch.

## SoA, AoS, And Vectorization

SoA means "structure of arrays": each local coordinate, state, or matrix entry has one contiguous array for all elements in the batch. This is the layout used by the generated kernels because it exposes independent element lanes to the compiler for SIMD vectorization.

AoS means "array of structures": one complete dense matrix is stored next to the next complete dense matrix:

```text
element_matrix_aos[e][row][col]
```

AoS is often more convenient for users after assembly. A common pattern is:

1. Gather global data into thread-local SoA batch buffers.
2. Call the generated `*_element_soa` function.
3. Convert the active `H_local[row * ndofs + col][lane]` entries to `element_matrix_aos[element][row][col]`.

## Minimal Batched Call

```cpp
static constexpr int VECTOR_SIZE = 16;
static constexpr int MAX_NDOFS = 81; // Scratch capacity, not the active element NDOFS.

const int dim = mesh->spatial_dimension(); // Example mesh-manager call
const int nshape = mesh->n_nodes_per_element(0); // Example mesh-manager call
const int ndofs = dim * nshape; // Active NDOFS for this element.
const int nelems = batch_nelems;

if (ndofs > MAX_NDOFS) {
    return SFEM_FAILURE;
}

real_t coords_soa[MAX_NDOFS][VECTOR_SIZE];
real_t u_soa[MAX_NDOFS][VECTOR_SIZE];
real_t hessian_soa[MAX_NDOFS * MAX_NDOFS][VECTOR_SIZE];

// Caller gathers coordinates and displacement/current state.
// dof = shape * dim + component.
for (int shape = 0; shape < nshape; ++shape) {
    for (int d = 0; d < dim; ++d) {
        const int dof = shape * dim + d;
        for (int lane = 0; lane < nelems; ++lane) {
            const idx_t node = gathered_element_nodes[lane][shape];
            coords_soa[dof][lane] = points[d][node];
            u_soa[dof][lane] = u[node * dim + d];
        }
    }
}

int status = SFEM_FAILURE;
// `coords`, `u_local`, and `H_local` are the generated API views over
// coords_soa, u_soa, and hessian_soa. The reference example shows the binding.
if (dim == 2) {
    status = sfem::codegen::neohookean_ogden_hessian_2d_element_soa<real_t, VECTOR_SIZE>(
            element_type, nelems, coords, lmbda, mu, u_local, H_local);
} else if (dim == 3) {
    status = sfem::codegen::neohookean_ogden_hessian_3d_element_soa<real_t, VECTOR_SIZE>(
            element_type, nelems, coords, lmbda, mu, u_local, H_local);
}

if (status != SFEM_SUCCESS) {
    // Unsupported element type or invalid call.
}
```

Choose `MAX_NDOFS` once for the largest element family you support. The fixed `*_soa` arrays are the only local storage capacity. All loops and dense matrix indexing use the active `ndofs` for the current element.

## SoA To AoS

```cpp
for (int row = 0; row < ndofs; ++row) {
    for (int col = 0; col < ndofs; ++col) {
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t e = batch_begin + lane;
            element_matrix_aos[e * ndofs * ndofs + row * ndofs + col] = hessian_soa[row * ndofs + col][lane];
        }
    }
}
```

## 3D Differences

Use the 3D dispatch wrapper:

```cpp
sfem::codegen::neohookean_ogden_hessian_3d_element_soa<real_t, VECTOR_SIZE>(
        element_type, nelems, coords, lmbda, mu, u_local, H_local);
```

The layout is the same, but:

```text
DIM = 3
NDOFS = 3 * N_SHAPE
dof = shape * 3 + component
```

Supported 2D dispatch elements include `TRI3`, `TRI6`, `QUAD4`, and `PROTEUS_QUAD4`. Supported 3D dispatch elements include `TET4`, `TET10`, `HEX8`, `HEX27`, and generated Proteus HEX variants.

For standard tensor-product meshes, pass gathered arrays in the standard SFEM local node order. The `QUAD4`, `HEX8`, and `HEX27` element API wrappers perform the pointer shuffle internally and delegate to their `PROTEUS_*` kernels. Use the explicit `PROTEUS_*` dispatch values only when your gathered arrays are already in Proteus order.

## Reference Example

See `drivers/bench/neohookean_assemble.exe.cpp` for a complete example that:

- creates a mesh,
- gathers thread-local SoA batches,
- uses fixed `MAX_NDOFS` storage and binds the generated API pointer views,
- calls the generated NeoHookean dense Hessian API,
- stores the result as global AoS dense element matrices,
- compares the result against current BSR assembly.
