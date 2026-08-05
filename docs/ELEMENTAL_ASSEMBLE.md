# Elemental Assembly API

This note explains how to call the generated header-only NeoHookean dense local Hessian API:

```cpp
#include "sfem_GeneratedNeoHookeanOgden_element_api.hpp"

sfem::codegen::neohookean_ogden_hessian_2d_element_soa<real_t, VECTOR_SIZE>(
        element_type, nelements, coords, lmbda, mu, u_streams, matrix_streams);
```

The function evaluates dense element matrices only. It does not gather from global arrays, does not use connectivity, and does not scatter into CRS/BSR. The caller gathers local element data before the call and scatters or stores the dense matrices after the call.

## Layout

Inputs and outputs are SoA streams over a batch of elements.

```text
DIM = 2
N_SHAPE = nodes per element
NDOFS = DIM * N_SHAPE

local dof stream = shape * DIM + component
coords[stream][element_lane]
u_streams[stream][element_lane]
matrix_streams[row * NDOFS + col][element_lane]
```

For example, in 2D:

```text
stream 0 = shape 0, x
stream 1 = shape 0, y
stream 2 = shape 1, x
stream 3 = shape 1, y
...
```

`matrix_streams` stores a full dense `NDOFS x NDOFS` matrix in row-major local-dof order, but each matrix entry is a vector stream over the batch.

## SoA, AoS, And Vectorization

SoA means "structure of arrays": each local coordinate, state, or matrix entry has one contiguous stream for all elements in the batch. This is the layout used by the generated kernels because it exposes independent element lanes to the compiler for SIMD vectorization.

AoS means "array of structures": one complete dense matrix is stored next to the next complete dense matrix:

```text
element_matrix_aos[e][row][col]
```

AoS is often more convenient for users after assembly. A common pattern is:

1. Gather global data into thread-local SoA batch buffers.
2. Call the generated `*_element_soa` function.
3. Convert `matrix_streams[row * NDOFS + col][lane]` to `element_matrix_aos[element][row][col]`.

## Minimal Batched Call

```cpp
static constexpr int VECTOR_SIZE = 16;
const int DIM = 2;
const int N_SHAPE = 3;        // TRI3
const int NDOFS = DIM * N_SHAPE;
const int nelems = batch_nelems;

real_t coord_storage[NDOFS * VECTOR_SIZE];
real_t state_storage[NDOFS * VECTOR_SIZE];
real_t matrix_storage[NDOFS * NDOFS * VECTOR_SIZE];

const real_t *coords[NDOFS];
const real_t *u_streams[NDOFS];
real_t *matrix_streams[NDOFS * NDOFS];

for (int s = 0; s < NDOFS; ++s) {
    coords[s] = coord_storage + s * VECTOR_SIZE;
    u_streams[s] = state_storage + s * VECTOR_SIZE;
}

for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
    matrix_streams[entry] = matrix_storage + entry * VECTOR_SIZE;
}

// Caller gathers coordinates and displacement/current state.
// stream = shape * DIM + component.
for (int shape = 0; shape < N_SHAPE; ++shape) {
    for (int d = 0; d < DIM; ++d) {
        const int stream = shape * DIM + d;
        for (int lane = 0; lane < nelems; ++lane) {
            const idx_t node = gathered_element_nodes[lane][shape];
            coord_storage[stream * VECTOR_SIZE + lane] = points[d][node];
            state_storage[stream * VECTOR_SIZE + lane] = u[node * DIM + d];
        }
    }
}

const int status =
        sfem::codegen::neohookean_ogden_hessian_2d_element_soa<real_t, VECTOR_SIZE>(
                smesh::TRI3, nelems, coords, lmbda, mu, u_streams, matrix_streams);

if (status != SFEM_SUCCESS) {
    // Unsupported element type or invalid call.
}
```

## SoA To AoS

```cpp
for (int row = 0; row < NDOFS; ++row) {
    for (int col = 0; col < NDOFS; ++col) {
        const real_t *stream = matrix_streams[row * NDOFS + col];
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t e = batch_begin + lane;
            element_matrix_aos[e * NDOFS * NDOFS + row * NDOFS + col] = stream[lane];
        }
    }
}
```

## 3D Differences

Use the 3D dispatch wrapper:

```cpp
sfem::codegen::neohookean_ogden_hessian_3d_element_soa<real_t, VECTOR_SIZE>(
        element_type, nelems, coords, lmbda, mu, u_streams, matrix_streams);
```

The layout is the same, but:

```text
DIM = 3
NDOFS = 3 * N_SHAPE
stream = shape * 3 + component
```

Supported 2D dispatch elements include `TRI3`, `TRI6`, `QUAD4`, and `PROTEUS_QUAD4`. Supported 3D dispatch elements include `TET4`, `TET10`, `HEX8`, `HEX27`, and generated Proteus HEX variants.

For standard tensor-product meshes, make sure the local node order passed to the element API matches the generated element specialization. Existing SFEM mesh-level wrappers may internally reorder `QUAD4`, `HEX8`, and `HEX27` to Proteus order before calling tensor-product kernels.

## Reference Example

See `drivers/bench/neohookean_assemble.exe.cpp` for a complete example that:

- creates a mesh,
- gathers thread-local SoA batches,
- calls the generated NeoHookean dense Hessian API,
- stores the result as global AoS dense element matrices,
- compares the result against current BSR assembly.
