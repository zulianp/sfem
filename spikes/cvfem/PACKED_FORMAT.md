# The packed mesh format

This pins the contract that `smesh::PackedMesh<pack_idx_t>` actually provides, so that
CPU layouts, CUDA kernels, and any future reader or writer agree on it. Every claim
below was checked against `external/smesh/src/frontend/smesh_packed_mesh.cpp` rather
than inferred from usage; the anchors are given so they can be rechecked.

The on-disk layout is documented here **as it is** and is deliberately *not* extended.

## 1. What a pack is

A pack is a contiguous range of elements together with the nodes those elements touch.
Element `e` belongs to pack `p = e / elements_per_pack`; the assignment is by element
index and nothing else (`smesh_packed_mesh.cpp:277-279`).

That has a consequence worth stating as a precondition rather than an optimisation:
**the mesh must be spatially reordered before packing**, or a pack's elements will be
scattered and its node set will be huge. The bench does this with an SFC pass before
calling `make_packed` (`cvfem_hex8_ns_upwind_bench.cpp`, `smesh::SFC::create_from_env`).

`PackedMesh::create(mesh, {}, modify_mesh = true, pack_size)` **renumbers the mesh
nodes in place**. Anything derived from node numbering — coordinates, the node-to-node
graph, the BSR sparsity pattern — must be built *after* that call, never before. This
ordering is load-bearing in the bench's setup.

## 2. Node identity within a pack

Each node has exactly one owning pack. Pack `p` owns a contiguous, monotonically
increasing global range, so the owned ranges partition `[0, nnodes)`:

```
n_contiguous(p) = owned_nodes_ptr[p+1] - owned_nodes_ptr[p]
```

A pack-local id `l` resolves to a global id as:

| range of `l` | meaning | global id |
|---|---|---|
| `0 <= l < n_contiguous` | owned by this pack | `owned_nodes_ptr[p] + l` |
| `l >= n_contiguous` | ghost | `ghost_idx[ghost_ptr[p] + (l - n_contiguous)]` |

Reference implementation: `pack_local_to_global`, `cvfem_hex8_layout_common.hpp`.
Ghost ids start exactly at `n_contiguous` — `smesh_packed_mesh.cpp:303`
(`d_packed_elements[v][e] = nowned + d_ghost_map[node]`).

Because owned ranges are contiguous and monotone in `p`, the owning pack of a global
node id can be found by binary search over `owned_nodes_ptr`. `cvfem_pack_coloring.hpp`
relies on this.

## 3. The shared/non-shared split — the invariant a GPU flush depends on

This is the least obvious property of the format and the one most worth writing down.

Within a pack's owned range the ids are ordered **non-shared first, then shared**
(`smesh_packed_mesh.cpp:213-214`):

```cpp
ptrdiff_t owned_idx  = next_id;
ptrdiff_t shared_idx = next_id + (nowned - nshared);
```

so, in pack-local terms:

| range of `l` | property |
|---|---|
| `0 <= l < n_contiguous - n_shared[p]` | touched by **no other pack** |
| `n_contiguous - n_shared[p] <= l < n_contiguous` | also ghosted by at least one other pack |

**Consequence.** When flushing a pack's accumulator to the global array, the first
sub-range can be written with a plain non-atomic `+=`; only the shared tail and the
ghosts need atomics. `bench/cuda/bench_packed_laplacian.cu:235-244` is the reference
implementation of exactly this split.

This is also the answer to an open question from the earlier survey: `n_shared` is
carried in `PackedData` but has no consumer in the CPU CVFEM layouts, because there a
pack is one thread and the whole flush is already race-free. It acquires a consumer on
the GPU, where a pack is a block of many threads.

## 4. Ghost lists

`ghost_idx[ghost_ptr[p] .. ghost_ptr[p+1])` holds the global ids that pack `p` touches
but does not own. Two properties, both from `smesh_packed_mesh.cpp:284-308`:

- **Deduplicated** per pack, via a flag mask.
- **Ordered by first appearance** in a node-major, element-minor sweep
  (`for v { for e { ... } }`) — **not sorted**. Do not assume sortedness or binary-search
  into a ghost list.

## 5. The ghost reduction graph

`ghost_reduce_{ptr,idx,dest}` is a CSR gather that replaces atomics entirely
(`smesh_packed_mesh.cpp:313-352`). Row `r` means:

```
global_field[ghost_reduce_dest[r]] += sum over j in [ghost_reduce_ptr[r], ghost_reduce_ptr[r+1])
                                      of ghost_buf[ghost_reduce_idx[j]]
```

The builder sorts `(global_id, entry_index)` pairs and groups them, so **each destination
appears in exactly one row** and rows are ordered by destination. Two things follow:

- the reduction is race-free without atomics, and
- it is **bit-deterministic**, unlike an atomic flush.

That determinism is a property of the *reduction graph*, and it is worth having. But
note what it does and does not buy: it makes the ghost gather reproducible, not the
whole residual. A GPU implementation that accumulates a pack's element contributions
with `atomicAdd` into shared memory is still non-reproducible run to run, because those
shared-memory atomics fix no order either -- measured, on the implementation in
cuda/cvfem_hex8_ns_cuda.cu, which is bit-unstable in both flush modes for exactly this
reason.

Making a GPU residual genuinely bit-reproducible therefore needs the *in-pack*
accumulation ordered as well, for instance by gathering per node from a pack-local
node-to-element adjacency instead of scattering per element. That is a real design
option, not a free consequence of using the two-pass flush.

## 6. Sizes — one trap

`PackedMesh::max_nodes_per_pack()` is **not a measured maximum**. It is the compile-time
constant `std::numeric_limits<pack_idx_t>::max() + 1` (`smesh_packed_mesh.cpp:357`),
i.e. 65,536 for `uint16_t`. It is the bound the packer sizes packs against, not the
number of nodes any pack actually has.

Anything sizing scratch or GPU shared memory must compute its own maximum:

```
max over p of  (owned_nodes_ptr[p+1] - owned_nodes_ptr[p]) + (ghost_ptr[p+1] - ghost_ptr[p])
```

`PackedData::max_actual_nodes_per_pack` (`cvfem_hex8_layout_common.hpp`) is that value.
Using the 65,536 constant to size shared memory would ask for 16 MB per block.

`pack_idx_t` also caps how large a pack may be: a pack may never reach more than
`max()+1` distinct nodes, which is what bounds `elements_per_pack`.

## 7. Element connectivity

`elements(block)` is SoA: `pack_idx_t **`, extent `[nodes_per_element][n_elements]`,
holding **pack-local** ids. Reading node `v` of element `e` is `elems[v][e]`, which is
the coalescing-friendly order for a GPU where `e` maps to the thread index.

## 8. Matrix layouts built on top (spike-local)

The node partition above supports two different pack-local sparse layouts. They are
distinct and both in use, so a spec that names only one is incomplete:

- **compact** — `local_rowptr` / `local_colidx` / `local_global_slot`
  (`cvfem_hex8_layout_packed.hpp`). Every row of the pack gets a compact pattern; each
  local nonzero maps to a global block id.
- **store** — owned rows adopt the **global** row pattern, so a pack's owned blocks are
  one contiguous slice of the global values array and can be flushed with a single
  streaming write, no zeroing pass (`cvfem_hex8_layout_store.hpp`).

Neither is part of the on-disk format; both are derived at load time.

## 9. On-disk layout (frozen)

Written by `PackedMesh::write` and `Block::write` (`smesh_packed_mesh.cpp:31-43` and the
`write` member), driver `external/smesh/src/drivers/conversions/mesh_to_packed.exe.cpp`.

Raw binary arrays, one per file, no header and no version field; the element count is
implied by file size / `sizeof(T)`. The type name is part of the filename, which is what
makes the files self-describing enough to read back.

At the top level:

| file | type |
|---|---|
| `node_map.<idx_t>` | `idx_t` |
| `x%d.<geom_t>` | `geom_t`, one file per coordinate |

Then per block, in a subdirectory named after the block:

| file | type |
|---|---|
| `i%d.<pack_idx_t>` | `pack_idx_t`, one file per element-local node |
| `owned_nodes_ptr.<ptrdiff_t>` | `ptrdiff_t` |
| `n_shared.<ptrdiff_t>` | `ptrdiff_t` |
| `ghost_ptr.<ptrdiff_t>` | `ptrdiff_t` |
| `ghost_idx.<idx_t>` | `idx_t` |
| `ghost_reduce_ptr.<ptrdiff_t>` | `ptrdiff_t`, optional |
| `ghost_reduce_idx.<ptrdiff_t>` | `ptrdiff_t`, optional |
| `ghost_reduce_dest.<idx_t>` | `idx_t`, optional |

The `ghost_reduce_*` triple is written only when it was built.

**Do not extend this format.** Everything a device path needs beyond it — flattened
CRS, precomputed BSR slots, boundary-face masks, colourings — is cheap to recompute at
load time and should be, rather than being persisted and versioned.

## 10. Known-unresolved divergence

`cvfem_hex8_ns_steady.cpp` defines its own `MeshData` and `BSR4`, distinct from the
bench's. Steady's carry `Lx/Ly/Lz`, the nodal pressure gradient `pgx/pgy/pgz`,
`rhie_chow_scale`, and `diag_slots`, which the bench has no notion of.

Only the **pack machinery** is unified (`cvfem_hex8_pack_common.hpp`). Unifying
`MeshData` and `BSR4` would mean either giving the bench fields it does not use or
teaching the solver the bench's layout enums, and neither is justified by the CUDA port.
Recorded here so the divergence is a decision rather than an oversight.
