#ifndef CVFEM_HEX8_LAYOUT_ATOMIC_HPP
#define CVFEM_HEX8_LAYOUT_ATOMIC_HPP

// Atomic layout: a flat parallel sweep over elements that writes into the global
// residual / matrix with #pragma omp atomic on every entry. No mesh partitioning
// and no scratch, which makes it the simplest and the reference for correctness,
// but assembly pays ~1024 atomic read-modify-writes per element.

#include "cvfem_hex8_layout_common.hpp"

static SFEM_NOINLINE void apply_jacobian_action_atomic(MeshData             &d,
                                                       const scalar_t        rho,
                                                       const scalar_t        mu,
                                                       const scalar_t *const dir,
                                                       scalar_t *const       jv) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g         = d.elems[a][e];
            const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * N_FIELDS;
            vx[a]                            = dv[0];
            vy[a]                            = dv[1];
            vz[a]                            = dv[2];
            q[a]                             = dv[3];
        }
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_jacobian_action_atomic_isoparam(MeshData             &d,
                                                                const scalar_t        rho,
                                                                const scalar_t        mu,
                                                                const scalar_t *const dir,
                                                                scalar_t *const       jv) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t                  g  = d.elems[a][e];
            const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * N_FIELDS;
            vx[a]                                  = dv[0];
            vy[a]                                  = dv[1];
            vz[a]                                  = dv[2];
            q[a]                                   = dv[3];
        }
        cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, x, y, z, ux, uy, uz, vx, vy, vz, q, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_residual(rho, mu, adj, det, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_sumfact(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_isoparam(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, x, y, z, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_sympy(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_residual(rho, mu, adj, det, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_fd(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.empty() ? nullptr : b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, adj, det, ux, uy, uz, p, ke);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t row = d.elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const smesh::count_t slot =
                        slots ? slots[(size_t)e * 64 + a * 8 + bnode] : find_bsr_slot(b.rowptr, b.colidx, row, d.elems[bnode][e]);
                scalar_t *const      blk  = values + (ptrdiff_t)slot * 16;
                for (int rf = 0; rf < 4; ++rf) {
                    for (int cf = 0; cf < 4; ++cf) {
                        const scalar_t v = ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)];
                        CVFEM_ATOMIC_ADD(blk[rf * 4 + cf], v);
                    }
                }
            }
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_block(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_row(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_face(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

// Split assembly: the viscous part of the Jacobian depends only on the mesh and mu, so
// in a Newton loop it is the same matrix every iteration. Build it once, then each
// iteration restore it and add only the velocity-dependent terms.
//
// `linear` is a buffer of the same shape as b.values. The pair is exact, not an
// approximation: linear + nonlinear reproduces assemble_jacobian_atomic_sumfact
// bit-for-bit, because they are the two halves of the same kernel.
static SFEM_NOINLINE void assemble_jacobian_atomic_linear(MeshData             &d,
                                                          BSR4                 &b,
                                                          const scalar_t        mu,
                                                          std::vector<scalar_t> &linear) {
    linear.assign((size_t)b.nnz * 16, scalar_t(0));
    scalar_t *const SFEM_RESTRICT             values = linear.data();
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_add_slots_linear<true>(mu, adj, det,
                                                             slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_nonlinear(MeshData                    &d,
                                                             BSR4                        &b,
                                                             const scalar_t               rho,
                                                             const scalar_t               mu,
                                                             const std::vector<scalar_t> &linear) {
    scalar_t *const SFEM_RESTRICT             values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

    // Restore the constant part. A streaming copy, in place of the scattered
    // accumulation it replaces.
    std::memcpy(values, linear.data(), linear.size() * sizeof(scalar_t));

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_add_slots_nonlinear<true>(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sumfact(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT                 values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT     slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_add_slots<true>(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_isoparam(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT             values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                rho, mu, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

#endif  // CVFEM_HEX8_LAYOUT_ATOMIC_HPP
