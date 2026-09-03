// What fraction of the assembled matrix does the velocity-dependent part ever touch?
// If it is small, the linear-only entries are written once and never again -- and the
// per-iteration restore only has to cover the rest.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <mpi.h>
#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    const int n = argc > 1 ? atoi(argv[1]) : 32;
    for (int i = 0; i < 64; ++i) g_identity_slots[i] = i;

    MeshData d;
    d.mesh = smesh::Mesh::create_hex8_cube(smesh::Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
    auto sfc = smesh::SFC::create_from_env(); sfc->reorder(*d.mesh);
    d.nnodes = d.mesh->n_nodes(); d.nelements = d.mesh->n_elements(0);
    d.elems = d.mesh->elements(0)->data(); d.points = d.mesh->points()->data();
    fill_fields(d); precompute_affine_geometry(d);
    BSR4 bsr = make_bsr4(d.mesh);
    precompute_element_bsr_slots(d, bsr);

    const scalar_t rho = 1.0, mu = 0.01;
    const size_t   N   = (size_t)bsr.nnz * 16;

    // Linear part alone.
    std::vector<scalar_t> lin;
    assemble_jacobian_atomic_linear(d, bsr, mu, lin);

    // Nonlinear alone: assemble the split against a zero linear buffer.
    std::vector<scalar_t> zero(N, 0.0), nl_only;
    assemble_jacobian_atomic_nonlinear(d, bsr, rho, mu, zero);
    nl_only.assign(bsr.values->data(), bsr.values->data() + N);

    // Full, for reference.
    assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
    const scalar_t *full = bsr.values->data();

    size_t nz_lin = 0, nz_nl = 0, both = 0, nz_full = 0, chk = 0;
    for (size_t i = 0; i < N; ++i) {
        const bool a = lin[i] != 0.0, b = nl_only[i] != 0.0;
        nz_lin += a; nz_nl += b; both += (a && b); nz_full += (full[i] != 0.0);
        if (std::fabs((lin[i] + nl_only[i]) - full[i]) > 1e-14) ++chk;
    }
    std::printf("n=%d  nnz blocks=%td  matrix entries=%zu\n", n, bsr.nnz, N);
    std::printf("  linear-only touched   : %8zu  (%5.1f%%)\n", nz_lin - both, 100.0 * (nz_lin - both) / N);
    std::printf("  nonlinear touched     : %8zu  (%5.1f%%)  <- all that ever needs re-touching\n",
                nz_nl, 100.0 * nz_nl / N);
    std::printf("  overlap (both)        : %8zu  (%5.1f%%)\n", both, 100.0 * both / N);
    std::printf("  structurally zero     : %8zu  (%5.1f%%)\n", N - nz_full, 100.0 * (N - nz_full) / N);
    std::printf("  entries where lin+nl != full: %zu (must be 0)\n", chk);

    // Block-level: a restore that copies whole 4x4 blocks stays coalesced (16 contiguous
    // doubles), so what matters in practice is how many BLOCKS the nonlinear part touches.
    size_t blk_nl = 0, blk_lin_only = 0, blk_zero = 0;
    for (ptrdiff_t b = 0; b < bsr.nnz; ++b) {
        bool nl = false, li = false;
        for (int k = 0; k < 16; ++k) {
            if (nl_only[(size_t)b * 16 + k] != 0.0) nl = true;
            if (lin[(size_t)b * 16 + k] != 0.0) li = true;
        }
        blk_nl += nl; blk_lin_only += (li && !nl); blk_zero += (!li && !nl);
    }
    std::printf("\n  blocks touched by nonlinear : %8zu of %td (%5.1f%%)  <- what a block-wise restore must copy\n",
                blk_nl, bsr.nnz, 100.0 * (double)blk_nl / (double)bsr.nnz);
    std::printf("  blocks linear-only          : %8zu (%5.1f%%)\n", blk_lin_only,
                100.0 * (double)blk_lin_only / (double)bsr.nnz);
    std::printf("  blocks entirely zero        : %8zu (%5.1f%%)\n", blk_zero,
                100.0 * (double)blk_zero / (double)bsr.nnz);
    MPI_Finalize();
    return 0;
}
