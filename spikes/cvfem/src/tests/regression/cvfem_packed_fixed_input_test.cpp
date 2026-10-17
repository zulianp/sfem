#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "frontend/staging/cvfem_hex8_packed_launch.hpp"
#include "sfem_context.hpp"
#include "smesh_sfc.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static bool same_finite_bits(const std::vector<scalar_t> &a,
                             const std::vector<scalar_t> &b) {
    if (a.empty() || a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(scalar_t)) == 0;
}

int main(int argc, char **argv) {
#ifndef _OPENMP
    std::puts("OpenMP is required for a cross-thread check");
    return 77;
#else
    sfem::Context context(argc, argv);
    omp_set_dynamic(0);

    const std::vector<scalar_t> control{scalar_t(1)};
    const std::vector<scalar_t> changed{std::nextafter(scalar_t(1), scalar_t(2))};
    const std::vector<scalar_t> nan{std::numeric_limits<scalar_t>::quiet_NaN()};
    if (!same_finite_bits(control, control) || same_finite_bits(control, changed) ||
        same_finite_bits(nan, nan) || same_finite_bits(control, {})) return 1;

    // n=24 includes non-dyadic coordinates and the input-tail case found in review.
    for (const int n : {8, 24}) {
        omp_set_num_threads(1);
        MeshData d;
        d.mesh = smesh::Mesh::create_hex8_cube(context.communicator(), n, n, n, 0, 0, 0, 1, 1, 1);
        auto sfc = smesh::SFC::create_from_env();
        sfc->reorder(*d.mesh);
        auto packed = make_packed(d.mesh, 64);
        d.nnodes = d.mesh->n_nodes();
        d.nelements = d.mesh->n_elements(0);
        d.elems = d.mesh->elements(0)->data();
        d.points = d.mesh->points()->data();
        precompute_affine_geometry(d);
        fill_fields(d);
        if (packed.n_packs < 4 || packed.n_ghost_reduce_rows == 0) return 1;

        // Initialize once: changing arithmetic in a parallel initializer would
        // compare different inputs instead of testing operator reproducibility.
        std::vector<scalar_t> direction(size_t(d.nnodes) * N_FIELDS);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const uint32_t h = uint32_t(i) * 2246822519u + 374761393u;
            const auto noise = [&](const int shift) {
                return scalar_t((h >> shift) & 255u) / scalar_t(255) - scalar_t(0.5);
            };
            d.ux[i] += scalar_t(0.03) * noise(0);
            d.uy[i] += scalar_t(0.03) * noise(8);
            d.uz[i] += scalar_t(0.03) * noise(16);
            d.p[i] += scalar_t(0.03) * noise(3);
            for (int f = 0; f < N_FIELDS; ++f)
                direction[size_t(i) * N_FIELDS + f] = scalar_t(0.25) + noise(f * 5);
        }
        const auto ux = d.ux, uy = d.uy, uz = d.uz, pressure = d.p;
        const auto saved_direction = direction;
        std::vector<scalar_t> buffer, gu, gv, baseline_residual, baseline_action;
        for (const int scheme : {-2, -1, 0, 1, 2, 3}) {
            d.rhie_chow_scale = scheme == -2 ? 0 : 1;
            for (const int threads : {1, 2, 3, 4, 2, 1}) {
                omp_set_num_threads(threads);
                int actual_threads = 0;
#pragma omp parallel
                {
#pragma omp single
                    actual_threads = omp_get_num_threads();
                }
                if (actual_threads != threads) {
                    std::fprintf(stderr, "requested %d threads, got %d\n", threads, actual_threads);
                    return 1;
                }
                if (scheme != -2) {
                    cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, d.p.data(), 1,
                                                         d.pgx, d.pgy, d.pgz, buffer);
                    cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, direction.data() + 3, N_FIELDS,
                                                         d.qgx, d.qgy, d.qgz, buffer);
                }
                if (scheme >= 0) {
                    const scalar_t *state[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
                    const scalar_t *dir[3] = {direction.data(), direction.data() + 1, direction.data() + 2};
                    const int strides[3] = {N_FIELDS, N_FIELDS, N_FIELDS};
                    cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, state, 3, gu, buffer);
                    cvfem_hex8_assemble_nodal_grads_packed(d, packed, 0, dir, 3, gv, buffer, strides);
                    apply_residual_packed_defcor(d, packed, scalar_t(1), scalar_t(0.01),
                                                gu.data(), scheme, scalar_t(0));
                } else {
                    apply_residual_packed(d, packed, scalar_t(1), scalar_t(0.01), GeomKind::Affine);
                }
                std::vector<scalar_t> residual;
                for (const auto *field : {&d.rx, &d.ry, &d.rz, &d.rc})
                    residual.insert(residual.end(), field->begin(), field->end());
                std::vector<scalar_t> action(direction.size());
                apply_jacobian_action_packed(d, packed, scalar_t(1), scalar_t(0.01),
                    GeomKind::Affine, direction.data(), action.data(),
                    scheme >= 0 ? gu.data() : nullptr, scheme >= 0 ? gv.data() : nullptr,
                    scheme >= 0 ? scheme : 0);
                if (baseline_residual.empty()) {
                    baseline_residual = residual;
                    baseline_action = action;
                }
                if (!same_finite_bits(baseline_residual, residual) ||
                    !same_finite_bits(baseline_action, action) ||
                    !same_finite_bits(ux, d.ux) || !same_finite_bits(uy, d.uy) ||
                    !same_finite_bits(uz, d.uz) || !same_finite_bits(pressure, d.p) ||
                    !same_finite_bits(saved_direction, direction)) {
                    std::fprintf(stderr, "FAIL: n=%d scheme=%d threads=%d\n", n, scheme, threads);
                    return 1;
                }
                bool residual_nonzero = false, action_nonzero = false;
                for (const auto v : residual) residual_nonzero |= v != scalar_t(0);
                for (const auto v : action) action_nonzero |= v != scalar_t(0);
                if (!residual_nonzero || !action_nonzero) return 1;
            }
            std::printf("PASS: n=%d pack=64 scheme=%d residual/action bits at 1/2/3/4 threads\n", n, scheme);
            baseline_residual.clear();
            baseline_action.clear();
        }
    }
    return 0;
#endif
}
