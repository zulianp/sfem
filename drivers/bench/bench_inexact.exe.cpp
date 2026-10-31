// The stored-tangent (INEXACT) operator's formats against the assembled tangent and the exact
// apply, on the host or the device (SFEM_EXECUTION_SPACE=device).
//
//   bench_inexact <HEX8|TET4|TET10|PROTEUS_HEX27> <n> [repeats] [material]
//
// The mesh is the unit cube: n^3 hexahedra, n^3 cubes of tetrahedra (TET10 promoted), or n^3 macro
// elements of degree 2.  The state is a smooth deformation (no element inverts), the increment white
// noise.  Every row reports the dofs, the thread count, the store's set-up and its apply per call,
// MDOF/s, the bytes the operator holds per dof (BSR: its values and column indices), and its relative
// l2 distance to the exact apply.
// The rows reproduce the format tables of spikes/inexact_apply_compare/qp/RESULTS_QP.md from the
// library alone.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_Function.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#ifdef SFEM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
    using clock_type = std::chrono::steady_clock;

    void synchronize(const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) cudaDeviceSynchronize();
#else
        (void)es;
#endif
    }

    template <class F>
    double seconds(const sfem::ExecutionSpace es, const int repeats, F &&f) {
        f();  // warm-up
        synchronize(es);
        const auto t0 = clock_type::now();
        for (int r = 0; r < repeats; ++r) f();
        synchronize(es);
        return std::chrono::duration<double>(clock_type::now() - t0).count() / repeats;
    }

    sfem::SharedBuffer<real_t> in_space(const std::vector<real_t> &v, const sfem::ExecutionSpace es) {
        auto host = sfem::create_host_buffer<real_t>((ptrdiff_t)v.size());
        std::copy(v.begin(), v.end(), host->data());
        return es == sfem::EXECUTION_SPACE_DEVICE ? smesh::to_device(host) : host;
    }

    std::vector<real_t> on_host(const sfem::SharedBuffer<real_t> &b, const sfem::ExecutionSpace es) {
        auto host = es == sfem::EXECUTION_SPACE_DEVICE ? smesh::to_host(b) : b;
        return std::vector<real_t>(host->data(), host->data() + host->size());
    }

    double relative_difference(const std::vector<real_t> &a, const std::vector<real_t> &b) {
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            num += ((double)a[i] - (double)b[i]) * ((double)a[i] - (double)b[i]);
            den += (double)b[i] * (double)b[i];
        }
        return std::sqrt(num / den);
    }

    std::shared_ptr<sfem::FunctionSpace> space(const std::string &element, const int n, const sfem::ExecutionSpace es) {
        auto comm = sfem::Communicator::world();
        std::shared_ptr<sfem::Mesh> mesh;
        if (element == "HEX8" || element == "PROTEUS_HEX27") {
            mesh = sfem::Mesh::create_hex8_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1);
        } else {
            mesh = sfem::Mesh::create_tet4_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1);
        }
        if (element == "TET10") mesh = smesh::promote_to(smesh::TET10, mesh);
        if (element == "PROTEUS_HEX27") mesh = smesh::to_semistructured(2, mesh, true, false);
        auto fs = sfem::FunctionSpace::create(mesh, 3);
        if (element == "PROTEUS_HEX27") fs->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);
        if (es == sfem::EXECUTION_SPACE_HOST && fs->initialize_packed_mesh() != SFEM_SUCCESS) return nullptr;
        return fs;
    }
}  // namespace

int main(int argc, char *argv[]) {
    auto context = sfem::initialize(argc, argv);
    if (argc < 3) {
        fprintf(stderr, "usage: %s <HEX8|TET4|TET10|PROTEUS_HEX27> <n> [repeats] [material]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string element  = argv[1];
    const int         n        = atoi(argv[2]);
    const int         repeats  = argc > 3 ? atoi(argv[3]) : 50;
    const std::string material = argc > 4 ? argv[4] : "GeneratedNeoHookeanOgden";
    const char *const es_name  = getenv("SFEM_EXECUTION_SPACE");
    const auto        es       = sfem::execution_space_from_string(es_name ? es_name : "host");

    auto fs = space(element, n, es);
    if (!fs) return EXIT_FAILURE;
    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, material.c_str(), es);
    if (!op || op->initialize() != SFEM_SUCCESS || !op->inexact_supported()) {
        fprintf(stderr, "bench_inexact: %s has no stored tangent on %s\n", material.c_str(), element.c_str());
        return EXIT_FAILURE;
    }
    f->add_operator(op);

    // A smooth deformation: a stretch and a bend, small enough that nothing inverts.
    const ptrdiff_t      ndofs  = fs->n_dofs();
    const ptrdiff_t      nnodes = ndofs / 3;
    geom_t **const       x      = fs->mesh()->points()->data();
    std::vector<real_t>  u(ndofs), h(ndofs);
    unsigned             seed = 11;
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        const real_t a = x[0][i], b = x[1][i], c = x[2][i];
        u[i * 3 + 0]   = 0.1 * a + 0.02 * std::sin(3 * b + c);
        u[i * 3 + 1]   = 0.05 * a * a + 0.02 * std::sin(2 * c + a);
        u[i * 3 + 2]   = 0.02 * std::sin(3 * a + b);
        for (int k = 0; k < 3; ++k) {
            seed         = seed * 1664525u + 1013904223u;
            h[i * 3 + k] = (real_t)(seed >> 8) / (real_t)(1u << 24) - real_t(0.5);
        }
    }
    auto du = in_space(u, es), dh = in_space(h, es), y = in_space(std::vector<real_t>(ndofs, 0), es);
    const real_t *const U = du->data(), *const H = dh->data();
    real_t *const       Y = y->data();
    auto blas = sfem::blas<real_t>(es);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("bench_inexact %s %s n=%d: %ld elements, %ld dofs, %s, %d threads, %d repeats\n", material.c_str(),
           element.c_str(), n, (long)fs->mesh()->n_elements(), (long)ndofs, es_name ? es_name : "host", threads, repeats);
    printf("%-26s %10s %10s %9s %9s %10s\n", "operator", "set-up ms", "apply ms", "MDOF/s", "B/dof", "rel diff");

    auto row = [&](const char *name, const double setup, const double apply, const double bytes, const std::vector<real_t> &out,
                   const std::vector<real_t> &exact) {
        printf("%-26s %10.3f %10.4f %9.1f %9.1f %10.2e\n", name, 1e3 * setup, 1e3 * apply, 1e-6 * ndofs / apply, bytes / ndofs,
               exact.empty() ? 0.0 : relative_difference(out, exact));
    };

    // The exact apply (matrix-free): the reference.
    blas->zeros(ndofs, Y);
    f->apply(U, H, Y);
    const std::vector<real_t> exact = on_host(y, es);
    const double              t_mf  = seconds(es, repeats, [&] { f->apply(U, H, Y); });
    row("matrix-free", 0, t_mf, 0, exact, {});

    // The assembled tangent, double and single precision values.
    for (const int bytes : {8, 4}) {
        auto         tangent = sfem::BSRTangent::create(f, op->execution_context(), bytes);
        const double setup   = seconds(es, 1, [&] { tangent->assemble(U); });
        auto         K       = tangent->op(0);
        blas->zeros(ndofs, Y);
        K->apply(H, Y);
        const std::vector<real_t> out = on_host(y, es);
        const double t = seconds(es, repeats, [&] { K->apply(H, Y); });
        const double read = (double)tangent->rowptr()[tangent->block_rows()] * (9 * bytes + sizeof(idx_t));
        row(bytes == 8 ? "BSR f64" : "BSR f32", setup, t, read, out, exact);
    }

    const char *const formats[][2] = {{"points", "f64"},          {"points", "f32"},          {"points", "f16"},
                                      {"points", "i8md"},         {"element_matrix", "f64"},  {"element_matrix", "f32"},
                                      {"element_matrix", "f16"},  {"element_matrix", "i8pair"}, {"bsr", "f16"},
                                      {"bsr", "i8"}};
    for (const auto &fmt : formats) {
        op->set_inexact_options(sfem::InexactOptions::from_strings(fmt[0], fmt[1]));
        const double setup = seconds(es, 3, [&] { op->inexact_update(U); });
        blas->zeros(ndofs, Y);
        op->inexact_apply(H, Y);
        const std::vector<real_t> out = on_host(y, es);
        const double t = seconds(es, repeats, [&] { op->inexact_apply(H, Y); });
        const std::string name = std::string(fmt[0]) + " " + fmt[1];
        row(name.c_str(), setup, t, (double)op->inexact_bytes(), out, exact);
    }
    return EXIT_SUCCESS;
}
