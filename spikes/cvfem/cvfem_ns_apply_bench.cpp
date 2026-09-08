// Throughput of the CVFEM Navier-Stokes operator kernels, in isolation.
//
// T1 established that the Krylov solve is where the time goes -- roughly 800 linear
// iterations per Newton step -- so `apply` is the kernel that matters, and the whole
// point of the semi-structured work is to make it faster. This measures it on its own,
// away from the Newton and Krylov machinery whose vector operations and reductions
// dominated and hid everything at the sizes first tried.
//
// Two lessons from T1 are built in. Sizes are swept rather than assumed, because the
// per-dof cost fell 7x between 10k and 560k dofs and conclusions drawn at the small end
// pointed the wrong way. And the traffic model below is a stated lower bound rather than
// a guess dressed as a measurement.
//
// Includes only the operator and the channel case: no MeshData, no kernels, no BSR4.

#include "cvfem_hex8_ns_op.hpp"
#include "cvfem_ns_channel_case.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_context.hpp"

#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

    constexpr int N_FIELDS = 4;

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    }

}  // namespace

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);

    const int         reps      = smesh::Env::read<int>("SFEM_BENCH_REPS", 20);
    const int         warmup    = smesh::Env::read<int>("SFEM_BENCH_WARMUP", 3);
    const std::string geom_name = smesh::Env::read_string("SFEM_GEOM", "affine");
    const int         pack_size = smesh::Env::read<int>("SFEM_PACK_SIZE", 2048);
    const real_t      Lx = 4, Ly = 1, Lz = 1;
    const real_t      rho = 1, mu = 0.01, U = 1;

    // Swept by default. The list runs from well below saturation to well past it, so the
    // saturated figure is visible as a plateau rather than taken on faith.
    std::vector<int> sizes;
    {
        const std::string spec = smesh::Env::read_string("SFEM_BENCH_SIZES", "4,8,16,24,32,40");
        size_t            pos  = 0;
        std::string       rest = spec;
        while (!rest.empty()) {
            pos            = rest.find(',');
            const auto tok = rest.substr(0, pos);
            if (!tok.empty()) sizes.push_back(std::atoi(tok.c_str()));
            if (pos == std::string::npos) break;
            rest = rest.substr(pos + 1);
        }
    }

    std::printf("geom=%s pack_size=%d reps=%d\n", geom_name.c_str(), pack_size, reps);
    std::printf("%-10s %-10s %-11s %-11s %-11s %-11s %-9s\n",
                "ndof", "nelem", "apply_ns/d", "grad_ns/d", "bdiag_ns/d", "apply_MDOF/s", "eff_GB/s");

    for (const int n : sizes) {
        const int ny = n;
        const int nx = std::max(1, (int)std::lround((double)ny * (double)Lx / (double)Ly));
        const int nz = std::max(1, (int)std::lround((double)ny * (double)Lz / (double)Ly));

        auto mesh = smesh::Mesh::create_hex8_cube(ctx->communicator(), nx, ny, nz, 0, 0, 0, Lx, Ly, Lz);
        auto fs   = sfem::FunctionSpace::create(mesh, N_FIELDS);

        auto op       = std::make_shared<sfem::CVFEMNavierStokes>(fs);
        op->rho       = rho;
        op->mu        = mu;
        op->geom      = (geom_name == "isoparam") ? sfem::CVFEMGeometry::Isoparam : sfem::CVFEMGeometry::Affine;
        op->pack_size = pack_size;
        if (op->initialize() != SFEM_SUCCESS) return EXIT_FAILURE;

        // initialize() renumbers the nodes for the packed layout, so the state is built
        // from the mesh only after it.
        const ptrdiff_t nnodes = mesh->n_nodes();
        const ptrdiff_t ndof   = nnodes * N_FIELDS;

        std::vector<real_t> x((size_t)ndof), h((size_t)ndof), y((size_t)ndof, 0);
        {
            const auto *const px = mesh->points()->data()[0];
            const auto *const py = mesh->points()->data()[1];
            const auto *const pz = mesh->points()->data()[2];
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                real_t ux, uy, uz, p;
                cvfem_case::exact_state(cvfem_case::FlowCase::Poiseuille, mu, U, Lx, Ly,
                                        (real_t)px[i], (real_t)py[i], (real_t)pz[i], ux, uy, uz, p);
                // A developed, non-symmetric state: a zero interior would leave the upwind
                // switch untaken and time a branch the real solve does take.
                x[(size_t)i * 4 + 0] = ux + real_t(0.1) * std::sin(real_t(3) * (real_t)px[i]);
                x[(size_t)i * 4 + 1] = uy + real_t(0.05) * std::cos(real_t(2) * (real_t)py[i]);
                x[(size_t)i * 4 + 2] = uz + real_t(0.05) * std::sin(real_t(2) * (real_t)pz[i]);
                x[(size_t)i * 4 + 3] = p;
                for (int c = 0; c < 4; ++c)
                    h[(size_t)i * 4 + c] = std::sin(real_t(0.7) * (real_t)(i * 4 + c) + real_t(0.3));
            }
        }

        std::vector<real_t> bdiag((size_t)nnodes * 16, 0);

        auto time_it = [&](auto &&fn) {
            for (int r = 0; r < warmup; ++r) fn();
            std::vector<double> t;
            t.reserve((size_t)reps);
            for (int r = 0; r < reps; ++r) {
                const double t0 = smesh::time_seconds();
                fn();
                t.push_back(smesh::time_seconds() - t0);
            }
            return median(t);
        };

        const double t_apply = time_it([&] { std::fill(y.begin(), y.end(), real_t(0)); op->apply(x.data(), h.data(), y.data()); });
        const double t_grad  = time_it([&] { std::fill(y.begin(), y.end(), real_t(0)); op->gradient(x.data(), y.data()); });
        const double t_bd    = time_it([&] { std::fill(bdiag.begin(), bdiag.end(), real_t(0)); op->hessian_block_diag(x.data(), bdiag.data()); });

        // Lower bound on traffic for `apply`, counting only what no implementation can
        // avoid: the state and direction read, the output read-modify-written because
        // Op::apply accumulates, and the affine geometry (adjugate + determinant per
        // element) the kernel loads. Connectivity, coordinates and the nodal pressure
        // gradient are all excluded, so the real figure is higher and the bandwidth
        // reported here is a floor, not an estimate.
        const double bytes = (double)ndof * 8.0 * 4.0 + (double)mesh->n_elements(0) * 10.0 * 8.0;

        std::printf("%-10td %-10td %-11.2f %-11.2f %-11.2f %-11.1f %-9.1f\n",
                    ndof,
                    mesh->n_elements(0),
                    1e9 * t_apply / (double)ndof,
                    1e9 * t_grad / (double)ndof,
                    1e9 * t_bd / (double)ndof,
                    1e-6 * (double)ndof / t_apply,
                    1e-9 * bytes / t_apply);
    }

    return EXIT_SUCCESS;
}
