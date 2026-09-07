// A spike copy of bench_op that separates setup from application.
//
// `drivers/bench/bench_op.exe.cpp` reports one rate per operator and folds the
// partial assembly into a `setup` column that is measured once and never
// compared.  For the split partial assembly those are the two numbers that
// matter and they have to be read together: the tangent is assembled once per
// Newton step and applied once per Krylov iteration, so an apply rate without
// its assembly cost is only half the story, and the break-even between them is
// the answer to "is this worth it".
//
// This is a spike, deliberately outside the driver tree: it must not perturb
// bench_op, whose numbers are the baseline in BASELINE.md.  It builds the same
// mesh with the same environment variables so the two are comparable, times the
// library's operators exactly as bench_op does, and then times the generated
// split kernels on the same mesh -- assembly and apply separately, at three
// storage precisions.
//
//   SFEM_ELEM_TYPE          TET4 | TET10 | HEX8   (must match -DELEMENT_*)
//   SFEM_BASE_RESOLUTION    cube divisions per axis
//   SFEM_PROMOTE_TO_P2      1 to promote a TET4 cube to TET10
//   SFEM_REPEAT             applies per timing window
#include <cstdio>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "sfem_Function.hpp"
#include "sfem_API.hpp"
#include "sfem_base.hpp"
#include "sfem_aliases.hpp"
#include "smesh_env.hpp"
#include "smesh_kernel_data.hpp"

#include "kernel_math.hpp"
#include MATERIAL_INEXACT_HEADER

#if defined(ELEMENT_HEX8)
#define NXE 8
#define ELEMENT_NAME "HEX8"
#elif defined(ELEMENT_TET10)
#define NXE 10
#define ELEMENT_NAME "TET10"
#else
#define NXE 4
#define ELEMENT_NAME "TET4"
#endif

// Symmetric tangent: an energy material's tangent is a Hessian.
#define TANGENT_COMPONENTS 45

struct Row {
    std::string what;
    double      setup = 0;     // seconds, once
    double      apply = 0;     // seconds, per application
    ptrdiff_t   ndof  = 0;

    double rate() const { return apply > 0 ? 1e-6 * ndof / apply : 0.0; }

    static void header() {
        std::printf("%-34s %12s %14s %14s\n",
                    "What", "Setup [s]", "Apply [s]", "Rate [MDOF/s]");
        std::printf("%-34s %12s %14s %14s\n",
                    "----------------------------------", "-----------",
                    "-------------", "-------------");
    }
    void print() const {
        std::printf("%-34s %12.4f %14.4e %14.2f\n", what.c_str(), setup, apply, rate());
    }
};

template <typename F>
static double time_best(int repeats, F &&fn) {
    fn();
    double best = 1e30;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const double t0 = MPI_Wtime();
        for (int r = 0; r < repeats; ++r) fn();
        const double t1 = MPI_Wtime();
        best = std::min(best, (t1 - t0) / repeats);
    }
    return best;
}

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    {
        auto comm = context.communicator();
        const int  resolution = smesh::Env::read("SFEM_BASE_RESOLUTION", 60);
        const int  repeats    = smesh::Env::read("SFEM_REPEAT", 5);
        const auto elem_type  = smesh::type_from_string(
                smesh::Env::read_string("SFEM_ELEM_TYPE", "TET4").c_str());

        auto m = sfem::Mesh::create_cube(comm, static_cast<smesh::ElemType>(elem_type),
                                         resolution, resolution, resolution,
                                         0, 0, 0, 1, 1, 1);
        if (smesh::Env::read("SFEM_PROMOTE_TO_P2", false)) {
            m = smesh::promote_to(smesh::TET10, m);
        }

        const ptrdiff_t nelements = m->n_elements();
        const ptrdiff_t nnodes    = m->n_nodes();
        const ptrdiff_t ndof      = 3 * nnodes;
        std::printf("%s, element %s, %ld elements, %ld nodes, ndof %ld, repeats %d\n\n",
                    MATERIAL_LABEL, type_to_string(m->element_type(0)),
                    (long)nelements, (long)nnodes, (long)ndof, repeats);

        std::vector<Row> rows;

        // ---- the library's operators, timed the way bench_op times them ----
        for (const char *name : {"NeoHookeanOgden", "GeneratedNeoHookeanOgden"}) {
            auto fs = sfem::FunctionSpace::create(m, 3);
            auto f  = sfem::Function::create(fs);
            auto op = sfem::create_op(fs, name, sfem::EXECUTION_SPACE_HOST);
            if (!op || op->initialize() != SFEM_SUCCESS) {
                std::printf("  (skipping %s: not available for this element)\n", name);
                continue;
            }
            f->add_operator(op);
            auto x      = sfem::create_buffer<real_t>(op->n_dofs_domain(), sfem::EXECUTION_SPACE_HOST);
            auto input  = sfem::create_buffer<real_t>(op->n_dofs_domain(), sfem::EXECUTION_SPACE_HOST);
            auto output = sfem::create_buffer<real_t>(op->n_dofs_image(), sfem::EXECUTION_SPACE_HOST);

            const double t0 = MPI_Wtime();
            f->update(x->data());
            auto linear_op = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, x,
                                                          sfem::EXECUTION_SPACE_HOST);
            const double t1 = MPI_Wtime();

            Row row;
            row.what  = std::string("library: ") + name;
            row.ndof  = ndof;
            row.setup = t1 - t0;
            row.apply = time_best(repeats, [&] { linear_op->apply(input->data(), output->data()); });
            rows.push_back(row);
        }

        // ---- the generated split, on the same mesh ----
        auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                m, smesh::MEMORY_SPACE_HOST, 0);
        if (!jacobian) {
            SFEM_ERROR("could not build the affine geometry cache\n");
            return SFEM_FAILURE;
        }
        auto        adj_soa    = jacobian->jacobian_adjugate_SoA()->data();
        const auto *determinant = jacobian->jacobian_determinant()->data();
        const geom_t *A[9];
        for (int i = 0; i < 9; ++i) A[i] = reinterpret_cast<const geom_t *>(adj_soa[i]);

        auto elements = m->elements(0)->data();

        const ptrdiff_t cstride = nelements + 64;   // padded off a power of two
        std::vector<double>  S64((size_t)cstride * TANGENT_COMPONENTS);
        std::vector<float>   S32((size_t)cstride * TANGENT_COMPONENTS);
        std::vector<half_t>  S16((size_t)cstride * TANGENT_COMPONENTS);
        std::vector<float>   scale(nelements, 1.0f);
        std::vector<real_t>  u(ndof, 0.0), h(ndof, 0.0), out(ndof, 0.0);

        // A smooth state, so the tangent is a real one rather than the identity's.
        auto points = m->points()->data();
        std::vector<double> ux(nnodes), uy(nnodes), uz(nnodes);
        std::vector<double> hx(nnodes), hy(nnodes), hz(nnodes);
        std::vector<double> ox(nnodes, 0), oy(nnodes, 0), oz(nnodes, 0);
        for (ptrdiff_t v = 0; v < nnodes; ++v) {
            const double x = points[0][v], y = points[1][v], z = points[2][v];
            ux[v] = 0.02*std::sin(3*x + y + 0.5*z);
            uy[v] = 0.02*std::sin(x + 3*y + 1.5*z);
            uz[v] = 0.02*std::sin(0.5*x + 1.5*y + 3*z);
            hx[v] = 0.05*std::sin(2*x + 0.7*y + 1.1*z);
            hy[v] = 0.05*std::sin(0.7*x + 2*y + 1.3*z);
            hz[v] = 0.05*std::sin(1.1*x + 1.3*y + 2*z);
        }

        const double mu = 1.0, lmbda = 1.0;
        auto assemble = [&] {
            TANGENT_KERNEL<double, geom_t, double>(
                nelements, elements, A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8],
                determinant, lmbda, mu, 1, ux.data(), uy.data(), uz.data(),
                1, cstride, S64.data());
        };

        Row setup_row;
        setup_row.what = "split: partial assembly";
        setup_row.ndof = ndof;
        setup_row.apply = time_best(repeats, assemble);
        setup_row.setup = setup_row.apply;   // it *is* the setup; shown in both
        rows.push_back(setup_row);

        for (ptrdiff_t e = 0; e < nelements; ++e) {
            double top = 0;
            for (int c = 0; c < TANGENT_COMPONENTS; ++c)
                top = std::max(top, std::fabs(S64[(size_t)c*cstride + e]));
            const double s = top > 65504.0 ? (top + 1e-8)/65504.0 : 1.0;
            scale[e] = (float)s;
            for (int c = 0; c < TANGENT_COMPONENTS; ++c) {
                const size_t at = (size_t)c*cstride + e;
                S32[at] = (float)S64[at];
                S16[at] = (half_t)(S64[at]/s);
            }
        }

        auto stored = [&](auto *store, const char *label) {
            Row row;
            row.what  = std::string("split: stored apply ") + label;
            row.ndof  = ndof;
            row.setup = setup_row.apply;
            row.apply = time_best(repeats, [&] {
                STORED_APPLY<double, typename std::remove_const<
                        typename std::remove_pointer<decltype(store)>::type>::type>(
                    nelements, elements, 1, cstride, store,
                    1, hx.data(), hy.data(), hz.data(),
                    1, ox.data(), oy.data(), oz.data());
            });
            rows.push_back(row);
        };
        stored(S64.data(), "f64");
        stored(S32.data(), "f32");
        {
            Row row;
            row.what  = "split: stored apply f16";
            row.ndof  = ndof;
            row.setup = setup_row.apply;
            row.apply = time_best(repeats, [&] {
                COMPRESSED_APPLY<double, half_t, float>(
                    nelements, elements, 1, cstride, S16.data(), scale.data(),
                    1, hx.data(), hy.data(), hz.data(),
                    1, ox.data(), oy.data(), oz.data());
            });
            rows.push_back(row);
        }

        Row::header();
        for (const auto &row : rows) row.print();

        // Break-even, stated against the right reference.
        //
        // It only means something against an operator that does *no* setup of
        // its own.  The hand-written NeoHookeanOgden turns partial assembly on
        // for HEX8 and TET10 and pays for it in its own setup column, so
        // comparing a stored apply's break-even against that one would charge
        // the split for an assembly the reference also performs.  The fused
        // generated kernel is the honest reference: it rebuilds the tangent
        // every apply and stores nothing.
        //
        // Against an operator that also assembles, the comparison to make is
        // simply apply against apply and setup against setup, both printed
        // above.
        double fused = -1.0, best_any = 1e30;
        std::string best_any_name;
        for (const auto &row : rows) {
            if (row.what == "library: GeneratedNeoHookeanOgden") fused = row.apply;
            if (row.what.rfind("library:", 0) == 0 && row.apply < best_any) {
                best_any = row.apply;
                best_any_name = row.what;
            }
        }
        if (fused > 0) {
            std::printf("\n  break-even applies per tangent, against the fused generated apply (%.4e s,\n"
                        "  which performs no assembly of its own):\n", fused);
            for (const auto &row : rows) {
                if (row.what.rfind("split: stored", 0) != 0) continue;
                const double gain = fused - row.apply;
                if (gain <= 0) {
                    std::printf("    %-30s never (apply is not faster)\n", row.what.c_str());
                } else {
                    std::printf("    %-30s %.2f\n", row.what.c_str(), setup_row.apply / gain);
                }
            }
        }
        if (best_any < 1e29 && best_any_name != "library: GeneratedNeoHookeanOgden") {
            std::printf("\n  against %s, which does assemble, compare directly:\n",
                        best_any_name.c_str());
            for (const auto &row : rows) {
                if (row.what.rfind("split: stored", 0) != 0) continue;
                std::printf("    %-30s apply %.2fx   (setup %.4f s against its own)\n",
                            row.what.c_str(), best_any / row.apply, setup_row.apply);
            }
        }
        std::printf("\n  store bytes/element: f64 %d  f32 %d  f16+scale %d\n",
                    TANGENT_COMPONENTS*8, TANGENT_COMPONENTS*4, TANGENT_COMPONENTS*2 + 4);
    }
    return SFEM_SUCCESS;
}
