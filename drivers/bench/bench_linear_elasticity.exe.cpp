#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "generated/linear_elasticity/op/sfem_GeneratedLinearElasticity.hpp"
#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_LinearElasticity.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"
#include "smesh_mesh_reorder.hpp"

namespace {

    struct ErrorMetrics {
        real_t max_abs{0};
        real_t rel_l2{0};
    };

    void print_rate(const char     *name,
                    const double    elapsed,
                    const ptrdiff_t nelements,
                    const ptrdiff_t ndofs,
                    const int       repeat,
                    const double    flops,
                    const size_t    memory_traffic_bytes) {
        const double seconds_per_call     = elapsed / repeat;
        const double melements_per_s      = 1e-6 * static_cast<double>(nelements) / seconds_per_call;
        const double mdofs_per_s          = 1e-6 * static_cast<double>(ndofs) / seconds_per_call;
        const double arithmetic_intensity = memory_traffic_bytes ? flops / static_cast<double>(memory_traffic_bytes) : 0;
        const double flops_per_element    = nelements > 0 ? flops / static_cast<double>(nelements) : 0;
        const double bytes_per_element =
                nelements > 0 ? static_cast<double>(memory_traffic_bytes) / static_cast<double>(nelements) : 0;
        const double gflops_per_s         = 1e-9 * flops / seconds_per_call;
        const double gbytes_per_s         = 1e-9 * static_cast<double>(memory_traffic_bytes) / seconds_per_call;

        printf("%-40s %12.6e %16.3f %13.3f %12.3f %12.3f %10.3f %13.3f %12.3f\n",
               name,
               seconds_per_call,
               melements_per_s,
               mdofs_per_s,
               flops_per_element,
               bytes_per_element,
               arithmetic_intensity,
               gflops_per_s,
               gbytes_per_s);
    }

    bool generated_linear_elasticity_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TRI3:
            case smesh::TRI6:
            case smesh::QUAD4:
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
            case smesh::HEX27:
            case smesh::PROTEUS_HEX8:
            case smesh::PROTEUS_HEX27:
            case smesh::PROTEUS_HEX64:
            case smesh::PROTEUS_HEX125:
            case smesh::PROTEUS_HEX729:
                return true;
            default:
                return false;
        }
    }

    bool baseline_linear_elasticity_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
            case smesh::PROTEUS_HEX8:
            case smesh::PROTEUS_HEX27:
            case smesh::PROTEUS_HEX64:
            case smesh::PROTEUS_HEX125:
            case smesh::PROTEUS_HEX729:
                return true;
            default:
                return false;
        }
    }

    bool packed_linear_elasticity_supported(const smesh::ElemType element_type, const std::string &operator_name) {
        if (operator_name == "GeneratedLinearElasticity") {
            return generated_linear_elasticity_supported(element_type);
        }

        return false;
    }

    void set_geometry_options(const std::shared_ptr<sfem::Op> &op, const bool assume_affine) {
        op->set_option("assume_affine", assume_affine);
        op->set_option("ASSUME_AFFINE", assume_affine);
    }

    void set_generated_material(const std::shared_ptr<sfem::Op> &op, const real_t mu, const real_t lambda) {
        op->set_value_in_block("default", "mu", mu);
        op->set_value_in_block("default", "lmbda", lambda);
    }

    void set_baseline_material(const std::shared_ptr<sfem::Op> &op, const real_t mu, const real_t lambda) {
        op->set_value_in_block("default", "mu", mu);
        op->set_value_in_block("default", "lambda", lambda);
    }

    void fill_test_vectors(const ptrdiff_t ndofs, real_t *const x, real_t *const h) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            x[i] = static_cast<real_t>(static_cast<int>(i % 131) - 65) / static_cast<real_t>(131);
            h[i] = static_cast<real_t>(static_cast<int>(i % 97) + 1) / static_cast<real_t>(97);
        }
    }

    ErrorMetrics compare_vectors(const ptrdiff_t ndofs, const real_t *const reference, const real_t *const candidate) {
        real_t diff_norm2 = 0;
        real_t ref_norm2  = 0;
        real_t max_abs    = 0;

#pragma omp parallel for reduction(+ : diff_norm2, ref_norm2) reduction(max : max_abs)
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            const real_t diff = candidate[i] - reference[i];
            diff_norm2 += diff * diff;
            ref_norm2 += reference[i] * reference[i];
            max_abs = std::max(max_abs, std::abs(diff));
        }

        const real_t denom = std::max<real_t>(std::sqrt(ref_norm2), 1);
        return {max_abs, std::sqrt(diff_norm2) / denom};
    }

    bool finite_vector(const ptrdiff_t ndofs, const real_t *const values) {
        int finite = 1;
#pragma omp parallel for reduction(&& : finite)
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            finite = finite && std::isfinite(values[i]);
        }
        return finite;
    }

    double time_gradient(const std::shared_ptr<sfem::Function>         &f,
                         const real_t *const                            x,
                         real_t *const                                  out,
                         const ptrdiff_t                                ndofs,
                         const int                                      repeat,
                         const std::shared_ptr<sfem::BLAS_Tpl<real_t>> &blas) {
        blas->zeros(ndofs, out);
        sfem::device_synchronize();
        const double t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            f->gradient(x, out);
        }
        sfem::device_synchronize();
        return MPI_Wtime() - t0;
    }

    double time_apply(const std::shared_ptr<sfem::Operator<real_t>> &op,
                      const real_t *const                            h,
                      real_t *const                                  out,
                      const ptrdiff_t                                ndofs,
                      const int                                      repeat,
                      const std::shared_ptr<sfem::BLAS_Tpl<real_t>> &blas) {
        blas->zeros(ndofs, out);
        sfem::device_synchronize();
        const double t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            op->apply(h, out);
        }
        sfem::device_synchronize();
        return MPI_Wtime() - t0;
    }

}  // namespace

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    auto          comm = context.communicator();

    if (comm->size() != 1) {
        SFEM_ERROR("bench_linear_elasticity.exe supports one MPI rank\n");
    }

    const int         resolution              = smesh::Env::read("SFEM_BASE_RESOLUTION", 16);
    const int         warmup                  = smesh::Env::read("SFEM_WARMUP", 3);
    const int         repeat                  = smesh::Env::read("SFEM_REPEAT", 10);
    const real_t      mu                       = smesh::Env::read("SFEM_SHEAR_MODULUS", 1.0);
    const real_t      lambda                   = smesh::Env::read("SFEM_FIRST_LAME_PARAMETER", 1.0);
    const real_t      compare_rtol             = smesh::Env::read("SFEM_COMPARE_RTOL", 1e-8);
    const real_t      compare_atol             = smesh::Env::read("SFEM_COMPARE_ATOL", 1e-8);
    const int         packed_elements_per_pack = smesh::Env::read("SFEM_ELEMENTS_PER_PACK", 0);
    const std::string generated_operator_name  = smesh::Env::read_string("SFEM_GENERATED_OPERATOR", "GeneratedLinearElasticity");
    const std::string baseline_operator_name   = smesh::Env::read_string("SFEM_BASELINE_OPERATOR", "LinearElasticity");
    const std::string packed_operator_name     = smesh::Env::read_string("SFEM_PACKED_OPERATOR", "GeneratedLinearElasticity");
    const std::string codegen_geometry         = smesh::Env::read_string("SFEM_CODEGEN_GEOMETRY", "isoparametric");
    const bool        run_baseline_requested   = smesh::Env::read("SFEM_RUN_BASELINE", true);
    const bool        run_packed_requested     = smesh::Env::read("SFEM_RUN_PACKED", true);
    const bool        run_packed_two_pass_requested = smesh::Env::read("SFEM_RUN_PACKED_TWO_PASS", true);

    if (repeat <= 0) {
        SFEM_ERROR("SFEM_REPEAT must be positive\n");
    }
    if (codegen_geometry != "affine" && codegen_geometry != "isoparametric") {
        SFEM_ERROR("SFEM_CODEGEN_GEOMETRY must be affine or isoparametric\n");
    }

    const auto element_type = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());
    auto       mesh         = sfem::Mesh::create_cube(
            comm, static_cast<smesh::ElemType>(element_type), resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
    auto sfc = smesh::SFC::create_from_env();
    sfc->reorder(*mesh);

    const int block_size = mesh->spatial_dimension();
    if (block_size != 3) {
        SFEM_ERROR("bench_linear_elasticity.exe requires a three-dimensional mesh\n");
    }
    if (!generated_linear_elasticity_supported(mesh->element_type(0))) {
        SFEM_ERROR("generated path does not support SFEM_ELEM_TYPE=%s\n", type_to_string(mesh->element_type(0)));
    }

    const bool assume_affine = codegen_geometry == "affine";

    auto fs = sfem::FunctionSpace::create(mesh, block_size);

    auto                      generated_f = sfem::Function::create(fs);
    std::shared_ptr<sfem::Op> generated_op;
    if (generated_operator_name == "GeneratedLinearElasticity") {
        generated_op = std::make_shared<sfem::GeneratedLinearElasticity>(fs);
    } else {
        generated_op = sfem::create_op(fs, generated_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!generated_op) {
            SFEM_ERROR("Unable to create generated operator %s\n", generated_operator_name.c_str());
        }
    }
    set_geometry_options(generated_op, assume_affine);
    if (generated_op->initialize() != SFEM_SUCCESS) {
        SFEM_ERROR("Unable to initialize generated operator %s\n", generated_operator_name.c_str());
    }
    set_generated_material(generated_op, mu, lambda);
    generated_f->add_operator(generated_op);

    const bool run_baseline = run_baseline_requested && baseline_linear_elasticity_supported(mesh->element_type(0));
    std::shared_ptr<sfem::Function> baseline_f;
    if (run_baseline) {
        auto baseline_op = sfem::create_op(fs, baseline_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!baseline_op) {
            SFEM_ERROR("Unable to create baseline operator %s\n", baseline_operator_name.c_str());
        }
        set_geometry_options(baseline_op, assume_affine);
        if (baseline_op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to initialize baseline operator %s\n", baseline_operator_name.c_str());
        }
        set_baseline_material(baseline_op, mu, lambda);

        baseline_f = sfem::Function::create(fs);
        baseline_f->add_operator(baseline_op);
    }

    const bool run_packed = run_packed_requested && packed_linear_elasticity_supported(mesh->element_type(0), packed_operator_name);
    const bool run_packed_two_pass =
            run_packed_two_pass_requested && run_packed && packed_operator_name == "GeneratedLinearElasticity";
    std::shared_ptr<sfem::FunctionSpace> packed_fs;
    std::shared_ptr<sfem::Function>      packed_f;
    std::shared_ptr<sfem::Function>      packed_two_pass_f;
    if (run_packed) {
        auto packed_mesh = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true, packed_elements_per_pack);
        packed_fs        = sfem::FunctionSpace::create(packed_mesh, block_size);

        setenv("SFEM_PACKED_TWO_PASS", "0", 1);
        auto packed_op = sfem::create_op(packed_fs, packed_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!packed_op) {
            SFEM_ERROR("Unable to create packed operator %s\n", packed_operator_name.c_str());
        }
        set_geometry_options(packed_op, assume_affine);
        if (packed_op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to initialize packed operator %s\n", packed_operator_name.c_str());
        }
        set_generated_material(packed_op, mu, lambda);

        packed_f = sfem::Function::create(packed_fs);
        packed_f->add_operator(packed_op);

        if (run_packed_two_pass) {
            setenv("SFEM_PACKED_TWO_PASS", "1", 1);
            auto packed_two_pass_op = sfem::create_op(packed_fs, packed_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
            if (!packed_two_pass_op) {
                SFEM_ERROR("Unable to create packed two-pass operator %s\n", packed_operator_name.c_str());
            }
            set_geometry_options(packed_two_pass_op, assume_affine);
            if (packed_two_pass_op->initialize() != SFEM_SUCCESS) {
                SFEM_ERROR("Unable to initialize packed two-pass operator %s\n", packed_operator_name.c_str());
            }
            set_generated_material(packed_two_pass_op, mu, lambda);
            setenv("SFEM_PACKED_TWO_PASS", "0", 1);

            packed_two_pass_f = sfem::Function::create(packed_fs);
            packed_two_pass_f->add_operator(packed_two_pass_op);
        }
    }

    const ptrdiff_t nelements          = mesh->n_elements();
    const ptrdiff_t ndofs              = fs->n_dofs();
    auto            x                  = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            h                  = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            generated_gradient = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            baseline_gradient  = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            generated_apply    = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            baseline_apply     = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            packed_gradient    = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            packed_apply       = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            packed_two_pass_gradient = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            packed_two_pass_apply    = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            blas               = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    fill_test_vectors(ndofs, x->data(), h->data());

    generated_f->update(x->data());
    auto generated_linear_op = sfem::create_linear_operator("MF", generated_f, x, sfem::EXECUTION_SPACE_HOST);

    std::shared_ptr<sfem::Operator<real_t>> baseline_linear_op;
    if (baseline_f) {
        baseline_f->update(x->data());
        baseline_linear_op = sfem::create_linear_operator("MF", baseline_f, x, sfem::EXECUTION_SPACE_HOST);
    }
    std::shared_ptr<sfem::Operator<real_t>> packed_linear_op;
    if (packed_f) {
        packed_f->update(x->data());
        packed_linear_op = sfem::create_linear_operator("MF", packed_f, x, sfem::EXECUTION_SPACE_HOST);
    }
    std::shared_ptr<sfem::Operator<real_t>> packed_two_pass_linear_op;
    if (packed_two_pass_f) {
        packed_two_pass_f->update(x->data());
        packed_two_pass_linear_op = sfem::create_linear_operator("MF", packed_two_pass_f, x, sfem::EXECUTION_SPACE_HOST);
    }

    for (int i = 0; i < warmup; ++i) {
        blas->zeros(ndofs, generated_gradient->data());
        generated_f->gradient(x->data(), generated_gradient->data());
        blas->zeros(ndofs, generated_apply->data());
        generated_linear_op->apply(h->data(), generated_apply->data());
        if (baseline_f) {
            blas->zeros(ndofs, baseline_gradient->data());
            baseline_f->gradient(x->data(), baseline_gradient->data());
            blas->zeros(ndofs, baseline_apply->data());
            baseline_linear_op->apply(h->data(), baseline_apply->data());
        }
        if (packed_f) {
            blas->zeros(ndofs, packed_gradient->data());
            packed_f->gradient(x->data(), packed_gradient->data());
            blas->zeros(ndofs, packed_apply->data());
            packed_linear_op->apply(h->data(), packed_apply->data());
        }
        if (packed_two_pass_f) {
            blas->zeros(ndofs, packed_two_pass_gradient->data());
            packed_two_pass_f->gradient(x->data(), packed_two_pass_gradient->data());
            blas->zeros(ndofs, packed_two_pass_apply->data());
            packed_two_pass_linear_op->apply(h->data(), packed_two_pass_apply->data());
        }
    }

    const double generated_gradient_elapsed =
            time_gradient(generated_f, x->data(), generated_gradient->data(), ndofs, repeat, blas);
    const double generated_apply_elapsed =
            time_apply(generated_linear_op, h->data(), generated_apply->data(), ndofs, repeat, blas);

    double baseline_gradient_elapsed = 0;
    double baseline_apply_elapsed    = 0;
    if (baseline_f) {
        baseline_gradient_elapsed = time_gradient(baseline_f, x->data(), baseline_gradient->data(), ndofs, repeat, blas);
        baseline_apply_elapsed    = time_apply(baseline_linear_op, h->data(), baseline_apply->data(), ndofs, repeat, blas);
    }
    double packed_gradient_elapsed = 0;
    double packed_apply_elapsed    = 0;
    if (packed_f) {
        packed_gradient_elapsed = time_gradient(packed_f, x->data(), packed_gradient->data(), ndofs, repeat, blas);
        packed_apply_elapsed    = time_apply(packed_linear_op, h->data(), packed_apply->data(), ndofs, repeat, blas);
    }
    double packed_two_pass_gradient_elapsed = 0;
    double packed_two_pass_apply_elapsed    = 0;
    if (packed_two_pass_f) {
        packed_two_pass_gradient_elapsed =
                time_gradient(packed_two_pass_f, x->data(), packed_two_pass_gradient->data(), ndofs, repeat, blas);
        packed_two_pass_apply_elapsed =
                time_apply(packed_two_pass_linear_op, h->data(), packed_two_pass_apply->data(), ndofs, repeat, blas);
    }

    blas->zeros(ndofs, generated_gradient->data());
    generated_f->gradient(x->data(), generated_gradient->data());
    blas->zeros(ndofs, generated_apply->data());
    generated_linear_op->apply(h->data(), generated_apply->data());

    ErrorMetrics gradient_error{};
    ErrorMetrics apply_error{};
    if (baseline_f) {
        blas->zeros(ndofs, baseline_gradient->data());
        baseline_f->gradient(x->data(), baseline_gradient->data());
        blas->zeros(ndofs, baseline_apply->data());
        baseline_linear_op->apply(h->data(), baseline_apply->data());
        gradient_error = compare_vectors(ndofs, baseline_gradient->data(), generated_gradient->data());
        apply_error    = compare_vectors(ndofs, baseline_apply->data(), generated_apply->data());
    }
    ErrorMetrics packed_gradient_error{};
    ErrorMetrics packed_apply_error{};
    ErrorMetrics packed_gradient_error_vs_standard{};
    ErrorMetrics packed_apply_error_vs_standard{};
    if (packed_f) {
        blas->zeros(ndofs, packed_gradient->data());
        packed_f->gradient(x->data(), packed_gradient->data());
        blas->zeros(ndofs, packed_apply->data());
        packed_linear_op->apply(h->data(), packed_apply->data());
        packed_gradient_error = compare_vectors(ndofs, generated_gradient->data(), packed_gradient->data());
        packed_apply_error    = compare_vectors(ndofs, generated_apply->data(), packed_apply->data());
        if (baseline_f) {
            packed_gradient_error_vs_standard = compare_vectors(ndofs, baseline_gradient->data(), packed_gradient->data());
            packed_apply_error_vs_standard    = compare_vectors(ndofs, baseline_apply->data(), packed_apply->data());
        }
    }
    ErrorMetrics packed_two_pass_gradient_error{};
    ErrorMetrics packed_two_pass_apply_error{};
    ErrorMetrics packed_two_pass_gradient_error_vs_atomic{};
    ErrorMetrics packed_two_pass_apply_error_vs_atomic{};
    if (packed_two_pass_f) {
        blas->zeros(ndofs, packed_two_pass_gradient->data());
        packed_two_pass_f->gradient(x->data(), packed_two_pass_gradient->data());
        blas->zeros(ndofs, packed_two_pass_apply->data());
        packed_two_pass_linear_op->apply(h->data(), packed_two_pass_apply->data());
        packed_two_pass_gradient_error =
                compare_vectors(ndofs, generated_gradient->data(), packed_two_pass_gradient->data());
        packed_two_pass_apply_error = compare_vectors(ndofs, generated_apply->data(), packed_two_pass_apply->data());
        if (packed_f) {
            packed_two_pass_gradient_error_vs_atomic =
                    compare_vectors(ndofs, packed_gradient->data(), packed_two_pass_gradient->data());
            packed_two_pass_apply_error_vs_atomic =
                    compare_vectors(ndofs, packed_apply->data(), packed_two_pass_apply->data());
        }
    }

    printf("generated_operator %s\n", generated_operator_name.c_str());
    printf("baseline_operator %s\n", baseline_f ? baseline_operator_name.c_str() : "disabled");
    printf("packed_operator %s\n", packed_f ? packed_operator_name.c_str() : "disabled");
    printf("packed_two_pass_operator %s\n", packed_two_pass_f ? packed_operator_name.c_str() : "disabled");
    printf("packed_reduction %s\n", packed_f ? "atomic" : "disabled");
    printf("packed_two_pass_reduction %s\n", packed_two_pass_f ? "two_pass" : "disabled");
    printf("geometry %s\n", codegen_geometry.c_str());
    printf("element_type %s\n", type_to_string(mesh->element_type(0)));
    printf("mu %.16g\n", static_cast<double>(mu));
    printf("lambda %.16g\n", static_cast<double>(lambda));
    printf("#elements %ld\n", static_cast<long>(nelements));
    printf("#nodes %ld\n", static_cast<long>(mesh->n_nodes()));
    printf("#dofs %ld\n", static_cast<long>(ndofs));
    printf("\n%-40s %12s %16s %13s %12s %12s %10s %13s %12s\n",
           "Operation",
           "Time [s]",
           "[MElem/s]",
           "[MDOF/s]",
           "[FLOP/Elem]",
           "[B/Elem]",
           "AI",
           "[GFLOP/s]",
           "[GB/s]");
    printf("------------------------------------------------------------------------------------------------------------------------------------------"
           "------\n");
    print_rate("generated_gradient",
               generated_gradient_elapsed,
               nelements,
               ndofs,
               repeat,
               generated_f->flops_gradient(),
               generated_f->memory_traffic_bytes_gradient());
    print_rate("generated_hessian_apply",
               generated_apply_elapsed,
               nelements,
               ndofs,
               repeat,
               generated_f->flops_apply(),
               generated_f->memory_traffic_bytes_apply());
    if (baseline_f) {
        print_rate("baseline_gradient",
                   baseline_gradient_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   baseline_f->flops_gradient(),
                   baseline_f->memory_traffic_bytes_gradient());
        print_rate("baseline_hessian_apply",
                   baseline_apply_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   baseline_f->flops_apply(),
                   baseline_f->memory_traffic_bytes_apply());
    } else if (run_baseline_requested) {
        printf("baseline_skipped unsupported_element %s\n", type_to_string(mesh->element_type(0)));
    }
    if (packed_f) {
        print_rate("packed_gradient",
                   packed_gradient_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   packed_f->flops_gradient(),
                   packed_f->memory_traffic_bytes_gradient());
        print_rate("packed_hessian_apply",
                   packed_apply_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   packed_f->flops_apply(),
                   packed_f->memory_traffic_bytes_apply());
    } else if (run_packed_requested) {
        printf("packed_skipped unsupported_element %s\n", type_to_string(mesh->element_type(0)));
    }
    if (packed_two_pass_f) {
        print_rate("packed_two_pass_gradient",
                   packed_two_pass_gradient_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   packed_two_pass_f->flops_gradient(),
                   packed_two_pass_f->memory_traffic_bytes_gradient());
        print_rate("packed_two_pass_hessian_apply",
                   packed_two_pass_apply_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   packed_two_pass_f->flops_apply(),
                   packed_two_pass_f->memory_traffic_bytes_apply());
        if (packed_f) {
            printf("packed_two_pass_gradient_speedup_vs_packed_atomic %g\n",
                   packed_gradient_elapsed / packed_two_pass_gradient_elapsed);
            printf("packed_two_pass_apply_speedup_vs_packed_atomic %g\n",
                   packed_apply_elapsed / packed_two_pass_apply_elapsed);
        }
        printf("packed_two_pass_gradient_speedup_vs_generated %g\n",
               generated_gradient_elapsed / packed_two_pass_gradient_elapsed);
        printf("packed_two_pass_apply_speedup_vs_generated %g\n",
               generated_apply_elapsed / packed_two_pass_apply_elapsed);
    } else if (run_packed_two_pass_requested && run_packed) {
        printf("packed_two_pass_skipped unsupported_operator %s\n", packed_operator_name.c_str());
    }

    if (baseline_f) {
        printf("\ngradient_max_abs %.16e\n", static_cast<double>(gradient_error.max_abs));
        printf("gradient_rel_l2 %.16e\n", static_cast<double>(gradient_error.rel_l2));
        printf("apply_max_abs %.16e\n", static_cast<double>(apply_error.max_abs));
        printf("apply_rel_l2 %.16e\n", static_cast<double>(apply_error.rel_l2));
    }
    if (packed_f) {
        printf("packed_gradient_max_abs_vs_generated %.16e\n", static_cast<double>(packed_gradient_error.max_abs));
        printf("packed_gradient_rel_l2_vs_generated %.16e\n", static_cast<double>(packed_gradient_error.rel_l2));
        printf("packed_apply_max_abs_vs_generated %.16e\n", static_cast<double>(packed_apply_error.max_abs));
        printf("packed_apply_rel_l2_vs_generated %.16e\n", static_cast<double>(packed_apply_error.rel_l2));
        if (baseline_f) {
            printf("packed_gradient_max_abs_vs_standard %.16e\n", static_cast<double>(packed_gradient_error_vs_standard.max_abs));
            printf("packed_gradient_rel_l2_vs_standard %.16e\n", static_cast<double>(packed_gradient_error_vs_standard.rel_l2));
            printf("packed_apply_max_abs_vs_standard %.16e\n", static_cast<double>(packed_apply_error_vs_standard.max_abs));
            printf("packed_apply_rel_l2_vs_standard %.16e\n", static_cast<double>(packed_apply_error_vs_standard.rel_l2));
        }
    }
    if (packed_two_pass_f) {
        printf("packed_two_pass_gradient_max_abs_vs_generated %.16e\n",
               static_cast<double>(packed_two_pass_gradient_error.max_abs));
        printf("packed_two_pass_gradient_rel_l2_vs_generated %.16e\n",
               static_cast<double>(packed_two_pass_gradient_error.rel_l2));
        printf("packed_two_pass_apply_max_abs_vs_generated %.16e\n",
               static_cast<double>(packed_two_pass_apply_error.max_abs));
        printf("packed_two_pass_apply_rel_l2_vs_generated %.16e\n",
               static_cast<double>(packed_two_pass_apply_error.rel_l2));
        if (packed_f) {
            printf("packed_two_pass_gradient_max_abs_vs_packed_atomic %.16e\n",
                   static_cast<double>(packed_two_pass_gradient_error_vs_atomic.max_abs));
            printf("packed_two_pass_gradient_rel_l2_vs_packed_atomic %.16e\n",
                   static_cast<double>(packed_two_pass_gradient_error_vs_atomic.rel_l2));
            printf("packed_two_pass_apply_max_abs_vs_packed_atomic %.16e\n",
                   static_cast<double>(packed_two_pass_apply_error_vs_atomic.max_abs));
            printf("packed_two_pass_apply_rel_l2_vs_packed_atomic %.16e\n",
                   static_cast<double>(packed_two_pass_apply_error_vs_atomic.rel_l2));
        }
    }

    const bool finite = finite_vector(ndofs, generated_gradient->data()) && finite_vector(ndofs, generated_apply->data()) &&
                        (!packed_f || (finite_vector(ndofs, packed_gradient->data()) && finite_vector(ndofs, packed_apply->data()))) &&
                        (!packed_two_pass_f ||
                         (finite_vector(ndofs, packed_two_pass_gradient->data()) &&
                          finite_vector(ndofs, packed_two_pass_apply->data())));
    if (!finite) {
        return SFEM_FAILURE;
    }

    if (baseline_f && (gradient_error.max_abs > compare_atol || gradient_error.rel_l2 > compare_rtol ||
                       apply_error.max_abs > compare_atol || apply_error.rel_l2 > compare_rtol)) {
        return SFEM_FAILURE;
    }
    if (packed_f && (packed_gradient_error.max_abs > compare_atol || packed_gradient_error.rel_l2 > compare_rtol ||
                     packed_apply_error.max_abs > compare_atol || packed_apply_error.rel_l2 > compare_rtol)) {
        return SFEM_FAILURE;
    }
    if (packed_two_pass_f &&
        (packed_two_pass_gradient_error.max_abs > compare_atol || packed_two_pass_gradient_error.rel_l2 > compare_rtol ||
         packed_two_pass_apply_error.max_abs > compare_atol || packed_two_pass_apply_error.rel_l2 > compare_rtol)) {
        return SFEM_FAILURE;
    }

    return SFEM_SUCCESS;
}
