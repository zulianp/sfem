#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeneratedLaplace.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"
#include "smesh_mesh_reorder.hpp"

namespace {

    double mdofs_per_second(const double elapsed, const ptrdiff_t ndofs, const int repeat) {
        return 1e-6 * static_cast<double>(ndofs) / (elapsed / repeat);
    }

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
        const double gflops_per_s = 1e-9 * flops / seconds_per_call;
        const double gbytes_per_s = 1e-9 * static_cast<double>(memory_traffic_bytes) / seconds_per_call;

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

    bool generated_laplace_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TRI3:
            case smesh::TRI6:
            case smesh::QUAD4:
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
            case smesh::HEX27:
            case smesh::PROTEUS_QUAD4:
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

    bool baseline_laplace_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
                return true;
            default:
                return false;
        }
    }

    bool packed_laplace_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
                return true;
            default:
                return false;
        }
    }

    bool generated_laplace_affine_geometry_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
                return true;
            default:
                return false;
        }
    }

    bool generated_laplace_packed_isoparametric_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TRI6:
            case smesh::QUAD4:
            case smesh::TET10:
            case smesh::HEX8:
            case smesh::HEX27:
            case smesh::PROTEUS_QUAD4:
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

    bool generated_laplace_packed_supported(const smesh::ElemType element_type, const bool assume_affine) {
        return assume_affine ? packed_laplace_supported(element_type)
                             : generated_laplace_packed_isoparametric_supported(element_type);
    }

    void set_geometry_options(const std::shared_ptr<sfem::Op> &op, const bool assume_affine) {
        op->set_option("ASSUME_AFFINE_RESIDUAL", assume_affine);
        op->set_option("ASSUME_AFFINE_GRADIENT", assume_affine);
        op->set_option("ASSUME_AFFINE_JACOBIAN_ACTION", assume_affine);
        op->set_option("ASSUME_AFFINE_APPLY", assume_affine);
        op->set_option("ASSUME_AFFINE", assume_affine);
        op->set_option("assume_affine", assume_affine);
    }

    void require_success(const int err, const char *const label) {
        if (err != SFEM_SUCCESS) {
            SFEM_ERROR("%s failed with code %d\n", label, err);
        }
    }

    std::shared_ptr<sfem::Mesh> create_benchmark_mesh(const std::shared_ptr<sfem::Communicator> &comm,
                                                      const smesh::ElemType                      element_type,
                                                      const int                                  resolution) {
        switch (element_type) {
            case smesh::TRI3:
            case smesh::QUAD4:
                return sfem::Mesh::create_square(comm, element_type, resolution, resolution, 0, 0, 1, 1);
            case smesh::PROTEUS_QUAD4:
                return sfem::Mesh::create_square(comm, smesh::QUAD4, resolution, resolution, 0, 0, 1, 1);
            case smesh::TRI6: {
                auto mesh = sfem::Mesh::create_square(comm, smesh::TRI3, resolution, resolution, 0, 0, 1, 1);
                return smesh::promote_to(smesh::TRI6, mesh);
            }
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
            case smesh::HEX27:
            case smesh::PROTEUS_HEX8:
            case smesh::PROTEUS_HEX27:
            case smesh::PROTEUS_HEX64:
            case smesh::PROTEUS_HEX125:
            case smesh::PROTEUS_HEX729:
                return sfem::Mesh::create_cube(comm, element_type, resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
            default:
                return nullptr;
        }
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
            require_success(f->gradient(x, out), "bench_laplace gradient");
        }
        sfem::device_synchronize();
        return MPI_Wtime() - t0;
    }

    double time_apply(const std::shared_ptr<sfem::Op>               &op,
                      const real_t *const                            state,
                      const real_t *const                            direction,
                      real_t *const                                  out,
                      const ptrdiff_t                                ndofs,
                      const int                                      repeat,
                      const std::shared_ptr<sfem::BLAS_Tpl<real_t>> &blas) {
        blas->zeros(ndofs, out);
        sfem::device_synchronize();
        const double t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            require_success(op->apply(state, direction, out), "bench_laplace apply");
        }
        sfem::device_synchronize();
        return MPI_Wtime() - t0;
    }

    real_t max_abs_diff(const real_t *const left, const real_t *const right, const ptrdiff_t n) {
        real_t max_diff = 0;
#pragma omp parallel for reduction(max : max_diff)
        for (ptrdiff_t i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, static_cast<real_t>(std::abs(left[i] - right[i])));
        }
        return max_diff;
    }

}  // namespace

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    auto          comm = context.communicator();

    if (comm->size() != 1) {
        SFEM_ERROR("bench_laplace.exe supports one MPI rank\n");
    }

    const int         resolution              = smesh::Env::read("SFEM_BASE_RESOLUTION", 64);
    const int         warmup                  = smesh::Env::read("SFEM_WARMUP", 3);
    const int         repeat                  = smesh::Env::read("SFEM_REPEAT", 20);
    const std::string generated_operator_name = smesh::Env::read_string("SFEM_GENERATED_OPERATOR", "GeneratedLaplace");
    const std::string baseline_operator_name  = smesh::Env::read_string("SFEM_BASELINE_OPERATOR", "Laplacian");
    const std::string generated_packed_operator_name =
            smesh::Env::read_string("SFEM_GENERATED_PACKED_OPERATOR", generated_operator_name.c_str());
    const std::string packed_operator_name      = smesh::Env::read_string("SFEM_PACKED_OPERATOR", "PackedLaplacian");
    const std::string codegen_geometry          = smesh::Env::read_string("SFEM_CODEGEN_GEOMETRY", "affine");
    const std::string generated_packed_geometry =
            smesh::Env::read_string("SFEM_GENERATED_PACKED_GEOMETRY", codegen_geometry.c_str());
    const bool        run_baseline              = smesh::Env::read("SFEM_RUN_BASELINE", true);
    const bool        run_generated_packed      = smesh::Env::read("SFEM_RUN_GENERATED_PACKED", true);
    const bool        run_packed                = smesh::Env::read("SFEM_RUN_PACKED", true);
    const bool        run_packed_two_pass       = smesh::Env::read("SFEM_RUN_PACKED_TWO_PASS", true);

    if (codegen_geometry != "affine" && codegen_geometry != "isoparametric") {
        SFEM_ERROR("SFEM_CODEGEN_GEOMETRY must be affine or isoparametric\n");
    }
    if (generated_packed_geometry != "affine" && generated_packed_geometry != "isoparametric") {
        SFEM_ERROR("SFEM_GENERATED_PACKED_GEOMETRY must be affine or isoparametric\n");
    }
    const auto element_type =
            static_cast<smesh::ElemType>(smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "TET4").c_str()));
    const bool assume_affine = codegen_geometry == "affine" && generated_laplace_affine_geometry_supported(element_type);
    const bool generated_packed_supports_isoparametric =
            generated_laplace_packed_isoparametric_supported(element_type);
    const bool generated_packed_assume_affine =
            generated_laplace_affine_geometry_supported(element_type) &&
            (generated_packed_geometry == "affine" ||
             (generated_packed_geometry == "isoparametric" && !generated_packed_supports_isoparametric));
    auto mesh = create_benchmark_mesh(comm, element_type, resolution);

    if (!mesh) {
        SFEM_ERROR("bench_laplace.exe cannot create mesh for SFEM_ELEM_TYPE=%s\n", type_to_string(element_type));
    }

    auto sfc = smesh::SFC::create_from_env();
    sfc->reorder(*mesh);
    if (element_type == smesh::PROTEUS_QUAD4) {
        mesh->block(0)->set_element_type(smesh::PROTEUS_QUAD4);
    }

    if (!generated_laplace_supported(mesh->element_type(0))) {
        SFEM_ERROR("generated Laplace path does not support SFEM_ELEM_TYPE=%s\n", type_to_string(mesh->element_type(0)));
    }

    auto fs = sfem::FunctionSpace::create(mesh, 1);

    auto                      generated_f = sfem::Function::create(fs);
    std::shared_ptr<sfem::Op> generated_op;
    if (generated_operator_name == "GeneratedLaplace") {
        generated_op = std::make_shared<sfem::GeneratedLaplace>(fs);
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
    generated_f->add_operator(generated_op);

    const bool run_baseline_checked = run_baseline && baseline_laplace_supported(mesh->element_type(0));
    const bool run_generated_packed_checked =
            run_generated_packed && generated_laplace_packed_supported(mesh->element_type(0), generated_packed_assume_affine);
    const bool                      run_packed_checked = run_packed && packed_laplace_supported(mesh->element_type(0));
    std::shared_ptr<sfem::Function> baseline_f;
    std::shared_ptr<sfem::Op>       baseline_op;
    if (run_baseline_checked) {
        baseline_f  = sfem::Function::create(fs);
        baseline_op = sfem::create_op(fs, baseline_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!baseline_op) {
            SFEM_ERROR("Unable to create baseline operator %s\n", baseline_operator_name.c_str());
        }
        set_geometry_options(baseline_op, assume_affine);
        if (baseline_op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to initialize baseline operator %s\n", baseline_operator_name.c_str());
        }
        baseline_f->add_operator(baseline_op);
    }

    std::shared_ptr<sfem::FunctionSpace> packed_fs;
    const bool                           run_packed_two_pass_checked =
            run_packed_two_pass && run_packed_checked && packed_operator_name == "PackedLaplacian";
    if (run_generated_packed_checked || run_packed_checked) {
        auto packed_mesh = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true);
        packed_fs        = sfem::FunctionSpace::create(packed_mesh, 1);
    }

    std::shared_ptr<sfem::Op> generated_packed_op;
    if (run_generated_packed_checked) {
        setenv("SFEM_PACKED_TWO_PASS", "0", 1);
        if (generated_packed_operator_name == "GeneratedLaplace") {
            generated_packed_op = std::make_shared<sfem::GeneratedLaplace>(packed_fs);
        } else {
            generated_packed_op = sfem::create_op(packed_fs, generated_packed_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
            if (!generated_packed_op) {
                SFEM_ERROR("Unable to create generated packed operator %s\n", generated_packed_operator_name.c_str());
            }
        }
        set_geometry_options(generated_packed_op, generated_packed_assume_affine);
        if (generated_packed_op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to initialize generated packed operator %s\n", generated_packed_operator_name.c_str());
        }
    }

    std::shared_ptr<sfem::Op> packed_op;
    if (run_packed_checked) {
        setenv("SFEM_PACKED_TWO_PASS", "0", 1);
        packed_op = sfem::create_op(packed_fs, packed_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!packed_op) {
            SFEM_ERROR("Unable to create packed operator %s\n", packed_operator_name.c_str());
        }
        if (packed_op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to initialize packed operator %s\n", packed_operator_name.c_str());
        }
    }

    std::shared_ptr<sfem::Op> packed_two_pass_op;
    if (run_packed_two_pass_checked) {
        setenv("SFEM_PACKED_TWO_PASS", "1", 1);
        packed_two_pass_op = sfem::create_op(packed_fs, packed_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!packed_two_pass_op) {
            SFEM_ERROR("Unable to create packed two-pass operator %s\n", packed_operator_name.c_str());
        }
        if (packed_two_pass_op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to initialize packed two-pass operator %s\n", packed_operator_name.c_str());
        }
        setenv("SFEM_PACKED_TWO_PASS", "0", 1);
    }

    const ptrdiff_t nelements        = mesh->n_elements();
    const ptrdiff_t ndofs            = fs->n_dofs();
    auto            x                = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            direction        = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            generated        = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            baseline         = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            generated_packed = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            packed           = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            packed_two_pass  = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            blas             = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        x->data()[i]         = static_cast<real_t>((i % 97) + 1) / 97;
        direction->data()[i] = static_cast<real_t>(((i * 7) % 101) + 1) / 101;
    }

    for (int i = 0; i < warmup; ++i) {
        blas->zeros(ndofs, generated->data());
        require_success(generated_f->gradient(x->data(), generated->data()), "generated warmup gradient");
        blas->zeros(ndofs, generated->data());
        require_success(generated_op->apply(x->data(), direction->data(), generated->data()), "generated warmup apply");
        if (baseline_f) {
            blas->zeros(ndofs, baseline->data());
            require_success(baseline_f->gradient(x->data(), baseline->data()), "baseline warmup gradient");
            blas->zeros(ndofs, baseline->data());
            require_success(baseline_op->apply(x->data(), direction->data(), baseline->data()), "baseline warmup apply");
        }
        if (generated_packed_op) {
            blas->zeros(ndofs, generated_packed->data());
            require_success(generated_packed_op->apply(x->data(), direction->data(), generated_packed->data()),
                            "generated packed warmup apply");
        }
        if (packed_op) {
            blas->zeros(ndofs, packed->data());
            require_success(packed_op->apply(x->data(), direction->data(), packed->data()), "packed warmup apply");
        }
        if (packed_two_pass_op) {
            blas->zeros(ndofs, packed_two_pass->data());
            require_success(packed_two_pass_op->apply(x->data(), direction->data(), packed_two_pass->data()),
                            "packed two-pass warmup apply");
        }
    }

    blas->zeros(ndofs, generated->data());
    require_success(generated_f->gradient(x->data(), generated->data()), "generated gradient");
    real_t gradient_max_abs_diff = 0;
    if (baseline_f) {
        blas->zeros(ndofs, baseline->data());
        require_success(baseline_f->gradient(x->data(), baseline->data()), "baseline gradient");
        gradient_max_abs_diff = max_abs_diff(generated->data(), baseline->data(), ndofs);
    }

    blas->zeros(ndofs, generated->data());
    require_success(generated_op->apply(x->data(), direction->data(), generated->data()), "generated apply");
    real_t apply_max_abs_diff = 0;
    if (baseline_op) {
        blas->zeros(ndofs, baseline->data());
        require_success(baseline_op->apply(x->data(), direction->data(), baseline->data()), "baseline apply");
        apply_max_abs_diff = max_abs_diff(generated->data(), baseline->data(), ndofs);
    }
    real_t generated_packed_apply_max_abs_diff = 0;
    if (generated_packed_op) {
        blas->zeros(ndofs, generated_packed->data());
        require_success(generated_packed_op->apply(x->data(), direction->data(), generated_packed->data()),
                        "generated packed apply");
        generated_packed_apply_max_abs_diff =
                max_abs_diff(generated_packed->data(), baseline_op ? baseline->data() : generated->data(), ndofs);
    }
    real_t packed_apply_max_abs_diff = 0;
    if (packed_op) {
        blas->zeros(ndofs, packed->data());
        require_success(packed_op->apply(x->data(), direction->data(), packed->data()), "packed apply");
        packed_apply_max_abs_diff = max_abs_diff(packed->data(), baseline_op ? baseline->data() : generated->data(), ndofs);
    }
    real_t packed_two_pass_apply_max_abs_diff = 0;
    if (packed_two_pass_op) {
        blas->zeros(ndofs, packed_two_pass->data());
        require_success(packed_two_pass_op->apply(x->data(), direction->data(), packed_two_pass->data()),
                        "packed two-pass apply");
        packed_two_pass_apply_max_abs_diff =
                max_abs_diff(packed_two_pass->data(), baseline_op ? baseline->data() : generated->data(), ndofs);
    }

    const double generated_gradient_elapsed = time_gradient(generated_f, x->data(), generated->data(), ndofs, repeat, blas);
    const double generated_apply_elapsed =
            time_apply(generated_op, x->data(), direction->data(), generated->data(), ndofs, repeat, blas);

    double baseline_gradient_elapsed      = 0;
    double baseline_apply_elapsed         = 0;
    double generated_packed_apply_elapsed = 0;
    if (generated_packed_op) {
        generated_packed_apply_elapsed =
                time_apply(generated_packed_op, x->data(), direction->data(), generated_packed->data(), ndofs, repeat, blas);
    }
    double packed_apply_elapsed = 0;
    if (packed_op) {
        packed_apply_elapsed = time_apply(packed_op, x->data(), direction->data(), packed->data(), ndofs, repeat, blas);
    }
    double packed_two_pass_apply_elapsed = 0;
    if (packed_two_pass_op) {
        packed_two_pass_apply_elapsed =
                time_apply(packed_two_pass_op, x->data(), direction->data(), packed_two_pass->data(), ndofs, repeat, blas);
    }
    if (baseline_f) {
        baseline_gradient_elapsed = time_gradient(baseline_f, x->data(), baseline->data(), ndofs, repeat, blas);
        baseline_apply_elapsed    = time_apply(baseline_op, x->data(), direction->data(), baseline->data(), ndofs, repeat, blas);
    }

    printf("generated_operator %s\n", generated_operator_name.c_str());
    printf("baseline_operator %s\n", baseline_f ? baseline_operator_name.c_str() : "disabled");
    printf("generated_packed_operator %s\n", generated_packed_op ? generated_packed_operator_name.c_str() : "disabled");
    printf("packed_operator %s\n", packed_op ? packed_operator_name.c_str() : "disabled");
    printf("packed_two_pass_operator %s\n", packed_two_pass_op ? packed_operator_name.c_str() : "disabled");
    printf("packed_reduction %s\n", packed_op ? "atomic" : "disabled");
    printf("packed_two_pass_reduction %s\n", packed_two_pass_op ? "two_pass" : "disabled");
    printf("geometry %s\n", assume_affine ? "affine" : "isoparametric");
    printf("generated_packed_geometry %s\n", generated_packed_assume_affine ? "affine" : "isoparametric");
    if (codegen_geometry == "affine" && !assume_affine) {
        printf("requested_geometry affine\n");
        printf("geometry_fallback unsupported_affine_cache %s\n", type_to_string(mesh->element_type(0)));
    }
    if (generated_packed_geometry == "affine" && !generated_packed_assume_affine) {
        printf("requested_generated_packed_geometry affine\n");
        printf("generated_packed_geometry_fallback unsupported_affine_cache %s\n", type_to_string(mesh->element_type(0)));
    }
    if (generated_packed_geometry == "isoparametric" && generated_packed_assume_affine) {
        printf("requested_generated_packed_geometry isoparametric\n");
        printf("generated_packed_geometry_fallback unsupported_isoparametric_packed %s\n",
               type_to_string(mesh->element_type(0)));
    }
    printf("element_type %s\n", type_to_string(mesh->element_type(0)));
    printf("#elements %ld\n", static_cast<long>(nelements));
    printf("#nodes %ld\n", static_cast<long>(mesh->n_nodes()));
    printf("#dofs %ld\n", static_cast<long>(ndofs));
    printf("gradient_max_abs_diff %g\n", static_cast<double>(gradient_max_abs_diff));
    printf("apply_max_abs_diff %g\n", static_cast<double>(apply_max_abs_diff));
    printf("generated_packed_apply_max_abs_diff %g\n", static_cast<double>(generated_packed_apply_max_abs_diff));
    printf("packed_apply_max_abs_diff %g\n", static_cast<double>(packed_apply_max_abs_diff));
    printf("packed_two_pass_apply_max_abs_diff %g\n", static_cast<double>(packed_two_pass_apply_max_abs_diff));
    printf("\n%-40s %12s %16s %13s %12s %12s %10s %13s %12s\n",
           "Operation",
           "Time [s]",
           "Rate [MElem/s]",
           "Rate [MDOF/s]",
           "[FLOP/Elem]",
           "[B/Elem]",
           "AI",
           "Rate [GFLOP/s]",
           "Rate [GB/s]");
    printf("---------------------------------------------------------------------------------------------------------------------"
           "\n");
    print_rate("generated_gradient",
               generated_gradient_elapsed,
               nelements,
               ndofs,
               repeat,
               generated_f->flops_gradient(),
               generated_f->memory_traffic_bytes_gradient());
    print_rate("generated_apply",
               generated_apply_elapsed,
               nelements,
               ndofs,
               repeat,
               generated_f->flops_apply(),
               generated_f->memory_traffic_bytes_apply());
    if (generated_packed_op) {
        print_rate("generated_packed_apply",
                   generated_packed_apply_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   generated_packed_op->flops_apply(),
                   generated_packed_op->memory_traffic_bytes_apply());
    }
    if (packed_op) {
        print_rate("packed_apply",
                   packed_apply_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   packed_op->flops_apply(),
                   packed_op->memory_traffic_bytes_apply());
    }
    if (packed_two_pass_op) {
        print_rate("packed_two_pass_apply",
                   packed_two_pass_apply_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   packed_two_pass_op->flops_apply(),
                   packed_two_pass_op->memory_traffic_bytes_apply());
    }
    if (baseline_f) {
        print_rate("baseline_gradient",
                   baseline_gradient_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   baseline_f->flops_gradient(),
                   baseline_f->memory_traffic_bytes_gradient());
        print_rate("baseline_apply",
                   baseline_apply_elapsed,
                   nelements,
                   ndofs,
                   repeat,
                   baseline_f->flops_apply(),
                   baseline_f->memory_traffic_bytes_apply());
        printf("gradient_speedup_vs_baseline %g\n", baseline_gradient_elapsed / generated_gradient_elapsed);
        printf("apply_speedup_vs_baseline %g\n", baseline_apply_elapsed / generated_apply_elapsed);
        if (generated_packed_op) {
            printf("generated_packed_apply_speedup_vs_baseline %g\n", baseline_apply_elapsed / generated_packed_apply_elapsed);
        }
        if (packed_op) {
            printf("packed_apply_speedup_vs_baseline %g\n", baseline_apply_elapsed / packed_apply_elapsed);
        }
        if (packed_two_pass_op) {
            printf("packed_two_pass_apply_speedup_vs_baseline %g\n",
                   baseline_apply_elapsed / packed_two_pass_apply_elapsed);
        }
    } else if (run_baseline) {
        printf("baseline_skipped unsupported_element %s\n", type_to_string(mesh->element_type(0)));
    }
    printf("generated_gradient_mdofs_per_s %g\n", mdofs_per_second(generated_gradient_elapsed, ndofs, repeat));
    printf("generated_apply_mdofs_per_s %g\n", mdofs_per_second(generated_apply_elapsed, ndofs, repeat));
    if (generated_packed_op) {
        printf("generated_packed_apply_mdofs_per_s %g\n", mdofs_per_second(generated_packed_apply_elapsed, ndofs, repeat));
    }
    if (packed_op) {
        printf("packed_apply_mdofs_per_s %g\n", mdofs_per_second(packed_apply_elapsed, ndofs, repeat));
    }
    if (packed_two_pass_op) {
        printf("packed_two_pass_apply_mdofs_per_s %g\n", mdofs_per_second(packed_two_pass_apply_elapsed, ndofs, repeat));
    }
    if (baseline_f) {
        printf("baseline_gradient_mdofs_per_s %g\n", mdofs_per_second(baseline_gradient_elapsed, ndofs, repeat));
        printf("baseline_apply_mdofs_per_s %g\n", mdofs_per_second(baseline_apply_elapsed, ndofs, repeat));
    }
    if (packed_op) {
        printf("packed_apply_speedup_vs_generated %g\n", generated_apply_elapsed / packed_apply_elapsed);
    } else if (run_packed) {
        printf("packed_skipped unsupported_element %s\n", type_to_string(mesh->element_type(0)));
    }
    if (packed_two_pass_op) {
        printf("packed_two_pass_apply_speedup_vs_generated %g\n",
               generated_apply_elapsed / packed_two_pass_apply_elapsed);
        if (packed_op) {
            printf("packed_two_pass_apply_speedup_vs_packed_atomic %g\n",
                   packed_apply_elapsed / packed_two_pass_apply_elapsed);
        }
    } else if (run_packed_two_pass && run_packed_checked) {
        printf("packed_two_pass_skipped unsupported_operator %s\n", packed_operator_name.c_str());
    }
    if (generated_packed_op) {
        printf("generated_packed_apply_speedup_vs_generated %g\n", generated_apply_elapsed / generated_packed_apply_elapsed);
    } else if (run_generated_packed) {
        printf("generated_packed_skipped unsupported_element %s\n", type_to_string(mesh->element_type(0)));
    }

    return SFEM_SUCCESS;
}
