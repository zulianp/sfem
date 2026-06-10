#include "sfem_KelvinVoigtNewmark.hpp"

#include "kelvin_voigt_newmark.hpp"
#include "sshex8_kelvin_voigt_newmark.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_mesh.hpp"

#include "hex8_jacobian.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>

namespace sfem {

    namespace {

        void kv_seed_material(MultiDomainOp &m,
                              const real_t k,
                              const real_t K,
                              const real_t eta,
                              const real_t dt,
                              const real_t gamma,
                              const real_t beta,
                              const real_t rho) {
            for (auto &kv : m.domains()) {
                kv.second.parameters->set_value("k", k);
                kv.second.parameters->set_value("K", K);
                kv.second.parameters->set_value("eta", eta);
                kv.second.parameters->set_value("dt", dt);
                kv.second.parameters->set_value("gamma", gamma);
                kv.second.parameters->set_value("beta", beta);
                kv.second.parameters->set_value("rho", rho);
            }
        }

        void kv_copy_material(const MultiDomainOp &from, MultiDomainOp &to) {
            for (const auto &kv : from.domains()) {
                auto it = to.domains().find(kv.first);
                if (it == to.domains().end()) {
                    continue;
                }
                it->second.parameters->set_value("k", kv.second.parameters->require_real_value("k"));
                it->second.parameters->set_value("K", kv.second.parameters->require_real_value("K"));
                it->second.parameters->set_value("eta", kv.second.parameters->require_real_value("eta"));
                it->second.parameters->set_value("dt", kv.second.parameters->require_real_value("dt"));
                it->second.parameters->set_value("gamma", kv.second.parameters->require_real_value("gamma"));
                it->second.parameters->set_value("beta", kv.second.parameters->require_real_value("beta"));
                it->second.parameters->set_value("rho", kv.second.parameters->require_real_value("rho"));
            }
        }

    }  // namespace

    class KVJacobians {
    public:
        std::shared_ptr<Buffer<jacobian_t>> adjugate;
        std::shared_ptr<Buffer<geom_t>>     determinant;

        KVJacobians(const ptrdiff_t n_elements, const int size_adjugate)
            : adjugate(sfem::create_host_buffer<jacobian_t>(n_elements * size_adjugate)),
              determinant(sfem::create_host_buffer<geom_t>(n_elements)) {}
    };

    class KelvinVoigtNewmark::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<MultiDomainOp>  domains;
        std::shared_ptr<Buffer<real_t>> vel_[3];
        std::shared_ptr<Buffer<real_t>> acc_[3];

        real_t k{4}, K{3}, eta{0.1}, dt{0.1}, gamma{0.5}, beta{0.25}, rho{1.0};
        bool   use_affine_approximation{true};

#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif

        explicit Impl(const std::shared_ptr<FunctionSpace> &space_) : space(space_) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "KelvinVoigtNewmark::apply");
#endif
        }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    ptrdiff_t KelvinVoigtNewmark::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t KelvinVoigtNewmark::n_dofs_image() const { return impl_->space->n_dofs(); }

    std::unique_ptr<Op> KelvinVoigtNewmark::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::create");

        auto ret = std::make_unique<KelvinVoigtNewmark>(space);

        int SFEM_HEX8_ASSUME_AFFINE = ret->impl_->use_affine_approximation ? 1 : 0;
        SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
        ret->impl_->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

        return ret;
    }

    std::shared_ptr<Op> KelvinVoigtNewmark::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::lor_op");

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type())) {
            fprintf(stderr, "[Error] KelvinVoigtNewmark::lor_op NOT IMPLEMENTED for semi-structured mesh!\n");
            assert(false);
            return nullptr;
        }

        auto ret            = std::make_shared<KelvinVoigtNewmark>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        kv_copy_material(*impl_->domains, *ret->impl_->domains);
        ret->impl_->k                        = impl_->k;
        ret->impl_->K                        = impl_->K;
        ret->impl_->eta                      = impl_->eta;
        ret->impl_->dt                       = impl_->dt;
        ret->impl_->gamma                    = impl_->gamma;
        ret->impl_->beta                     = impl_->beta;
        ret->impl_->rho                      = impl_->rho;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
        return ret;
    }

    std::shared_ptr<Op> KelvinVoigtNewmark::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::derefine_op");

        if (space->has_semi_structured_mesh() && is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<KelvinVoigtNewmark>(space);
            kv_copy_material(*impl_->domains, *ret->impl_->domains);
            ret->impl_->k                        = impl_->k;
            ret->impl_->K                        = impl_->K;
            ret->impl_->eta                      = impl_->eta;
            ret->impl_->dt                       = impl_->dt;
            ret->impl_->gamma                    = impl_->gamma;
            ret->impl_->beta                     = impl_->beta;
            ret->impl_->rho                      = impl_->rho;
            ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
            return ret;
        }

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type()) &&
            !is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<KelvinVoigtNewmark>(space);
            kv_copy_material(*impl_->domains, *ret->impl_->domains);
            ret->impl_->k                        = impl_->k;
            ret->impl_->K                        = impl_->K;
            ret->impl_->eta                      = impl_->eta;
            ret->impl_->dt                       = impl_->dt;
            ret->impl_->gamma                    = impl_->gamma;
            ret->impl_->beta                     = impl_->beta;
            ret->impl_->rho                      = impl_->rho;
            ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
            assert(space->n_blocks() == 1);
            ret->override_element_types({space->element_type()});
            return ret;
        }

        auto ret            = std::make_shared<KelvinVoigtNewmark>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        kv_copy_material(*impl_->domains, *ret->impl_->domains);
        ret->impl_->k                        = impl_->k;
        ret->impl_->K                        = impl_->K;
        ret->impl_->eta                      = impl_->eta;
        ret->impl_->dt                       = impl_->dt;
        ret->impl_->gamma                    = impl_->gamma;
        ret->impl_->beta                     = impl_->beta;
        ret->impl_->rho                      = impl_->rho;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
        return ret;
    }

    KelvinVoigtNewmark::KelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {
        initialize({});
    }

    KelvinVoigtNewmark::~KelvinVoigtNewmark() = default;

    int KelvinVoigtNewmark::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::initialize");

        auto mesh      = impl_->space->mesh_ptr();
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        real_t SFEM_SHEAR_STIFFNESS_KV = 4;
        real_t SFEM_BULK_MODULUS       = 3;
        real_t SFEM_DAMPING_RATIO      = 0.1;
        real_t SFEM_DT                 = 0.1;
        real_t SFEM_GAMMA              = 0.5;
        real_t SFEM_BETA               = 0.25;
        real_t SFEM_DENSITY            = 1.0;

        SFEM_READ_ENV(SFEM_SHEAR_STIFFNESS_KV, atof);
        SFEM_READ_ENV(SFEM_BULK_MODULUS, atof);
        SFEM_READ_ENV(SFEM_DAMPING_RATIO, atof);
        SFEM_READ_ENV(SFEM_DT, atof);
        SFEM_READ_ENV(SFEM_GAMMA, atof);
        SFEM_READ_ENV(SFEM_BETA, atof);
        SFEM_READ_ENV(SFEM_DENSITY, atof);

        impl_->k     = SFEM_SHEAR_STIFFNESS_KV;
        impl_->K     = SFEM_BULK_MODULUS;
        impl_->eta   = SFEM_DAMPING_RATIO;
        impl_->dt    = SFEM_DT;
        impl_->gamma = SFEM_GAMMA;
        impl_->beta  = SFEM_BETA;
        impl_->rho   = SFEM_DENSITY;

        kv_seed_material(*impl_->domains,
                         impl_->k,
                         impl_->K,
                         impl_->eta,
                         impl_->dt,
                         impl_->gamma,
                         impl_->beta,
                         impl_->rho);

        for (auto &n2d : impl_->domains->domains()) {
            OpDomain &domain = n2d.second;
            int       dim      = mesh->spatial_dimension();
            auto      block    = domain.block;

            if (domain.element_type == smesh::HEX8 && !is_semistructured_type(domain.element_type)) {
                auto jacobians = std::make_shared<KVJacobians>(block->n_elements(), dim * dim);
                hex8_adjugate_and_det_fill(block->n_elements(),
                                           block->elements()->data(),
                                           mesh->points()->data(),
                                           jacobians->adjugate->data(),
                                           jacobians->determinant->data());
                domain.user_data = std::static_pointer_cast<void>(jacobians);
            } else if (is_semistructured_type(domain.element_type)) {
                auto jacobians = std::make_shared<KVJacobians>(block->n_elements(), dim * dim);
                const int level = smesh::semistructured_level(domain.element_type);
                sshex8_macro_hex8_adjugate_and_det_fill(level,
                                                        block->n_elements(),
                                                        block->elements()->data(),
                                                        mesh->points()->data(),
                                                        jacobians->adjugate->data(),
                                                        jacobians->determinant->data());
                domain.user_data = std::static_pointer_cast<void>(jacobians);
            }
        }

        return SFEM_SUCCESS;
    }

    void KelvinVoigtNewmark::set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &vel, int component) {
        if (strcmp(name, "velocity") == 0) {
            impl_->vel_[component] = vel;
        } else if (strcmp(name, "acceleration") == 0) {
            impl_->acc_[component] = vel;
        } else {
            SFEM_ERROR(
                    "Invalid field name! Call set_field(\"velocity\", buffer, 0/1/2) or set_field(\"acceleration\", buffer, "
                    "0/1/2) first.\n");
        }
    }

    void KelvinVoigtNewmark::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void KelvinVoigtNewmark::set_option(const std::string &name, const bool val) {
        if (name == "ASSUME_AFFINE") {
            impl_->use_affine_approximation = val;
        }
    }

    void KelvinVoigtNewmark::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    real_t KelvinVoigtNewmark::get_k() const { return impl_->k; }
    void   KelvinVoigtNewmark::set_k(const real_t val) {
        impl_->k = val;
        if (impl_->domains) {
            for (auto &kv : impl_->domains->domains()) {
                kv.second.parameters->set_value("k", val);
            }
        }
    }
    real_t KelvinVoigtNewmark::get_K() const { return impl_->K; }
    void   KelvinVoigtNewmark::set_K(const real_t val) {
        impl_->K = val;
        if (impl_->domains) {
            for (auto &kv : impl_->domains->domains()) {
                kv.second.parameters->set_value("K", val);
            }
        }
    }
    real_t KelvinVoigtNewmark::get_eta() const { return impl_->eta; }
    void   KelvinVoigtNewmark::set_eta(const real_t val) {
        impl_->eta = val;
        if (impl_->domains) {
            for (auto &kv : impl_->domains->domains()) {
                kv.second.parameters->set_value("eta", val);
            }
        }
    }
    real_t KelvinVoigtNewmark::get_dt() const { return impl_->dt; }
    void   KelvinVoigtNewmark::set_dt(const real_t val) {
        impl_->dt = val;
        if (impl_->domains) {
            for (auto &kv : impl_->domains->domains()) {
                kv.second.parameters->set_value("dt", val);
            }
        }
    }
    real_t KelvinVoigtNewmark::get_gamma() const { return impl_->gamma; }
    void   KelvinVoigtNewmark::set_gamma(const real_t val) {
        impl_->gamma = val;
        if (impl_->domains) {
            for (auto &kv : impl_->domains->domains()) {
                kv.second.parameters->set_value("gamma", val);
            }
        }
    }
    real_t KelvinVoigtNewmark::get_beta() const { return impl_->beta; }
    void   KelvinVoigtNewmark::set_beta(const real_t val) {
        impl_->beta = val;
        if (impl_->domains) {
            for (auto &kv : impl_->domains->domains()) {
                kv.second.parameters->set_value("beta", val);
            }
        }
    }
    real_t KelvinVoigtNewmark::get_rho() const { return impl_->rho; }
    void   KelvinVoigtNewmark::set_rho(const real_t val) {
        impl_->rho = val;
        if (impl_->domains) {
            for (auto &kv : impl_->domains->domains()) {
                kv.second.parameters->set_value("rho", val);
            }
        }
    }

    int KelvinVoigtNewmark::hessian_crs(const real_t *const /*x*/,
                                        const count_t *const /*rowptr*/,
                                        const idx_t *const /*colidx*/,
                                        real_t *const /*values*/) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_crs");
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int KelvinVoigtNewmark::hessian_bsr(const real_t *const /*x*/,
                                        const count_t *const /*rowptr*/,
                                        const idx_t *const /*colidx*/,
                                        real_t *const /*values*/) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_bsr");
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int KelvinVoigtNewmark::hessian_bcrs_sym(const real_t *const /*x*/,
                                             const count_t *const /*rowptr*/,
                                             const idx_t *const /*colidx*/,
                                             const ptrdiff_t /*block_stride*/,
                                             real_t **const /*diag_values*/,
                                             real_t **const /*off_diag_values*/) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_bcrs_sym");
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int KelvinVoigtNewmark::hessian_block_diag_sym(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_block_diag_sym");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        real_t *const out0 = &values[0];
        real_t *const out1 = &values[1];
        real_t *const out2 = &values[2];
        real_t *const out3 = &values[3];
        real_t *const out4 = &values[4];
        real_t *const out5 = &values[5];

        err = impl_->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto params       = domain.parameters;
            const real_t beta  = params->require_real_value("beta");
            const real_t gamma = params->require_real_value("gamma");
            const real_t dt    = params->require_real_value("dt");
            const real_t k     = params->require_real_value("k");
            const real_t K     = params->require_real_value("K");
            const real_t eta   = params->require_real_value("eta");
            const real_t rho   = params->require_real_value("rho");

            SFEM_TRACE_SCOPE("kelvin_voigt_newmark_block_diag_sym");

            return kelvin_voigt_newmark_block_diag_sym(domain.element_type,
                                                       block->n_elements(),
                                                       mesh->n_nodes(),
                                                       block->elements()->data(),
                                                       mesh->points()->data(),
                                                       beta,
                                                       gamma,
                                                       dt,
                                                       k,
                                                       K,
                                                       eta,
                                                       rho,
                                                       6,
                                                       out0,
                                                       out1,
                                                       out2,
                                                       out3,
                                                       out4,
                                                       out5);
        });

        return err;
    }

    int KelvinVoigtNewmark::hessian_diag(const real_t *const /*x*/, real_t *const out) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block  = domain.block;
            auto params = domain.parameters;
            const real_t beta  = params->require_real_value("beta");
            const real_t gamma = params->require_real_value("gamma");
            const real_t dt    = params->require_real_value("dt");
            const real_t k     = params->require_real_value("k");
            const real_t K     = params->require_real_value("K");
            const real_t eta   = params->require_real_value("eta");
            const real_t rho   = params->require_real_value("rho");

            return kelvin_voigt_newmark_assemble_diag_aos(domain.element_type,
                                                          block->n_elements(),
                                                          mesh->n_nodes(),
                                                          block->elements()->data(),
                                                          mesh->points()->data(),
                                                          beta,
                                                          gamma,
                                                          dt,
                                                          k,
                                                          K,
                                                          eta,
                                                          rho,
                                                          out);
        });

        return err;
    }

    int KelvinVoigtNewmark::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::gradient");
        auto          mesh = impl_->space->mesh_ptr();
        const real_t *u    = x;

        const real_t *vbase = impl_->vel_[0]->data();
        const real_t *vx    = &vbase[0];
        const real_t *vy    = &vbase[1];
        const real_t *vz    = &vbase[2];

        const real_t *abase = impl_->acc_[0]->data();
        const real_t *ax    = &abase[0];
        const real_t *ay    = &abase[1];
        const real_t *az    = &abase[2];

        const double tick = MPI_Wtime();
        int          err  = SFEM_SUCCESS;

        err = impl_->iterate([&](const OpDomain &domain) {
            auto          block        = domain.block;
            auto          element_type = domain.element_type;
            auto          params       = domain.parameters;
            const real_t k             = params->require_real_value("k");
            const real_t K             = params->require_real_value("K");
            const real_t eta           = params->require_real_value("eta");
            const real_t rho           = params->require_real_value("rho");

            SFEM_TRACE_SCOPE("kelvin_voigt_newmark_gradient_aos");

            if (!domain.user_data) {
                SFEM_ERROR(
                        "KelvinVoigtNewmark::gradient: Jacobian cache missing; initialize() must run for HEX8 and "
                        "semi-structured hex domains.\n");
            }
            auto jacobians = std::static_pointer_cast<KVJacobians>(domain.user_data);
            return kelvin_voigt_newmark_gradient_aos(element_type,
                                                     block->n_elements(),
                                                     mesh->n_nodes(),
                                                     block->elements()->data(),
                                                     jacobians->adjugate->data(),
                                                     jacobians->determinant->data(),
                                                     k,
                                                     K,
                                                     eta,
                                                     rho,
                                                     u,
                                                     vx,
                                                     vy,
                                                     vz,
                                                     ax,
                                                     ay,
                                                     az,
                                                     out);
        });

        return err;
    }

    int KelvinVoigtNewmark::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::apply");
        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        err = impl_->iterate([&](const OpDomain &domain) {
            auto          block        = domain.block;
            auto          element_type = domain.element_type;
            auto          params       = domain.parameters;
            const real_t dt            = params->require_real_value("dt");
            const real_t gamma         = params->require_real_value("gamma");
            const real_t beta          = params->require_real_value("beta");
            const real_t k             = params->require_real_value("k");
            const real_t K             = params->require_real_value("K");
            const real_t eta           = params->require_real_value("eta");
            const real_t rho           = params->require_real_value("rho");

            SFEM_TRACE_SCOPE("kelvin_voigt_newmark_apply_adjugate_aos");

            if (!domain.user_data) {
                SFEM_ERROR(
                        "KelvinVoigtNewmark::apply: Jacobian cache missing; initialize() must run for HEX8 and "
                        "semi-structured hex domains.\n");
            }
            auto jacobians = std::static_pointer_cast<KVJacobians>(domain.user_data);
            return kelvin_voigt_newmark_apply_adjugate_aos(element_type,
                                                           block->n_elements(),
                                                           mesh->n_nodes(),
                                                           block->elements()->data(),
                                                           jacobians->adjugate->data(),
                                                           jacobians->determinant->data(),
                                                           dt,
                                                           gamma,
                                                           beta,
                                                           k,
                                                           K,
                                                           eta,
                                                           rho,
                                                           h,
                                                           out);
        });

        return err;
    }

    int KelvinVoigtNewmark::value(const real_t * /*x*/, real_t *const /*out*/) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::value");
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int KelvinVoigtNewmark::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> KelvinVoigtNewmark::clone() const {
        auto ret                             = std::make_shared<KelvinVoigtNewmark>(impl_->space);
        ret->impl_->domains = impl_->domains;
        for (int c = 0; c < 3; c++) {
            ret->impl_->vel_[c] = impl_->vel_[c];
            ret->impl_->acc_[c] = impl_->acc_[c];
        }
        ret->impl_->k                        = impl_->k;
        ret->impl_->K                        = impl_->K;
        ret->impl_->eta                      = impl_->eta;
        ret->impl_->dt                       = impl_->dt;
        ret->impl_->gamma                    = impl_->gamma;
        ret->impl_->beta                     = impl_->beta;
        ret->impl_->rho                      = impl_->rho;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
        return ret;
    }

    std::unique_ptr<Op> create_kelvin_voigt_newmark(const std::shared_ptr<FunctionSpace> &space) {
        return KelvinVoigtNewmark::create(space);
    }

}  // namespace sfem
