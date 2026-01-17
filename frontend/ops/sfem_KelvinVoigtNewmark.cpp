#include "sfem_KelvinVoigtNewmark.hpp"

// C includes
#include "kelvin_voigt_newmark.h"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_Tracer.hpp"

#include <mpi.h>
#include "hex8_jacobian.h"

namespace sfem {

    class KVJacobians {
    public:
        std::shared_ptr<Buffer<jacobian_t>> adjugate;
        std::shared_ptr<Buffer<jacobian_t>> determinant;

        KVJacobians(const ptrdiff_t n_elements, const int size_adjugate)
            : adjugate(sfem::create_host_buffer<jacobian_t>(n_elements * size_adjugate)),
              determinant(sfem::create_host_buffer<jacobian_t>(n_elements)) {}
    };

    class KelvinVoigtNewmark::Impl {
    public:
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<MultiDomainOp>  domains;
        std::shared_ptr<Buffer<real_t>> vel_[3];
        std::shared_ptr<Buffer<real_t>> acc_[3];
        enum ElemType                   element_type { INVALID };

        real_t k{4}, K{3}, eta{0.1}, dt{0.1}, gamma{0.5}, beta{0.25}, rho{1.0};

        long   calls{0};
        double total_time{0};

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

    // Factory
    std::unique_ptr<Op> KelvinVoigtNewmark::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::create");

        auto ret = std::make_unique<KelvinVoigtNewmark>(space);

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

        ret->impl_->k            = SFEM_SHEAR_STIFFNESS_KV;
        ret->impl_->K            = SFEM_BULK_MODULUS;
        ret->impl_->eta          = SFEM_DAMPING_RATIO;
        ret->impl_->dt           = SFEM_DT;
        ret->impl_->gamma        = SFEM_GAMMA;
        ret->impl_->beta         = SFEM_BETA;
        ret->impl_->rho          = SFEM_DENSITY;
        ret->impl_->element_type = (enum ElemType)space->element_type();

        return ret;
    }

    // LOR / derefine
    std::shared_ptr<Op> KelvinVoigtNewmark::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::lor_op");

        auto ret            = std::make_shared<KelvinVoigtNewmark>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        ret->impl_->k       = impl_->k;
        ret->impl_->K       = impl_->K;
        ret->impl_->eta     = impl_->eta;
        ret->impl_->dt      = impl_->dt;
        ret->impl_->gamma   = impl_->gamma;
        ret->impl_->beta    = impl_->beta;
        ret->impl_->rho     = impl_->rho;
        return ret;
    }

    std::shared_ptr<Op> KelvinVoigtNewmark::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::derefine_op");

        auto ret            = std::make_shared<KelvinVoigtNewmark>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        ret->impl_->k       = impl_->k;
        ret->impl_->K       = impl_->K;
        ret->impl_->eta     = impl_->eta;
        ret->impl_->dt      = impl_->dt;
        ret->impl_->gamma   = impl_->gamma;
        ret->impl_->beta    = impl_->beta;
        ret->impl_->rho     = impl_->rho;
        return ret;
    }

    // Lifecycle
    KelvinVoigtNewmark::KelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {
        initialize({});
    }

    KelvinVoigtNewmark::~KelvinVoigtNewmark() {
        if (impl_->calls) {
            printf("KelvinVoigtNewmark::apply called %ld times. Total: %g [s], Avg: %g [s], TP %g [MDOF/s]\n",
                   impl_->calls,
                   impl_->total_time,
                   impl_->total_time / impl_->calls,
                   1e-6 * impl_->space->n_dofs() / (impl_->total_time / impl_->calls));
        }
    }

    int KelvinVoigtNewmark::initialize(const std::vector<std::string> &block_names) {
        auto mesh      = impl_->space->mesh_ptr();
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        if (impl_->space->element_type() == HEX8) {
            int dim = mesh->spatial_dimension();
            for (auto &domain : impl_->domains->domains()) {
                auto block     = domain.second.block;
                auto jacobians = std::make_shared<KVJacobians>(block->n_elements(), dim * dim);
                hex8_adjugate_and_det_fill(block->n_elements(),
                                           block->elements()->data(),
                                           mesh->points()->data(),
                                           jacobians->adjugate->data(),
                                           jacobians->determinant->data());
                domain.second.user_data = std::static_pointer_cast<void>(jacobians);
            }
        }

        return SFEM_SUCCESS;
    }

    // Fields / options
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

    void KelvinVoigtNewmark::set_value_in_block(const std::string & /*block_name*/,
                                                const std::string &var_name,
                                                const real_t       value) {
        impl_->domains->set_value_in_block("", var_name, value);
    }

    // Accessors
    real_t KelvinVoigtNewmark::get_k() const { return impl_->k; }
    void   KelvinVoigtNewmark::set_k(real_t val) { impl_->k = val; }
    real_t KelvinVoigtNewmark::get_K() const { return impl_->K; }
    void   KelvinVoigtNewmark::set_K(real_t val) { impl_->K = val; }
    real_t KelvinVoigtNewmark::get_eta() const { return impl_->eta; }
    void   KelvinVoigtNewmark::set_eta(real_t val) { impl_->eta = val; }
    real_t KelvinVoigtNewmark::get_dt() const { return impl_->dt; }
    void   KelvinVoigtNewmark::set_dt(real_t val) { impl_->dt = val; }
    real_t KelvinVoigtNewmark::get_gamma() const { return impl_->gamma; }
    void   KelvinVoigtNewmark::set_gamma(real_t val) { impl_->gamma = val; }
    real_t KelvinVoigtNewmark::get_beta() const { return impl_->beta; }
    void   KelvinVoigtNewmark::set_beta(real_t val) { impl_->beta = val; }
    real_t KelvinVoigtNewmark::get_rho() const { return impl_->rho; }
    void   KelvinVoigtNewmark::set_rho(real_t val) { impl_->rho = val; }

    // Assembly / actions
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

    int KelvinVoigtNewmark::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_block_diag_sym");

        auto          mesh = impl_->space->mesh_ptr();
        int err = SFEM_SUCCESS;

        real_t *const out0 = &values[0];
        real_t *const out1 = &values[1];
        real_t *const out2 = &values[2];
        real_t *const out3 = &values[3];
        real_t *const out4 = &values[4];
        real_t *const out5 = &values[5];

        err = impl_->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto beta         = domain.parameters->get_real_value("beta", impl_->beta);
            auto gamma        = domain.parameters->get_real_value("gamma", impl_->gamma);
            auto dt           = domain.parameters->get_real_value("dt", impl_->dt);
            auto k            = domain.parameters->get_real_value("k", impl_->k);
            auto K            = domain.parameters->get_real_value("K", impl_->K);
            auto eta          = domain.parameters->get_real_value("eta", impl_->eta);
            auto rho          = domain.parameters->get_real_value("rho", impl_->rho);
            auto element_type = domain.element_type;

            auto jacobians = std::static_pointer_cast<KVJacobians>(domain.user_data);
            SFEM_TRACE_SCOPE("kelvin_voigt_newmark_gradient_aos");

            return kelvin_voigt_newmark_block_diag_sym(element_type,
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

    int KelvinVoigtNewmark::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto beta         = domain.parameters->get_real_value("beta", impl_->beta);
            auto gamma        = domain.parameters->get_real_value("gamma", impl_->gamma);
            auto dt           = domain.parameters->get_real_value("dt", impl_->dt);
            auto k            = domain.parameters->get_real_value("k", impl_->k);
            auto K            = domain.parameters->get_real_value("K", impl_->K);
            auto eta          = domain.parameters->get_real_value("eta", impl_->eta);
            auto rho          = domain.parameters->get_real_value("rho", impl_->rho);
            auto element_type = domain.element_type;

            return kelvin_voigt_newmark_assemble_diag_aos(element_type,
                                                          block->n_elements(),
                                                          mesh->n_nodes(),
                                                          block->elements()->data(),
                                                          mesh->points()->data(),
                                                          impl_->beta,
                                                          impl_->gamma,
                                                          impl_->dt,
                                                          impl_->k,
                                                          impl_->K,
                                                          impl_->eta,
                                                          impl_->rho,
                                                          out);
        });

        return err;
    }

    int KelvinVoigtNewmark::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::gradient");
        auto          mesh = impl_->space->mesh_ptr();
        const real_t *u    = x;

        // AoS view: pass three component pointers with stride=3
        const real_t *vbase = impl_->vel_[0]->data();
        const real_t *vx    = &vbase[0];
        const real_t *vy    = &vbase[1];
        const real_t *vz    = &vbase[2];

        const real_t *abase = impl_->acc_[0]->data();
        const real_t *ax    = &abase[0];
        const real_t *ay    = &abase[1];
        const real_t *az    = &abase[2];

        double tick = MPI_Wtime();
        int    err  = SFEM_SUCCESS;

        err = impl_->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto element_type = domain.element_type;
            auto params       = domain.parameters;
            auto k            = params->get_real_value("k", impl_->k);
            auto K            = params->get_real_value("K", impl_->K);
            auto eta          = params->get_real_value("eta", impl_->eta);
            auto rho          = params->get_real_value("rho", impl_->rho);

            auto jacobians = std::static_pointer_cast<KVJacobians>(domain.user_data);
            SFEM_TRACE_SCOPE("kelvin_voigt_newmark_gradient_aos");
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

        double tock = MPI_Wtime();
        impl_->total_time += (tock - tick);
        impl_->calls++;
        return err;
    }

    int KelvinVoigtNewmark::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::apply");
        auto   mesh = impl_->space->mesh_ptr();
        double tick = MPI_Wtime();
        int    err  = SFEM_SUCCESS;

        err = impl_->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto element_type = domain.element_type;
            auto params       = domain.parameters;
            auto dt           = params->get_real_value("dt", impl_->dt);
            auto gamma        = params->get_real_value("gamma", impl_->gamma);
            auto beta         = params->get_real_value("beta", impl_->beta);
            auto k            = params->get_real_value("k", impl_->k);
            auto K            = params->get_real_value("K", impl_->K);
            auto eta          = params->get_real_value("eta", impl_->eta);
            auto rho          = params->get_real_value("rho", impl_->rho);

            auto jacobians = std::static_pointer_cast<KVJacobians>(domain.user_data);
            SFEM_TRACE_SCOPE("kelvin_voigt_newmark_apply_adjugate_aos");
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

        double tock = MPI_Wtime();
        impl_->total_time += (tock - tick);
        impl_->calls++;
        return err;
    }

    int KelvinVoigtNewmark::value(const real_t * /*x*/, real_t *const /*out*/) {
        SFEM_TRACE_SCOPE("KelvinVoigtNewmark::value");
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int KelvinVoigtNewmark::report(const real_t *const) { return SFEM_SUCCESS; }

    // Convenience factory
    std::unique_ptr<Op> create_kelvin_voigt_newmark(const std::shared_ptr<FunctionSpace> &space) {
        return KelvinVoigtNewmark::create(space);
    }

}  // namespace sfem
