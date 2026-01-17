#include "sfem_SemiStructuredKelvinVoigtNewmark.hpp"

// C includes
#include "sshex8_kelvin_voigt_newmark.h"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#include <mpi.h>

namespace sfem {

    void SemiStructuredKelvinVoigtNewmark::set_field(const char                            *name,
                                                     const std::shared_ptr<Buffer<real_t>> &vel,
                                                     int                                    component) {
        if (strcmp(name, "velocity") == 0) {
            vel_[component] = vel;
        } else if (strcmp(name, "acceleration") == 0) {
            acc_[component] = vel;
        } else {
            SFEM_ERROR(
                    "Invalid field name! Call set_field(\"velocity\", buffer, 0/1/2) or set_field(\"acceleration\", buffer, "
                    "0/1/2) first.\n");
        }
    }

    std::unique_ptr<Op> SemiStructuredKelvinVoigtNewmark::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredKelvinVoigtNewmark::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr, "[Error] SemiStructuredKelvinVoigtNewmark::create requires space with semi_structured_mesh!\n");
            return nullptr;
        }

        assert(space->element_type() == SSHEX8);
        auto ret = std::make_unique<SemiStructuredKelvinVoigtNewmark>(space);

        real_t SFEM_SHEAR_STIFFNESS_KV = 4.0;
        real_t SFEM_BULK_MODULUS       = 3.0;
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
        ret->k            = SFEM_SHEAR_STIFFNESS_KV;
        ret->K            = SFEM_BULK_MODULUS;
        ret->eta          = SFEM_DAMPING_RATIO;
        ret->dt           = SFEM_DT;
        ret->gamma        = SFEM_GAMMA;
        ret->beta         = SFEM_BETA;
        ret->rho          = SFEM_DENSITY;
        ret->element_type = (enum ElemType)space->element_type();

        int SFEM_HEX8_ASSUME_AFFINE = ret->use_affine_approximation;
        SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
        ret->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

        return ret;
    }

    SemiStructuredKelvinVoigtNewmark::SemiStructuredKelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space)
        : space(space) {}

    int SemiStructuredKelvinVoigtNewmark::hessian_crs(const real_t *const,
                                                      const count_t *const,
                                                      const idx_t *const,
                                                      real_t *const) {
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredKelvinVoigtNewmark::hessian_bsr(const real_t *const,
                                                      const count_t *const,
                                                      const idx_t *const,
                                                      real_t *const) {
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredKelvinVoigtNewmark::hessian_crs_sym(const real_t *const,
                                                          const count_t *const,
                                                          const idx_t *const,
                                                          real_t *const,
                                                          real_t *const) {
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredKelvinVoigtNewmark::hessian_bcrs_sym(const real_t *const,
                                                           const count_t *const,
                                                           const idx_t *const,
                                                           const ptrdiff_t,
                                                           real_t **const,
                                                           real_t **const) {
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredKelvinVoigtNewmark::hessian_block_diag_sym(const real_t *const, real_t *const values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredKelvinVoigtNewmark[%d]::hessian_block_diag_sym", ssm.level());

        return affine_sshex8_kelvin_voigt_newmark_block_diag_sym(ssm.level(),
                                                                 ssm.n_elements(),
                                                                 ssm.interior_start(),
                                                                 ssm.element_data(),
                                                                 ssm.point_data(),
                                                                 beta,
                                                                 gamma,
                                                                 dt,
                                                                 k,
                                                                 K,
                                                                 eta,
                                                                 rho,
                                                                 6,
                                                                 &values[0],
                                                                 &values[1],
                                                                 &values[2],
                                                                 &values[3],
                                                                 &values[4],
                                                                 &values[5]);
    }

    int SemiStructuredKelvinVoigtNewmark::hessian_diag(const real_t *const x, real_t *const values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredKelvinVoigtNewmark[%d]::hessian_diag", ssm.level());

        return affine_sshex8_kelvin_voigt_newmark_diag(ssm.level(),
                                                       ssm.n_elements(),
                                                       ssm.interior_start(),
                                                       ssm.element_data(),
                                                       ssm.point_data(),
                                                       beta,
                                                       gamma,
                                                       dt,
                                                       k,
                                                       K,
                                                       eta,
                                                       rho,
                                                       3,
                                                       &values[0],
                                                       &values[1],
                                                       &values[2]);
    }

    int SemiStructuredKelvinVoigtNewmark::gradient(const real_t *const x, real_t *const out) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredKelvinVoigtNewmark[%d]::gradient", ssm.level());

        assert(element_type == SSHEX8);

        const real_t *u = x;

        // Velocity provided as a single AoS buffer (same Buffer passed for 0/1/2).
        // Create three component views with stride=3.
        const real_t *vbase = vel_[0]->data();
        const real_t *vx    = &vbase[0];
        const real_t *vy    = &vbase[1];
        const real_t *vz    = &vbase[2];

        const real_t *abase = acc_[0]->data();
        const real_t *ax    = &abase[0];
        const real_t *ay    = &abase[1];
        const real_t *az    = &abase[2];

        calls++;
        double tick = MPI_Wtime();
        int    err;

        // We only provide affine path for now (consistent with sshex8_kv.c)
        err = affine_sshex8_kelvin_voigt_newmark_gradient(ssm.level(),
                                                          ssm.n_elements(),
                                                          ssm.interior_start(),
                                                          ssm.element_data(),
                                                          ssm.point_data(),
                                                          k,
                                                          K,
                                                          eta,
                                                          rho,
                                                          /*u_stride*/ 3,
                                                          /*u*/ &u[0],
                                                          &u[1],
                                                          &u[2],
                                                          /*v*/ vx,
                                                          vy,
                                                          vz,
                                                          /*a*/ ax,
                                                          ay,
                                                          az,
                                                          /*out_stride*/ 3,
                                                          /*out*/ &out[0],
                                                          &out[1],
                                                          &out[2]);

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        return err;
    }

    int SemiStructuredKelvinVoigtNewmark::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredKelvinVoigtNewmark[%d]::apply", ssm.level());

        assert(element_type == SSHEX8);

        calls++;
        double tick = MPI_Wtime();
        int    err;

        // We only provide affine path for now (consistent with sshex8_kv.c)
        err = affine_sshex8_kelvin_voigt_newmark_apply(ssm.level(),
                                                       ssm.n_elements(),
                                                       ssm.interior_start(),
                                                       ssm.element_data(),
                                                       ssm.point_data(),
                                                       k,
                                                       K,
                                                       eta,
                                                       rho,
                                                       dt,
                                                       gamma,
                                                       beta,
                                                       3,
                                                       &h[0],
                                                       &h[1],
                                                       &h[2],
                                                       3,
                                                       &out[0],
                                                       &out[1],
                                                       &out[2]);

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        return err;
    }

    int SemiStructuredKelvinVoigtNewmark::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("Called unimplemented method!\n");
        return SFEM_FAILURE;
    }

    std::shared_ptr<Op> SemiStructuredKelvinVoigtNewmark::clone() const {
        auto ret = std::make_shared<SemiStructuredKelvinVoigtNewmark>(space);
        *ret     = *this;
        return ret;
    }

    SemiStructuredKelvinVoigtNewmark::~SemiStructuredKelvinVoigtNewmark() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredKelvinVoigtNewmark[%d]::apply called %ld times. Total: %g [s], Avg: %g [s]\n",
                   space->semi_structured_mesh().level(),
                   calls,
                   total_time,
                   total_time / calls);
        }
    }

    int SemiStructuredKelvinVoigtNewmark::initialize(const std::vector<std::string> &) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredKelvinVoigtNewmark::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredKelvinVoigtNewmark::derefine_op");
        if (space->has_semi_structured_mesh()) {
            auto ret                      = std::make_shared<SemiStructuredKelvinVoigtNewmark>(space);
            ret->element_type             = element_type;
            ret->use_affine_approximation = use_affine_approximation;
            ret->k                        = k;
            ret->K                        = K;
            ret->eta                      = eta;
            ret->dt                       = dt;
            ret->gamma                    = gamma;
            ret->beta                     = beta;
            ret->rho                      = rho;
            return ret;
        } else {
            assert(space->element_type() == macro_base_elem(element_type));
            auto ret = std::make_shared<KelvinVoigtNewmark>(space);
            ret->initialize();
            ret->set_k(k);
            ret->set_K(K);
            ret->set_eta(eta);
            ret->set_dt(dt);
            ret->set_gamma(gamma);
            ret->set_beta(beta);
            ret->set_rho(rho);
            ret->override_element_types({macro_base_elem(element_type)});

            return ret;
        }
    }

    std::shared_ptr<Op> SemiStructuredKelvinVoigtNewmark::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        assert(false);
        fprintf(stderr, "[Error] ss:KelvinVoigtNewmark::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    const char *SemiStructuredKelvinVoigtNewmark::name() const { return "ss:KelvinVoigtNewmark"; }

    void SemiStructuredKelvinVoigtNewmark::set_option(const std::string &name, bool val) {
        if (name == "ASSUME_AFFINE") {
            use_affine_approximation = val;
        }
    }

    int SemiStructuredKelvinVoigtNewmark::report(const real_t *const) { return SFEM_SUCCESS; }
}  // namespace sfem