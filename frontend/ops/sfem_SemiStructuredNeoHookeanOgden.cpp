#include "sfem_SemiStructuredNeoHookeanOgden.hpp"

// C includes
#include "sshex8_neohookean_ogden.h"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_NeoHookeanOgden.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#include <mpi.h>

namespace sfem {

    std::unique_ptr<Op> SemiStructuredNeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredNeoHookeanOgden::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredNeoHookeanOgden::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
        auto ret = std::make_unique<SemiStructuredNeoHookeanOgden>(space);

        real_t SFEM_SHEAR_MODULUS        = 1;
        real_t SFEM_FIRST_LAME_PARAMETER = 1;

        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

        ret->mu           = SFEM_SHEAR_MODULUS;
        ret->lambda       = SFEM_FIRST_LAME_PARAMETER;
        ret->element_type = (enum ElemType)space->element_type();

        int SFEM_HEX8_ASSUME_AFFINE = ret->use_affine_approximation;
        SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
        ret->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

        return ret;
    }

    SemiStructuredNeoHookeanOgden::SemiStructuredNeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int SemiStructuredNeoHookeanOgden::hessian_crs(const real_t *const  x,
                                                   const count_t *const rowptr,
                                                   const idx_t *const   colidx,
                                                   real_t *const        values) {
        assert(false);
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_bsr(const real_t *const  x,
                                                   const count_t *const rowptr,
                                                   const idx_t *const   colidx,
                                                   real_t *const        values) {
        auto &ssm = space->semi_structured_mesh();
        // SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::hessian_bsr", ssm.level());

        // return affine_sshex8_elasticity_bsr(ssm.level(),
        //                                     ssm.n_elements(),
        //                                     ssm.interior_start(),
        //                                     ssm.element_data(),
        //                                     ssm.point_data(),
        //                                     this->mu,
        //                                     this->lambda,
        //                                     rowptr,
        //                                     colidx,
        //                                     values);

        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_diag(const real_t *const x, real_t *const values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::hessian_diag", ssm.level());

        // return affine_sshex8_linear_elasticity_diag(ssm.level(),
        //                                             ssm.n_elements(),
        //                                             ssm.interior_start(),
        //                                             ssm.element_data(),
        //                                             ssm.point_data(),
        //                                             mu,
        //                                             lambda,
        //                                             3,
        //                                             &values[0],
        //                                             &values[1],
        //                                             &values[2]);
        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::hessian_block_diag_sym", ssm.level());

        //     return affine_sshex8_linear_elasticity_block_diag_sym(ssm.level(),
        //                                                           ssm.n_elements(),
        //                                                           ssm.interior_start(),
        //                                                           ssm.element_data(),
        //                                                           ssm.point_data(),
        //                                                           mu,
        //                                                           lambda,
        //                                                           6,
        //                                                           &values[0],
        //                                                           &values[1],
        //                                                           &values[2],
        //                                                           &values[3],
        //                                                           &values[4],
        //                                                           &values[5]);
        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredNeoHookeanOgden::gradient");
        auto &ssm = space->semi_structured_mesh();
        return sshex8_neohookean_ogden_gradient(ssm.level(),
                                                ssm.n_elements(),
                                                1,
                                                ssm.n_nodes(),
                                                ssm.element_data(),
                                                ssm.point_data(),
                                                this->mu,
                                                this->lambda,
                                                3,
                                                &x[0],
                                                &x[1],
                                                &x[2],
                                                3,
                                                &out[0],
                                                &out[1],
                                                &out[2]);
    }

    int SemiStructuredNeoHookeanOgden::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        // auto &ssm = space->semi_structured_mesh();
        // SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::apply", ssm.level());

        // assert(element_type == SSHEX8);  // REMOVEME once generalized approach

        // calls++;

        // double tick = MPI_Wtime();
        // int    err;
        // if (use_affine_approximation) {
        //     err = affine_sshex8_linear_elasticity_apply(ssm.level(),
        //                                                 ssm.n_elements(),
        //                                                 ssm.interior_start(),
        //                                                 ssm.element_data(),
        //                                                 ssm.point_data(),
        //                                                 mu,
        //                                                 lambda,
        //                                                 3,
        //                                                 &h[0],
        //                                                 &h[1],
        //                                                 &h[2],
        //                                                 3,
        //                                                 &out[0],
        //                                                 &out[1],
        //                                                 &out[2]);

        // } else {
        //     err = sshex8_linear_elasticity_apply(ssm.level(),
        //                                          ssm.n_elements(),
        //                                          ssm.interior_start(),
        //                                          ssm.element_data(),
        //                                          ssm.point_data(),
        //                                          mu,
        //                                          lambda,
        //                                          3,
        //                                          &h[0],
        //                                          &h[1],
        //                                          &h[2],
        //                                          3,
        //                                          &out[0],
        //                                          &out[1],
        //                                          &out[2]);
        // }

        // double tock = MPI_Wtime();
        // total_time += (tock - tick);

        // return err;

        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::value(const real_t *x, real_t *const out) {
        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    std::shared_ptr<Op> SemiStructuredNeoHookeanOgden::clone() const {
        SFEM_IMPLEMENT_ME();
        auto ret = std::make_shared<SemiStructuredNeoHookeanOgden>(space);
        *ret     = *this;
        return ret;
    }

    SemiStructuredNeoHookeanOgden::~SemiStructuredNeoHookeanOgden() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredNeoHookeanOgden[%d]::apply(%s) called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   space->semi_structured_mesh().level(),
                   use_affine_approximation ? "affine" : "isoparametric",
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    int SemiStructuredNeoHookeanOgden::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredNeoHookeanOgden::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredNeoHookeanOgden::derefine_op");

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));

        if (space->has_semi_structured_mesh()) {
            auto ret                      = std::make_shared<SemiStructuredNeoHookeanOgden>(space);
            ret->element_type             = element_type;
            ret->use_affine_approximation = use_affine_approximation;
            ret->mu                       = mu;
            ret->lambda                   = lambda;
            // ret->initialize();
            return ret;
        } else {
            assert(space->element_type() == macro_base_elem(element_type));
            auto ret = std::make_shared<NeoHookeanOgden>(space);
            ret->initialize();
            ret->set_mu(mu);
            ret->set_lambda(lambda);
            ret->override_element_types({macro_base_elem(element_type)});

            return ret;
        }
    }

    std::shared_ptr<Op> SemiStructuredNeoHookeanOgden::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        assert(false);
        fprintf(stderr, "[Error] ss:NeoHookeanOgden::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    const char *SemiStructuredNeoHookeanOgden::name() const { return "ss:NeoHookeanOgden"; }

    void SemiStructuredNeoHookeanOgden::set_option(const std::string &name, bool val) {
        if (name == "ASSUME_AFFINE") {
            use_affine_approximation = val;
        }
    }

    int SemiStructuredNeoHookeanOgden::hessian_crs_sym(const real_t *const  x,
                                                       const count_t *const rowptr,
                                                       const idx_t *const   colidx,
                                                       real_t *const        diag_values,
                                                       real_t *const        off_diag_values) {
        SFEM_ERROR("[Error] ss:NeoHookeanOgden::hessian_crs_sym NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_bcrs_sym(const real_t *const  x,
                                                        const count_t *const rowidx,
                                                        const idx_t *const   colidx,
                                                        const ptrdiff_t      block_stride,
                                                        real_t **const       diag_values,
                                                        real_t **const       off_diag_values) {
        SFEM_ERROR("[Error] ss:NeoHookeanOgden::hessian_bcrs_sym NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::report(const real_t *const) { return SFEM_SUCCESS; }
}  // namespace sfem
