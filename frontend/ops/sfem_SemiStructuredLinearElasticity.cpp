#include "sfem_SemiStructuredLinearElasticity.hpp"

// C includes
#include "sshex8_linear_elasticity.h"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_LinearElasticity.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#include <mpi.h>

namespace sfem {

    std::unique_ptr<Op> SemiStructuredLinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredLinearElasticity::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredLinearElasticity::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
        auto ret = std::make_unique<SemiStructuredLinearElasticity>(space);

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

    SemiStructuredLinearElasticity::SemiStructuredLinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int SemiStructuredLinearElasticity::hessian_crs(const real_t *const  x,
                                                    const count_t *const rowptr,
                                                    const idx_t *const   colidx,
                                                    real_t *const        values) {
        assert(false);
        return SFEM_FAILURE;
    }

    int SemiStructuredLinearElasticity::hessian_bsr(const real_t *const  x,
                                                    const count_t *const rowptr,
                                                    const idx_t *const   colidx,
                                                    real_t *const        values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::hessian_bsr", ssm.level());

        return affine_sshex8_elasticity_bsr(ssm.level(),
                                            ssm.n_elements(),
                                            ssm.interior_start(),
                                            ssm.element_data(),
                                            ssm.point_data(),
                                            this->mu,
                                            this->lambda,
                                            rowptr,
                                            colidx,
                                            values);
    }

    int SemiStructuredLinearElasticity::hessian_diag(const real_t *const x, real_t *const values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::hessian_diag", ssm.level());

        return affine_sshex8_linear_elasticity_diag(ssm.level(),
                                                    ssm.n_elements(),
                                                    ssm.interior_start(),
                                                    ssm.element_data(),
                                                    ssm.point_data(),
                                                    mu,
                                                    lambda,
                                                    3,
                                                    &values[0],
                                                    &values[1],
                                                    &values[2]);
    }

    int SemiStructuredLinearElasticity::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::hessian_block_diag_sym", ssm.level());

        return affine_sshex8_linear_elasticity_block_diag_sym(ssm.level(),
                                                              ssm.n_elements(),
                                                              ssm.interior_start(),
                                                              ssm.element_data(),
                                                              ssm.point_data(),
                                                              mu,
                                                              lambda,
                                                              6,
                                                              &values[0],
                                                              &values[1],
                                                              &values[2],
                                                              &values[3],
                                                              &values[4],
                                                              &values[5]);
    }

    int SemiStructuredLinearElasticity::gradient(const real_t *const x, real_t *const out) { return apply(nullptr, x, out); }

    int SemiStructuredLinearElasticity::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLinearElasticity[%d]::apply", ssm.level());

        assert(element_type == SSHEX8);  // REMOVEME once generalized approach

        calls++;

        double tick = MPI_Wtime();
        int    err;
        if (use_affine_approximation) {
            err = affine_sshex8_linear_elasticity_apply(ssm.level(),
                                                        ssm.n_elements(),
                                                        ssm.interior_start(),
                                                        ssm.element_data(),
                                                        ssm.point_data(),
                                                        mu,
                                                        lambda,
                                                        3,
                                                        &h[0],
                                                        &h[1],
                                                        &h[2],
                                                        3,
                                                        &out[0],
                                                        &out[1],
                                                        &out[2]);

        } else {
            err = sshex8_linear_elasticity_apply(ssm.level(),
                                                 ssm.n_elements(),
                                                 ssm.interior_start(),
                                                 ssm.element_data(),
                                                 ssm.point_data(),
                                                 mu,
                                                 lambda,
                                                 3,
                                                 &h[0],
                                                 &h[1],
                                                 &h[2],
                                                 3,
                                                 &out[0],
                                                 &out[1],
                                                 &out[2]);
        }

        double tock = MPI_Wtime();
        total_time += (tock - tick);

        return err;
    }

    int SemiStructuredLinearElasticity::value(const real_t *x, real_t *const out) {
        assert(false);
        return SFEM_FAILURE;
    }

    std::shared_ptr<Op> SemiStructuredLinearElasticity::clone() const {
        auto ret = std::make_shared<SemiStructuredLinearElasticity>(space);
        *ret     = *this;
        return ret;
    }

    SemiStructuredLinearElasticity::~SemiStructuredLinearElasticity() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredLinearElasticity[%d]::apply(%s) called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   space->semi_structured_mesh().level(),
                   use_affine_approximation ? "affine" : "isoparametric",
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    int SemiStructuredLinearElasticity::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredLinearElasticity::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredLinearElasticity::derefine_op");

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));

        if (space->has_semi_structured_mesh()) {
            auto ret                      = std::make_shared<SemiStructuredLinearElasticity>(space);
            ret->element_type             = element_type;
            ret->use_affine_approximation = use_affine_approximation;
            ret->mu                       = mu;
            ret->lambda                   = lambda;
            // ret->initialize();
            return ret;
        } else {
            assert(space->element_type() == macro_base_elem(element_type));
            auto ret          = std::make_shared<LinearElasticity>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->mu           = mu;
            ret->lambda       = lambda;
            ret->initialize();
            return ret;
        }
    }

    std::shared_ptr<Op> SemiStructuredLinearElasticity::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        assert(false);
        fprintf(stderr, "[Error] ss:LinearElasticity::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    const char *SemiStructuredLinearElasticity::name() const { return "ss:LinearElasticity"; }

    void SemiStructuredLinearElasticity::set_option(const std::string &name, bool val) {
        if (name == "ASSUME_AFFINE") {
            use_affine_approximation = val;
        }
    }

    int SemiStructuredLinearElasticity::hessian_crs_sym(const real_t *const  x,
                                                        const count_t *const rowptr,
                                                        const idx_t *const   colidx,
                                                        real_t *const        diag_values,
                                                        real_t *const        off_diag_values) {
        SFEM_ERROR("[Error] ss:LinearElasticity::hessian_crs_sym NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredLinearElasticity::hessian_bcrs_sym(const real_t *const  x,
                                                         const count_t *const rowidx,
                                                         const idx_t *const   colidx,
                                                         const ptrdiff_t      block_stride,
                                                         real_t **const       diag_values,
                                                         real_t **const       off_diag_values) {
        SFEM_ERROR("[Error] ss:LinearElasticity::hessian_bcrs_sym NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredLinearElasticity::report(const real_t *const) { return SFEM_SUCCESS; }
}  // namespace sfem