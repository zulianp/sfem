#include "sfem_SemiStructuredLaplacian.hpp"

// C includes
#include "sshex8_laplacian.h"
#include "hex8_fff.h"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_glob.hpp"
#include "sfem_LinearElasticity.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_Laplacian.hpp"


namespace sfem {

    std::unique_ptr<Op> SemiStructuredLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredLaplacian::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredLaplacian::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
        auto ret = std::make_unique<SemiStructuredLaplacian>(space);

        ret->element_type = (enum ElemType)space->element_type();

        int SFEM_HEX8_ASSUME_AFFINE = ret->use_affine_approximation;
        SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
        ret->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

        int SFEM_ENABLE_HEX8_STENCIL = ret->use_stencil;
        SFEM_READ_ENV(SFEM_ENABLE_HEX8_STENCIL, atoi);
        ret->use_stencil = SFEM_ENABLE_HEX8_STENCIL;

        int SFEM_SS_LAPLACIAN_FFF = 1;
        SFEM_READ_ENV(SFEM_SS_LAPLACIAN_FFF, atoi);

        if (SFEM_SS_LAPLACIAN_FFF) {
            ret->fff = create_host_buffer<jacobian_t>(space->mesh_ptr()->n_elements() * 6);

            if (SFEM_SUCCESS != hex8_fff_fill(space->mesh_ptr()->n_elements(),
                                              space->mesh_ptr()->elements()->data(),
                                              space->mesh_ptr()->points()->data(),
                                              ret->fff->data())) {
                SFEM_ERROR("Unable to create fff");
            }
        }

        return ret;
    }

    SemiStructuredLaplacian::SemiStructuredLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    SemiStructuredLaplacian::~SemiStructuredLaplacian() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredLaplacian[%d]::apply(%s) called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   space->semi_structured_mesh().level(),
                   use_affine_approximation ? (use_stencil ? "stencil" : "affine") : "isoparametric",
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    std::shared_ptr<Op> SemiStructuredLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        fprintf(stderr, "[Error] ss:Laplacian::lor_op NOT IMPLEMENTED!\n");
        assert(false);
        return nullptr;
    }

    std::shared_ptr<Op> SemiStructuredLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredLaplacian::derefine_op");

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
        if (space->has_semi_structured_mesh()) {
            auto ret                      = std::make_shared<SemiStructuredLaplacian>(space);
            ret->element_type             = element_type;
            ret->use_affine_approximation = use_affine_approximation;
            ret->use_stencil              = use_stencil;
            return ret;
        } else {
            auto ret          = std::make_shared<Laplacian>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }
    }

    const char *SemiStructuredLaplacian::name() const {
        return "ss:Laplacian";
    }

    int SemiStructuredLaplacian::initialize() {
        return SFEM_SUCCESS;
    }

    int SemiStructuredLaplacian::hessian_crs(const real_t *const  x,
                                            const count_t *const rowptr,
                                            const idx_t *const   colidx,
                                            real_t *const        values) {
        SFEM_ERROR("[Error] ss:Laplacian::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    // int SemiStructuredLaplacian::hessian_bsr(const real_t *const x,
    //                                          const count_t *const rowptr,
    //                                          const idx_t *const colidx,
    //                                          real_t *const values) {
    //     auto &ssm = space->semi_structured_mesh();
    //     SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLaplacian[%d]::hessian_bsr", ssm.level());

    //     return affine_sshex8_laplacian_bsr(ssm.level(),
    //                                        ssm.n_elements(),
    //                                        ssm.interior_start(),
    //                                        ssm.element_data(),
    //                                        ssm.point_data(),
    //                                        rowptr,
    //                                        colidx,
    //                                        values);
    // }

    int SemiStructuredLaplacian::hessian_diag(const real_t *const x, real_t *const values) {
        auto &ssm = space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredLaplacian[%d]::hessian_diag", ssm.level());

        return affine_sshex8_laplacian_diag(ssm.level(),
                                            ssm.n_elements(),
                                            ssm.interior_start(),
                                            ssm.element_data(),
                                            ssm.point_data(),
                                            values);
    }

    int SemiStructuredLaplacian::gradient(const real_t *const x, real_t *const out) {
        return apply(nullptr, x, out);
    }

    int SemiStructuredLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredLaplacian::apply");

        assert(element_type == SSHEX8);  // REMOVEME once generalized approach

        auto &ssm = space->semi_structured_mesh();

        double tick = MPI_Wtime();

        int err = 0;

        if (this->fff) {
            SFEM_TRACE_SCOPE("affine_sshex8_laplacian_stencil_apply_fff");
            affine_sshex8_laplacian_stencil_apply_fff(
                    ssm.level(), ssm.n_elements(), ssm.element_data(), this->fff->data(), h, out);
        } else {
            if (use_stencil) {
                err = affine_sshex8_laplacian_stencil_apply(
                        ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);
            } else if (use_affine_approximation) {
                err = affine_sshex8_laplacian_apply(
                        ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);

            } else {
                err = sshex8_laplacian_apply(
                        ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);
            }
        }

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int SemiStructuredLaplacian::value(const real_t *x, real_t *const out) {
        assert(false);
        return SFEM_FAILURE;
    }

    int SemiStructuredLaplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredLaplacian::clone() const {
        auto ret = std::make_shared<SemiStructuredLaplacian>(space);
        *ret     = *this;
        return ret;
    }

    void SemiStructuredLaplacian::set_option(const std::string &name, bool val) {
        if (name == "use_affine_approximation") {
            use_affine_approximation = val;
        }
    }

} // namespace sfem 