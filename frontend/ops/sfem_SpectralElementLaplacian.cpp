#include "sfem_SpectralElementLaplacian.hpp"

// C includes   
#include "spectral_hex_laplacian.h"

// C++ includes
#include "sfem_Laplacian.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_glob.hpp"
#include "sfem_LinearElasticity.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"

namespace sfem {

    std::unique_ptr<Op> SpectralElementLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SpectralElementLaplacian::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SpectralElementLaplacian::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
        auto ret          = std::make_unique<SpectralElementLaplacian>(space);
        ret->element_type = (enum ElemType)space->element_type();

        return ret;
    }

    SpectralElementLaplacian::SpectralElementLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    SpectralElementLaplacian::~SpectralElementLaplacian() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SpectralElementLaplacian[%d]::apply called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   space->semi_structured_mesh().level(),
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    std::shared_ptr<Op> SpectralElementLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        fprintf(stderr, "[Error] SpectralElementLaplacian::lor_op NOT IMPLEMENTED!\n");
        assert(false);
        return nullptr;
    }

    std::shared_ptr<Op> SpectralElementLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SpectralElementLaplacian::derefine_op");

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
        if (space->has_semi_structured_mesh()) {
            auto ret          = std::make_shared<SpectralElementLaplacian>(space);
            ret->element_type = element_type;
            return ret;
        } else {
            auto ret          = std::make_shared<Laplacian>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }
    }

    const char *SpectralElementLaplacian::name() const {
        return "ss:SpectralElementLaplacian";
    }

    int SpectralElementLaplacian::initialize() {
        return SFEM_SUCCESS;
    }

    int SpectralElementLaplacian::hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) {
        SFEM_ERROR("[Error] SpectralElementLaplacian::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SpectralElementLaplacian::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("SpectralElementLaplacian::hessian_diag");
        SFEM_ERROR("[Error] SpectralElementLaplacian::hessian_diag NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SpectralElementLaplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_ERROR("[Error] SpectralElementLaplacian::gradient NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SpectralElementLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SpectralElementLaplacian::apply");

        assert(element_type == SSHEX8);  // REMOVEME once generalized approach

        auto &ssm = space->semi_structured_mesh();

        double tick = MPI_Wtime();

        int err = spectral_hex_laplacian_apply(
                ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), h, out);

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int SpectralElementLaplacian::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("[Error] SpectralElementLaplacian::value NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    std::shared_ptr<Op> SpectralElementLaplacian::clone() const {
        auto ret = std::make_shared<SpectralElementLaplacian>(space);
        *ret     = *this;
        return ret;
    }

} // namespace sfem 