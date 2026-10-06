#include "sfem_SpectralElementLaplacian.hpp"

// C includes
#include "spectral_hex_laplacian.hpp"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_Laplacian.hpp"
#include "sfem_LinearElasticity.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include "smesh_glob.hpp"
#include "sfem_logger.hpp"

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

        assert(is_semistructured_type(space->element_type()));  // REMOVEME once generalized approach
        auto ret          = std::make_unique<SpectralElementLaplacian>(space);
        ret->element_type = (smesh::ElemType)space->element_type();

        return ret;
    }

    SpectralElementLaplacian::SpectralElementLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    SpectralElementLaplacian::~SpectralElementLaplacian() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SpectralElementLaplacian[%d]::apply called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   smesh::semistructured_level(space->mesh()),
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    std::shared_ptr<Op> SpectralElementLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SMESH_ERROR("SpectralElementLaplacian::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    std::shared_ptr<Op> SpectralElementLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SpectralElementLaplacian::derefine_op");

        if (space->n_blocks() > 1) {
            SFEM_ERROR("SpectralElementLaplacian::derefine_op: multi-block is not implemented\n");
            return nullptr;
        }

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
        if (space->has_semi_structured_mesh()) {
            auto ret          = std::make_shared<SpectralElementLaplacian>(space);
            ret->element_type = element_type;
            return ret;
        }

        auto ret = std::make_shared<Laplacian>(space);
        ret->initialize({});
        return ret;
    }

    const char *SpectralElementLaplacian::name() const { return "SpectralElementLaplacian"; }

    int SpectralElementLaplacian::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

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

        assert(is_semistructured_type(element_type));  // REMOVEME once generalized approach

        auto &ssm = space->mesh();
        if (ssm.n_blocks() != 1) {
            SFEM_ERROR("SpectralElementLaplacian::apply: multi-block is not implemented\n");
            return SFEM_FAILURE;
        }
        if (!smesh::is_hex_ss_family(ssm.element_type(0))) {
            SFEM_ERROR("SpectralElementLaplacian::apply: HEX SS family required (got %s)\n",
                       smesh::type_to_string(ssm.element_type(0)));
            return SFEM_FAILURE;
        }

        auto block = ssm.block(0);

        double tick = smesh::time_seconds();

        int err = spectral_hex_laplacian_apply(smesh::semistructured_level(ssm),
                                               block->n_elements(),
                                               smesh::semistructured_interior_start(ssm),
                                               block->elements()->data(),
                                               ssm.points()->data(),
                                               h,
                                               out);

        double tock = smesh::time_seconds();
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

}  // namespace sfem

