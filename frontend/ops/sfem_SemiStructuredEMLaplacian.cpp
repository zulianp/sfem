#include "sfem_SemiStructuredEMLaplacian.hpp"

// C includes
#include "sshex8_laplacian.hpp"
#include "sshex8_stencil_element_matrix_apply.hpp"

// C++ includes
#include "sfem_Laplacian.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_mesh.hpp"

#include "smesh_glob.hpp"

namespace sfem {

    std::unique_ptr<Op> SemiStructuredEMLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredEMLaplacian::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(is_semistructured_type(space->element_type()));  // REMOVEME once generalized approach
        auto ret          = std::make_unique<SemiStructuredEMLaplacian>(space);
        ret->element_type = (smesh::ElemType)space->element_type();
        return ret;
    }

    SemiStructuredEMLaplacian::SemiStructuredEMLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    SemiStructuredEMLaplacian::~SemiStructuredEMLaplacian() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredEMLaplacian[%d]::apply() called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   smesh::semistructured_level(space->mesh()),
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    std::shared_ptr<Op> SemiStructuredEMLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SMESH_ERROR("SemiStructuredEMLaplacian::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    std::shared_ptr<Op> SemiStructuredEMLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::derefine_op");

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
        if (space->has_semi_structured_mesh()) {
            auto ret          = std::make_shared<SemiStructuredEMLaplacian>(space);
            ret->element_type = element_type;
            // FIXME every level stores a variatin of it with different scaling
            // It woud be usefull to revisit
            // ret->element_matrix = element_matrix;
            ret->initialize();
            return ret;
        } else {
            auto ret = std::make_shared<Laplacian>(space);
            assert(space->n_blocks() == 1);  // FIXME
            ret->override_element_types({macro_base_elem(element_type)});
            return ret;
        }
    }

    const char *SemiStructuredEMLaplacian::name() const { return "ss:em:Laplacian"; }

    int SemiStructuredEMLaplacian::initialize(const std::vector<std::string> &block_names) {
        auto &ssm  = space->mesh();
        auto  mesh = space->has_semi_structured_mesh() ? smesh::derefine(space->mesh_ptr(), 1) : space->mesh_ptr();
        element_matrix = sfem::create_host_buffer<real_t>(mesh->n_elements() * 64);
        return sshex8_laplacian_element_matrix(smesh::semistructured_level(ssm),
                                               mesh->n_elements(),
                                               mesh->n_nodes(),
                                               mesh->elements(0)->data(),
                                               mesh->points()->data(),
                                               element_matrix->data());
    }

    int SemiStructuredEMLaplacian::hessian_crs(const real_t *const  x,
                                               const count_t *const rowptr,
                                               const idx_t *const   colidx,
                                               real_t *const        values) {
        SFEM_ERROR("[Error] ss:em:Laplacian::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLaplacian::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::hessian_diag");

        auto &ssm = space->mesh();
        return affine_sshex8_laplacian_diag(
                smesh::semistructured_level(ssm), ssm.n_elements(), ssm.elements(0)->data(), ssm.points()->data(), out);
    }

    int SemiStructuredEMLaplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_ERROR("[Error] ss:em:Laplacian::gradient NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::apply");

        assert(is_semistructured_type(element_type));  // REMOVEME once generalized approach

        auto &ssm = space->mesh();

        double tick = MPI_Wtime();

        int err = sshex8_stencil_element_matrix_apply(
                smesh::semistructured_level(ssm), ssm.n_elements(), ssm.elements(0)->data(), element_matrix->data(), h, out);

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int SemiStructuredEMLaplacian::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("[Error] ss:em:Laplacian::value NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLaplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredEMLaplacian::clone() const {
        auto ret = std::make_shared<SemiStructuredEMLaplacian>(space);
        *ret     = *this;
        return ret;
    }

}  // namespace sfem
