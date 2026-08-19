#include "sfem_SemiStructuredEMLaplacian.hpp"

// C includes
#include "sshex8_laplacian.hpp"
#include "sshex8_stencil_element_matrix_apply.hpp"

// C++ includes
#include "sfem_Laplacian.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include "smesh_glob.hpp"

#include <algorithm>
#include <vector>

namespace sfem {

    namespace {

        static bool block_is_selected(const std::string &name, const std::vector<std::string> &block_names) {
            return block_names.empty() || std::find(block_names.begin(), block_names.end(), name) != block_names.end();
        }

        static int require_sshex8_block(const char *const op_name, const smesh::Mesh &mesh, const size_t block_id) {
            const auto element_type = mesh.element_type(static_cast<smesh::block_idx_t>(block_id));
            if (!is_semistructured_type(element_type) || macro_base_elem(element_type) != smesh::PROTEUS_HEX8) {
                SFEM_ERROR("%s supports homogeneous SSHEX8 blocks\n", op_name);
                return SFEM_FAILURE;
            }

            return SFEM_SUCCESS;
        }

        static std::shared_ptr<smesh::Mesh> element_matrix_mesh(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = space->has_semi_structured_mesh() ? smesh::derefine(space->mesh_ptr(), 1) : space->mesh_ptr();
            if (mesh && mesh->element_type(0) == smesh::PROTEUS_HEX8) {
                mesh = smesh::sshex_to_hex8(mesh);
            }
            return mesh;
        }

    }  // namespace

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
            auto                         ret = std::make_shared<Laplacian>(space);
            std::vector<smesh::ElemType> element_types(space->n_blocks(), macro_base_elem(element_type));
            ret->override_element_types(element_types);
            return ret;
        }
    }

    const char *SemiStructuredEMLaplacian::name() const { return "em:Laplacian"; }

    int SemiStructuredEMLaplacian::initialize(const std::vector<std::string> &block_names) {
        auto &ssm  = space->mesh();
        auto  mesh = element_matrix_mesh(space);
        if (!mesh) {
            return SFEM_FAILURE;
        }

        const auto n_blocks = ssm.n_blocks();
        element_matrices.assign(n_blocks, nullptr);
        element_matrix = nullptr;

        int err = SFEM_SUCCESS;
        for (size_t b = 0; b < n_blocks; ++b) {
            const auto block_id = static_cast<smesh::block_idx_t>(b);
            if (!block_is_selected(ssm.block(b)->name(), block_names)) {
                continue;
            }

            err = require_sshex8_block(name(), ssm, b);
            if (err != SFEM_SUCCESS) {
                return err;
            }

            auto matrix = sfem::create_host_buffer<real_t>(mesh->n_elements(block_id) * 64);
            err         = sshex8_laplacian_element_matrix_cartesian(smesh::semistructured_level(ssm),
                                                            mesh->n_elements(block_id),
                                                            mesh->n_nodes(),
                                                            mesh->elements(block_id)->data(),
                                                            mesh->points()->data(),
                                                            matrix->data());
            if (err != SFEM_SUCCESS) {
                return err;
            }

            element_matrices[b] = matrix;
            if (!element_matrix) {
                element_matrix = matrix;
            }
        }

        return SFEM_SUCCESS;
    }

    int SemiStructuredEMLaplacian::hessian_crs(const real_t *const  x,
                                               const count_t *const rowptr,
                                               const idx_t *const   colidx,
                                               real_t *const        values) {
        SFEM_ERROR("[Error] em:Laplacian::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLaplacian::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::hessian_diag");

        auto &ssm = space->mesh();
        int   err = SFEM_SUCCESS;
        for (size_t b = 0; b < element_matrices.size(); ++b) {
            if (!element_matrices[b]) {
                continue;
            }

            const auto block_id = static_cast<smesh::block_idx_t>(b);
            err                 = affine_sshex8_laplacian_diag(smesh::semistructured_level(ssm),
                                               ssm.n_elements(block_id),
                                               ssm.elements(block_id)->data(),
                                               ssm.points()->data(),
                                               out);
            if (err != SFEM_SUCCESS) {
                return err;
            }
        }

        return SFEM_SUCCESS;
    }

    int SemiStructuredEMLaplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_ERROR("[Error] em:Laplacian::gradient NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLaplacian::apply");

        assert(is_semistructured_type(element_type));  // REMOVEME once generalized approach

        auto &ssm = space->mesh();

        double tick = smesh::time_seconds();

        int err = SFEM_SUCCESS;
        for (size_t b = 0; b < element_matrices.size(); ++b) {
            auto &matrix = element_matrices[b];
            if (!matrix) {
                continue;
            }

            const auto block_id = static_cast<smesh::block_idx_t>(b);
            err                 = sshex8_stencil_element_matrix_apply(smesh::semistructured_level(ssm),
                                                      ssm.n_elements(block_id),
                                                      ssm.elements(block_id)->data(),
                                                      matrix->data(),
                                                      h,
                                                      out);
            if (err != SFEM_SUCCESS) {
                return err;
            }
        }

        double tock = smesh::time_seconds();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int SemiStructuredEMLaplacian::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("[Error] em:Laplacian::value NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLaplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredEMLaplacian::clone() const {
        auto ret = std::make_shared<SemiStructuredEMLaplacian>(space);
        *ret     = *this;
        return ret;
    }

}  // namespace sfem
