#include "sfem_SemiStructuredEMLinearElasticity.hpp"

#include "sshex8_linear_elasticity.hpp"
#include "sshex8_stencil_element_matrix_apply.hpp"

#include "sfem_LinearElasticity.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include "smesh_glob.hpp"

#include <algorithm>

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

    std::unique_ptr<Op> SemiStructuredEMLinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredEMLinearElasticity::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(is_semistructured_type(space->element_type()));
        auto ret          = std::make_unique<SemiStructuredEMLinearElasticity>(space);
        ret->element_type = (smesh::ElemType)space->element_type();
        return ret;
    }

    SemiStructuredEMLinearElasticity::SemiStructuredEMLinearElasticity(const std::shared_ptr<FunctionSpace> &space)
        : space(space) {}

    SemiStructuredEMLinearElasticity::~SemiStructuredEMLinearElasticity() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredEMLinearElasticity[%d]::apply() called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   smesh::semistructured_level(space->mesh()),
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    std::shared_ptr<Op> SemiStructuredEMLinearElasticity::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SMESH_ERROR("SemiStructuredEMLinearElasticity::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    std::shared_ptr<Op> SemiStructuredEMLinearElasticity::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::derefine_op");

        if (space->has_semi_structured_mesh()) {
            auto ret          = std::make_shared<SemiStructuredEMLinearElasticity>(space);
            ret->element_type = element_type;
            ret->mu           = mu;
            ret->lambda       = lambda;
            ret->initialize();
            return ret;
        }

        auto ret = std::make_shared<LinearElasticity>(space);
        ret->initialize();
        ret->set_value_in_block("", "mu", mu);
        ret->set_value_in_block("", "lambda", lambda);
        std::vector<smesh::ElemType> element_types(space->n_blocks(), macro_base_elem(element_type));
        ret->override_element_types(element_types);
        return ret;
    }

    const char *SemiStructuredEMLinearElasticity::name() const { return "em:LinearElasticity"; }

    int SemiStructuredEMLinearElasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::initialize");

        auto &ssm = space->mesh();

        real_t SFEM_SHEAR_MODULUS        = mu;
        real_t SFEM_FIRST_LAME_PARAMETER = lambda;
        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
        mu     = SFEM_SHEAR_MODULUS;
        lambda = SFEM_FIRST_LAME_PARAMETER;

        auto mesh = element_matrix_mesh(space);
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

            auto matrix = sfem::create_host_buffer<scalar_t>(mesh->n_elements(block_id) * 24 * 24);
            err         = sshex8_linear_elasticity_element_matrix_cartesian(smesh::semistructured_level(ssm),
                                                                    mesh->n_elements(block_id),
                                                                    mesh->n_nodes(),
                                                                    mesh->elements(block_id)->data(),
                                                                    mesh->points()->data(),
                                                                    mu,
                                                                    lambda,
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

    int SemiStructuredEMLinearElasticity::hessian_crs(const real_t *const,
                                                      const count_t *const,
                                                      const idx_t *const,
                                                      real_t *const) {
        SFEM_ERROR("[Error] em:LinearElasticity::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLinearElasticity::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::hessian_diag");

        auto &ssm = space->mesh();
        int   err = SFEM_SUCCESS;
        for (size_t b = 0; b < element_matrices.size(); ++b) {
            if (!element_matrices[b]) {
                continue;
            }

            const auto block_id = static_cast<smesh::block_idx_t>(b);
            err                 = affine_sshex8_linear_elasticity_diag(smesh::semistructured_level(ssm),
                                                       ssm.n_elements(block_id),
                                                       ssm.n_nodes(),
                                                       ssm.elements(block_id)->data(),
                                                       ssm.points()->data(),
                                                       mu,
                                                       lambda,
                                                       3,
                                                       &out[0],
                                                       &out[1],
                                                       &out[2]);
            if (err != SFEM_SUCCESS) {
                return err;
            }
        }

        return SFEM_SUCCESS;
    }

    int SemiStructuredEMLinearElasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_ERROR("[Error] em:LinearElasticity::gradient NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLinearElasticity::apply(const real_t *const, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::apply");

        assert(is_semistructured_type(element_type));
        auto &ssm = space->mesh();

        double tick = smesh::time_seconds();
        int    err  = SFEM_SUCCESS;
        for (size_t b = 0; b < element_matrices.size(); ++b) {
            auto &matrix = element_matrices[b];
            if (!matrix) {
                continue;
            }

            const auto block_id = static_cast<smesh::block_idx_t>(b);
            err                 = sshex8_stencil_element_matrix_apply3(smesh::semistructured_level(ssm),
                                                       ssm.n_elements(block_id),
                                                       ssm.elements(block_id)->data(),
                                                       matrix->data(),
                                                       3,
                                                       &h[0],
                                                       &h[1],
                                                       &h[2],
                                                       3,
                                                       &out[0],
                                                       &out[1],
                                                       &out[2]);
            if (err != SFEM_SUCCESS) {
                return err;
            }
        }

        double tock = smesh::time_seconds();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int SemiStructuredEMLinearElasticity::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("[Error] em:LinearElasticity::value NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLinearElasticity::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredEMLinearElasticity::clone() const {
        auto ret = std::make_shared<SemiStructuredEMLinearElasticity>(space);
        *ret     = *this;
        return ret;
    }

    void SemiStructuredEMLinearElasticity::set_value_in_block(const std::string &block_name,
                                                              const std::string &var_name,
                                                              const real_t       value) {
        bool changed = false;
        if (var_name == "mu") {
            mu      = value;
            changed = true;
        } else if (var_name == "lambda") {
            lambda  = value;
            changed = true;
        }

        if (changed && element_matrix) {
            auto &ssm  = space->mesh();
            auto  mesh = element_matrix_mesh(space);
            if (!mesh) {
                return;
            }

            for (size_t b = 0; b < element_matrices.size(); ++b) {
                auto &matrix = element_matrices[b];
                if (!matrix) {
                    continue;
                }

                if (!block_name.empty() && block_name != ssm.block(b)->name()) {
                    continue;
                }

                const auto block_id = static_cast<smesh::block_idx_t>(b);
                sshex8_linear_elasticity_element_matrix_cartesian(smesh::semistructured_level(ssm),
                                                                  mesh->n_elements(block_id),
                                                                  mesh->n_nodes(),
                                                                  mesh->elements(block_id)->data(),
                                                                  mesh->points()->data(),
                                                                  mu,
                                                                  lambda,
                                                                  matrix->data());
            }
        }
    }

}  // namespace sfem
