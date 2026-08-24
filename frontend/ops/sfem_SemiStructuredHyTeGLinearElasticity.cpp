#include "sfem_SemiStructuredHyTeGLinearElasticity.hpp"

#include "sshex8_linear_elasticity.hpp"
#include "sshex8_stencil_element_matrix_apply.hpp"

#include "sfem_LinearElasticity.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>

namespace sfem {

    namespace {
        static bool block_is_selected(const std::string &name, const std::vector<std::string> &block_names) {
            return block_names.empty() || std::find(block_names.begin(), block_names.end(), name) != block_names.end();
        }

        static int require_sshex8_block(const char *const op_name, const smesh::Mesh &mesh, const size_t block_id) {
            const auto element_type = mesh.element_type(static_cast<smesh::block_idx_t>(block_id));
            if (!is_semistructured_type(element_type)) {
                SFEM_ERROR("%s supports semistructured blocks\n", op_name);
                return SFEM_FAILURE;
            }

            if (smesh::ss_source_family(element_type) != smesh::HEX8) {
                SFEM_ERROR("%s supports SSHEX8 blocks\n", op_name);
                return SFEM_FAILURE;
            }

            return SFEM_SUCCESS;
        }

        static smesh::ElemType standard_base_elem(const smesh::ElemType element_type) { return macro_base_elem(element_type); }

        static std::shared_ptr<smesh::Mesh> element_matrix_mesh(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = space->has_semi_structured_mesh() ? smesh::derefine(space->mesh_ptr(), 1) : space->mesh_ptr();
            if (mesh && mesh->element_type(0) == smesh::PROTEUS_HEX8) {
                mesh = smesh::sshex_to_hex8(mesh);
            }

            return mesh;
        }

        static int build_hyteg_stencils(const std::shared_ptr<FunctionSpace>          &space,
                                        const real_t                                   mu,
                                        const real_t                                   lambda,
                                        const std::vector<std::string>                &block_names,
                                        std::vector<std::shared_ptr<Buffer<scalar_t>>> &category_stencils) {
            auto &ssm  = space->mesh();
            auto  mesh = element_matrix_mesh(space);
            if (!mesh) {
                return SFEM_FAILURE;
            }

            const auto n_blocks = ssm.n_blocks();
            if (block_names.empty() || category_stencils.size() != n_blocks) {
                category_stencils.assign(n_blocks, nullptr);
            }

            int err = SFEM_SUCCESS;
            for (size_t b = 0; b < n_blocks; ++b) {
                const auto block_id = static_cast<smesh::block_idx_t>(b);
                if (!block_is_selected(ssm.block(b)->name(), block_names)) {
                    continue;
                }

                err = require_sshex8_block("LinearElasticityHyTeG", ssm, b);
                if (err != SFEM_SUCCESS) {
                    return err;
                }

                const ptrdiff_t ne = mesh->n_elements(block_id);
                auto            matrix = sfem::create_host_buffer<scalar_t>(ne * 24 * 24);
                err = sshex8_linear_elasticity_element_matrix_cartesian(smesh::semistructured_level(ssm),
                                                                        ne,
                                                                        mesh->n_nodes(),
                                                                        mesh->elements(block_id)->data(),
                                                                        mesh->points()->data(),
                                                                        mu,
                                                                        lambda,
                                                                        matrix->data());
                if (err != SFEM_SUCCESS) {
                    return err;
                }

                auto categories = sfem::create_host_buffer<scalar_t>(ne * 27 * 27 * 9);
                err             = sshex8_linear_elasticity_element_matrix_to_category_stencils(
                        ne, matrix->data(), categories->data());
                if (err != SFEM_SUCCESS) {
                    return err;
                }

                category_stencils[b] = categories;
            }

            return SFEM_SUCCESS;
        }
    }  // namespace

    std::unique_ptr<Op> SemiStructuredHyTeGLinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredHyTeGLinearElasticity::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredHyTeGLinearElasticity::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(is_semistructured_type(space->element_type()));
        auto ret          = std::make_unique<SemiStructuredHyTeGLinearElasticity>(space);
        ret->element_type = (smesh::ElemType)space->element_type();
        return ret;
    }

    SemiStructuredHyTeGLinearElasticity::SemiStructuredHyTeGLinearElasticity(
            const std::shared_ptr<FunctionSpace> &space)
        : space(space) {}

    SemiStructuredHyTeGLinearElasticity::~SemiStructuredHyTeGLinearElasticity() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredHyTeGLinearElasticity[%d]::apply() called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   smesh::semistructured_level(space->mesh()),
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    std::shared_ptr<Op> SemiStructuredHyTeGLinearElasticity::lor_op(const std::shared_ptr<FunctionSpace> &) {
        SFEM_ERROR("[Error] LinearElasticityHyTeG::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    std::shared_ptr<Op> SemiStructuredHyTeGLinearElasticity::derefine_op(
            const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredHyTeGLinearElasticity::derefine_op");

        if (space->has_semi_structured_mesh()) {
            auto ret          = std::make_shared<SemiStructuredHyTeGLinearElasticity>(space);
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
        std::vector<smesh::ElemType> element_types(space->n_blocks(), standard_base_elem(element_type));
        ret->override_element_types(element_types);
        return ret;
    }

    const char *SemiStructuredHyTeGLinearElasticity::name() const { return "LinearElasticityHyTeG"; }

    int SemiStructuredHyTeGLinearElasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("SemiStructuredHyTeGLinearElasticity::initialize");

        real_t SFEM_SHEAR_MODULUS        = mu;
        real_t SFEM_FIRST_LAME_PARAMETER = lambda;
        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
        mu     = SFEM_SHEAR_MODULUS;
        lambda = SFEM_FIRST_LAME_PARAMETER;

        return build_hyteg_stencils(space, mu, lambda, block_names, category_stencils);
    }

    int SemiStructuredHyTeGLinearElasticity::hessian_crs(const real_t *const,
                                                        const count_t *const,
                                                        const idx_t *const,
                                                        real_t *const) {
        SFEM_ERROR("[Error] LinearElasticityHyTeG::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredHyTeGLinearElasticity::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredHyTeGLinearElasticity::hessian_diag");

        auto &ssm = space->mesh();
        int   err = SFEM_SUCCESS;
        for (size_t b = 0; b < category_stencils.size(); ++b) {
            if (!category_stencils[b]) {
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

    int SemiStructuredHyTeGLinearElasticity::hessian_block_diag_sym(const real_t *const, real_t *const values) {
        SFEM_TRACE_SCOPE("SemiStructuredHyTeGLinearElasticity::hessian_block_diag_sym");

        auto &ssm = space->mesh();
        int   err = SFEM_SUCCESS;
        for (size_t b = 0; b < category_stencils.size(); ++b) {
            if (!category_stencils[b]) {
                continue;
            }

            const auto block_id = static_cast<smesh::block_idx_t>(b);
            err                 = affine_sshex8_linear_elasticity_block_diag_sym(smesh::semistructured_level(ssm),
                                                                 ssm.n_elements(block_id),
                                                                 ssm.n_nodes(),
                                                                 ssm.elements(block_id)->data(),
                                                                 ssm.points()->data(),
                                                                 mu,
                                                                 lambda,
                                                                 6,
                                                                 &values[0],
                                                                 &values[1],
                                                                 &values[2],
                                                                 &values[3],
                                                                 &values[4],
                                                                 &values[5]);
            if (err != SFEM_SUCCESS) {
                return err;
            }
        }

        return SFEM_SUCCESS;
    }

    int SemiStructuredHyTeGLinearElasticity::gradient(const real_t *const, real_t *const) {
        SFEM_ERROR("[Error] LinearElasticityHyTeG::gradient NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredHyTeGLinearElasticity::apply(const real_t *const, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredHyTeGLinearElasticity::apply");

        assert(is_semistructured_type(element_type));
        auto &ssm = space->mesh();

        double tick = smesh::time_seconds();
        int    err  = SFEM_SUCCESS;
        for (size_t b = 0; b < category_stencils.size(); ++b) {
            auto &categories = category_stencils[b];
            if (!categories) {
                continue;
            }

            const auto block_id = static_cast<smesh::block_idx_t>(b);
            err                 = sshex8_stencil_element_matrix_apply3_hyteg_stencil(smesh::semistructured_level(ssm),
                                                                    ssm.n_elements(block_id),
                                                                    ssm.elements(block_id)->data(),
                                                                    categories->data(),
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

    int SemiStructuredHyTeGLinearElasticity::value(const real_t *, real_t *const) {
        SFEM_ERROR("[Error] LinearElasticityHyTeG::value NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredHyTeGLinearElasticity::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredHyTeGLinearElasticity::clone() const {
        auto ret = std::make_shared<SemiStructuredHyTeGLinearElasticity>(space);
        *ret     = *this;
        return ret;
    }

    void SemiStructuredHyTeGLinearElasticity::set_value_in_block(const std::string &block_name,
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

        if (!changed) {
            return;
        }

        std::vector<std::string> block_names;
        if (!block_name.empty()) {
            block_names.push_back(block_name);
        }

        if (build_hyteg_stencils(space, mu, lambda, block_names, category_stencils) != SFEM_SUCCESS) {
            SFEM_ERROR("Failed to rebuild SSHEX8 HyTeG linear elasticity stencils\n");
        }
    }

}  // namespace sfem
