#include "sfem_SemiStructuredEMLinearElasticity.hpp"

#include "sshex8_linear_elasticity.hpp"
#include "sshex8_stencil_element_matrix_apply.hpp"

#include "sfem_LinearElasticity.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include "smesh_glob.hpp"

namespace sfem {

    namespace {
        static bool accepts_block_names(const smesh::Mesh &mesh, const std::vector<std::string> &block_names) {
            if (block_names.empty()) {
                return true;
            }

            if (mesh.n_blocks() != 1) {
                return false;
            }

            const std::string &name = mesh.block(0)->name();
            for (const auto &block_name : block_names) {
                if (block_name == name) {
                    return true;
                }
            }

            return false;
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
        assert(space->n_blocks() == 1);
        ret->override_element_types({macro_base_elem(element_type)});
        return ret;
    }

    const char *SemiStructuredEMLinearElasticity::name() const { return "ss:em:LinearElasticity"; }

    int SemiStructuredEMLinearElasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::initialize");

        auto &ssm = space->mesh();
        if (!accepts_block_names(ssm, block_names)) {
            SFEM_ERROR("SemiStructuredEMLinearElasticity supports one semistructured block\n");
            return SFEM_FAILURE;
        }

        real_t SFEM_SHEAR_MODULUS        = mu;
        real_t SFEM_FIRST_LAME_PARAMETER = lambda;
        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
        mu     = SFEM_SHEAR_MODULUS;
        lambda = SFEM_FIRST_LAME_PARAMETER;

        auto mesh      = space->has_semi_structured_mesh() ? smesh::derefine(space->mesh_ptr(), 1) : space->mesh_ptr();
        element_matrix = sfem::create_host_buffer<scalar_t>(mesh->n_elements() * 24 * 24);

        return sshex8_linear_elasticity_element_matrix(smesh::semistructured_level(ssm),
                                                       mesh->n_elements(),
                                                       mesh->n_nodes(),
                                                       mesh->elements(0)->data(),
                                                       mesh->points()->data(),
                                                       mu,
                                                       lambda,
                                                       element_matrix->data());
    }

    int SemiStructuredEMLinearElasticity::hessian_crs(const real_t *const,
                                                      const count_t *const,
                                                      const idx_t *const,
                                                      real_t *const) {
        SFEM_ERROR("[Error] ss:em:LinearElasticity::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLinearElasticity::hessian_diag(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::hessian_diag");

        auto &ssm = space->mesh();
        return affine_sshex8_linear_elasticity_diag(smesh::semistructured_level(ssm),
                                                    ssm.n_elements(),
                                                    ssm.n_nodes(),
                                                    ssm.elements(0)->data(),
                                                    ssm.points()->data(),
                                                    mu,
                                                    lambda,
                                                    3,
                                                    &out[0],
                                                    &out[1],
                                                    &out[2]);
    }

    int SemiStructuredEMLinearElasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_ERROR("[Error] ss:em:LinearElasticity::gradient NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredEMLinearElasticity::apply(const real_t *const, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredEMLinearElasticity::apply");

        assert(is_semistructured_type(element_type));
        auto &ssm = space->mesh();

        double tick = smesh::time_seconds();
        int    err  = sshex8_stencil_element_matrix_apply3(smesh::semistructured_level(ssm),
                                                          ssm.n_elements(),
                                                          ssm.elements(0)->data(),
                                                          element_matrix->data(),
                                                          3,
                                                          &h[0],
                                                          &h[1],
                                                          &h[2],
                                                          3,
                                                          &out[0],
                                                          &out[1],
                                                          &out[2]);

        double tock = smesh::time_seconds();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int SemiStructuredEMLinearElasticity::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("[Error] ss:em:LinearElasticity::value NOT IMPLEMENTED!\n");
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
        if (!block_name.empty() && space->mesh().n_blocks() == 1 && block_name != space->mesh().block(0)->name()) {
            return;
        }

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
            auto  mesh = space->has_semi_structured_mesh() ? smesh::derefine(space->mesh_ptr(), 1) : space->mesh_ptr();

            sshex8_linear_elasticity_element_matrix(smesh::semistructured_level(ssm),
                                                    mesh->n_elements(),
                                                    mesh->n_nodes(),
                                                    mesh->elements(0)->data(),
                                                    mesh->points()->data(),
                                                    mu,
                                                    lambda,
                                                    element_matrix->data());
        }
    }

}  // namespace sfem
