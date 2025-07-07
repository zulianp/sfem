#include "sfem_SemiStructuredLumpedMass.hpp"

#include "sshex8_mass.h"

#include "sfem_FunctionSpace.hpp"
#include "sfem_glob.hpp"
#include "sfem_LinearElasticity.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_LumpedMass.hpp"


namespace sfem {
// ... Implementation copied from sfem_Function.cpp ...
// (I will fill in the full implementation after creating all headers)

    std::unique_ptr<Op> SemiStructuredLumpedMass::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredLumpedMass::create");

        if (!space->has_semi_structured_mesh()) {
            SFEM_ERROR(
                    "[Error] SemiStructuredLumpedMass::create requires space with "
                    "semi_structured_mesh!\n");
        }

        assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
        auto ret          = std::make_unique<SemiStructuredLumpedMass>(space);
        ret->element_type = (enum ElemType)space->element_type();

        return ret;
    }

    std::shared_ptr<Op> SemiStructuredLumpedMass::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredLumpedMass::derefine_op");

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(element_type));
        if (space->has_semi_structured_mesh()) {
            auto ret          = std::make_shared<SemiStructuredLumpedMass>(space);
            ret->element_type = element_type;
            return ret;
        } else {
            auto ret          = std::make_shared<LumpedMass>(space);
            ret->element_type = macro_base_elem(element_type);
            return ret;
        }
    }

    int SemiStructuredLumpedMass::initialize() { return SFEM_SUCCESS; }

    SemiStructuredLumpedMass::SemiStructuredLumpedMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int SemiStructuredLumpedMass::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("SemiStructuredLumpedMass::hessian_diag");

        auto &ssm = space->semi_structured_mesh();
        if (space->block_size() == 1) {
            return affine_sshex8_mass_lumped(
                    ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), values);
        } else {
            const ptrdiff_t n = space->n_dofs() / space->block_size();

            auto    buff = create_host_buffer<real_t>(n);
            real_t *temp = buff->data();
            int     err  = affine_sshex8_mass_lumped(
                    ssm.level(), ssm.n_elements(), ssm.interior_start(), ssm.element_data(), ssm.point_data(), temp);

            if (err) SFEM_ERROR("Failure in affine_sshex8_mass_lumped\n");

            int bs = space->block_size();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                for (int b = 0; b < bs; b++) {
                    values[i * bs + b] += temp[i];
                }
            }
        }

        return SFEM_SUCCESS;
    }

    int SemiStructuredLumpedMass::hessian_crs(const real_t *const  x,
                                              const count_t *const rowptr,
                                              const idx_t *const   colidx,
                                              real_t *const        values) {
        SFEM_ERROR("IMEPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredLumpedMass::gradient(const real_t *const x, real_t *const out) {
        SFEM_ERROR("IMEPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredLumpedMass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_ERROR("IMEPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredLumpedMass::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("IMEPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredLumpedMass::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredLumpedMass::clone() const {
        auto ret = std::make_shared<SemiStructuredLumpedMass>(space);
        *ret     = *this;
        return ret;
    }

    SemiStructuredLumpedMass::~SemiStructuredLumpedMass() = default;
} // namespace sfem 