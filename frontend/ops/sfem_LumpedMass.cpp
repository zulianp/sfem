#include "sfem_LumpedMass.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "mass.h"

#include "sfem_Mesh.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_CRSGraph.hpp"

namespace sfem {

    std::unique_ptr<Op> LumpedMass::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("LumpedMass::create");

        auto ret          = std::make_unique<LumpedMass>(space);
        ret->element_type = (enum ElemType)space->element_type();
        return ret;
    }

    LumpedMass::LumpedMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
    
    int LumpedMass::hessian_crs(const real_t *const  x,
                                const count_t *const rowptr,
                                const idx_t *const   colidx,
                                real_t *const        values) {
        assert(0);
        return SFEM_FAILURE;
    }

    int LumpedMass::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("LumpedMass::hessian_diag");

        auto mesh = space->mesh_ptr();

        if (space->block_size() == 1) {
            assemble_lumped_mass(element_type,
                                 mesh->n_elements(),
                                 mesh->n_nodes(),
                                 mesh->elements()->data(),
                                 mesh->points()->data(),
                                 values);
        } else {
            const ptrdiff_t n_nodes = mesh->n_nodes();
            real_t         *temp    = (real_t *)calloc(n_nodes, sizeof(real_t));
            assemble_lumped_mass(
                    element_type, mesh->n_elements(), n_nodes, mesh->elements()->data(), mesh->points()->data(), temp);

            int bs = space->block_size();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                for (int b = 0; b < bs; b++) {
                    values[i * bs + b] += temp[i];
                }
            }

            free(temp);
        }

        return SFEM_SUCCESS;
    }

    int LumpedMass::gradient(const real_t *const x, real_t *const out) {
        assert(0);
        return SFEM_FAILURE;
    }

    int LumpedMass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        assert(0);
        return SFEM_FAILURE;
    }

    int LumpedMass::value(const real_t *x, real_t *const out) {
        assert(0);
        return SFEM_FAILURE;
    }

    int LumpedMass::report(const real_t *const) {
        return SFEM_SUCCESS;
    }

    std::shared_ptr<Op> LumpedMass::clone() const {
        auto ret = std::make_shared<LumpedMass>(space);
        *ret     = *this;
        return ret;
    }

} // namespace sfem 