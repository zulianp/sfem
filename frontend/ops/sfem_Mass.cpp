#include "sfem_Mass.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "mass.h"

#include "sfem_Mesh.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_CRSGraph.hpp"

namespace sfem {

    std::unique_ptr<Op> Mass::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Mass::create");

        assert(1 == space->block_size());

        auto ret          = std::make_unique<Mass>(space);
        ret->element_type = (enum ElemType)space->element_type();
        return ret;
    }

    Mass::Mass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int Mass::hessian_crs(const real_t *const  x,
                          const count_t *const rowptr,
                          const idx_t *const   colidx,
                          real_t *const        values) {
        SFEM_TRACE_SCOPE("Mass::hessian_crs");

        auto mesh  = space->mesh_ptr();
        auto graph = space->dof_to_dof_graph();

        assemble_mass(element_type,
                      mesh->n_elements(),
                      mesh->n_nodes(),
                      mesh->elements()->data(),
                      mesh->points()->data(),
                      graph->rowptr()->data(),
                      graph->colidx()->data(),
                      values);
        return SFEM_SUCCESS;
    }

    int Mass::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Mass::gradient");

        auto mesh = space->mesh_ptr();
        // Use apply_mass for gradient computation
        apply_mass(element_type,
                   mesh->n_elements(),
                   mesh->n_nodes(),
                   mesh->elements()->data(),
                   mesh->points()->data(),
                   1, x, 1, out);
        return SFEM_SUCCESS;
    }

    int Mass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Mass::apply");

        auto   mesh = space->mesh_ptr();
        double tick = MPI_Wtime();

        apply_mass(element_type,
                   mesh->n_elements(),
                   mesh->n_nodes(),
                   mesh->elements()->data(),
                   mesh->points()->data(),
                   1, h, 1, out);

        double tock = MPI_Wtime();
        return SFEM_SUCCESS;
    }

    int Mass::value(const real_t *x, real_t *const out) {
        assert(0);
        return SFEM_FAILURE;
    }

    // int Mass::report(const real_t *const) { return SFEM_SUCCESS; }

    // std::shared_ptr<Op> Mass::clone() const {
    //     auto ret = std::make_shared<Mass>(space);
    //     *ret     = *this;
    //     return ret;
    // }

} // namespace sfem 