#include "sfem_BoundaryMass.hpp"
#include "sfem_glob.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_Mesh.hpp"
#include "mass.h"
#include <mpi.h>

namespace sfem {

    std::unique_ptr<Op> BoundaryMass::create(const std::shared_ptr<FunctionSpace> &space, const std::shared_ptr<Buffer<idx_t *>> &boundary_elements) {
        SFEM_TRACE_SCOPE("BoundaryMass::create");

        auto ret          = std::make_unique<BoundaryMass>(space);
        auto element_type = (enum ElemType)space->element_type();
        ret->element_type = shell_type(side_type(element_type));
        if (ret->element_type == INVALID) {
            std::cerr << "Invalid element type for BoundaryMass, Bulk element type: " << type_to_string(element_type) << "\n";
            return nullptr;
        }
        ret->boundary_elements = boundary_elements;
        return ret;
    }

    BoundaryMass::BoundaryMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int BoundaryMass::initialize() {
        return SFEM_SUCCESS;
    }

    int BoundaryMass::hessian_crs(const real_t *const  x,
                                  const count_t *const rowptr,
                                  const idx_t *const   colidx,
                                  real_t *const        values) {
        // auto mesh = (mesh_t *)space->mesh().impl_mesh();

        // auto graph = space->dof_to_dof_graph();

        // assemble_mass(element_type,
        //               boundary_elements->extent(1),
        //               mesh->nnodes,
        //               boundary_elements->data(),
        //               mesh->points,
        //               graph->rowptr()->data(),
        //               graph->colidx()->data(),
        //               values);

        // return SFEM_SUCCESS;

        assert(0);
        return SFEM_FAILURE;
    }

    int BoundaryMass::gradient(const real_t *const x, real_t *const out) {
        // auto mesh = (mesh_t *)space->mesh().impl_mesh();

        // assert(1 == space->block_size());

        // apply_mass(element_type,
        //            boundary_elements->extent(1),
        //            mesh->nnodes,
        //            boundary_elements->data(),
        //            mesh->points,
        //            x,
        //            out);

        // return SFEM_SUCCESS;

        assert(0);
        return SFEM_FAILURE;
    }

    int BoundaryMass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("BoundaryMass::apply");

        auto mesh = space->mesh_ptr();

        int  block_size = space->block_size();
        auto data       = boundary_elements->data();

        for (int d = 0; d < block_size; d++) {
            apply_mass(element_type,
                       boundary_elements->extent(1),
                       mesh->n_nodes(),
                       boundary_elements->data(),
                       mesh->points()->data(),
                       block_size,
                       &h[d],
                       block_size,
                       &out[d]);
        }

        return SFEM_SUCCESS;
    }

    int BoundaryMass::value(const real_t *x, real_t *const out) {
        // auto mesh = (mesh_t *)space->mesh().impl_mesh();

        // mass_assemble_value((enum ElemType)space->element_type(),
        //                     mesh->nelements,
        //                     mesh->nnodes,
        //                     mesh->elements,
        //                     mesh->points,
        //                     x,
        //                     out);

        // return SFEM_SUCCESS;

        assert(0);
        return SFEM_FAILURE;
    }

    int BoundaryMass::report(const real_t *const) {
        return SFEM_SUCCESS;
    }

} // namespace sfem 