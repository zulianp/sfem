#include "sfem_Laplacian.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "laplacian.h"

#include "sfem_Mesh.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_CRSGraph.hpp"

namespace sfem {

    std::unique_ptr<Op> Laplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Laplacian::create");

        assert(1 == space->block_size());

        auto ret          = std::make_unique<Laplacian>(space);
        ret->element_type = (enum ElemType)space->element_type();
        return ret;
    }

    std::shared_ptr<Op> Laplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret          = std::make_shared<Laplacian>(space);
        ret->element_type = macro_type_variant(element_type);
        return ret;
    }

    std::shared_ptr<Op> Laplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret          = std::make_shared<Laplacian>(space);
        ret->element_type = macro_base_elem(element_type);
        return ret;
    }

    Laplacian::Laplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
    
    Laplacian::~Laplacian() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("Laplacian::apply called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    int Laplacian::hessian_crs(const real_t *const  x,
                               const count_t *const rowptr,
                               const idx_t *const   colidx,
                               real_t *const        values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs");

        auto mesh  = space->mesh_ptr();
        auto graph = space->dof_to_dof_graph();

        return laplacian_crs(element_type,
                             mesh->n_elements(),
                             mesh->n_nodes(),
                             mesh->elements()->data(),
                             mesh->points()->data(),
                             graph->rowptr()->data(),
                             graph->colidx()->data(),
                             values);
    }

    int Laplacian::hessian_crs_sym(const real_t *const  x,
                                   const count_t *const rowptr,
                                   const idx_t *const   colidx,
                                   real_t *const        diag_values,
                                   real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs_sym");

        // auto graph = space->node_to_node_graph_upper_triangular();

        auto mesh = space->mesh_ptr();

        return laplacian_crs_sym(element_type,
                                 mesh->n_elements(),
                                 mesh->n_nodes(),
                                 mesh->elements()->data(),
                                 mesh->points()->data(),
                                 rowptr,
                                 colidx,
                                 diag_values,
                                 off_diag_values);
    }

    int Laplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_diag");

        auto mesh = space->mesh_ptr();
        return laplacian_diag(
                element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), values);
    }

    int Laplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::gradient");

        auto mesh = space->mesh_ptr();
        return laplacian_assemble_gradient(
                element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), x, out);
    }

    int Laplacian::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::apply");

        auto   mesh = space->mesh_ptr();
        double tick = MPI_Wtime();

        int err = laplacian_apply(
                element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), h, out);

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int Laplacian::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::value");

        auto mesh = space->mesh_ptr();
        return laplacian_assemble_value(
                element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), x, out);
    }

} // namespace sfem 