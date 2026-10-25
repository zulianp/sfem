#ifndef SFEM_TRANSFER_HPP
#define SFEM_TRANSFER_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_execution_context.hpp"
#include "smesh_crs_graph.hpp"
#include "smesh_restrict.hpp"

#include <memory>

// Grid transfers between nested function spaces (multigrid prolongation and restriction).
namespace sfem {

    SharedBuffer<idx_t> create_edge_idx(CRSGraph &crs_graph);

    std::shared_ptr<CRSGraph> create_derefined_crs_graph(FunctionSpace &space);

    bool mesh_is_mpi(const std::shared_ptr<Mesh> &mesh);

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation(const std::shared_ptr<FunctionSpace> &from_space,
                                                                       const std::shared_ptr<FunctionSpace> &to_space,
                                                                       const ExecutionContext               &context);

    std::shared_ptr<Operator<real_t>> make_op(const std::shared_ptr<smesh::AbstractRestrict<real_t>> &op);

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction(const std::shared_ptr<FunctionSpace> &from_space,
                                                                      const std::shared_ptr<FunctionSpace> &to_space,
                                                                      const ExecutionSpace                  es);

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction_from_graph(const ptrdiff_t                  n_fine_nodes,
                                                                                 const int                        block_size,
                                                                                 const std::shared_ptr<CRSGraph> &crs_graph,
                                                                                 const SharedBuffer<idx_t>       &edges,
                                                                                 const ExecutionContext          &context);

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation_from_graph(const std::shared_ptr<Function> &function,
                                                                                  const std::shared_ptr<CRSGraph> &crs_graph,
                                                                                  const SharedBuffer<idx_t>       &edges,
                                                                                  const ExecutionContext          &context);

}  // namespace sfem

#endif  // SFEM_TRANSFER_HPP
