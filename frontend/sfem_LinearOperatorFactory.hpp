#ifndef SFEM_LINEAR_OPERATOR_FACTORY_HPP
#define SFEM_LINEAR_OPERATOR_FACTORY_HPP

#include "sfem_CRS.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

// Linear operators of a Function: matrix-free, the stored tangent, or an assembled Hessian in one
// of the sparse formats (op_type in sfem_defs.hpp).
namespace sfem {

    std::shared_ptr<Operator<real_t>> make_linear_op(const std::shared_ptr<Function> &f);

    std::shared_ptr<Operator<real_t>> make_linear_op_variant(const std::shared_ptr<Function>                &f,
                                                             const std::vector<std::pair<std::string, int>> &opts);

    std::shared_ptr<sfem::CRS<count_t, idx_t, real_t, real_t>> hessian_crs(sfem::Function                  &f,
                                                                           const std::shared_ptr<CRSGraph> &crs_graph,
                                                                           const sfem::ExecutionSpace       es);

    std::shared_ptr<sfem::CRS<count_t, idx_t, real_t, real_t>> hessian_crs(const std::shared_ptr<sfem::Function> &f,
                                                                           const SharedBuffer<real_t>            &x,
                                                                           const sfem::ExecutionSpace             es);

    std::shared_ptr<sfem::CRS<count_t, idx_t, real_t, real_t>> node_block_crs_to_scalar_crs(
            const std::shared_ptr<CRSGraph> &crs_graph,
            const int                        block_size,
            const SharedBuffer<real_t>      &node_block_values);

    std::shared_ptr<Operator<real_t>> compose_constraints_op(const std::shared_ptr<sfem::Function>         &f,
                                                             const std::shared_ptr<sfem::Operator<real_t>> &op);

    std::shared_ptr<sfem::Operator<real_t>> hessian_bsr(const std::shared_ptr<sfem::Function> &f,
                                                        const SharedBuffer<real_t>            &x,
                                                        const sfem::ExecutionSpace             es);

    std::shared_ptr<sfem::Operator<real_t>> create_linear_operator(const std::string                           &format,
                                                                   const std::shared_ptr<sfem::Function>       &f,
                                                                   const std::shared_ptr<sfem::Buffer<real_t>> &u,
                                                                   enum sfem::ExecutionSpace                    es);

    SharedInPlaceOperator<real_t> create_zero_constraints_op(const std::shared_ptr<Function> &f);

}  // namespace sfem

#endif  // SFEM_LINEAR_OPERATOR_FACTORY_HPP
