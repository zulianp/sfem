#ifndef CU_NODE_INTERPOLATE_HPP
#define CU_NODE_INTERPOLATE_HPP

#include "node_interpolate_kernel.hpp"

// sample_node<order> (node_interpolate_kernel.hpp) at every node, on the device: idx, disp, the
// points, the grid's coefficients and the outputs are device memory. order is 1 or 2.
int cu_sample_nodes(const int                         order,
                    const ptrdiff_t                   nnodes,
                    const idx_t *const SFEM_RESTRICT  idx,
                    const NodeSamplerPoints          &points,
                    const real_t *const SFEM_RESTRICT disp,
                    const NodeSamplerGrid            &grid,
                    const NodeSamplerOutput          &out,
                    void *const                       stream);

#endif  // CU_NODE_INTERPOLATE_HPP
