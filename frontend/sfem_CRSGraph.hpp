#ifndef SFEM_CRS_GRAPH_HPP
#define SFEM_CRS_GRAPH_HPP

#include "sfem_base.hpp"
#include "sfem_Buffer.hpp"
#include "smesh_crs_graph.hpp"
#include "smesh_crs_graph.impl.hpp"

namespace sfem {

using CRSGraph = smesh::CRSGraph<count_t, idx_t>;

}  // namespace sfem

#endif  // SFEM_CRS_GRAPH_HPP
