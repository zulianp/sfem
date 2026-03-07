#ifndef SFEM_MESH_HPP
#define SFEM_MESH_HPP

#include "smesh_mesh.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_Communicator.hpp"
#include "sfem_CRSGraph.hpp"
#include "smesh_mesh.hpp"

namespace sfem {

using Mesh = smesh::Mesh;
using SharedMesh = std::shared_ptr<Mesh>;
using SharedBlock = std::shared_ptr<Mesh::Block>;

}  // namespace sfem

#endif  // SFEM_MESH_HPP
