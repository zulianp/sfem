#ifndef SFEM_ALIASES_HPP
#define SFEM_ALIASES_HPP

#include "sfem_context.hpp"
#include "smesh_context.hpp"
#include "smesh_elem_type.hpp"
#include "smesh_forward_declarations.hpp"
#include "smesh_mesh.hpp"
#include "smesh_tracer.hpp"
#include "smesh_types.hpp"

#include <memory>

namespace sfem {

    using count_t = smesh::count_t;
    using idx_t   = smesh::idx_t;
    using mask_t  = smesh::mask_t;
    using real_t  = smesh::real_t;

    using Context       = smesh::Context;
    using SharedContext = std::shared_ptr<Context>;

    using ElemType     = smesh::ElemType;
    using Communicator = smesh::Communicator;
    using Mesh         = smesh::Mesh;
    using SharedMesh   = std::shared_ptr<Mesh>;
    using SharedBlock  = std::shared_ptr<smesh::Mesh::Block>;
    using CRSGraph     = smesh::CRSGraph<count_t, idx_t>;
    using Sideset      = smesh::Sideset;
    template <typename pack_idx_t>
    using PackedMesh = smesh::PackedMesh<pack_idx_t>;

    template <typename T>
    using Buffer = smesh::Buffer<T>;

    template <typename T>
    using SharedBuffer = smesh::SharedBuffer<T>;

    using smesh::astype;
    using smesh::convert_host_buffer_to_fake_SoA;
    using smesh::copy;
    using smesh::create_buffer;
    using smesh::create_host_buffer;
    using smesh::create_host_buffer_fake_SoA;
    using smesh::manage_host_buffer;
    using smesh::soa_to_aos;
    using smesh::sub;
    using smesh::view;
    using smesh::zeros_like;

    using smesh::EXECUTION_SPACE_DEVICE;
    using smesh::EXECUTION_SPACE_HOST;
    using smesh::EXECUTION_SPACE_INVALID;
    using smesh::ExecutionSpace;

    using smesh::MEMORY_SPACE_DEVICE;
    using smesh::MEMORY_SPACE_HOST;
    using smesh::MEMORY_SPACE_INVALID;
    using smesh::MemorySpace;

    using smesh::execution_space_from_string;

}  // namespace sfem

#define SFEM_TRACE_SCOPE(name) SMESH_TRACE_SCOPE(name)
#define SFEM_TRACE_SCOPE_VARIANT(format, num) SMESH_TRACE_SCOPE_VARIANT(format, num)
#endif
