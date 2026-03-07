#ifndef SFEM_BUFFER_HPP
#define SFEM_BUFFER_HPP

#include "smesh_buffer.hpp"
#include "smesh_buffer.impl.hpp"
#include "smesh_distributed_buffer.hpp"

namespace sfem {

using smesh::ExecutionSpace;
using smesh::MemorySpace;
using smesh::execution_space_from_string;

using smesh::EXECUTION_SPACE_DEVICE;
using smesh::EXECUTION_SPACE_HOST;
using smesh::EXECUTION_SPACE_INVALID;

using smesh::MEMORY_SPACE_DEVICE;
using smesh::MEMORY_SPACE_HOST;
using smesh::MEMORY_SPACE_INVALID;

template <typename T>
using Buffer = smesh::Buffer<T>;

template <typename T>
using SharedBuffer = smesh::SharedBuffer<T>;

using smesh::astype;
using smesh::convert_host_buffer_to_fake_SoA;
using smesh::copy;
using smesh::create_host_buffer;
using smesh::create_host_buffer_fake_SoA;
using smesh::manage_host_buffer;
using smesh::soa_to_aos;
using smesh::sub;
using smesh::view;
using smesh::zeros_like;

template <typename T>
inline std::shared_ptr<Buffer<T>> create_buffer_from_file(const std::shared_ptr<smesh::Communicator> &comm, const char *path) {
    return smesh::create_buffer_from_file<T>(comm, path);
}

}  // namespace sfem

#endif  // SFEM_BUFFER_HPP
