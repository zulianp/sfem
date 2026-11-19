#ifndef SFEM_CONTEXT_HPP
#define SFEM_CONTEXT_HPP

#include "smesh_context.hpp"

#include <memory>

namespace sfem {
    std::shared_ptr<smesh::Context> initialize(int argc, char *argv[]);
    std::shared_ptr<smesh::Context> initialize_serial(int argc, char *argv[]);
}  // namespace sfem

#endif  // SFEM_CONTEXT_HPP