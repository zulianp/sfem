#pragma once

#include "sfem_base.hpp"
#include "sfem_aliases.hpp"

namespace sfem {

template <typename T>
class SparseBlockVector {
public:
    SharedBuffer<T> values;
    ptrdiff_t n_blocks{0};
    ptrdiff_t block_size{0};

    ptrdiff_t n_blocks() const { return n_blocks; }
    ptrdiff_t block_size() const { return block_size; }
};

template <typename T>
std::shared_ptr<Operator<T>> create_sparse_block_vector_mult(const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                                             const SharedBuffer<T>& diag);

} // namespace sfem 