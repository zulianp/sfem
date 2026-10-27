#ifndef SFEM_SPARSE_BLOCK_VECTOR_HPP
#define SFEM_SPARSE_BLOCK_VECTOR_HPP

#include "sfem_base.hpp"

#include "sfem_aliases.hpp"

#include <memory>
#include <ostream>

namespace sfem {

    template <typename T>
    class SparseBlockVector /*: public Operator<T>*/ {
    public:
        int                 block_size_{0};
        SharedBuffer<idx_t> idx_;
        SharedBuffer<T>     data_;

        inline int                 block_size() const { return block_size_; }
        const SharedBuffer<idx_t>& idx() const { return idx_; }
        const SharedBuffer<T>&     data() const { return data_; }
        ptrdiff_t                  n_blocks() const { return idx_->size(); }

        enum MemorySpace mem_space() const {
            if (data_) {
                return data_->mem_space();
            }

            return MEMORY_SPACE_INVALID;
        }

        void print(std::ostream& os) const {
            for (ptrdiff_t i = 0; i < n_blocks(); i++) {
                os << idx_->data()[i] << ") ";

                for (int d = 0; d < block_size(); d++) {
                    os << data_->data()[i * 6 + d];
                    os << " ";
                }

                os << "\n";
            }

            os << "\n";
        }

        // TODO maybe
        // int apply(const T* const x, T* const y) override {

        //     return SFEM_SUCCESS;
        // }

        // std::ptrdiff_t rows() const override { return data_->size(); }
        // std::ptrdiff_t cols() const override { return data_->size(); }
        // ExecutionSpace execution_space() const override { return data_->mem_space(); }
    };

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> create_sparse_block_vector(const SharedBuffer<idx_t>& idx,
                                                                     const SharedBuffer<T>&     data) {
        auto ret         = std::make_shared<SparseBlockVector<T>>();
        ret->block_size_ = idx->size() ? static_cast<int>(data->size() / idx->size()) : 0;
        ret->idx_        = idx;
        ret->data_       = data;
        return ret;
    }

}  // namespace sfem

#endif  // SFEM_SPARSE_BLOCK_VECTOR_HPP
