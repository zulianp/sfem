#ifndef SFEM_BCRS_SYM_SPMV_HPP
#define SFEM_BCRS_SYM_SPMV_HPP

#include <iostream>

namespace sfem {
    template <typename R, typename C, typename T>
    class BCRSSymSpMV : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return 0;
        }

        std::ptrdiff_t rows() const override { return block_size_ * block_rows_; }
        std::ptrdiff_t cols() const override { return block_size_ * block_cols_; }
        inline int block_size() const { return block_size_; }

        std::shared_ptr<Buffer<R>> rowptr;
        std::shared_ptr<Buffer<C>> colidx;
        std::shared_ptr<Buffer<T*>> diag_values;
        std::shared_ptr<Buffer<T*>> off_diag_values;

        int block_size_{0};
        ptrdiff_t block_rows_{0};
        ptrdiff_t block_cols_{0};
        ptrdiff_t block_stride_{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        ExecutionSpace execution_space() const override { return execution_space_; }

        void print(std::ostream& os) const {
            os << "BCRSSymSpMV (" << rows() << " x " << cols() << ")\n";
            os << "Block size: " << block_size_ << "\n";
            os << "Block rows: " << block_rows_ << "\n";
            os << "Block cols: " << block_cols_ << "\n";
            os << "Block stride: " << block_stride_ << "\n";

            // Get raw pointers
            auto rowptr = this->rowptr->data();
            auto colidx = this->colidx->data();
            auto diag_values = this->diag_values->data();
            auto off_diag_values = this->off_diag_values->data();

            for (ptrdiff_t i = 0; i < block_rows_; i++) {
                os << "Row: " << i << " (" << (rowptr[i + 1] - rowptr[i]) << "):\n";

                os << i << ")\n";
                int d_idx = 0;
                for (int d = 0; d < block_size_; d++) {
                    os << "\t";

                    for (int d2 = 0; d2 < d; d2++) {
                        os << "x ";
                    }
                    
                    for (int d2 = d; d2 < block_size_; d2++, d_idx++) {
                        os << diag_values[d_idx][i * block_stride_] << " ";
                    }

                    os << "\n";
                }

                for (count_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
                    os << colidx[j] << ")\n";

                    int d_idx = 0;
                    for (int d = 0; d < block_size_; d++) {
                        os << "\t";
                        
                        for (int d2 = 0; d2 < d; d2++) {
                            os << "x ";
                        }

                        for (int d2 = d; d2 < block_size_; d2++, d_idx++) {
                            os << off_diag_values[d_idx][j * block_stride_] << " ";
                        }

                        os << "\n";
                    }
                }
                os << "\n";
            }
        }
    };

    template <typename R, typename C, typename T>
    std::shared_ptr<BCRSSymSpMV<R, C, T>> h_bcrs_sym_spmv(
            const ptrdiff_t block_rows,
            const ptrdiff_t block_cols,
            const int block_size,
            const std::shared_ptr<Buffer<R>>& rowptr,
            const std::shared_ptr<Buffer<C>>& colidx,
            const ptrdiff_t block_stride,
            const std::shared_ptr<Buffer<T*>>& diag_values,
            const std::shared_ptr<Buffer<T*>>& off_diag_values,
            const T scale_output) {
        auto ret = std::make_shared<BCRSSymSpMV<R, C, T>>();
        ret->rowptr = rowptr;
        ret->colidx = colidx;
        ret->diag_values = diag_values;
        ret->off_diag_values = off_diag_values;
        ret->block_rows_ = block_rows;
        ret->block_cols_ = block_cols;
        ret->block_size_ = block_size;
        ret->block_stride_ = block_stride;
        ret->execution_space_ = EXECUTION_SPACE_HOST;

        ret->apply_ = [=](const T* const x, T* const y) {
            double tick = MPI_Wtime();
            const ptrdiff_t nnz = ret->colidx->size();
            const ptrdiff_t block_rows = ret->block_rows_;
            const ptrdiff_t block_size = ret->block_size_;
            const ptrdiff_t block_stride = ret->block_stride_;

            // Check preconditions
            assert(ret->colidx->size() == nnz);
            assert(ret->off_diag_values->extent(1) == nnz);
            assert(ret->diag_values->extent(1) == block_rows);

            // Get raw pointers
            auto rowptr = ret->rowptr->data();
            auto colidx = ret->colidx->data();
            auto diag_values = ret->diag_values->data();
            auto off_diag_values = ret->off_diag_values->data();

            if (scale_output != 1) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < block_rows * block_size; ++i) {
                    y[i] *= scale_output;
                }
            }   

            // Apply diagonal block
#pragma omp parallel for
            for (ptrdiff_t row = 0; row < block_rows; row++) {
                ptrdiff_t diag_idx = row * block_stride;
                int d_idx = 0;
                for (int d1 = 0; d1 < block_size; d1++) {
                    y[row * block_size + d1] += diag_values[d_idx][row] * x[row * block_size + d1];

                    for (int d2 = d1 + 1; d2 < block_size; d2++, d_idx++) {
                        y[row * block_size + d1] +=
                                diag_values[d_idx][diag_idx] * x[row * block_size + d2];

                        y[row * block_size + d2] +=
                                diag_values[d_idx][diag_idx] * x[row * block_size + d1];
                    }
                }
            }

            // Apply off-diagonal blocks
#pragma omp parallel for
            for (ptrdiff_t row = 0; row < block_rows; row++) {
                const count_t lenrow = rowptr[row + 1] - rowptr[row];
                const idx_t* cols = &colidx[rowptr[row]];

                for (count_t k = 0; k < lenrow; k++) {
                    const auto col = cols[k];
                    const ptrdiff_t off_diag_idx = (rowptr[row] + k) * block_stride;

                    int d_idx = 0;
                    for (int d1 = 0; d1 < block_size; d1++) {
#pragma omp atomic
                        y[row * block_size + d1] +=
                                off_diag_values[d_idx][off_diag_idx] * x[col * block_size + d1];

                        for (int d2 = d1 + 1; d2 < block_size; d2++, d_idx++) {
#pragma omp atomic
                            y[row * block_size + d1] +=
                                    off_diag_values[d_idx][off_diag_idx] * x[col * block_size + d2];

#pragma omp atomic
                            y[col * block_size + d2] +=
                                    off_diag_values[d_idx][off_diag_idx] * x[row * block_size + d1];
                        }
                    }
                }
            }

            double tock = MPI_Wtime();
            printf("Time BCRS_sym: %f\n", tock - tick);
        };

        return ret;
    }
}  // namespace sfem

#endif  // SFEM_BC_SYM_SPMV_HPP
