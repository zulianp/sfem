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

        template <int BLOCK_SIZE>
        void apply_sym_block(const T* const x, const T scale_output, T* const y) const {
            const ptrdiff_t nnz = this->colidx->size();
            const ptrdiff_t block_rows = this->block_rows_;
            const ptrdiff_t block_stride = this->block_stride_;

            // Check preconditions
            assert(this->colidx->size() == nnz);
            assert(this->off_diag_values->extent(1) == nnz);
            assert(this->diag_values->extent(1) == block_rows);

            // Get raw pointers
            auto rowptr = this->rowptr->data();
            auto colidx = this->colidx->data();
            auto diag_values = this->diag_values->data();
            auto off_diag_values = this->off_diag_values->data();

            if (scale_output != 1) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < block_rows * BLOCK_SIZE; ++i) {
                    y[i] *= scale_output;
                }
            }

            // Apply diagonal block
#pragma omp parallel for
            for (ptrdiff_t row = 0; row < block_rows; row++) {
                ptrdiff_t diag_idx = row * block_stride;

                T block_diag[BLOCK_SIZE * BLOCK_SIZE];
                T x_local[BLOCK_SIZE];
                T y_local[BLOCK_SIZE];

                for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                    x_local[d1] = x[row * BLOCK_SIZE + d1];
                    y_local[d1] = 0;
                }

                {
                    int d_idx = 0;
                    for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                        for (int d2 = d1; d2 < BLOCK_SIZE; d2++) {
                            block_diag[d1 * BLOCK_SIZE + d2] = diag_values[d_idx++][diag_idx];
                            block_diag[d2 * BLOCK_SIZE + d1] = block_diag[d1 * BLOCK_SIZE + d2];
                        }
                    }
                }

#pragma unroll(BLOCK_SIZE)
                for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
#pragma unroll(BLOCK_SIZE)
                    for (int d2 = 0; d2 < BLOCK_SIZE; d2++) {
                        y_local[d1] += block_diag[d1 * BLOCK_SIZE + d2] * x_local[d2];
                    }
                }

                for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
#pragma omp atomic
                    y[row * BLOCK_SIZE + d1] += y_local[d1];
                }
            }

            // Apply off-diagonal blocks
#pragma omp parallel for
            for (ptrdiff_t row = 0; row < block_rows; row++) {
                const count_t lenrow = rowptr[row + 1] - rowptr[row];
                const idx_t* cols = &colidx[rowptr[row]];

                T y_local[BLOCK_SIZE];
                for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                    y_local[d1] = 0;
                }

                for (count_t k = 0; k < lenrow; k++) {
                    const auto col = cols[k];
                    const ptrdiff_t off_diag_idx = (rowptr[row] + k) * block_stride;

                    // Construct block
                    T block[BLOCK_SIZE * BLOCK_SIZE];
                    {
                        int d_idx = 0;
                        for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                            block[d1 * BLOCK_SIZE + d1] = off_diag_values[d_idx++][off_diag_idx];
                            for (int d2 = d1 + 1; d2 < BLOCK_SIZE; d2++) {
                                block[d1 * BLOCK_SIZE + d2] =
                                        off_diag_values[d_idx++][off_diag_idx];
                                block[d2 * BLOCK_SIZE + d1] = block[d1 * BLOCK_SIZE + d2];
                            }
                        }
                    }

                    T x_local[BLOCK_SIZE];
                    for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                        x_local[d1] = x[col * BLOCK_SIZE + d1];
                    }

#pragma unroll(BLOCK_SIZE)
                    for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
#pragma unroll(BLOCK_SIZE)
                        for (int d2 = 0; d2 < BLOCK_SIZE; d2++) {
                            y_local[d1] += block[d1 * BLOCK_SIZE + d2] * x_local[d2];
                        }
                    }
                }

                for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
#pragma omp atomic
                    y[row * BLOCK_SIZE + d1] += y_local[d1];
                }

                for (count_t k = 0; k < lenrow; k++) {
                    const auto col = cols[k];
                    const ptrdiff_t off_diag_idx = (rowptr[row] + k) * block_stride;

                    // Construct block
                    T block[BLOCK_SIZE * BLOCK_SIZE];
                    {
                        int d_idx = 0;
                        for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                            block[d1 * BLOCK_SIZE + d1] = off_diag_values[d_idx++][off_diag_idx];
                            for (int d2 = d1 + 1; d2 < BLOCK_SIZE; d2++) {
                                block[d1 * BLOCK_SIZE + d2] =
                                        off_diag_values[d_idx++][off_diag_idx];
                                block[d2 * BLOCK_SIZE + d1] = block[d1 * BLOCK_SIZE + d2];
                            }
                        }
                    }

                    T x_local[BLOCK_SIZE];
                    for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                        x_local[d1] = x[row * BLOCK_SIZE + d1];
                    }
                    for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
                        y_local[d1] = 0;
                    }

#pragma unroll(BLOCK_SIZE)
                    for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
#pragma unroll(BLOCK_SIZE)
                        for (int d2 = 0; d2 < BLOCK_SIZE; d2++) {
                            y_local[d1] += block[d1 * BLOCK_SIZE + d2] * x_local[d2];
                        }
                    }

                    for (int d1 = 0; d1 < BLOCK_SIZE; d1++) {
#pragma omp atomic
                        y[col * BLOCK_SIZE + d1] += y_local[d1];
                    }
                }
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

        switch (block_size) {
            case 2: {
                ret->apply_ = [=](const T* const x, T* const y) {
                    ret->template apply_sym_block<2>(x, scale_output, y);
                };
                break;
            }
            case 3: {
                ret->apply_ = [=](const T* const x, T* const y) {
                    ret->template apply_sym_block<3>(x, scale_output, y);
                };
                break;
            }
            default:
                break;
        }

        return ret;
    }
}  // namespace sfem

#endif  // SFEM_BC_SYM_SPMV_HPP
