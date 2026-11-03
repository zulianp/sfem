#ifndef SFEM_PARALLEL_OPERATOR_HPP
#define SFEM_PARALLEL_OPERATOR_HPP

#include "sfem_Operator.hpp"
#include "sfem_Team.hpp"
#include "sfem_TeamOperator.hpp"
#include "sfem_aliases.hpp"

#include <cassert>
#include <functional>
#include <memory>

namespace sfem {

    template <typename T>
    class ParallelOperator : public Operator<T> {
    public:
        ~ParallelOperator() override = default;

        virtual std::shared_ptr<Communicator> comm() const = 0;

        /// Buffer capacity for range vectors (y), including ghosts/aura.
        virtual std::ptrdiff_t row_allocation_size() const = 0;

        /// Buffer capacity for domain vectors (x), including ghosts/aura.
        virtual std::ptrdiff_t col_allocation_size() const = 0;
    };

    template <typename T>
    class LambdaParallelOperator final : public ParallelOperator<T> {
    public:
        LambdaParallelOperator(const std::shared_ptr<Communicator>&          comm,
                               const std::ptrdiff_t                          rows,
                               const std::ptrdiff_t                          cols,
                               const std::ptrdiff_t                          row_allocation_size,
                               const std::ptrdiff_t                          col_allocation_size,
                               std::function<void(const T* const, T* const)> apply,
                               const ExecutionContext&                       context)
            : comm_(comm),
              rows_(rows),
              cols_(cols),
              row_allocation_size_(row_allocation_size),
              col_allocation_size_(col_allocation_size),
              apply_(std::move(apply)),
              context_(context) {
            assert(comm_);
            assert(rows_ >= 0);
            assert(cols_ >= 0);
            assert(row_allocation_size_ >= rows_);
            assert(col_allocation_size_ >= cols_);
        }

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return rows_; }
        std::ptrdiff_t cols() const override { return cols_; }
        ExecutionContext execution_context() const override { return context_; }

        std::shared_ptr<Communicator> comm() const override { return comm_; }
        std::ptrdiff_t                row_allocation_size() const override { return row_allocation_size_; }
        std::ptrdiff_t                col_allocation_size() const override { return col_allocation_size_; }

    private:
        std::shared_ptr<Communicator>          comm_;
        std::ptrdiff_t                         rows_{0};
        std::ptrdiff_t                         cols_{0};
        std::ptrdiff_t                         row_allocation_size_{0};
        std::ptrdiff_t                         col_allocation_size_{0};
        std::function<void(const T* const, T* const)> apply_;
        ExecutionContext                              context_{EXECUTION_SPACE_INVALID};
    };

    /// A team operator (sfem_TeamOperator.hpp) from a function of the team, the rows its shares
    /// split (the owned rows; under MPI thread 0 takes none).
    template <typename T>
    class TeamParallelOperator final : public ParallelOperator<T>, public TeamOperator<T> {
    public:
        using TeamApply = std::function<void(const Team&, const T* const, T* const)>;
        using Prepare   = std::function<int()>;

        TeamParallelOperator(const std::shared_ptr<Communicator>& comm,
                             const std::ptrdiff_t                 rows,
                             const std::ptrdiff_t                 cols,
                             const std::ptrdiff_t                 row_allocation_size,
                             const std::ptrdiff_t                 col_allocation_size,
                             const bool                           input_in_share,
                             TeamApply                            team_apply,
                             Prepare                              prepare = {})
            : comm_(comm),
              rows_(rows),
              cols_(cols),
              row_allocation_size_(row_allocation_size),
              col_allocation_size_(col_allocation_size),
              input_in_share_(input_in_share),
              team_apply_(std::move(team_apply)),
              prepare_(std::move(prepare)) {
            assert(comm_);
            assert(rows_ >= 0);
            assert(cols_ >= 0);
            assert(row_allocation_size_ >= rows_);
            assert(col_allocation_size_ >= cols_);
        }

        int apply(const T* const x, T* const y) override { return team_region_apply<T>(*this, x, y); }

        Team team() const override { return Team::current(rows_, comm_->size() > 1); }

        void team_apply(const Team& team, const T* const x, T* const y) const override { team_apply_(team, x, y); }

        bool input_in_share() const override { return input_in_share_; }

        int prepare() override { return prepare_ ? prepare_() : SFEM_SUCCESS; }

        std::ptrdiff_t rows() const override { return rows_; }
        std::ptrdiff_t cols() const override { return cols_; }
        ExecutionContext execution_context() const override { return EXECUTION_SPACE_HOST; }

        std::shared_ptr<Communicator> comm() const override { return comm_; }
        std::ptrdiff_t                row_allocation_size() const override { return row_allocation_size_; }
        std::ptrdiff_t                col_allocation_size() const override { return col_allocation_size_; }

    private:
        std::shared_ptr<Communicator> comm_;
        std::ptrdiff_t                rows_{0};
        std::ptrdiff_t                cols_{0};
        std::ptrdiff_t                row_allocation_size_{0};
        std::ptrdiff_t                col_allocation_size_{0};
        bool                          input_in_share_{false};
        TeamApply                     team_apply_;
        Prepare                       prepare_;
    };

    template <typename T>
    inline std::shared_ptr<TeamParallelOperator<T>> make_team_op(const std::shared_ptr<Communicator>&          comm,
                                                                 const std::ptrdiff_t                          rows,
                                                                 const std::ptrdiff_t                          cols,
                                                                 const std::ptrdiff_t                          row_allocation_size,
                                                                 const std::ptrdiff_t                          col_allocation_size,
                                                                 const bool                                    input_in_share,
                                                                 typename TeamParallelOperator<T>::TeamApply   team_apply,
                                                                 typename TeamParallelOperator<T>::Prepare     prepare = {}) {
        return std::make_shared<TeamParallelOperator<T>>(
                comm, rows, cols, row_allocation_size, col_allocation_size, input_in_share, std::move(team_apply), std::move(prepare));
    }

    template <typename T>
    inline std::shared_ptr<ParallelOperator<T>> make_parallel_op(const std::shared_ptr<Communicator>&          comm,
                                                                 const std::ptrdiff_t                          rows,
                                                                 const std::ptrdiff_t                          cols,
                                                                 std::function<void(const T* const, T* const)> op,
                                                                 const ExecutionContext&                       context) {
        return std::make_shared<LambdaParallelOperator<T>>(comm, rows, cols, rows, cols, std::move(op), context);
    }

    template <typename T>
    inline std::shared_ptr<ParallelOperator<T>> make_parallel_op(const std::shared_ptr<Communicator>& comm,
                                                                 const std::ptrdiff_t                 rows,
                                                                 const std::ptrdiff_t                 cols,
                                                                 const std::ptrdiff_t                 row_allocation_size,
                                                                 const std::ptrdiff_t                 col_allocation_size,
                                                                 std::function<void(const T* const, T* const)> op,
                                                                 const ExecutionContext&                       context) {
        return std::make_shared<LambdaParallelOperator<T>>(
                comm, rows, cols, row_allocation_size, col_allocation_size, std::move(op), context);
    }

}  // namespace sfem

#endif  // SFEM_PARALLEL_OPERATOR_HPP
