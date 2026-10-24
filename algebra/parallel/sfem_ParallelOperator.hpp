#ifndef SFEM_PARALLEL_OPERATOR_HPP
#define SFEM_PARALLEL_OPERATOR_HPP

#include "sfem_Operator.hpp"
#include "sfem_Team.hpp"
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
        LambdaParallelOperator(const std::shared_ptr<Communicator>&           comm,
                               const std::ptrdiff_t                           rows,
                               const std::ptrdiff_t                           cols,
                               const std::ptrdiff_t                           row_allocation_size,
                               const std::ptrdiff_t                           col_allocation_size,
                               std::function<void(const T* const, T* const)>  apply,
                               const ExecutionSpace                           es)
            : comm_(comm),
              rows_(rows),
              cols_(cols),
              row_allocation_size_(row_allocation_size),
              col_allocation_size_(col_allocation_size),
              apply_(std::move(apply)),
              execution_space_(es) {
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
        ExecutionContext execution_context() const override { return execution_space_; }

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
        ExecutionSpace                         execution_space_{EXECUTION_SPACE_INVALID};
    };

    /// A host operator whose product runs inside a parallel region its caller opened: each thread
    /// of the region calls team_apply with its team(), may wait for the others (team.barrier())
    /// and, on return, has computed the rows of its share, [team.begin, team.end). apply() is one
    /// region around it. Under MPI thread 0 takes no rows: it keeps the messages moving (a
    /// rendezvous advances only while a thread is in MPI). With input_in_share() team_apply reads
    /// x beyond the calling thread's share only after a barrier of its own, so a caller that has
    /// just written its share of x need not wait for the other threads first.
    template <typename T>
    class TeamParallelOperator final : public ParallelOperator<T> {
    public:
        using TeamApply = std::function<void(const Team&, const T* const, T* const)>;

        TeamParallelOperator(const std::shared_ptr<Communicator>& comm,
                             const std::ptrdiff_t                 rows,
                             const std::ptrdiff_t                 cols,
                             const std::ptrdiff_t                 row_allocation_size,
                             const std::ptrdiff_t                 col_allocation_size,
                             const bool                           input_in_share,
                             TeamApply                            team_apply)
            : comm_(comm),
              rows_(rows),
              cols_(cols),
              row_allocation_size_(row_allocation_size),
              col_allocation_size_(col_allocation_size),
              input_in_share_(input_in_share),
              team_apply_(std::move(team_apply)) {
            assert(comm_);
            assert(rows_ >= 0);
            assert(cols_ >= 0);
            assert(row_allocation_size_ >= rows_);
            assert(col_allocation_size_ >= cols_);
        }

        int apply(const T* const x, T* const y) override {
#pragma omp parallel
            team_apply_(team(), x, y);
            return SFEM_SUCCESS;
        }

        /// Inside a region, the calling thread's share of the rows.
        Team team() const { return Team::current(rows_, comm_->size() > 1); }

        /// From inside a region, every thread with its own team().
        void team_apply(const Team& team, const T* const x, T* const y) const { team_apply_(team, x, y); }

        bool input_in_share() const { return input_in_share_; }

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
    };

    template <typename T>
    inline std::shared_ptr<TeamParallelOperator<T>> make_team_op(const std::shared_ptr<Communicator>&          comm,
                                                                 const std::ptrdiff_t                          rows,
                                                                 const std::ptrdiff_t                          cols,
                                                                 const std::ptrdiff_t                          row_allocation_size,
                                                                 const std::ptrdiff_t                          col_allocation_size,
                                                                 const bool                                    input_in_share,
                                                                 typename TeamParallelOperator<T>::TeamApply   team_apply) {
        return std::make_shared<TeamParallelOperator<T>>(
                comm, rows, cols, row_allocation_size, col_allocation_size, input_in_share, std::move(team_apply));
    }

    template <typename T>
    inline std::shared_ptr<ParallelOperator<T>> make_parallel_op(
            const std::shared_ptr<Communicator>&          comm,
            const std::ptrdiff_t                          rows,
            const std::ptrdiff_t                          cols,
            std::function<void(const T* const, T* const)> op,
            const ExecutionSpace                          es) {
        return std::make_shared<LambdaParallelOperator<T>>(comm, rows, cols, rows, cols, std::move(op), es);
    }

    template <typename T>
    inline std::shared_ptr<ParallelOperator<T>> make_parallel_op(
            const std::shared_ptr<Communicator>&          comm,
            const std::ptrdiff_t                          rows,
            const std::ptrdiff_t                          cols,
            const std::ptrdiff_t                          row_allocation_size,
            const std::ptrdiff_t                          col_allocation_size,
            std::function<void(const T* const, T* const)> op,
            const ExecutionSpace                          es) {
        return std::make_shared<LambdaParallelOperator<T>>(
                comm, rows, cols, row_allocation_size, col_allocation_size, std::move(op), es);
    }

}  // namespace sfem

#endif  // SFEM_PARALLEL_OPERATOR_HPP
