#ifndef SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
#define SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP

#include <cstddef>
#include <memory>
#include <functional>

namespace sfem {

	template<typename T>
	class Operator {
	public:
		virtual ~Operator() = default;
	    virtual int apply(const T* const x, T* const y) = 0;
	};

    template<typename T>
    class LambdaOperator final : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;
        int apply(const T* const x, T* const y) override{
        	apply_(x, y);
        	return 0;
        }
    };

    template<typename T>
    inline std::shared_ptr<Operator<T>> make_op(std::function<void(const T* const, T* const)> op) {
    	auto ret = std::make_shared<LambdaOperator<T>>();
    	ret->apply_ = op;
    	return ret;
    }

    template <typename T>
    class MatrixFreeLinearSolver : public Operator<T> {
    public:
    	virtual ~MatrixFreeLinearSolver() = default;
        virtual void set_op(const std::shared_ptr<Operator<T>> &op) = 0;
        virtual void set_preconditioner(const std::shared_ptr<Operator<T>> &op) = 0;
        virtual void set_max_it(const int it) = 0;
        virtual void set_n_dofs(const ptrdiff_t n) = 0;
    };
}

#endif //SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
