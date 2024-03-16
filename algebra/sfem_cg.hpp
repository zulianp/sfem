#ifndef SFEM_CG_HPP
#define SFEM_CG_HPP

#include <function>

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {
	template<typename T>
	class ConjugateGradient {
	public:
		std::function<void(const T*const, T*const)> apply;
		std::function<T*(const std::size_t)> allocate;
		std::function<void(const T*)> destroy;
		std::function<T (const T*const, const T*const)> dot;
		std::function<void(const T, const T*const, const T, T*const)> axpby;
		T tol{1e-10};

		int apply(const size_t n, const T*const b, T*const x)
		{
			T * r = allocate(n);
			T * p = allocate(n);
			T * ptAp = allocate(n);

			apply(x, r);
			axpby(1, b, -1, r);

			T rtr = dot(r, r);
			
			if(rtr < tol) {
				return 0;
			}

			for(int k = 0; k < max_it; k++) {
				apply(p, ptAp);

				T ptAp = dot(p, Ap);
				T alpha = rtr/ptAp;

				// Opt use 2 cuda streams?
				axpby(alpha, p, 1, x);
				axpby(-alpha, Ap, 1, r);

				T rtr_new = dot(r, r);
				T beta = rtr_new/rtr;
				rtr = rtr_new;

				axpby(1, r, beta, p);
			}

			// clean-up
			destroy(r);
			destroy(p);
			destroy(ptAp);
		}
	};
}

#endif SFEM_CG_HPP