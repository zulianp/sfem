#ifndef SFEM_MPRGP_IMPL_HPP
#define SFEM_MPRGP_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <functional>

namespace sfem {

    template <typename T>
    struct MPRGP_Tpl {
        std::function<
                void(const ptrdiff_t n_dofs, const T* const lb, const T* const ub, T* const x)>
                project;

        std::function<T(const ptrdiff_t n_dofs,
                        const T* const lb,
                        const T* const ub,
                        const T* const x,
                        const T* const g,
                        const T eps)>
                norm_projected_gradient;

        std::function<void(const ptrdiff_t n_dofs,
                           const T* const lb,
                           const T* const ub,
                           const T* const x,
                           const T* const g,
                           T* const norm_free_gradient,
                           T* const norm_chopped_gradient,
                           const T eps)>
                norm_gradients;

        std::function<void(const ptrdiff_t n_dofs,
                           const T* const lb,
                           const T* const ub,
                           const T* const x,
                           const T* const g,
                           T* gc,
                           const T eps)>
                chopped_gradient;

        std::function<void(const ptrdiff_t n_dofs,
                           const T* const lb,
                           const T* const ub,
                           const T* const x,
                           const T* const g,
                           T* gf)>
                free_gradient;

        std::function<T(const ptrdiff_t n_dofs,
                        const T* const lb,
                        const T* const ub,
                        const T* const x,
                        const T* const p,
                        const T infty)>
                max_alpha;

        bool good() const {
            assert(project);
            assert(norm_projected_gradient);
            assert(norm_gradients);
            assert(chopped_gradient);
            assert(free_gradient);
            assert(max_alpha);
            return project && norm_projected_gradient && norm_gradients && chopped_gradient &&
                   free_gradient && max_alpha;
        }
    };

    template <typename T>
    struct OpenMP_MPRGP {
        static inline T gf_lb_ub(const T lbi, const T ubi, const T xi, const T gi) {
#if 1
            return (xi <= lbi || xi >= ubi) ? T(0) : gi;
#else
            return ((std::abs(lbi - xi) < eps) || (std::abs(ubi - xi) < eps)) ? T(0) : gi;
#endif
        }

        static inline T gf_lb(const T lbi, const T xi, const T gi) {
#if 1
            return (xi <= lbi) ? T(0) : gi;
#else
            return (std::abs(lbi - xi) < eps) ? T(0) : gi;
#endif
        }

        static inline T gf_ub(const T ubi, const T xi, const T gi) {
#if 1
            return (xi >= ubi) ? T(0) : gi;
#else
            return (std::abs(ubi - xi) < eps) ? T(0) : gi;
#endif
        }

        static inline T gc_lb_ub(const T lbi, const T ubi, const T xi, const T gi, const T eps) {
            return ((std::abs(lbi - xi) < eps)
                            ? std::min(T(0), gi)
                            : ((std::abs(ubi - xi) < eps) ? std::max(T(0), gi) : T(0)));
        }

        static inline T gc_lb(const T lbi, const T xi, const T gi, const T eps) {
            return ((std::abs(lbi - xi) < eps) ? std::min(T(0), gi) : T(0));
        }

        static inline T gc_ub(const T ubi, const T xi, const T gi, const T eps) {
            return ((std::abs(ubi - xi) < eps) ? std::max(T(0), gi) : T(0));
        }

        static void project(const ptrdiff_t n_dofs,
                            const T* const lb,
                            const T* const ub,
                            T* const x) {
            if (lb && ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    x[i] = std::max(std::min(x[i], ub[i]), lb[i]);
                }
            } else if (ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    x[i] = std::min(x[i], ub[i]);
                }
            } else if (lb) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    x[i] = std::max(x[i], lb[i]);
                }
            }
        }

        static T norm_projected_gradient(const ptrdiff_t n_dofs,
                                         const T* const lb,
                                         const T* const ub,
                                         const T* const x,
                                         const T* const g,
                                         const T eps) {
            T ret = 0;
            if (lb && ub) {
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T d = gf_lb_ub(lb[i], ub[i], x[i], g[i]) +
                                gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);
                    ret += d * d;
                }
            } else if (ub) {
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T d = gf_ub(ub[i], x[i], g[i]) + gc_ub(ub[i], x[i], g[i], eps);
                    ret += d * d;
                }
            } else if (lb) {
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T d = gf_lb(lb[i], x[i], g[i]) + gc_lb(lb[i], x[i], g[i], eps);
                    ret += d * d;
                }
            } else {
                assert(false);
            }

            return sqrt(ret);
        }

        static void norm_gradients(const ptrdiff_t n_dofs,
                                   const T* const lb,
                                   const T* const ub,
                                   const T* const x,
                                   const T* const g,
                                   T* const norm_free_gradient,
                                   T* const norm_chopped_gradient,
                                   const T eps) {
            T acc_gf = 0;
            T acc_gc = 0;

            if (lb && ub) {
#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
                    const T val_gc = gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);

                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else if (ub) {
#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = gf_ub(ub[i], x[i], g[i]);
                    const T val_gc = gc_ub(ub[i], x[i], g[i], eps);

                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else if (lb) {
#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = gf_lb(lb[i], x[i], g[i]);
                    const T val_gc = gc_lb(lb[i], x[i], g[i], eps);

                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else {
                assert(false);
            }

            *norm_free_gradient = sqrt(acc_gf);
            *norm_chopped_gradient = sqrt(acc_gc);
        }

        static void chopped_gradient(const ptrdiff_t n_dofs,
                                     const T* const lb,
                                     const T* const ub,
                                     const T* const x,
                                     const T* const g,
                                     T* gc,
                                     const T eps) {
            if (lb && ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);
                }
            } else if (ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = gc_ub(ub[i], x[i], g[i], eps);
                }
            } else if (lb) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = gc_lb(lb[i], x[i], g[i], eps);
                }
            }
        }

        static void free_gradient(const ptrdiff_t n_dofs,
                                  const T* const lb,
                                  const T* const ub,
                                  const T* const x,
                                  const T* const g,
                                  T* gf) {
            if (lb && ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
                }
            } else if (ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = gf_ub(ub[i], x[i], g[i]);
                }
            } else if (lb) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = gf_lb(lb[i], x[i], g[i]);
                }
            }
        }

        static T max_alpha(const ptrdiff_t n_dofs,
                           const T* const lb,
                           const T* const ub,
                           const T* const x,
                           const T* const p,
                           const T infty) {
            T ret = infty;

            if (lb && ub) {
#pragma omp parallel for reduction(min : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T alpha_lb = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
                    const T alpha_ub = (p[i] < 0) ? ((x[i] - ub[i]) / p[i]) : infty;
                    const T alpha = std::min(alpha_lb, alpha_ub);
                    ret = std::min(alpha, ret);
                }

            } else if (ub) {
#pragma omp parallel for reduction(min : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T alpha = (p[i] < 0) ? ((x[i] - ub[i]) / p[i]) : infty;
                    ret = std::min(alpha, ret);
                }
            } else if (lb) {
#pragma omp parallel for reduction(min : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T alpha = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
                    ret = std::min(alpha, ret);
                }
            }

            return ret;
        }

        static void build_mprgp(struct MPRGP_Tpl<T>& tpl) {
            tpl.project = project;
            tpl.norm_projected_gradient = norm_projected_gradient;
            tpl.norm_gradients = norm_gradients;
            tpl.chopped_gradient = chopped_gradient;
            tpl.free_gradient = free_gradient;
            tpl.max_alpha = max_alpha;
        }
    };

}  // namespace sfem

#endif  // SFEM_MPRGP_IMPL_HPP
