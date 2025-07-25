#ifndef SFEM_API_HPP
#define SFEM_API_HPP

// C includes
#include "adj_table.h"
#include "crs_graph.h"
#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_mask.h"
#include "sfem_mesh.h"
#include "sfem_prolongation_restriction.h"
#include "sshex8.h"
#include "sshex8_interpolate.h"
#include "ssquad4.h"

// C++ includes
#include "sfem_CRSGraph.hpp"
#include "sfem_Chebyshev3.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_Context.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_Function.hpp"
#include "sfem_MixedPrecisionShiftableBlockSymJacobi.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_Restrict.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_bcrs_sym_SpMV.hpp"
#include "sfem_bsr_SpMV.hpp"
#include "sfem_cg.hpp"
#include "sfem_crs_SpMV.hpp"
#include "sfem_crs_sym_SpMV.hpp"
#include "sfem_glob.hpp"
#include "sfem_mprgp.hpp"

// CUDA includes
#ifdef SFEM_ENABLE_CUDA
#include "cu_sshex8_interpolate.h"
#include "cu_tet4_prolongation_restriction.h"
#include "sfem_ContactConditions_cuda.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_ShiftableJacobi.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_crs_SpMV.hpp"
#include "sfem_cuda_mprgp_impl.hpp"
#include "sfem_cuda_solver.hpp"
#else
namespace sfem {
    static void device_synchronize() {}
    static bool is_ptr_device(const void *) { return false; }
    template <typename T>
    inline T &to_host(T &ptr) {
        return ptr;
    }
}  // namespace sfem
#endif

// Externals
#include "matrixio_crs.h"

// System
#include <sys/stat.h>

namespace sfem {

    template <typename T>
    auto blas(const ExecutionSpace es) {
        auto blas = std::make_shared<BLAS_Tpl<T>>();

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE)
            CUDA_BLAS<T>::build_blas(*blas);
        else
#endif  // SFEM_ENABLE_CUDA
        {
            OpenMP_BLAS<T>::build_blas(*blas);
        }
        return blas;
    }

    template <typename T>
    static std::shared_ptr<Operator<T>> diag_op(const SharedBuffer<T> &diagonal_scaling, const ExecutionSpace es) {
        const std::ptrdiff_t n = diagonal_scaling->size();

        // // FIXME make simpler version
        auto impl = sfem::blas<T>(es)->xypaz;
        return sfem::make_op<T>(
                n,
                n,
                [n, diagonal_scaling, impl](const T *const x, T *const y) {
                    auto d = diagonal_scaling->data();
                    impl(n, x, d, 0, y);
                },
                es);
    }

    template <typename T>
    static SharedBuffer<T*> create_buffer(const std::ptrdiff_t n0, const std::ptrdiff_t n1, const MemorySpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == MEMORY_SPACE_DEVICE) return sfem::create_device_buffer<T>(n0, n1);
#endif  // SFEM_ENABLE_CUDA
        return sfem::create_host_buffer<T>(n0, n1);
    }

    template <typename T>
    static SharedBuffer<T*> create_buffer(const std::ptrdiff_t n0, const std::ptrdiff_t n1, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::create_device_buffer<T>(n0, n1);
#endif  // SFEM_ENABLE_CUDA
        return sfem::create_host_buffer<T>(n0, n1);
    }

    template <typename T>
    static SharedBuffer<T> create_buffer(const std::ptrdiff_t n, const MemorySpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == MEMORY_SPACE_DEVICE) return sfem::create_device_buffer<T>(n);
#endif  // SFEM_ENABLE_CUDA
        return sfem::create_host_buffer<T>(n);
    }

    template <typename T>
    static SharedBuffer<T> create_buffer(const std::ptrdiff_t n, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::create_device_buffer<T>(n);
#endif  // SFEM_ENABLE_CUDA
        return sfem::create_host_buffer<T>(n);
    }

    static std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space,
                                         const std::string                    &name,
                                         const ExecutionSpace                  es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::Factory::create_op_gpu(space, name.c_str());
#endif  // SFEM_ENABLE_CUDA
        return sfem::Factory::create_op(space, name.c_str());
    }

    template <typename T>
    static std::shared_ptr<ConjugateGradient<T>> create_cg(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        std::shared_ptr<ConjugateGradient<T>> cg;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            cg = sfem::d_cg<T>();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            cg = sfem::h_cg<T>();
        }

        cg->set_n_dofs(op->rows());
        cg->set_op(op);
        return cg;
    }

    template <typename T>
    static std::shared_ptr<ShiftableJacobi<T>> create_shiftable_jacobi(const SharedBuffer<T> &diag, const ExecutionSpace es) {
        auto ret = std::make_shared<sfem::ShiftableJacobi<T>>();

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            CUDA_BLAS<T>::build_blas(ret->blas);
            ret->execution_space_ = es;
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->default_init();
        }

        ret->set_diag(diag);
        return ret;
    }

    template <typename HP, typename LP>
    std::shared_ptr<MixedPrecisionShiftableBlockSymJacobi<HP, LP>> create_mixed_precision_shiftable_block_sym_jacobi(
            const int                   dim,
            const SharedBuffer<HP>     &diag,
            const SharedBuffer<mask_t> &constraints_mask,
            const ExecutionSpace        es) {
        auto ret = std::make_shared<sfem::MixedPrecisionShiftableBlockSymJacobi<HP, LP>>();

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            CUDA_BLAS<LP>::build_blas(ret->blas);
            ShiftableBlockSymJacobi_CUDA<HP, LP>::build(dim, ret->impl);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            OpenMP_BLAS<LP>::build_blas(ret->blas);
            ShiftableBlockSymJacobi_OpenMP<HP, LP>::build(dim, ret->impl);
        }

        ret->execution_space_ = es;
        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    std::shared_ptr<ShiftableBlockSymJacobi<T>> create_shiftable_block_sym_jacobi(const int                   dim,
                                                                                  const SharedBuffer<T>      &diag,
                                                                                  const SharedBuffer<mask_t> &constraints_mask,
                                                                                  const ExecutionSpace        es) {
        auto ret = std::make_shared<sfem::ShiftableBlockSymJacobi<T>>();

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            CUDA_BLAS<T>::build_blas(ret->blas);
            ShiftableBlockSymJacobi_CUDA<T>::build(dim, ret->impl);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            OpenMP_BLAS<T>::build_blas(ret->blas);
            ShiftableBlockSymJacobi_OpenMP<T>::build(dim, ret->impl);
        }

        ret->execution_space_ = es;
        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    static std::shared_ptr<StationaryIteration<T>> create_stationary(const std::shared_ptr<Operator<T>> &op,
                                                                     const std::shared_ptr<Operator<T>> &preconditioner,
                                                                     const ExecutionSpace                es) {
        auto ret            = std::make_shared<StationaryIteration<T>>();
        ret->op             = op;
        ret->preconditioner = preconditioner;
        ret->n_dofs         = op->cols();

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            CUDA_BLAS<T>::build_blas(ret->blas);
            ret->execution_space_ = es;
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->default_init();
        }

        return ret;
    }

    template <typename T>
    static std::shared_ptr<BiCGStab<T>> create_bcgs(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        std::shared_ptr<BiCGStab<T>> bcgs;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            bcgs = sfem::d_bcgs<T>();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            bcgs = sfem::h_bcgs<T>();
        }

        bcgs->set_n_dofs(op->rows());
        bcgs->set_op(op);
        return bcgs;
    }

    template <typename T>
    static std::shared_ptr<Chebyshev3<T>> create_cheb3(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        std::shared_ptr<Chebyshev3<T>> cheb;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            cheb = sfem::d_cheb3<T>(op);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            cheb = sfem::h_cheb3<T>(op);
        }

        return cheb;
    }

    template <typename T>
    static std::shared_ptr<MPRGP<T>> create_mprgp(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        auto mprgp = std::make_shared<sfem::MPRGP<real_t>>();
        mprgp->set_op(op);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            CUDA_BLAS<T>::build_blas(mprgp->blas);
            CUDA_MPRGP<T>::build_mprgp(mprgp->impl);
            mprgp->execution_space_ = es;

        } else
#endif  // SFEM_ENABLE_CUDA
        {
            mprgp->default_init();
        }

        return mprgp;
    }

    template <typename T>
    static std::shared_ptr<Multigrid<T>> create_mg(const ExecutionSpace es) {
        std::shared_ptr<Multigrid<T>> mg;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            mg = sfem::d_mg<T>();

        } else
#endif  // SFEM_ENABLE_CUDA
        {
            mg = sfem::h_mg<T>();
        }

        return mg;
    }

    static std::shared_ptr<Constraint> create_dirichlet_conditions_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                            const ExecutionSpace                  es) {
        auto conds = sfem::DirichletConditions::create_from_env(space);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::to_device(conds);
        }
#endif  // SFEM_ENABLE_CUDA

        return conds;
    }

    static std::shared_ptr<Constraint> create_dirichlet_conditions(const std::shared_ptr<FunctionSpace>              &space,
                                                                   const std::vector<DirichletConditions::Condition> &conditions,
                                                                   const ExecutionSpace                               es) {
        auto conds = sfem::DirichletConditions::create(space, conditions);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::to_device(conds);
        }
#endif  // SFEM_ENABLE_CUDA

        return conds;
    }

    static std::shared_ptr<Op> create_neumann_conditions(const std::shared_ptr<FunctionSpace>            &space,
                                                         const std::vector<NeumannConditions::Condition> &conditions,
                                                         const ExecutionSpace                             es) {
        auto conds = sfem::NeumannConditions::create(space, conditions);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::to_device(conds);
        }
#endif  // SFEM_ENABLE_CUDA

        return conds;
    }

    static std::shared_ptr<Constraint> create_contact_conditions_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                          const ExecutionSpace                  es) {
        auto conds = sfem::AxisAlignedContactConditions::create_from_env(space);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::to_device(conds);
        }
#endif  // SFEM_ENABLE_CUDA

        return conds;
    }

    static SharedBuffer<idx_t> create_edge_idx(CRSGraph &crs_graph) {
        const ptrdiff_t rows        = crs_graph.n_nodes();
        auto            p2_vertices = create_host_buffer<idx_t>(crs_graph.nnz());

        build_p1_to_p2_edge_map(rows, crs_graph.rowptr()->data(), crs_graph.colidx()->data(), p2_vertices->data());

        return p2_vertices;
    }

    static std::shared_ptr<CRSGraph> create_derefined_crs_graph(FunctionSpace &space) {
        auto et        = (enum ElemType)space.element_type();
        auto coarse_et = macro_base_elem(et);
        auto crs_graph = space.mesh().create_node_to_node_graph(coarse_et);
        return crs_graph;
    }

    static std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation(const std::shared_ptr<FunctionSpace> &from_space,
                                                                              const std::shared_ptr<FunctionSpace> &to_space,
                                                                              const ExecutionSpace                  es) {
#ifdef SFEM_ENABLE_CUDA
        if (EXECUTION_SPACE_DEVICE == es) {
            auto elements = to_space->device_elements();
            if (!elements) {
                elements = create_device_elements(to_space, to_space->element_type());
                from_space->set_device_elements(elements);
            }

            if (to_space->has_semi_structured_mesh()) {
                if (from_space->has_semi_structured_mesh()) {
                    auto from_elements = from_space->device_elements();
                    if (!from_elements) {
                        from_elements = create_device_elements(from_space, from_space->element_type());
                        from_space->set_device_elements(from_elements);
                    }

                    return make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t *const from, real_t *const to) {
                                SFEM_TRACE_SCOPE("cu_sshex8_prolongate");

                                auto &from_ssm = from_space->semi_structured_mesh();
                                auto &to_ssm   = to_space->semi_structured_mesh();

                                cu_sshex8_prolongate(from_ssm.n_elements(),
                                                     from_ssm.level(),
                                                     1,
                                                     from_elements->data(),
                                                     to_ssm.level(),
                                                     1,
                                                     elements->data(),
                                                     from_space->block_size(),
                                                     SFEM_REAL_DEFAULT,
                                                     1,
                                                     from,
                                                     SFEM_REAL_DEFAULT,
                                                     1,
                                                     to,
                                                     SFEM_DEFAULT_STREAM);
                            },
                            es);

                } else {
                    return make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t *const from, real_t *const to) {
                                SFEM_TRACE_SCOPE("cu_sshex8_hierarchical_prolongation");

                                auto &ssm = to_space->semi_structured_mesh();
                                cu_sshex8_hierarchical_prolongation(ssm.level(),
                                                                    ssm.n_elements(),
                                                                    elements->data(),
                                                                    from_space->block_size(),
                                                                    SFEM_REAL_DEFAULT,
                                                                    1,
                                                                    from,
                                                                    SFEM_REAL_DEFAULT,
                                                                    1,
                                                                    to,
                                                                    SFEM_DEFAULT_STREAM);
                            },
                            es);
                }
            } else {
                return make_op<real_t>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            SFEM_TRACE_SCOPE("cu_macrotet4_to_tet4_prolongation_element_based");

                            cu_macrotet4_to_tet4_prolongation_element_based(from_space->mesh().n_elements(),
                                                                            elements->data(),
                                                                            from_space->block_size(),
                                                                            SFEM_REAL_DEFAULT,
                                                                            1,
                                                                            from,
                                                                            SFEM_REAL_DEFAULT,
                                                                            1,
                                                                            to,
                                                                            SFEM_DEFAULT_STREAM);
                        },
                        es);
            }

        } else
#endif
        {
            if (to_space->has_semi_structured_mesh()) {
                if (!from_space->has_semi_structured_mesh()) {
                    return make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t *const from, real_t *const to) {
                                SFEM_TRACE_SCOPE("sshex8_hierarchical_prolongation");

                                auto &ssm = to_space->semi_structured_mesh();
                                sshex8_hierarchical_prolongation(
                                        ssm.level(), ssm.n_elements(), ssm.element_data(), from_space->block_size(), from, to);
                            },
                            EXECUTION_SPACE_HOST);
                } else {
                    assert(from_space->semi_structured_mesh().level() > 1);

                    return make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t *const from, real_t *const to) {
                                SFEM_TRACE_SCOPE("sshex8_prolongate");

                                auto &from_ssm = from_space->semi_structured_mesh();
                                auto &to_ssm   = to_space->semi_structured_mesh();

                                sshex8_prolongate(from_ssm.n_elements(),     // nelements,
                                                  from_ssm.level(),          // from_level
                                                  1,                         // from_level_stride
                                                  from_ssm.element_data(),   // from_elements
                                                  to_ssm.level(),            // to_level
                                                  1,                         // to_level_stride
                                                  to_ssm.element_data(),     // to_elements
                                                  from_space->block_size(),  // vec_size
                                                  from,
                                                  to);
                            },
                            EXECUTION_SPACE_HOST);
                }
            } else {
                return make_op<real_t>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            SFEM_TRACE_SCOPE("hierarchical_prolongation");

                            hierarchical_prolongation(from_space->element_type(),
                                                      to_space->element_type(),
                                                      to_space->mesh().n_elements(),
                                                      to_space->mesh().elements()->data(),
                                                      from_space->block_size(),
                                                      from,
                                                      to);
                        },
                        EXECUTION_SPACE_HOST);
            }
        }
    }

    static std::shared_ptr<Operator<real_t>> create_hierarchical_restriction(const std::shared_ptr<FunctionSpace> &from_space,
                                                                             const std::shared_ptr<FunctionSpace> &to_space,
                                                                             const ExecutionSpace                  es) {
        return sfem::Restrict<real_t>::create(from_space, to_space, es, from_space->block_size());
    }

    static std::shared_ptr<Operator<real_t>> create_hierarchical_restriction_from_graph(
            const ptrdiff_t                  n_fine_nodes,
            const int                        block_size,
            const std::shared_ptr<CRSGraph> &crs_graph,
            const SharedBuffer<idx_t>       &edges,
            const ExecutionSpace             es) {
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_coarse_nodes * block_size;
        ptrdiff_t cols = n_fine_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_edges     = to_device(edges);
            auto d_crs_graph = to_device(crs_graph);

            return make_op<real_t>(
                    rows,
                    cols,
                    [=](const real_t *const from, real_t *const to) {
                        // FIXME make it generic for all elements!
                        cu_macrotet4_to_tet4_restriction(n_coarse_nodes,
                                                         d_crs_graph->rowptr()->data(),
                                                         d_crs_graph->colidx()->data(),
                                                         d_edges->data(),
                                                         block_size,
                                                         SFEM_REAL_DEFAULT,
                                                         from,
                                                         SFEM_REAL_DEFAULT,
                                                         to,
                                                         SFEM_DEFAULT_STREAM);
                    },
                    EXECUTION_SPACE_DEVICE);
        }
#endif  // SFEM_ENABLE_CUDA

        return make_op<real_t>(
                rows,
                cols,
                [=](const real_t *const from, real_t *const to) {
                    ::hierarchical_restriction_with_edge_map(n_coarse_nodes,
                                                             crs_graph->rowptr()->data(),
                                                             crs_graph->colidx()->data(),
                                                             edges->data(),
                                                             block_size,
                                                             from,
                                                             to);
                },
                EXECUTION_SPACE_HOST);
    }

    static std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation_from_graph(
            const std::shared_ptr<Function> &function,
            const std::shared_ptr<CRSGraph> &crs_graph,
            const SharedBuffer<idx_t>       &edges,
            const ExecutionSpace             es) {
        const ptrdiff_t n_fine_nodes   = function->space()->mesh().n_nodes();
        int             block_size     = function->space()->block_size();
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_fine_nodes * block_size;
        ptrdiff_t cols = n_coarse_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_edges     = to_device(edges);
            auto d_crs_graph = to_device(crs_graph);

            return make_op<real_t>(
                    rows,
                    cols,
                    [=](const real_t *const from, real_t *const to) {
                        // FIXME make it generic for all elements!
                        cu_tet4_to_macrotet4_prolongation(n_coarse_nodes,
                                                          d_crs_graph->rowptr()->data(),
                                                          d_crs_graph->colidx()->data(),
                                                          d_edges->data(),
                                                          block_size,
                                                          SFEM_REAL_DEFAULT,
                                                          from,
                                                          SFEM_REAL_DEFAULT,
                                                          to,
                                                          SFEM_DEFAULT_STREAM);

                        function->apply_zero_constraints(to);
                    },
                    EXECUTION_SPACE_DEVICE);
        }

#endif  // SFEM_ENABLE_CUDA

        return make_op<real_t>(
                rows,
                cols,
                [=](const real_t *const from, real_t *const to) {
                    ::hierarchical_prolongation_with_edge_map(n_coarse_nodes,
                                                              crs_graph->rowptr()->data(),
                                                              crs_graph->colidx()->data(),
                                                              edges->data(),
                                                              block_size,
                                                              from,
                                                              to);

                    function->apply_zero_constraints(to);
                },
                EXECUTION_SPACE_HOST);
    }

    template <typename T>
    static std::shared_ptr<Operator<T>> create_inverse_diagonal_scaling(const SharedBuffer<T> &diag, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_diag = to_device(diag);
            return sfem::make_op<T>(
                    d_diag->size(),
                    d_diag->size(),
                    [=](const T *const x, T *const y) {
                        auto d = d_diag->data();
                        // FIXME (only supports real_t)
                        d_ediv(d_diag->size(), x, d, y);
                    },
                    EXECUTION_SPACE_DEVICE);
        }
#endif  // SFEM_ENABLE_CUDA

        return sfem::make_op<T>(
                diag->size(),
                diag->size(),
                [=](const T *const x, T *const y) {
                    auto d = diag->data();

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < diag->size(); ++i) {
                        y[i] = x[i] / d[i];
                    }
                },
                EXECUTION_SPACE_HOST);
    }

    static std::shared_ptr<Operator<real_t>> make_linear_op(const std::shared_ptr<Function> &f) {
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { f->apply(nullptr, x, y); },
                f->execution_space());
    }

    static std::shared_ptr<Operator<real_t>> make_linear_op_variant(const std::shared_ptr<Function>                &f,
                                                                    const std::vector<std::pair<std::string, int>> &opts) {
        auto variant = f->linear_op_variant(opts);
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { variant->apply(x, y); },
                f->execution_space());
    }

    static auto hessian_crs(sfem::Function &f, const std::shared_ptr<CRSGraph> &crs_graph, const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);
            auto values      = sfem::create_buffer<real_t>(d_crs_graph->nnz(), es);

            f.hessian_crs(nullptr, d_crs_graph->rowptr()->data(), d_crs_graph->colidx()->data(), values->data());

            return sfem::d_crs_spmv(d_crs_graph->n_nodes(),
                                    d_crs_graph->n_nodes(),
                                    d_crs_graph->rowptr(),
                                    d_crs_graph->colidx(),
                                    values,
                                    (real_t)1);
        }
#endif
        auto values = sfem::create_host_buffer<real_t>(crs_graph->nnz());

        f.hessian_crs(nullptr, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

        // Owns the pointers
        return sfem::h_crs_spmv(
                crs_graph->n_nodes(), crs_graph->n_nodes(), crs_graph->rowptr(), crs_graph->colidx(), values, (real_t)1);
    }

    static auto hessian_crs(const std::shared_ptr<sfem::Function> &f,
                            const SharedBuffer<real_t>            &x,
                            const sfem::ExecutionSpace             es) {
        auto crs_graph = f->crs_graph();

#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);
            auto values      = sfem::create_buffer<real_t>(d_crs_graph->nnz(), es);

            f->hessian_crs(x->data(), d_crs_graph->rowptr()->data(), d_crs_graph->colidx()->data(), values->data());

            return sfem::d_crs_spmv(d_crs_graph->n_nodes(),
                                    d_crs_graph->n_nodes(),
                                    d_crs_graph->rowptr(),
                                    d_crs_graph->colidx(),
                                    values,
                                    (real_t)1);
        }
#endif
        auto values = sfem::create_host_buffer<real_t>(crs_graph->nnz());

        const real_t *const x_data = (x) ? x->data() : nullptr;
        f->hessian_crs(x_data, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

        // Owns the pointers
        return sfem::h_crs_spmv(
                crs_graph->n_nodes(), crs_graph->n_nodes(), crs_graph->rowptr(), crs_graph->colidx(), values, (real_t)1);
    }

    static auto hessian_bsr(const std::shared_ptr<sfem::Function> &f,
                            const SharedBuffer<real_t>            &x,
                            const sfem::ExecutionSpace             es) {
        // Get the mesh node-to-node graph instead of the FunctionSpace scalar adapted graph
        auto      crs_graph  = f->space()->node_to_node_graph();
        const int block_size = f->space()->block_size();

#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);
            auto values      = sfem::create_buffer<real_t>(d_crs_graph->nnz() * block_size * block_size, es);

            f->hessian_bsr(x->data(), d_crs_graph->rowptr()->data(), d_crs_graph->colidx()->data(), values->data());

            auto spmv = sfem::d_bsr_spmv(d_crs_graph->n_nodes(),
                                         d_crs_graph->n_nodes(),
                                         block_size,
                                         d_crs_graph->rowptr(),
                                         d_crs_graph->colidx(),
                                         values,
                                         (real_t)1);

            // Owns the pointers
            return sfem::make_op<real_t>(
                    f->space()->n_dofs(),
                    f->space()->n_dofs(),
                    [=](const real_t *const x, real_t *const y) {
                        spmv->apply(x, y);
                        f->copy_constrained_dofs(x, y);
                    },
                    es);
        }
#endif
        auto values = sfem::create_host_buffer<real_t>(crs_graph->nnz() * block_size * block_size);

        real_t *x_data = (x) ? x->data() : nullptr;

        f->hessian_bsr(x_data, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

        // Owns the pointers
        auto spmv = sfem::h_bsr_spmv(crs_graph->n_nodes(),
                                     crs_graph->n_nodes(),
                                     block_size,
                                     crs_graph->rowptr(),
                                     crs_graph->colidx(),
                                     values,
                                     (real_t)1);
        // return spmv;
        // Owns the pointers
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) {
                    spmv->apply(x, y);
                    f->copy_constrained_dofs(x, y);
                },
                es);
    }

    static auto hessian_bcrs_sym(const std::shared_ptr<sfem::Function> &f,
                                 const SharedBuffer<real_t>            &x,
                                 const sfem::ExecutionSpace             es) {
        assert(es == sfem::EXECUTION_SPACE_HOST);

        auto      crs_graph  = f->space()->mesh().node_to_node_graph_upper_triangular();
        const int block_size = f->space()->block_size();

        int       nblock_entries = ((block_size + 1) * block_size) / 2;
        ptrdiff_t block_stride   = 1;

        bool SFEM_BCRS_SYM_USE_AOS = false;
        SFEM_READ_ENV(SFEM_BCRS_SYM_USE_AOS, atoi);

        SharedBuffer<real_t *> diag_values;
        SharedBuffer<real_t *> off_diag_values;

        if (SFEM_BCRS_SYM_USE_AOS) {
            block_stride    = nblock_entries;
            off_diag_values = sfem::create_host_buffer_fake_SoA<real_t>(nblock_entries, crs_graph->nnz());
            diag_values     = sfem::create_host_buffer_fake_SoA<real_t>(nblock_entries, f->space()->n_dofs() / block_size);
        } else {
            off_diag_values = sfem::create_host_buffer<real_t>(nblock_entries, crs_graph->nnz());
            diag_values     = sfem::create_host_buffer<real_t>(nblock_entries, f->space()->n_dofs() / block_size);
        }

        real_t *x_data = (x) ? x->data() : nullptr;
        f->hessian_bcrs_sym(x_data,
                            crs_graph->rowptr()->data(),
                            crs_graph->colidx()->data(),
                            block_stride,
                            diag_values->data(),
                            off_diag_values->data());

        auto spmv = sfem::h_bcrs_sym<count_t, idx_t, real_t>(crs_graph->n_nodes(),
                                                             crs_graph->n_nodes(),
                                                             block_size,
                                                             crs_graph->rowptr(),
                                                             crs_graph->colidx(),
                                                             block_stride,
                                                             diag_values,
                                                             off_diag_values,
                                                             (real_t)1);
        // Owns the pointers
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) {
                    spmv->apply(x, y);
                    f->copy_constrained_dofs(x, y);
                },
                es);
    }

    static auto hessian_coo_sym(const std::shared_ptr<sfem::Function> &f,
                                const SharedBuffer<real_t>            &x,
                                const sfem::ExecutionSpace             es) {
        auto fs        = f->space();
        auto crs_graph = fs->mesh_ptr()->node_to_node_graph_upper_triangular();

        auto diag_values     = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        auto off_diag_values = sfem::create_buffer<real_t>(crs_graph->nnz(), es);

        real_t *x_data = nullptr;
        if (x) {
            x_data = x->data();
        }

        std::shared_ptr<sfem::Operator<real_t>> spmv;
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);

            f->hessian_crs_sym(x_data,
                               d_crs_graph->rowptr()->data(),
                               d_crs_graph->colidx()->data(),
                               diag_values->data(),
                               off_diag_values->data());

            auto h_row_idx = sfem::create_buffer<idx_t>(crs_graph->nnz(), sfem::EXECUTION_SPACE_HOST);
            crs_to_coo(fs->n_dofs(), crs_graph->rowptr()->data(), h_row_idx->data());
            auto row_idx = sfem::to_device(h_row_idx);

            spmv = sfem::d_sym_coo_spmv(fs->n_dofs(), row_idx, crs_graph->colidx(), off_diag_values, diag_values, 1);

        } else
#endif
        {
            f->hessian_crs_sym(x_data,
                               crs_graph->rowptr()->data(),
                               crs_graph->colidx()->data(),
                               diag_values->data(),
                               off_diag_values->data());

            auto row_idx = sfem::create_buffer<idx_t>(crs_graph->nnz(), es);
            crs_to_coo(fs->n_dofs(), crs_graph->rowptr()->data(), row_idx->data());
            // auto mask = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
            // f->constaints_mask(mask->data());

            spmv = sfem::h_coosym<idx_t, real_t>(nullptr, row_idx, crs_graph->colidx(), off_diag_values, diag_values);
        }

        // Owns the pointers
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) {
                    spmv->apply(x, y);
                    f->copy_constrained_dofs(x, y);
                },
                es);
    }

    static auto hessian_crs_sym(const std::shared_ptr<sfem::Function> &f,
                                const std::shared_ptr<Buffer<real_t>> &x,
                                const sfem::ExecutionSpace             es) {
        auto fs              = f->space();
        auto crs_graph       = fs->mesh_ptr()->node_to_node_graph_upper_triangular();
        auto diag_values     = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        auto off_diag_values = sfem::create_buffer<real_t>(crs_graph->nnz(), es);

        real_t *x_data = nullptr;
        if (x) {
            x_data = x->data();
        }

        std::shared_ptr<sfem::Operator<real_t>> spmv;
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            // TODO
            // spmv = sfem::d_crs_sym_spmv<count_t, idx_t, real_t>(fs->n_dofs(),
            //                                                          fs->n_dofs(),
            //                                                          crs_graph->rowptr(),
            //                                                          crs_graph->colidx(),
            //                                                          diag_values,
            //                                                          off_diag_values,
            //                                                          (real_t)1);
            assert(false);
        } else
#endif
        {
            f->hessian_crs_sym(x_data,
                               crs_graph->rowptr()->data(),
                               crs_graph->colidx()->data(),
                               diag_values->data(),
                               off_diag_values->data());

            spmv = sfem::h_crs_sym<count_t, idx_t, real_t>(fs->n_dofs(),
                                                           fs->n_dofs(),
                                                           crs_graph->rowptr(),
                                                           crs_graph->colidx(),
                                                           diag_values,
                                                           off_diag_values,
                                                           (real_t)1);
        }

        // Owns the pointers
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) {
                    spmv->apply(x, y);
                    f->copy_constrained_dofs(x, y);
                },
                es);
    }

    static real_t residual(sfem::Operator<real_t> &op, const real_t *const rhs, const real_t *const x, real_t *const r) {
#ifdef SFEM_ENABLE_CUDA
        if (op.execution_space() == sfem::EXECUTION_SPACE_DEVICE) {
            d_memset(r, 0, op.rows() * sizeof(real_t));
            op.apply(x, r);
            d_axpby(op.rows(), 1, rhs, -1, r);
            return d_nrm2(op.rows(), r);
        } else
#endif
        {
            memset(r, 0, op.rows() * sizeof(real_t));
            op.apply(x, r);

            real_t ret = 0;
#pragma omp parallel for reduction(+ : ret)
            for (ptrdiff_t i = 0; i < op.rows(); i++) {
                r[i] = rhs[i] - r[i];
                ret += r[i] * r[i];
            }
            return sqrt(ret);
        }
    }

    static int write_crs(const std::string &path, CRSGraph &graph, sfem::Buffer<real_t> &values) {
        sfem::create_directory(path.c_str());
        crs_t crs_out;
        crs_out.rowptr      = (char *)graph.rowptr()->data();
        crs_out.colidx      = (char *)graph.colidx()->data();
        crs_out.values      = (char *)values.data();
        crs_out.grows       = graph.rowptr()->size() - 1;
        crs_out.lrows       = graph.rowptr()->size() - 1;
        crs_out.lnnz        = values.size();
        crs_out.gnnz        = values.size();
        crs_out.start       = 0;
        crs_out.rowoffset   = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = SFEM_MPI_REAL_T;
        return crs_write_folder(MPI_COMM_SELF, path.c_str(), &crs_out);
    }

    static std::shared_ptr<sfem::Operator<real_t>> create_linear_operator(const std::string                           &format,
                                                                          const std::shared_ptr<sfem::Function>       &f,
                                                                          const std::shared_ptr<sfem::Buffer<real_t>> &x,
                                                                          enum sfem::ExecutionSpace                    es) {
        if (format == MATRIX_FREE) {
            return sfem::make_linear_op(f);
        }

        if (f->space()->block_size() == 1) {
            if (format == CRS_SYM)
                return sfem::hessian_crs_sym(f, nullptr, es);
            else if (format == COO_SYM)
                return sfem::hessian_coo_sym(f, nullptr, es);

            if (format != CRS) {
                fprintf(stderr, "[Warning] fallback to CRS format as \"%s\" is not supported!\n", format.c_str());
            }

            return sfem::hessian_crs(f, nullptr, es);
        }

        if (format == BSR) return sfem::hessian_bsr(f, nullptr, es);
        if (format != BSR_SYM) {
            fprintf(stderr, "[Warning] fallback to BCRS_SYM format as \"%s\" is not supported!\n", format.c_str());
        }

        return sfem::hessian_bcrs_sym(f, nullptr, es);
    }

    static SharedBuffer<idx_t *> sshex8_derefine_element_connectivity(const int                    from_level,
                                                                      const int                    to_level,
                                                                      const SharedBuffer<idx_t *> &elements) {
        const int       step_factor = from_level / to_level;
        const int       nxe         = (to_level + 1) * (to_level + 1) * (to_level + 1);
        const ptrdiff_t nelements   = elements->extent(1);

#ifdef SFEM_ENABLE_CUDA
        if (elements->mem_space() == MEMORY_SPACE_DEVICE) {
            std::vector<idx_t *> host_buff_from(elements->extent(0));
            buffer_device_to_host(elements->extent(0) * sizeof(idx_t *), elements->data(), host_buff_from.data());

            std::vector<idx_t *> host_dev_ptrs(nxe);
            for (int zi = 0; zi <= to_level; zi++) {
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        const int from_lidx = sshex8_lidx(from_level, xi * step_factor, yi * step_factor, zi * step_factor);
                        const int to_lidx   = sshex8_lidx(to_level, xi, yi, zi);

                        assert(from_lidx < elements->extent(0));
                        assert(to_lidx < host_dev_ptrs.size());

                        host_dev_ptrs[to_lidx] = host_buff_from[from_lidx];
                    }
                }
            }

            idx_t **dev_buff_to = (idx_t **)d_buffer_alloc(nxe * sizeof(idx_t *));
            buffer_host_to_device(nxe * sizeof(idx_t *), host_dev_ptrs.data(), dev_buff_to);
            return std::make_shared<Buffer<idx_t *>>(
                    nxe,
                    nelements,
                    dev_buff_to,
                    [nxe, host_dev_ptrs](int n, void **ptr) { d_buffer_destroy(ptr); },
                    MEMORY_SPACE_DEVICE);
        }
#endif

        auto view = std::make_shared<Buffer<idx_t *>>(
                nxe,
                nelements,
                (idx_t **)malloc(nxe * sizeof(idx_t *)),
                [keep_alive = elements](int, void **v) {
                    (void)keep_alive;
                    free(v);
                },
                elements->mem_space());

        auto d     = view->data();
        auto elems = elements->data();

        for (int zi = 0; zi <= to_level; zi++) {
            for (int yi = 0; yi <= to_level; yi++) {
                for (int xi = 0; xi <= to_level; xi++) {
                    const int from_lidx = sshex8_lidx(from_level, xi * step_factor, yi * step_factor, zi * step_factor);
                    const int to_lidx   = sshex8_lidx(to_level, xi, yi, zi);
                    d[to_lidx]          = elems[from_lidx];
                }
            }
        }

        return view;
    }

    static SharedBuffer<idx_t *> ssquad4_derefine_element_connectivity(const int                    from_level,
                                                                       const int                    to_level,
                                                                       const SharedBuffer<idx_t *> &elements) {
        const int       step_factor = from_level / to_level;
        const int       nxe         = (to_level + 1) * (to_level + 1);
        const ptrdiff_t nelements   = elements->extent(1);

        auto view = std::make_shared<Buffer<idx_t *>>(
                nxe,
                nelements,
                (idx_t **)malloc(nxe * sizeof(idx_t *)),
                [keep_alive = elements](int, void **v) {
                    (void)keep_alive;
                    free(v);
                },
                elements->mem_space());

        for (int yi = 0; yi <= to_level; yi++) {
            for (int xi = 0; xi <= to_level; xi++) {
                const int from_lidx   = ssquad4_lidx(from_level, xi * step_factor, yi * step_factor);
                const int to_lidx     = ssquad4_lidx(to_level, xi, yi);
                view->data()[to_lidx] = elements->data()[from_lidx];
            }
        }

        return view;
    }

    static ptrdiff_t ss_elements_max_node_id(const SharedBuffer<idx_t *> &elements) {
        ptrdiff_t max_node_id{-1};
        {
            auto            vv        = elements->data();
            const ptrdiff_t nelements = elements->extent(1);
            for (int v = 0; v < elements->extent(0); v++) {
                for (ptrdiff_t e = 0; e < nelements; e++) {
                    max_node_id = (vv[v][e] > max_node_id ? vv[v][e] : max_node_id);
                }
            }
        }

        return max_node_id;
    }

    static std::shared_ptr<sfem::Sideset> create_skin_sideset(const std::shared_ptr<sfem::Mesh> &mesh) {
        ptrdiff_t      n_surf_elements = 0;
        element_idx_t *parent          = 0;
        int16_t       *side_idx        = 0;

        if (extract_skin_sideset(mesh->n_elements(),
                                 mesh->n_nodes(),
                                 mesh->element_type(),
                                 mesh->elements()->data(),
                                 &n_surf_elements,
                                 &parent,
                                 &side_idx) != SFEM_SUCCESS) {
            SFEM_ERROR("Failed to extract skin!\n");
        }

        auto sideset = std::make_shared<sfem::Sideset>(mesh->comm(),
                                                       sfem::manage_host_buffer(n_surf_elements, parent),
                                                       sfem::manage_host_buffer(n_surf_elements, side_idx));

        return sideset;
    }

    static SharedInPlaceOperator<real_t> create_zero_constraints_op(const std::shared_ptr<Function> &f) {
        return make_in_place_op<real_t>(
                f->space()->n_dofs(),
                [=](real_t *const x) { f->apply_zero_constraints(x); },
                f->execution_space());
    }

}  // namespace sfem

#endif  // SFEM_API_HPP
