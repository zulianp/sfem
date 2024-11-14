#ifndef SFEM_API_HPP
#define SFEM_API_HPP

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_mesh.h"

#include "sfem_Chebyshev3.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_bcrs_sym_SpMV.hpp"
#include "sfem_bsr_SpMV.hpp"
#include "sfem_cg.hpp"
#include "sfem_crs_SpMV.hpp"
#include "sfem_mprgp.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "cu_proteus_hex8_interpolate.h"
#include "cu_tet4_prolongation_restriction.h"
#include "sfem_ContactConditions_cuda.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_crs_SpMV.hpp"
#include "sfem_cuda_mprgp_impl.hpp"
#include "sfem_cuda_solver.hpp"

#else
namespace sfem {
    void device_synchronize() {}
}  // namespace sfem
#endif

#include "proteus_hex8.h"
#include "proteus_hex8_interpolate.h"
#include "sfem_prolongation_restriction.h"

#include <sys/stat.h>
#include "matrixio_crs.h"

namespace sfem {

    template <typename T>
    std::shared_ptr<Buffer<T>> create_buffer(const std::ptrdiff_t n, const MemorySpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == MEMORY_SPACE_DEVICE) return sfem::d_buffer<T>(n);
#endif  // SFEM_ENABLE_CUDA
        return sfem::h_buffer<T>(n);
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> create_buffer(const std::ptrdiff_t n, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::d_buffer<T>(n);
#endif  // SFEM_ENABLE_CUDA
        return sfem::h_buffer<T>(n);
    }

    std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space,
                                  const char *name,
                                  const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::Factory::create_op_gpu(space, name);
#endif  // SFEM_ENABLE_CUDA
        return sfem::Factory::create_op(space, name);
    }

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> create_cg(const std::shared_ptr<Operator<T>> &op,
                                                    const ExecutionSpace es) {
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
    std::shared_ptr<BiCGStab<T>> create_bcgs(const std::shared_ptr<Operator<T>> &op,
                                             const ExecutionSpace es) {
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
    std::shared_ptr<Chebyshev3<T>> create_cheb3(const std::shared_ptr<Operator<T>> &op,
                                                const ExecutionSpace es) {
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
    std::shared_ptr<MPRGP<T>> create_mprgp(const std::shared_ptr<Operator<T>> &op,
                                           const ExecutionSpace es) {
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
    std::shared_ptr<Multigrid<T>> create_mg(const ExecutionSpace es) {
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

    std::shared_ptr<Constraint> create_dirichlet_conditions_from_env(
            const std::shared_ptr<FunctionSpace> &space,
            const ExecutionSpace es) {
        auto conds = sfem::DirichletConditions::create_from_env(space);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::to_device(conds);
        }
#endif  // SFEM_ENABLE_CUDA

        return conds;
    }

    std::shared_ptr<Constraint> create_contact_conditions_from_env(
            const std::shared_ptr<FunctionSpace> &space,
            const ExecutionSpace es) {
        auto conds = sfem::ContactConditions::create_from_env(space);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::to_device(conds);
        }
#endif  // SFEM_ENABLE_CUDA

        return conds;
    }

    std::shared_ptr<Buffer<idx_t>> create_edge_idx(CRSGraph &crs_graph) {
        const ptrdiff_t rows = crs_graph.n_nodes();
        auto p2_vertices = h_buffer<idx_t>(crs_graph.nnz());

        build_p1_to_p2_edge_map(
                rows, crs_graph.rowptr()->data(), crs_graph.colidx()->data(), p2_vertices->data());

        return p2_vertices;
    }

    std::shared_ptr<CRSGraph> create_derefined_crs_graph(FunctionSpace &space) {
        auto et = (enum ElemType)space.element_type();
        auto coarse_et = macro_base_elem(et);
        auto crs_graph = space.mesh().create_node_to_node_graph(coarse_et);
        return crs_graph;
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation(
            const std::shared_ptr<FunctionSpace> &from_space,
            const std::shared_ptr<FunctionSpace> &to_space,
            const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (EXECUTION_SPACE_DEVICE == es) {
            auto elements = to_space->device_elements();
            if (!elements) {
                elements = create_device_elements(to_space, to_space->element_type());
                from_space->set_device_elements(elements);
            }

            if (to_space->has_semi_structured_mesh()) {
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto &ssm = to_space->semi_structured_mesh();
                            cu_proteus_hex8_hierarchical_prolongation(ssm.level(),
                                                                      ssm.n_elements(),
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
            } else {
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto mesh = (mesh_t *)from_space->mesh().impl_mesh();
                            cu_macrotet4_to_tet4_prolongation_element_based(
                                    mesh->nelements,
                                    mesh->nelements,
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
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto &ssm = to_space->semi_structured_mesh();
                            proteus_hex8_hierarchical_prolongation(ssm.level(),
                                                                   ssm.n_elements(),
                                                                   ssm.element_data(),
                                                                   from_space->block_size(),
                                                                   from,
                                                                   to);
                        },
                        EXECUTION_SPACE_HOST);
            } else {
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto mesh = (mesh_t *)from_space->mesh().impl_mesh();
                            hierarchical_prolongation(from_space->element_type(),
                                                      to_space->element_type(),
                                                      mesh->nelements,
                                                      mesh->elements,
                                                      from_space->block_size(),
                                                      from,
                                                      to);
                        },
                        EXECUTION_SPACE_HOST);
            }
        }
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction(
            const std::shared_ptr<FunctionSpace> &from_space,
            const std::shared_ptr<FunctionSpace> &to_space,
            const ExecutionSpace es) {
        auto mesh = (mesh_t *)from_space->mesh().impl_mesh();
        auto from_element = (enum ElemType)from_space->element_type();
        auto to_element = (enum ElemType)to_space->element_type();
        const int block_size = from_space->block_size();

        ptrdiff_t nnodes = 0;
        idx_t **elements = nullptr;
        int nxe;
        if (from_space->has_semi_structured_mesh()) {
            auto &mesh = from_space->semi_structured_mesh();
            nxe = proteus_hex8_nxe(mesh.level());
            elements = mesh.element_data();
            nnodes = mesh.n_nodes();
        } else {
            nxe = elem_num_nodes(from_element);
            elements = mesh->elements;
            nnodes = mesh->nnodes;
        }

        auto element_to_node_incidence_count = create_buffer<uint16_t>(nnodes, MEMORY_SPACE_HOST);
        {
            auto buff = element_to_node_incidence_count->data();

            for (int d = 0; d < nxe; d++) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < mesh->nelements; ++i) {
#pragma omp atomic update
                    buff[elements[d][i]]++;
                }
            }
        }

        // element_to_node_incidence_count->print(std::cout);

#ifdef SFEM_ENABLE_CUDA
        if (EXECUTION_SPACE_DEVICE == es) {
            auto dbuff = to_device(element_to_node_incidence_count);

            auto elements = from_space->device_elements();
            if (!elements) {
                elements = create_device_elements(from_space, from_space->element_type());
                from_space->set_device_elements(elements);
            }

            if (from_space->has_semi_structured_mesh()) {
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto &ssm = from_space->semi_structured_mesh();
                            cu_proteus_hex8_hierarchical_restriction(ssm.level(),
                                                                     ssm.n_elements(),
                                                                     ssm.n_elements(),
                                                                     elements->data(),
                                                                     dbuff->data(),
                                                                     block_size,
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
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto mesh = (mesh_t *)from_space->mesh().impl_mesh();
                            cu_macrotet4_to_tet4_restriction_element_based(mesh->nelements,
                                                                           mesh->nelements,
                                                                           elements->data(),
                                                                           dbuff->data(),
                                                                           block_size,
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
            if (from_space->has_semi_structured_mesh()) {
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto &ssm = from_space->semi_structured_mesh();
                            proteus_hex8_hierarchical_restriction(
                                    ssm.level(),
                                    ssm.n_elements(),
                                    ssm.element_data(),
                                    element_to_node_incidence_count->data(),
                                    block_size,
                                    from,
                                    to);
                        },
                        EXECUTION_SPACE_HOST);
            } else {
                return std::make_shared<LambdaOperator<real_t>>(
                        to_space->n_dofs(),
                        from_space->n_dofs(),
                        [=](const real_t *const from, real_t *const to) {
                            auto mesh = (mesh_t *)from_space->mesh().impl_mesh();
                            hierarchical_restriction_with_counting(
                                    from_element,
                                    to_element,
                                    mesh->nelements,
                                    mesh->elements,
                                    element_to_node_incidence_count->data(),
                                    block_size,
                                    from,
                                    to);
                        },
                        EXECUTION_SPACE_HOST);
            }
        }
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction_from_graph(
            const ptrdiff_t n_fine_nodes,
            const int block_size,
            const std::shared_ptr<CRSGraph> &crs_graph,
            const std::shared_ptr<Buffer<idx_t>> &edges,
            const ExecutionSpace es) {
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_coarse_nodes * block_size;
        ptrdiff_t cols = n_fine_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_edges = to_device(edges);
            auto d_crs_graph = to_device(crs_graph);

            return std::make_shared<LambdaOperator<real_t>>(
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

        return std::make_shared<LambdaOperator<real_t>>(
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

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation_from_graph(
            const std::shared_ptr<Function> &function,
            const std::shared_ptr<CRSGraph> &crs_graph,
            const std::shared_ptr<Buffer<idx_t>> &edges,
            const ExecutionSpace es) {
        const ptrdiff_t n_fine_nodes = function->space()->mesh().n_nodes();
        int block_size = function->space()->block_size();
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_fine_nodes * block_size;
        ptrdiff_t cols = n_coarse_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_edges = to_device(edges);
            auto d_crs_graph = to_device(crs_graph);

            return std::make_shared<LambdaOperator<real_t>>(
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

        return std::make_shared<LambdaOperator<real_t>>(
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
    std::shared_ptr<Operator<T>> create_inverse_diagonal_scaling(
            const std::shared_ptr<Buffer<T>> &diag,
            const ExecutionSpace es) {
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

    std::shared_ptr<Operator<real_t>> make_linear_op(const std::shared_ptr<Function> &f) {
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { f->apply(nullptr, x, y); },
                f->execution_space());
    }

    std::shared_ptr<Operator<real_t>> make_linear_op_variant(
            const std::shared_ptr<Function> &f,
            const std::vector<std::pair<std::string, int>> &opts) {
        auto variant = f->linear_op_variant(opts);
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { variant->apply(x, y); },
                f->execution_space());
    }

    auto hessian_crs(sfem::Function &f,
                     const std::shared_ptr<CRSGraph> &crs_graph,
                     const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);
            auto values = sfem::create_buffer<real_t>(d_crs_graph->nnz(), es);

            f.hessian_crs(nullptr,
                          d_crs_graph->rowptr()->data(),
                          d_crs_graph->colidx()->data(),
                          values->data());

            return sfem::d_crs_spmv(d_crs_graph->n_nodes(),
                                    d_crs_graph->n_nodes(),
                                    d_crs_graph->rowptr(),
                                    d_crs_graph->colidx(),
                                    values,
                                    (real_t)1);
        }
#endif
        auto values = sfem::h_buffer<real_t>(crs_graph->nnz());

        f.hessian_crs(
                nullptr, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

        // Owns the pointers
        return sfem::h_crs_spmv(crs_graph->n_nodes(),
                                crs_graph->n_nodes(),
                                crs_graph->rowptr(),
                                crs_graph->colidx(),
                                values,
                                (real_t)1);
    }

    auto hessian_crs(std::shared_ptr<sfem::Function> &f,
                     const std::shared_ptr<Buffer<real_t>> &x,
                     const sfem::ExecutionSpace es) {
        auto crs_graph = f->crs_graph();

#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);
            auto values = sfem::create_buffer<real_t>(d_crs_graph->nnz(), es);

            f->hessian_crs(x->data(),
                           d_crs_graph->rowptr()->data(),
                           d_crs_graph->colidx()->data(),
                           values->data());

            return sfem::d_crs_spmv(d_crs_graph->n_nodes(),
                                    d_crs_graph->n_nodes(),
                                    d_crs_graph->rowptr(),
                                    d_crs_graph->colidx(),
                                    values,
                                    (real_t)1);
        }
#endif
        auto values = sfem::h_buffer<real_t>(crs_graph->nnz());

        f->hessian_crs(x->data(),
                       crs_graph->rowptr()->data(),
                       crs_graph->colidx()->data(),
                       values->data());

        // Owns the pointers
        return sfem::h_crs_spmv(crs_graph->n_nodes(),
                                crs_graph->n_nodes(),
                                crs_graph->rowptr(),
                                crs_graph->colidx(),
                                values,
                                (real_t)1);
    }

    auto hessian_bsr(std::shared_ptr<sfem::Function> &f,
                     const std::shared_ptr<Buffer<real_t>> &x,
                     const sfem::ExecutionSpace es) {
        // Get the mesh node-to-node graph instead of the FunctionSpace scalar adapted graph
        auto crs_graph = f->space()->node_to_node_graph();
        const int block_size = f->space()->block_size();

#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);
            auto values =
                    sfem::create_buffer<real_t>(d_crs_graph->nnz() * block_size * block_size, es);

            f->hessian_bsr(x->data(),
                           d_crs_graph->rowptr()->data(),
                           d_crs_graph->colidx()->data(),
                           values->data());

            return sfem::d_bsr_spmv(d_crs_graph->n_nodes(),
                                    d_crs_graph->n_nodes(),
                                    block_size,
                                    d_crs_graph->rowptr(),
                                    d_crs_graph->colidx(),
                                    values,
                                    (real_t)1);
        }
#endif
        auto values = sfem::h_buffer<real_t>(crs_graph->nnz() * block_size * block_size);

        real_t *x_data = (x) ? x->data() : nullptr;

        f->hessian_bsr(
                x_data, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

        // Owns the pointers
        return sfem::h_bsr_spmv(crs_graph->n_nodes(),
                                crs_graph->n_nodes(),
                                block_size,
                                crs_graph->rowptr(),
                                crs_graph->colidx(),
                                values,
                                (real_t)1);
    }

    auto hessian_bcrs_sym(std::shared_ptr<sfem::Function> &f,
                          const std::shared_ptr<Buffer<real_t>> &x,
                          const sfem::ExecutionSpace es) {
        auto crs_graph = f->space()->mesh().node_to_node_graph_upper_triangular();
        const int block_size = f->space()->block_size();

        int nblock_entries = ((block_size + 1) * block_size) / 2;

        // We build them as AoS for now
        auto off_diag_values = sfem::h_buffer<real_t>(nblock_entries, crs_graph->nnz());
        auto diag_values =
                sfem::h_buffer<real_t>(nblock_entries, f->space()->n_dofs() / block_size);

        real_t *x_data = (x) ? x->data() : nullptr;
        f->hessian_bcrs_sym(x_data,
                            crs_graph->rowptr()->data(),
                            crs_graph->colidx()->data(),
                            1,
                            diag_values->data(),
                            off_diag_values->data());

        auto spmv = sfem::h_bcrs_sym_spmv<count_t, idx_t, real_t>(crs_graph->n_nodes(),
                                                                  crs_graph->n_nodes(),
                                                                  block_size,
                                                                  crs_graph->rowptr(),
                                                                  crs_graph->colidx(),
                                                                  1,
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

    real_t residual(sfem::Operator<real_t> &op,
                    const real_t *const rhs,
                    const real_t *const x,
                    real_t *const r) {
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

    //     template <typename R, typename C, typename T>
    //     std::shared_ptr<CRSSpMV<R, C, T>> create_crs_spmv(const ptrdiff_t rows,
    //                                                  const ptrdiff_t cols,
    //                                                  const std::shared_ptr<Buffer<R>> &rowptr,
    //                                                  const std::shared_ptr<Buffer<C>> &colidx,
    //                                                  const std::shared_ptr<Buffer<T>> &values,
    //                                                  const T scale_output, ExecutionSpace es)
    //     {
    //         #ifdef SFEM_ENABLE_CUDA
    //                 if (op.execution_space() == sfem::EXECUTION_SPACE_DEVICE) {

    //                 } else
    // #endif
    //                 {

    //                 }
    //     }

    int write_crs(const std::string &path, CRSGraph &graph, sfem::Buffer<real_t> &values) {
        struct stat st = {0};
        if (stat(path.c_str(), &st) == -1) {
            mkdir(path.c_str(), 0700);
        }

        crs_t crs_out;
        crs_out.rowptr = (char *)graph.rowptr()->data();
        crs_out.colidx = (char *)graph.colidx()->data();
        crs_out.values = (char *)values.data();
        crs_out.grows = graph.rowptr()->size() - 1;
        crs_out.lrows = graph.rowptr()->size() - 1;
        crs_out.lnnz = values.size();
        crs_out.gnnz = values.size();
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = SFEM_MPI_REAL_T;
        return crs_write_folder(MPI_COMM_SELF, path.c_str(), &crs_out);
    }

}  // namespace sfem

#endif  // SFEM_API_HPP
