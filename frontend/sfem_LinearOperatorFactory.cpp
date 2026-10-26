#include "sfem_LinearOperatorFactory.hpp"

#include "sfem_ACRS.hpp"
#include "sfem_SCRS.hpp"
#include "sfem_SDACRS.hpp"
#include "sfem_sell_spmv.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_DIA.hpp"
#include "sfem_Function.hpp"
#include "sfem_ParallelMatrixFreeOperator.hpp"
#include "sfem_bcrs_sym_SpMV.hpp"
#include "sfem_crs_sym_SpMV.hpp"
#include "sfem_defs.hpp"
#include "smesh_env.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_crs_SpMV.hpp"
#include "smesh_device_buffer.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <vector>

namespace sfem {

    namespace {

        SharedBuffer<int> node_graph_diagonal_offsets(const std::shared_ptr<CRSGraph> &crs_graph) {
            std::vector<int> offsets;
            offsets.reserve(crs_graph->nnz());

            const count_t *const rowptr = crs_graph->rowptr()->data();
            const idx_t *const   colidx = crs_graph->colidx()->data();
            const ptrdiff_t      nrows  = crs_graph->n_nodes();

            for (ptrdiff_t row = 0; row < nrows; ++row) {
                for (count_t k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                    offsets.push_back(static_cast<int>(colidx[k] - row));
                }
            }

            std::sort(offsets.begin(), offsets.end());
            offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

            auto ret = sfem::create_host_buffer<int>(offsets.size());
            std::copy(offsets.begin(), offsets.end(), ret->data());
            return ret;
        }

        std::shared_ptr<sfem::Operator<real_t>> hessian_dia(const std::shared_ptr<sfem::Function> &f,
                                                            const SharedBuffer<real_t>            &x,
                                                            const sfem::ExecutionSpace             es) {
            if (es != sfem::EXECUTION_SPACE_HOST) {
                SFEM_ERROR("hessian_dia supports host execution only\n");
                return nullptr;
            }

            const int block_size = f->space()->block_size();
            auto      graph      = f->space()->node_to_node_graph();
            auto      offsets    = node_graph_diagonal_offsets(graph);
            auto      values     = sfem::create_host_buffer<real_t>(offsets->size() * graph->n_nodes() * block_size * block_size);

            for (ptrdiff_t i = 0, n = values->size(); i < n; ++i) {
                values->data()[i] = 0;
            }

            const real_t *const x_data = x ? x->data() : nullptr;
            if (f->hessian_dia(x_data, offsets->data(), offsets->size(), values->data()) != SFEM_SUCCESS) {
                return nullptr;
            }

            return compose_constraints_op(
                    f,
                    sfem::h_dia_spmv<real_t, real_t>(graph->n_nodes(), graph->n_nodes(), block_size, offsets, values, (real_t)1));
        }

        std::shared_ptr<sfem::Operator<real_t>> hessian_node_block_crs(const std::shared_ptr<sfem::Function> &f,
                                                                       const SharedBuffer<real_t>            &x,
                                                                       const sfem::ExecutionSpace             es) {
            if (es != sfem::EXECUTION_SPACE_HOST) {
                SFEM_ERROR("hessian_node_block_crs supports host execution only\n");
                return nullptr;
            }

            const int block_size = f->space()->block_size();
            if (block_size == 1) {
                return sfem::hessian_crs(f, x, es);
            }

            auto crs_graph = f->space()->node_to_node_graph();
            auto values    = sfem::create_host_buffer<real_t>(crs_graph->nnz() * block_size * block_size);

            for (ptrdiff_t i = 0, n = values->size(); i < n; ++i) {
                values->data()[i] = 0;
            }

            const real_t *const x_data = (x) ? x->data() : nullptr;
            if (f->hessian_crs(x_data, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data()) !=
                SFEM_SUCCESS) {
                return nullptr;
            }

            return compose_constraints_op(f, node_block_crs_to_scalar_crs(crs_graph, block_size, values));
        }

        std::shared_ptr<Operator<real_t>> hessian_bcrs_sym(const std::shared_ptr<sfem::Function> &f,
                                                           const SharedBuffer<real_t>            &x,
                                                           const sfem::ExecutionSpace             es) {
            SFEM_UNUSED(es);
            assert(es == sfem::EXECUTION_SPACE_HOST);

            auto      crs_graph  = f->space()->mesh()->node_to_node_graph_upper_triangular();
            const int block_size = f->space()->block_size();

            int       nblock_entries = ((block_size + 1) * block_size) / 2;
            ptrdiff_t block_stride   = 1;

            bool SFEM_BCRS_SYM_USE_AOS = false;
            SFEM_READ_ENV(SFEM_BCRS_SYM_USE_AOS, atoi);

            int prec = smesh::Env::read("SFEM_ENABLE_MIXED_PRECISION", (int)sizeof(real_t));

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

            switch (prec) {
                case 2:
                    return compose_constraints_op(
                            f,
                            sfem::h_bcrs_sym<count_t, idx_t, half_t, real_t>(crs_graph->n_nodes(),
                                                                             crs_graph->n_nodes(),
                                                                             block_size,
                                                                             crs_graph->rowptr(),
                                                                             crs_graph->colidx(),
                                                                             block_stride,
                                                                             astype<half_t>(diag_values),
                                                                             astype<half_t>(off_diag_values),
                                                                             (real_t)1));
#if !SFEM_REAL_T_IS_FLOAT32
                case 4:
                    return compose_constraints_op(f,
                                                  sfem::h_bcrs_sym<count_t, idx_t, float, real_t>(crs_graph->n_nodes(),
                                                                                                  crs_graph->n_nodes(),
                                                                                                  block_size,
                                                                                                  crs_graph->rowptr(),
                                                                                                  crs_graph->colidx(),
                                                                                                  block_stride,
                                                                                                  astype<float>(diag_values),
                                                                                                  astype<float>(off_diag_values),
                                                                                                  (real_t)1));
#endif
                default:
                    return compose_constraints_op(f,
                                                  sfem::h_bcrs_sym<count_t, idx_t, real_t>(crs_graph->n_nodes(),
                                                                                           crs_graph->n_nodes(),
                                                                                           block_size,
                                                                                           crs_graph->rowptr(),
                                                                                           crs_graph->colidx(),
                                                                                           block_stride,
                                                                                           diag_values,
                                                                                           off_diag_values,
                                                                                           (real_t)1));
            }
        }

        std::shared_ptr<Operator<real_t>> hessian_coo_sym(const std::shared_ptr<sfem::Function> &f,
                                                          const SharedBuffer<real_t>            &x,
                                                          const sfem::ExecutionSpace             es) {
            auto fs        = f->space();
            auto crs_graph = fs->mesh()->node_to_node_graph_upper_triangular();

            auto diag_values     = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            auto off_diag_values = sfem::create_buffer<real_t>(crs_graph->nnz(), es);

            real_t *x_data = nullptr;
            if (x) {
                x_data = x->data();
            }

            std::shared_ptr<sfem::Operator<real_t>> spmv;
#ifdef SFEM_ENABLE_CUDA
            if (es == sfem::EXECUTION_SPACE_DEVICE) {
                auto d_crs_graph = smesh::to_device(crs_graph);

                f->hessian_crs_sym(x_data,
                                   d_crs_graph->rowptr()->data(),
                                   d_crs_graph->colidx()->data(),
                                   diag_values->data(),
                                   off_diag_values->data());

                auto h_row_idx = sfem::create_buffer<idx_t>(crs_graph->nnz(), sfem::EXECUTION_SPACE_HOST);
                crs_to_coo(fs->n_dofs(), crs_graph->rowptr()->data(), h_row_idx->data());
                auto row_idx = smesh::to_device(h_row_idx);

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
                // f->constraints_mask(mask->data());

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

        std::shared_ptr<Operator<real_t>> hessian_crs_sym(const std::shared_ptr<sfem::Function> &f,
                                                          const std::shared_ptr<Buffer<real_t>> &x,
                                                          const sfem::ExecutionSpace             es) {
            auto fs              = f->space();
            auto crs_graph       = fs->mesh()->node_to_node_graph_upper_triangular();
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
                SFEM_ERROR("hessian_crs_sym: DEVICE is not implemented\n");
                return nullptr;
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

        std::shared_ptr<sfem::Operator<real_t>> hessian_sdacrs(const std::shared_ptr<sfem::Function> &f,
                                                               const std::shared_ptr<Buffer<real_t>> &x,
                                                               const sfem::ExecutionSpace             es) {
            auto crs = hessian_crs(f, x, es);
            if (!crs) {
                return nullptr;
            }

            int prec = smesh::Env::read("SFEM_ENABLE_MIXED_PRECISION", (int)sizeof(real_t));
            switch (prec) {
                case 2:
                    return sfem::sdacrs_from_crs<count_t, idx_t, real_t, int16_t, real_t, half_t>(
                            crs->row_ptr, crs->col_idx, crs->values, es);
                case 4:
                    return sfem::sdacrs_from_crs<count_t, idx_t, real_t, int16_t, real_t, float>(
                            crs->row_ptr, crs->col_idx, crs->values, es);
                default:
                    return sfem::sdacrs_from_crs<count_t, idx_t, real_t, int16_t>(crs->row_ptr, crs->col_idx, crs->values, es);
            }
        }

        std::shared_ptr<sfem::Operator<real_t>> hessian_acrs(const std::shared_ptr<sfem::Function> &f,
                                                             const std::shared_ptr<Buffer<real_t>> &x,
                                                             const sfem::ExecutionSpace             es) {
            int  prec = smesh::Env::read("SFEM_ENABLE_MIXED_PRECISION", (int)sizeof(real_t));
            auto temp = sfem::hessian_crs(f, x, es);
            if (!temp) {
                return nullptr;
            }
            switch (prec) {
                case 2:
                    return sfem::acrs_from_crs<count_t, idx_t, real_t, real_t, half_t>(
                            temp->row_ptr, temp->col_idx, temp->values, es);
                case 4:
                    return sfem::acrs_from_crs<count_t, idx_t, real_t, real_t, float>(
                            temp->row_ptr, temp->col_idx, temp->values, es);
                default:
                    return sfem::acrs_from_crs<count_t, idx_t, real_t>(temp->row_ptr, temp->col_idx, temp->values, es);
            }
        }

        std::shared_ptr<sfem::Operator<real_t>> hessian_scrs(const std::shared_ptr<sfem::Function> &f,
                                                             const std::shared_ptr<Buffer<real_t>> &x,
                                                             const sfem::ExecutionSpace             es) {
            int  prec = smesh::Env::read("SFEM_ENABLE_MIXED_PRECISION", (int)sizeof(real_t));
            auto temp = sfem::hessian_crs(f, x, es);
            if (!temp) {
                return nullptr;
            }
            switch (prec) {
                case 2:
                    return sfem::scrs_from_crs<count_t, idx_t, real_t, uint16_t, real_t, half_t>(
                            temp->row_ptr, temp->col_idx, temp->values, es);
                case 4:
                    return sfem::scrs_from_crs<count_t, idx_t, real_t, uint16_t, real_t, float>(
                            temp->row_ptr, temp->col_idx, temp->values, es);
                default:
                    return sfem::scrs_from_crs<count_t, idx_t, real_t, uint16_t>(temp->row_ptr, temp->col_idx, temp->values, es);
            }
        }

        std::shared_ptr<sfem::Operator<real_t>> hessian_sell(const std::shared_ptr<sfem::Function> &f,
                                                             const std::shared_ptr<Buffer<real_t>> &x,
                                                             const sfem::ExecutionSpace             es) {
            int prec         = smesh::Env::read("SFEM_ENABLE_MIXED_PRECISION", (int)sizeof(real_t));
            int slice_height = smesh::Env::read("SFEM_SELL_SLICE_HEIGHT", 32);

            auto temp = sfem::hessian_crs(f, x, es);
            if (!temp) {
                return nullptr;
            }
            switch (prec) {
                case 2:
                    return sfem::sell_from_crs<count_t, idx_t, real_t, real_t, half_t>(
                            temp->row_ptr, temp->col_idx, temp->values, es, slice_height);
                case 4:
                    return sfem::sell_from_crs<count_t, idx_t, real_t, real_t, float>(
                            temp->row_ptr, temp->col_idx, temp->values, es, slice_height);
                default:
                    return sfem::sell_from_crs<count_t, idx_t, real_t>(
                            temp->row_ptr, temp->col_idx, temp->values, es, slice_height);
            }
        }

    }  // namespace

    std::shared_ptr<Operator<real_t>> make_linear_op(const std::shared_ptr<Function> &f) {
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { f->apply(nullptr, x, y); },
                f->execution_space());
    }

    std::shared_ptr<Operator<real_t>> make_linear_op_variant(const std::shared_ptr<Function>                &f,
                                                             const std::vector<std::pair<std::string, int>> &opts) {
        auto variant = f->linear_op_variant(opts);
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { variant->apply(x, y); },
                f->execution_space());
    }

    std::shared_ptr<sfem::CRS<count_t, idx_t, real_t, real_t>> hessian_crs(sfem::Function                  &f,
                                                                           const std::shared_ptr<CRSGraph> &crs_graph,
                                                                           const sfem::ExecutionSpace       es) {
        SFEM_UNUSED(es);
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = smesh::to_device(crs_graph);
            auto values      = sfem::create_buffer<real_t>(d_crs_graph->nnz(), es);

            if (f.hessian_crs(nullptr, d_crs_graph->rowptr()->data(), d_crs_graph->colidx()->data(), values->data()) !=
                SFEM_SUCCESS) {
                return nullptr;
            }

            return sfem::d_crs_spmv(d_crs_graph->n_nodes(),
                                    d_crs_graph->n_nodes(),
                                    d_crs_graph->rowptr(),
                                    d_crs_graph->colidx(),
                                    values,
                                    (real_t)1);
        }
#endif
        auto values = sfem::create_host_buffer<real_t>(crs_graph->nnz());

        if (f.hessian_crs(nullptr, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data()) != SFEM_SUCCESS) {
            return nullptr;
        }

        // Owns the pointers
        return sfem::h_crs_spmv(
                crs_graph->n_nodes(), crs_graph->n_nodes(), crs_graph->rowptr(), crs_graph->colidx(), values, (real_t)1);
    }

    std::shared_ptr<sfem::CRS<count_t, idx_t, real_t, real_t>> hessian_crs(const std::shared_ptr<sfem::Function> &f,
                                                                           const SharedBuffer<real_t>            &x,
                                                                           const sfem::ExecutionSpace             es) {
        SFEM_UNUSED(es);
        auto crs_graph = f->crs_graph();

#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = smesh::to_device(crs_graph);
            auto values      = sfem::create_buffer<real_t>(d_crs_graph->nnz(), es);

            if (f->hessian_crs(x->data(), d_crs_graph->rowptr()->data(), d_crs_graph->colidx()->data(), values->data()) !=
                SFEM_SUCCESS) {
                return nullptr;
            }

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
        if (f->hessian_crs(x_data, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data()) != SFEM_SUCCESS) {
            return nullptr;
        }

        // Owns the pointers
        return sfem::h_crs_spmv(
                crs_graph->n_nodes(), crs_graph->n_nodes(), crs_graph->rowptr(), crs_graph->colidx(), values, (real_t)1);
    }

    std::shared_ptr<sfem::CRS<count_t, idx_t, real_t, real_t>> node_block_crs_to_scalar_crs(
            const std::shared_ptr<CRSGraph> &crs_graph,
            const int                        block_size,
            const SharedBuffer<real_t>      &node_block_values) {
        assert(block_size > 0);

        const ptrdiff_t n_nodes       = crs_graph->n_nodes();
        const ptrdiff_t n_scalar_rows = n_nodes * block_size;
        const ptrdiff_t n_scalar_nnz  = crs_graph->nnz() * block_size * block_size;
        auto            scalar_rowptr = sfem::create_host_buffer<count_t>(n_scalar_rows + 1);
        auto            scalar_colidx = sfem::create_host_buffer<idx_t>(n_scalar_nnz);
        auto            scalar_values = sfem::create_host_buffer<real_t>(n_scalar_nnz);
        const count_t  *node_rowptr   = crs_graph->rowptr()->data();
        const idx_t    *node_colidx   = crs_graph->colidx()->data();
        const real_t   *block_values  = node_block_values->data();

        count_t next = 0;
        for (ptrdiff_t node_i = 0; node_i < n_nodes; ++node_i) {
            const count_t row_begin = node_rowptr[node_i];
            const count_t row_end   = node_rowptr[node_i + 1];
            const int     lenrow    = static_cast<int>(row_end - row_begin);

            for (int bi = 0; bi < block_size; ++bi) {
                scalar_rowptr->data()[node_i * block_size + bi] = next;
                const real_t *const block_row                   = &block_values[row_begin * block_size * block_size];
                const real_t *const row_values                  = &block_row[bi * lenrow * block_size];

                for (count_t k = row_begin; k < row_end; ++k) {
                    const idx_t node_j = node_colidx[k];
                    const int   local  = static_cast<int>(k - row_begin);

                    for (int bj = 0; bj < block_size; ++bj) {
                        scalar_colidx->data()[next] = node_j * block_size + bj;
                        scalar_values->data()[next] = row_values[local * block_size + bj];
                        ++next;
                    }
                }
            }
        }

        scalar_rowptr->data()[n_scalar_rows] = next;

        return sfem::h_crs_spmv<count_t, idx_t, real_t, real_t>(
                n_scalar_rows, n_scalar_rows, scalar_rowptr, scalar_colidx, scalar_values, (real_t)1);
    }

    std::shared_ptr<Operator<real_t>> compose_constraints_op(const std::shared_ptr<sfem::Function>         &f,
                                                             const std::shared_ptr<sfem::Operator<real_t>> &op) {
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) {
                    op->apply(x, y);
                    f->copy_constrained_dofs(x, y);
                },
                op->execution_context());
    }

    std::shared_ptr<sfem::Operator<real_t>> hessian_bsr(const std::shared_ptr<sfem::Function> &f,
                                                        const SharedBuffer<real_t>            &x,
                                                        const sfem::ExecutionSpace             es) {
        auto tangent = sfem::BSRTangent::create(f, es, smesh::Env::read("SFEM_ENABLE_MIXED_PRECISION", (int)sizeof(real_t)));
        if (!tangent || tangent->assemble(x ? x->data() : nullptr) != SFEM_SUCCESS) {
            return nullptr;
        }
        return tangent->op();
    }

    std::shared_ptr<sfem::Operator<real_t>> create_linear_operator(const std::string                           &format,
                                                                   const std::shared_ptr<sfem::Function>       &f,
                                                                   const std::shared_ptr<sfem::Buffer<real_t>> &u,
                                                                   enum sfem::ExecutionSpace                    es) {
        if (format == op_type::MATRIX_FREE) {
            return sfem::create_parallel_matrix_free_operator(f, u, es);
        }

        if (format == op_type::INEXACT) {
            if (!f->inexact_supported()) {
                fprintf(stderr,
                        "[Warning] no operator in this function stores a tangent; "
                        "falling back to matrix-free\n");
                return sfem::create_parallel_matrix_free_operator(f, u, es);
            }
            // The tangent is assembled by whoever owns the linearization -- a
            // Newton step calls `Function::inexact_update` once and then applies
            // this for every Krylov iteration.  It is not assembled here,
            // because this object outlives the state it would have been
            // assembled at.
            const ptrdiff_t ndofs = f->space()->n_dofs();
            return sfem::make_op<real_t>(
                    ndofs, ndofs, [f](const real_t *const h, real_t *const out) { f->inexact_apply(h, out); }, es);
        }

        if (f->space()->block_size() == 1) {
            if (format == op_type::CRS_SYM)
                return sfem::hessian_crs_sym(f, u, es);
            else if (format == op_type::COO_SYM)
                return sfem::hessian_coo_sym(f, u, es);
            else if (format == op_type::DIA)
                return sfem::hessian_dia(f, u, es);
            else if (format == op_type::SPLITCRS) {
                return sfem::hessian_scrs(f, u, es);
            } else if (format == op_type::ALIGNEDCRS) {
                return hessian_acrs(f, u, es);
            } else if (format == op_type::SPLITDACRS) {
                return sfem::hessian_sdacrs(f, u, es);
            } else if (format == op_type::SELL) {
                return sfem::hessian_sell(f, u, es);
            }

            if (format != op_type::CRS) {
                fprintf(stderr, "[Warning] fallback to CRS format as \"%s\" is not supported!\n", format.c_str());
            }

            // FIXME: This is a hack to support mixed precision
            int  prec = smesh::Env::read("SFEM_ENABLE_MIXED_PRECISION", (int)sizeof(real_t));
            auto crs  = sfem::hessian_crs(f, u, es);
            if (!crs) {
                return nullptr;
            }
            switch (prec) {
                case 2:
                    return sfem::h_crs_spmv<count_t, idx_t, half_t, real_t>(
                            crs->rows(), crs->cols(), crs->row_ptr, crs->col_idx, astype<half_t>(crs->values), (real_t)1);
#if !SFEM_REAL_T_IS_FLOAT32
                case 4:
                    return sfem::h_crs_spmv<count_t, idx_t, float, real_t>(
                            crs->rows(), crs->cols(), crs->row_ptr, crs->col_idx, astype<float>(crs->values), (real_t)1);
#endif
                default:
                    return crs;
            }
        } else {
            if (format == op_type::CRS) {
                return sfem::hessian_node_block_crs(f, u, es);
            } else if (format == op_type::BSR) {
                return sfem::hessian_bsr(f, u, es);
            } else if (format == op_type::BSR_SYM) {
                return sfem::hessian_bcrs_sym(f, u, es);
            } else if (format == op_type::DIA) {
                return sfem::hessian_dia(f, u, es);
            }
        }

        return sfem::hessian_bcrs_sym(f, u, es);
    }

    SharedInPlaceOperator<real_t> create_zero_constraints_op(const std::shared_ptr<Function> &f) {
        return make_in_place_op<real_t>(
                f->space()->n_dofs(), [=](real_t *const x) { f->apply_zero_constraints(x); }, f->execution_space());
    }

}  // namespace sfem
