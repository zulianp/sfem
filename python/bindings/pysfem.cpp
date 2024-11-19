#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <mpi.h>
#include <memory>

#include "sfem_Function.hpp"
#include "sfem_base.h"

#include "sfem_Chebyshev3.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_mprgp.hpp"

#include "sfem_API.hpp"
#include "sfem_Multigrid.hpp"

namespace nb = nanobind;

void SFEM_init() {
    char name[] = "SFEM_init";
    char **argv = (char **)malloc(sizeof(char *));
    argv[0] = name;
    int argc = 1;

    MPI_Init(&argc, (char ***)&argv);

    free(argv);
}

void SFEM_finalize() { MPI_Finalize(); }

NB_MODULE(pysfem, m) {
    using namespace sfem;

    using LambdaOperator_t = sfem::LambdaOperator<real_t>;
    using Operator_t = sfem::Operator<real_t>;
    using Multigrid_t = sfem::Multigrid<real_t>;
    using ConjugateGradient_t = sfem::ConjugateGradient<real_t>;
    using BiCGStab_t = sfem::BiCGStab<real_t>;
    using Chebyshev3_t = sfem::Chebyshev3<real_t>;
    using MPRGP_t = sfem::MPRGP<real_t>;
    using IdxBuffer2D = sfem::Buffer<idx_t *>;
    using IdxBuffer = sfem::Buffer<idx_t>;
    using CountBuffer = sfem::Buffer<count_t>;

    m.def("init", &SFEM_init);
    m.def("finalize", &SFEM_finalize);

    nb::enum_<ExecutionSpace>(m, "ExecutionSpace")
            .value("EXECUTION_SPACE_HOST", EXECUTION_SPACE_HOST)
            .value("EXECUTION_SPACE_DEVICE", EXECUTION_SPACE_DEVICE)
            .value("EXECUTION_SPACE_INVALID", EXECUTION_SPACE_INVALID);

    nb::enum_<MemorySpace>(m, "MemorySpace")
            .value("MEMORY_SPACE_HOST", MEMORY_SPACE_HOST)
            .value("MEMORY_SPACE_DEVICE", MEMORY_SPACE_DEVICE)
            .value("MEMORY_SPACE_INVALID", MEMORY_SPACE_INVALID);

    nb::class_<Mesh>(m, "Mesh")  //
            .def(nb::init<>())
            .def("read", &Mesh::read)
            .def("write", &Mesh::write)
            .def("n_nodes", &Mesh::n_nodes)
            .def("n_elements", &Mesh::n_elements)
            .def("convert_to_macro_element_mesh", &Mesh::convert_to_macro_element_mesh)
            .def("spatial_dimension", &Mesh::spatial_dimension);

    m.def("mesh_connectivity_from_file", [](const char *folder) -> std::shared_ptr<IdxBuffer2D> {
        return sfem::mesh_connectivity_from_file(MPI_COMM_WORLD, folder);
    });

    nb::class_<IdxBuffer2D>(m, "IdxBuffer2D");
    nb::class_<sfem::Buffer<int>>(m, "Buffer<int>").def(nb::init<>());
    nb::class_<sfem::Buffer<long>>(m, "Buffer<long>").def(nb::init<>());
    nb::class_<sfem::Buffer<float>>(m, "Buffer<float>").def(nb::init<>());
    nb::class_<sfem::Buffer<double>>(m, "Buffer<double>").def(nb::init<>());

    m.def("len", [](const std::shared_ptr<sfem::Buffer<int>> &b) -> size_t { return b->size(); });

    m.def("len",
          [](const std::shared_ptr<sfem::Buffer<double>> &b) -> size_t { return b->size(); });

    m.def("create_real_buffer", [](const ptrdiff_t n) -> std::shared_ptr<sfem::Buffer<real_t>> {
        return sfem::h_buffer<real_t>(n);
    });

    m.def("numpy_view",
          [](const std::shared_ptr<sfem::Buffer<int>> &b) -> nb::ndarray<nb::numpy, int> {
              return nb::ndarray<nb::numpy, int>(b->data(), {(size_t)b->size()}, nb::handle());
          });

    m.def("numpy_view",
          [](const std::shared_ptr<sfem::Buffer<long>> &b) -> nb::ndarray<nb::numpy, long> {
              return nb::ndarray<nb::numpy, long>(b->data(), {(size_t)b->size()}, nb::handle());
          });

    m.def("numpy_view",
          [](const std::shared_ptr<sfem::Buffer<double>> &b) -> nb::ndarray<nb::numpy, double> {
              return nb::ndarray<nb::numpy, double>(b->data(), {(size_t)b->size()}, nb::handle());
          });

    m.def("numpy_view",
          [](const std::shared_ptr<sfem::Buffer<float>> &b) -> nb::ndarray<nb::numpy, float> {
              return nb::ndarray<nb::numpy, float>(b->data(), {(size_t)b->size()}, nb::handle());
          });

    m.def("view", [](nb::ndarray<nb::numpy, float> &v) -> std::shared_ptr<sfem::Buffer<float>> {
        return sfem::Buffer<float>::wrap(v.size(), v.data(), sfem::MEMORY_SPACE_HOST);
    });

    m.def("view", [](nb::ndarray<nb::numpy, double> &v) -> std::shared_ptr<sfem::Buffer<double>> {
        return sfem::Buffer<double>::wrap(v.size(), v.data(), sfem::MEMORY_SPACE_HOST);
    });

    m.def("view", [](nb::ndarray<nb::numpy, int> &v) -> std::shared_ptr<sfem::Buffer<int>> {
        return sfem::Buffer<int>::wrap(v.size(), v.data(), sfem::MEMORY_SPACE_HOST);
    });

    m.def("view", [](nb::ndarray<nb::numpy, long> &v) -> std::shared_ptr<sfem::Buffer<long>> {
        return sfem::Buffer<long>::wrap(v.size(), v.data(), sfem::MEMORY_SPACE_HOST);
    });

    m.def("create_mesh",
          [](const char *elem_type_name,
             nb::ndarray<idx_t> idx,
             nb::ndarray<geom_t> p) -> std::shared_ptr<Mesh> {
              size_t n = idx.shape(0);
              enum ElemType element_type = type_from_string(elem_type_name);

              int nnxe = idx.shape(0);
              ptrdiff_t nelements = idx.shape(1);
              int spatial_dimension = p.shape(0);
              ptrdiff_t nnodes = p.shape(1);

              idx_t **elements = (idx_t **)malloc(nnxe * sizeof(idx_t *));
              geom_t **points = (geom_t **)malloc(spatial_dimension * sizeof(geom_t *));

              for (int d = 0; d < nnxe; d++) {
                  elements[d] = (idx_t *)malloc(nelements * sizeof(idx_t));

                  for (ptrdiff_t i = 0; i < nelements; i++) {
                      elements[d][i] = idx.data()[d * nelements + i];
                  }
              }

              for (int d = 0; d < spatial_dimension; d++) {
                  points[d] = (geom_t *)malloc(nnodes * sizeof(geom_t));
                  for (ptrdiff_t i = 0; i < nnodes; i++) {
                      points[d][i] = p.data()[d * nnodes + i];
                  }
              }

              // Transfer ownership to mesh
              return std::make_shared<Mesh>(
                      spatial_dimension, element_type, nelements, elements, nnodes, points);
          });

    m.def("points",
          [](std::shared_ptr<Mesh> &mesh, int coord) -> nb::ndarray<nb::numpy, const geom_t> {
              return nb::ndarray<nb::numpy, const geom_t>(
                      mesh->points(coord), {(size_t)mesh->n_nodes()}, nb::handle());
          });

    nb::class_<FunctionSpace>(m, "FunctionSpace")
            .def(nb::init<std::shared_ptr<Mesh>>())
            .def(nb::init<std::shared_ptr<Mesh>, const int>())
            .def("derefine", &FunctionSpace::derefine)
            .def("mesh", &FunctionSpace::mesh_ptr)
            .def("n_dofs", &FunctionSpace::n_dofs)
            .def("block_size", &FunctionSpace::block_size)
            .def("promote_to_semi_structured", &FunctionSpace::promote_to_semi_structured);

    m.def("create_derefined_crs_graph",
          [](const std::shared_ptr<FunctionSpace> &space) -> std::shared_ptr<CRSGraph> {
              return sfem::create_derefined_crs_graph(*space);
          });

    m.def("create_edge_idx",
          [](const std::shared_ptr<CRSGraph> &crs_graph) -> std::shared_ptr<Buffer<idx_t>> {
              return sfem::create_edge_idx(*crs_graph);
          });

    m.def("create_hierarchical_restriction", &sfem::create_hierarchical_restriction);
    m.def("create_hierarchical_prolongation", &sfem::create_hierarchical_prolongation);

    nb::class_<Op>(m, "Op");
    m.def("create_op", &Factory::create_op);
    m.def("create_boundary_op", &Factory::create_boundary_op);

    nb::class_<Output>(m, "Output")
            .def("set_output_dir", &Output::set_output_dir)
            .def("write", &Output::write)
            .def("write_time_step", &Output::write_time_step);

    m.def("write_time_step",
          [](std::shared_ptr<Output> &out,
             const char *name,
             const real_t t,
             nb::ndarray<real_t> x) { out->write_time_step(name, t, x.data()); });

    m.def("write", [](std::shared_ptr<Output> &out, const char *name, nb::ndarray<real_t> x) {
        out->write(name, x.data());
    });

    m.def("set_field",
          [](std::shared_ptr<Op> &op,
             const char *name,
             const int component,
             nb::ndarray<real_t> v) {
              size_t n = v.size();
              auto c_v = (real_t *)malloc(n * sizeof(real_t));
              memcpy(c_v, v.data(), n * sizeof(real_t));
              op->set_field(name, component, c_v);
          });

    m.def("hessian_diag",
          [](std::shared_ptr<Op> &op, nb::ndarray<real_t> x, nb::ndarray<real_t> d) {
              op->hessian_diag(x.data(), d.data());
          });

    m.def("hessian_diag",
          [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x, nb::ndarray<real_t> d) {
              fun->hessian_diag(x.data(), d.data());
          });

    m.def("hessian_crs",
          [](std::shared_ptr<Function> fun,
             std::shared_ptr<sfem::Buffer<real_t>> x,
             std::shared_ptr<sfem::Buffer<count_t>> rowptr,
             std::shared_ptr<sfem::Buffer<idx_t>> colidx,
             std::shared_ptr<sfem::Buffer<real_t>> values) {
              fun->hessian_crs(x->data(), rowptr->data(), colidx->data(), values->data());
          });

    // m.def("crs_spmv",
    //       [](nb::ndarray<count_t> rowptr,
    //          nb::ndarray<idx_t> colidx,
    //          nb::ndarray<real_t> values) -> std::shared_ptr<Operator_t> {
    //           return sfem::h_crs_spmv(
    //                   rowptr.size() - 1,
    //                   rowptr.size() - 1,
    //                   sfem::Buffer<count_t>::wrap(
    //                           rowptr.size(), rowptr.data(), sfem::MEMORY_SPACE_HOST),
    //                   sfem::Buffer<idx_t>::wrap(
    //                           colidx.size(), colidx.data(), sfem::MEMORY_SPACE_HOST),
    //                   sfem::Buffer<real_t>::wrap(
    //                           values.size(), values.data(), sfem::MEMORY_SPACE_HOST),
    //                   real_t(0));
    //       });

    m.def("crs_spmv",
          [](std::shared_ptr<sfem::Buffer<count_t>> rowptr,
             std::shared_ptr<sfem::Buffer<idx_t>> colidx,
             std::shared_ptr<sfem::Buffer<real_t>> values) -> std::shared_ptr<Operator_t> {
              return sfem::h_crs_spmv(
                      rowptr->size() - 1, rowptr->size() - 1, rowptr, colidx, values, real_t(0));
          });

    nb::class_<CRSGraph>(m, "CRSGraph")
            .def("n_nodes", &CRSGraph::n_nodes)
            .def("nnz", &CRSGraph::nnz)
            .def("rowptr", &CRSGraph::rowptr)
            .def("colidx", &CRSGraph::colidx);

    nb::class_<Function>(m, "Function")
            .def(nb::init<std::shared_ptr<FunctionSpace>>())
            .def("add_operator", &Function::add_operator)
            .def("space", &Function::space)
            .def("add_dirichlet_conditions", &Function::add_dirichlet_conditions)
            .def("set_output_dir", &Function::set_output_dir)
            .def("output", &Function::output)
            .def("crs_graph", &Function::crs_graph);

    m.def("diag", [](nb::ndarray<real_t> d) -> std::shared_ptr<Operator_t> {
        auto op = std::make_shared<LambdaOperator<real_t>>(
                d.size(),
                d.size(),
                [=](const real_t *const x, real_t *const y) {
                    ptrdiff_t n = d.size();
                    const real_t *d_ = d.data();

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        y[i] += d_[i] * x[i];
                    }
                },
                EXECUTION_SPACE_HOST);

        return op;
    });

    m.def("apply",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<real_t> x,
             nb::ndarray<real_t> h,
             nb::ndarray<real_t> y) { fun->apply(x.data(), h.data(), y.data()); });

    m.def("value", [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x) -> real_t {
        real_t value = 0;
        fun->value(x.data(), &value);
        return value;
    });

    m.def("gradient",
          [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x, nb::ndarray<real_t> y) {
              fun->gradient(x.data(), y.data());
          });

    m.def("apply_constraints", [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x) {
        fun->apply_constraints(x.data());
    });

    m.def("apply_zero_constraints", [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x) {
        fun->apply_zero_constraints(x.data());
    });

    m.def("copy_constrained_dofs",
          [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x, nb::ndarray<real_t> g) {
              fun->copy_constrained_dofs(x.data(), g.data());
          });

    m.def("constraints_gradient",
          [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x, nb::ndarray<real_t> g) {
              fun->constraints_gradient(x.data(), g.data());
          });

    m.def("report_solution", [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> x) {
        fun->report_solution(x.data());
    });

    nb::class_<DirichletConditions>(m, "DirichletConditions")
            .def(nb::init<std::shared_ptr<FunctionSpace>>());

    nb::class_<NeumannConditions, Op>(m, "NeumannConditions")
            .def(nb::init<std::shared_ptr<FunctionSpace>>());

    m.def("add_condition",
          [](std::shared_ptr<DirichletConditions> &dc,
             nb::ndarray<idx_t> idx,
             const int component,
             const real_t value) {
              size_t n = idx.size();
              auto c_idx = (idx_t *)malloc(n * sizeof(idx_t));
              memcpy(c_idx, idx.data(), n * sizeof(idx_t));

              dc->add_condition(n, n, c_idx, component, value);
          });

    m.def("apply_value",
          [](std::shared_ptr<DirichletConditions> &dc, real_t value, nb::ndarray<real_t> y) {
              dc->apply_value(value, y.data());
          });

    m.def("add_condition",
          [](std::shared_ptr<NeumannConditions> &nc,
             nb::ndarray<idx_t> idx,
             const int component,
             const real_t value) {
              size_t n = idx.size();
              auto c_idx = (idx_t *)malloc(n * sizeof(idx_t));
              memcpy(c_idx, idx.data(), n * sizeof(idx_t));

              nc->add_condition(n, n, c_idx, component, value);
          });

    nb::class_<Operator_t>(m, "Operator")
            .def("__add__",
                 [](const std::shared_ptr<Operator_t> &l, const std::shared_ptr<Operator_t> &r) {
                     assert(l->cols() == r->rows());
                     return sfem::make_op<real_t>(
                             l->rows(),
                             r->cols(),
                             [=](const real_t *const x, real_t *const y) {
                                 l->apply(x, y);
                                 r->apply(x, y);
                             },
                             l->execution_space());
                 })
            .def("__mul__",
                 [](const std::shared_ptr<Operator_t> &l, const std::shared_ptr<Operator_t> &r) {
                     assert(l->cols() == r->rows());

                     auto temp = sfem::create_buffer<real_t>(l->rows(), l->execution_space());

                     return sfem::make_op<real_t>(
                             l->rows(),
                             r->cols(),
                             [=](const real_t *const x, real_t *const y) {
                                 auto data = temp->data();
                                 ptrdiff_t n = l->rows();
#pragma omp parallel for
                                 for (ptrdiff_t i = 0; i < n; i++) {
                                     data[i] = 0;
                                 }
                                 r->apply(x, data);
                                 l->apply(data, y);
                             },
                             l->execution_space());
                 });

    m.def("make_op",
          [](std::shared_ptr<Function> &fun, nb::ndarray<real_t> u) -> std::shared_ptr<Operator_t> {
              return sfem::make_op<real_t>(
                      u.size(),
                      u.size(),
                      [=](const real_t *const x, real_t *const y) {
                          memset(y, 0, u.size() * sizeof(real_t));
                          fun->apply(u.data(), x, y);
                      },
                      fun->execution_space());
          });

    m.def("apply",
          [](const std::shared_ptr<Operator_t> &op, nb::ndarray<real_t> x, nb::ndarray<real_t> y) {
              op->apply(x.data(), y.data());
          });

    nb::class_<ConjugateGradient_t>(m, "ConjugateGradient")
            .def(nb::init<>())
            .def("default_init", &ConjugateGradient_t::default_init)
            .def("set_op", &ConjugateGradient_t::set_op)
            .def("set_preconditioner_op", &ConjugateGradient_t::set_preconditioner_op)
            .def("set_max_it", &ConjugateGradient_t::set_max_it)
            .def("set_verbose", &ConjugateGradient_t::set_verbose)
            .def("set_rtol", &ConjugateGradient_t::set_rtol)
            .def("set_atol", &ConjugateGradient_t::set_atol);

    m.def("apply",
          [](std::shared_ptr<ConjugateGradient_t> &cg,
             nb::ndarray<real_t> x,
             nb::ndarray<real_t> y) { cg->apply(x.data(), y.data()); });

    nb::class_<BiCGStab_t>(m, "BiCGStab")
            .def(nb::init<>())
            .def("default_init", &BiCGStab_t::default_init)
            .def("set_op", &BiCGStab_t::set_op)
            .def("set_preconditioner_op", &BiCGStab_t::set_preconditioner_op)
            .def("set_max_it", &BiCGStab_t::set_max_it);
    // .def("set_verbose", &BiCGStab_t::set_verbose)
    // .def("set_rtol", &BiCGStab_t::set_rtol)
    // .def("set_atol", &BiCGStab_t::set_atol);

    nb::class_<Chebyshev3_t>(m, "Chebyshev3")
            .def(nb::init<>())
            .def("default_init", &Chebyshev3_t::default_init)
            .def("set_op", &Chebyshev3_t::set_op)
            // .def("set_preconditioner_op", &Chebyshev3_t::set_preconditioner_op)
            .def("set_max_it", &Chebyshev3_t::set_max_it)
            .def("set_verbose", &Chebyshev3_t::set_verbose)
            .def("init_with_ones", &Chebyshev3_t::init_with_ones)
            // .def("set_rtol", &Chebyshev3_t::set_rtol)
            .def("set_atol", &Chebyshev3_t::set_atol);

    m.def("apply",
          [](std::shared_ptr<Chebyshev3_t> &op, nb::ndarray<real_t> x, nb::ndarray<real_t> y) {
              op->apply(x.data(), y.data());
          });

    m.def("apply",
          [](std::shared_ptr<BiCGStab_t> &op, nb::ndarray<real_t> x, nb::ndarray<real_t> y) {
              op->apply(x.data(), y.data());
          });

    nb::class_<MPRGP_t>(m, "MPRGP")
            .def(nb::init<>())
            .def("default_init", &MPRGP_t::default_init)
            .def("set_op", &MPRGP_t::set_op)
            .def("set_max_it", &MPRGP_t::set_max_it)
            .def("set_verbose", &MPRGP_t::set_verbose)
            .def("set_rtol", &MPRGP_t::set_rtol)
            .def("set_atol", &MPRGP_t::set_atol);

    m.def("set_upper_bound", [](std::shared_ptr<MPRGP_t> &op, nb::ndarray<real_t> &x) {
        op->set_upper_bound(sfem::Buffer<real_t>::wrap(x.size(), x.data()));
    });

    m.def("apply", [](std::shared_ptr<MPRGP_t> &op, nb::ndarray<real_t> x, nb::ndarray<real_t> y) {
        op->apply(x.data(), y.data());
    });

    nb::class_<Multigrid_t>(m, "Multigrid")
            .def(nb::init<>())
            .def("default_init", &Multigrid_t::default_init)
            .def("set_max_it", &Multigrid_t::set_max_it)
            .def("set_atol", &Multigrid_t::set_atol)
            .def("set_max_it", &Multigrid_t::set_max_it)
            .def("add_level", &Multigrid_t::add_level);

    m.def("apply",
          [](std::shared_ptr<Multigrid_t> &op, nb::ndarray<real_t> x, nb::ndarray<real_t> y) {
              op->apply(x.data(), y.data());
          });
}
