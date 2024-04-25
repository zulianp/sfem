#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <mpi.h>
#include <memory>

#include "sfem_Function.hpp"
#include "sfem_base.h"

#include "sfem_cg.hpp"

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

    using LambdaOperator_t = sfem::LambdaOperator<isolver_scalar_t>;
    using Operator_t = sfem::Operator<isolver_scalar_t>;
    using ConjugateGradient_t = sfem::ConjugateGradient<isolver_scalar_t>;

    m.def("init", &SFEM_init);
    m.def("finalize", &SFEM_finalize);
    nb::class_<Mesh>(m, "Mesh")  //
        .def(nb::init<>())
        .def("read", &Mesh::read)
        .def("write", &Mesh::write)
        .def("n_nodes", &Mesh::n_nodes)
        .def("convert_to_macro_element_mesh", &Mesh::convert_to_macro_element_mesh)
        .def("spatial_dimension", &Mesh::spatial_dimension);

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
        .def("n_dofs", &FunctionSpace::n_dofs);

    nb::class_<Op>(m, "Op");
    m.def("create_op", &Factory::create_op);

    nb::class_<Output>(m, "Output")
        .def("set_output_dir", &Output::set_output_dir)
        .def("write", &Output::write)
        .def("write_time_step", &Output::write_time_step);

    m.def("write_time_step",
          [](std::shared_ptr<Output> &out,
             const char *name,
             const isolver_scalar_t t,
             nb::ndarray<isolver_scalar_t> x) { out->write_time_step(name, t, x.data()); });

    m.def("set_field",
          [](std::shared_ptr<Op> &op,
             const char *name,
             const int component,
             nb::ndarray<isolver_scalar_t> v) {
              size_t n = v.size();
              auto c_v = (isolver_scalar_t *)malloc(n * sizeof(isolver_scalar_t));
              memcpy(c_v, v.data(), n * sizeof(isolver_scalar_t));
              op->set_field(name, component, c_v);
          });

    m.def("hessian_diag",
          [](std::shared_ptr<Op> &op,
             nb::ndarray<isolver_scalar_t> x,
             nb::ndarray<isolver_scalar_t> d) { op->hessian_diag(x.data(), d.data()); });

    m.def("hessian_diag",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<isolver_scalar_t> x,
             nb::ndarray<isolver_scalar_t> d) { fun->hessian_diag(x.data(), d.data()); });

    nb::class_<Function>(m, "Function")
        .def(nb::init<std::shared_ptr<FunctionSpace>>())
        .def("add_operator", &Function::add_operator)
        .def("add_dirichlet_conditions", &Function::add_dirichlet_conditions)
        .def("set_output_dir", &Function::set_output_dir)
        .def("output", &Function::output);

    m.def("diag", [](nb::ndarray<isolver_scalar_t> d) -> std::shared_ptr<Operator_t> {
        auto op = std::make_shared<LambdaOperator<isolver_scalar_t>>(
            d.size(), d.size(), [=](const isolver_scalar_t *const x, isolver_scalar_t *const y) {
                ptrdiff_t n = d.size();
                const isolver_scalar_t *d_ = d.data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    y[i] = d_[i] * x[i];
                }
            });

        return op;
    });

    m.def("apply",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<isolver_scalar_t> x,
             nb::ndarray<isolver_scalar_t> h,
             nb::ndarray<isolver_scalar_t> y) { fun->apply(x.data(), h.data(), y.data()); });

    m.def("value",
          [](std::shared_ptr<Function> &fun, nb::ndarray<isolver_scalar_t> x) -> isolver_scalar_t {
              isolver_scalar_t value = 0;
              fun->value(x.data(), &value);
              return value;
          });

    m.def("gradient",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<isolver_scalar_t> x,
             nb::ndarray<isolver_scalar_t> y) { fun->gradient(x.data(), y.data()); });

    m.def("apply_constraints", [](std::shared_ptr<Function> &fun, nb::ndarray<isolver_scalar_t> x) {
        fun->apply_constraints(x.data());
    });

    m.def("apply_zero_constraints",
          [](std::shared_ptr<Function> &fun, nb::ndarray<isolver_scalar_t> x) {
              fun->apply_zero_constraints(x.data());
          });

    m.def("constraints_gradient",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<isolver_scalar_t> x,
             nb::ndarray<isolver_scalar_t> g) { fun->constraints_gradient(x.data(), g.data()); });

    m.def("report_solution", [](std::shared_ptr<Function> &fun, nb::ndarray<isolver_scalar_t> x) {
        fun->report_solution(x.data());
    });

    nb::class_<DirichletConditions>(m, "DirichletConditions")
        .def(nb::init<std::shared_ptr<FunctionSpace>>());

    nb::class_<NeumannConditions, Op>(m, "NeumannConditions")
        .def(nb::init<std::shared_ptr<FunctionSpace>>());

    m.def("add_condition",
          [](std::shared_ptr<DirichletConditions> &dc,
             nb::ndarray<isolver_idx_t> idx,
             const int component,
             const isolver_scalar_t value) {
              size_t n = idx.size();
              auto c_idx = (isolver_idx_t *)malloc(n * sizeof(isolver_idx_t));
              memcpy(c_idx, idx.data(), n * sizeof(isolver_idx_t));

              dc->add_condition(n, n, c_idx, component, value);
          });

    m.def("apply_value",
          [](std::shared_ptr<DirichletConditions> &dc,
             isolver_scalar_t value,
             nb::ndarray<isolver_scalar_t> y) { dc->apply_value(value, y.data()); });

    m.def("add_condition",
          [](std::shared_ptr<NeumannConditions> &nc,
             nb::ndarray<isolver_idx_t> idx,
             const int component,
             const isolver_scalar_t value) {
              size_t n = idx.size();
              auto c_idx = (isolver_idx_t *)malloc(n * sizeof(isolver_idx_t));
              memcpy(c_idx, idx.data(), n * sizeof(isolver_idx_t));

              nc->add_condition(n, n, c_idx, component, value);
          });

    nb::class_<Operator_t>(m, "Operator");
    m.def("make_op",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<isolver_scalar_t> u) -> std::shared_ptr<Operator_t> {
              return sfem::make_op<isolver_scalar_t>(
                  u.size(),
                  u.size(),
                  [=](const isolver_scalar_t *const x, isolver_scalar_t *const y) {
                      memset(y, 0, u.size() * sizeof(isolver_scalar_t));
                      fun->apply(u.data(), x, y);
                  });
          });

    nb::class_<ConjugateGradient_t>(m, "ConjugateGradient")
        .def(nb::init<>())
        .def("default_init", &ConjugateGradient_t::default_init)
        .def("set_op", &ConjugateGradient_t::set_op)
        .def("set_preconditioner_op", &ConjugateGradient_t::set_preconditioner_op)
        .def("set_max_it", &ConjugateGradient_t::set_max_it);

    m.def("apply",
          [](std::shared_ptr<ConjugateGradient_t> &cg,
             nb::ndarray<isolver_scalar_t> x,
             nb::ndarray<isolver_scalar_t> y) {
              size_t n = x.size();
              cg->apply(n, x.data(), y.data());
          });
}
