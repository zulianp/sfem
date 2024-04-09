#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <mpi.h>
#include <__nullptr>
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

NB_MODULE(pysfem, m) {
    using namespace sfem;

    m.def("init", &SFEM_init);
    nb::class_<Mesh>(m, "Mesh")  //
        .def(nb::init<>())
        .def("read", &Mesh::read)
        .def("convert_to_macro_element_mesh", &Mesh::convert_to_macro_element_mesh)
        .def("spatial_dimension", &Mesh::spatial_dimension);

        // return nb::ndarray<nb::numpy, const float, nb::shape<2, -1>>(
        //            /* data = */ data,
        //            /* ndim = */ 2,
        //            /* shape pointer = */ shape,
        //            /* owner = */ nb::handle());

    nb::class_<FunctionSpace>(m, "FunctionSpace")
        .def(nb::init<std::shared_ptr<Mesh>>())
        .def(nb::init<std::shared_ptr<Mesh>, const int>())
        .def("n_dofs", &FunctionSpace::n_dofs);

    nb::class_<Op>(m, "Op");
    m.def("create_op", &Factory::create_op);

    nb::class_<Function>(m, "Function")
        .def(nb::init<std::shared_ptr<FunctionSpace>>())
        .def("add_operator", &Function::add_operator)
        .def("add_dirichlet_conditions", &Function::add_dirichlet_conditions)
        .def("set_output_dir", &Function::set_output_dir);

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
              size_t n = idx.shape(0);
              auto c_idx = (idx_t *)malloc(n * sizeof(isolver_idx_t));
              memcpy(c_idx, idx.data(), n * sizeof(isolver_idx_t));

              dc->add_condition(n, n, c_idx, component, value);
          });

    m.def("add_condition",
          [](std::shared_ptr<NeumannConditions> &nc,
             nb::ndarray<isolver_idx_t> idx,
             const int component,
             const isolver_scalar_t value) {
              size_t n = idx.shape(0);
              auto c_idx = (idx_t *)malloc(n * sizeof(isolver_idx_t));
              memcpy(c_idx, idx.data(), n * sizeof(isolver_idx_t));

              nc->add_condition(n, n, c_idx, component, value);
          });

    using Operator_t = Operator<isolver_scalar_t>;

    nb::class_<Operator_t>(m, "Operator");
    m.def("make_op",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<isolver_scalar_t> u) -> std::shared_ptr<Operator_t> {
              auto ret = std::make_shared<Operator_t>();

              ret->apply = [=](const isolver_scalar_t *const x, isolver_scalar_t *const y) {
                  memset(y, 0, u.shape(0) * sizeof(isolver_scalar_t));
                  fun->apply(u.data(), x, y);
              };

              return ret;
          });

    using ConjugateGradient_t = sfem::ConjugateGradient<isolver_scalar_t>;

    nb::class_<ConjugateGradient_t>(m, "ConjugateGradient")
        .def(nb::init<>())
        .def("default_init", &ConjugateGradient_t::default_init)
        .def("set_op", &ConjugateGradient_t::set_op)
        .def("set_max_it", &ConjugateGradient_t::set_max_it);

    m.def("apply",
          [](std::shared_ptr<ConjugateGradient_t> &cg,
             nb::ndarray<isolver_scalar_t> x,
             nb::ndarray<isolver_scalar_t> y) {
              size_t n = x.shape(0);
              cg->apply(n, x.data(), y.data());
          });
}
