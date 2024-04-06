#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <mpi.h>
#include <__nullptr>
#include <memory>

#include "sfem_Function.hpp"
#include "sfem_base.h"

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
        .def("read", &Mesh::read);

    nb::class_<FunctionSpace>(m, "FunctionSpace")
        .def(nb::init<std::shared_ptr<Mesh>>())
        .def(nb::init<std::shared_ptr<Mesh>, const int>())
        .def("n_dofs", &FunctionSpace::n_dofs);

    nb::class_<Op>(m, "Op");
    m.def("create_op", &Factory::create_op);

    nb::class_<Function>(m, "Function")
        .def(nb::init<std::shared_ptr<FunctionSpace>>())
        .def("add_operator", &Function::add_operator);

    m.def("apply",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<real_t> x,
             nb::ndarray<real_t> h,
             nb::ndarray<real_t> y) { fun->apply(x.data(), h.data(), y.data()); });

    m.def("gradient",
          [](std::shared_ptr<Function> &fun,
             nb::ndarray<real_t> x,
             nb::ndarray<real_t> y) { fun->gradient(x.data(), y.data()); });
}
