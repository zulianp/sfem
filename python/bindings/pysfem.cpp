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
#include "sfem_SSMultigrid.hpp"
#include "sfem_ShiftedPenalty.hpp"
#include "sfem_ShiftedPenaltyMultigrid.hpp"

// Add missing includes
#include "openmp/sfem_openmp_blas.hpp"
#include "sfem_Grid.hpp"
#include "sfem_Input.hpp"
#include "sfem_ssmgc.hpp"
#include "sfem_tpl_blas.hpp"

namespace nb = nanobind;

void SFEM_init() {
    char   name[] = "SFEM_init";
    char **argv   = (char **)malloc(sizeof(char *));
    argv[0]       = name;
    int argc      = 1;

    MPI_Init(&argc, (char ***)&argv);

    free(argv);
}

void SFEM_finalize() { MPI_Finalize(); }

NB_MODULE(pysfem, m) {
    using namespace sfem;

    using LambdaOperator_t          = sfem::LambdaOperator<real_t>;
    using MatrixFreeLinearSolver_t  = sfem::MatrixFreeLinearSolver<real_t>;
    using Operator_t                = sfem::Operator<real_t>;
    using Multigrid_t               = sfem::Multigrid<real_t>;
    using ShiftedPenalty_t          = sfem::ShiftedPenalty<real_t>;
    using ShiftedPenaltyMultigrid_t = sfem::ShiftedPenaltyMultigrid<real_t>;
    using ConjugateGradient_t       = sfem::ConjugateGradient<real_t>;
    using BiCGStab_t                = sfem::BiCGStab<real_t>;
    using Chebyshev3_t              = sfem::Chebyshev3<real_t>;
    using MPRGP_t                   = sfem::MPRGP<real_t>;
    using IdxBuffer2D               = sfem::Buffer<idx_t *>;
    using IdxBuffer                 = sfem::Buffer<idx_t>;
    using CountBuffer               = sfem::Buffer<count_t>;
    using SSMGC_t                   = sfem::SSMGC<real_t>;

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

    // Add Grid class bindings with wrapper functions
    nb::class_<Grid<geom_t>>(m, "Grid")
            .def("create_from_file", [](const std::string &path) { return Grid<geom_t>::create_from_file(sfem::Communicator::wrap(MPI_COMM_WORLD), path); })
            .def("create",
                 [](const ptrdiff_t nx,
                    const ptrdiff_t ny,
                    const ptrdiff_t nz,
                    const geom_t    xmin,
                    const geom_t    ymin,
                    const geom_t    zmin,
                    const geom_t    xmax,
                    const geom_t    ymax,
                    const geom_t    zmax) {
                     return Grid<geom_t>::create(sfem::Communicator::wrap(MPI_COMM_WORLD), nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax);
                 })
            .def("to_file", &Grid<geom_t>::to_file)
            .def("extent", &Grid<geom_t>::extent)
            .def("size", &Grid<geom_t>::size)
            .def("spatial_dimension", &Grid<geom_t>::spatial_dimension)
            .def("block_size", &Grid<geom_t>::block_size)
            .def("buffer", &Grid<geom_t>::buffer)
            .def("data", &Grid<geom_t>::data);

    // Add Sideset class bindings with wrapper functions
    nb::class_<Buffer<int16_t>>(m, "Int16Buffer").def("size", &Buffer<int16_t>::size).def("data", [](const Buffer<int16_t> &buf) {
        return nb::ndarray<const int16_t>(buf.data(), {buf.size()});
    });

    nb::class_<Sideset>(m, "Sideset")
            .def("create_from_file",
                 [](const std::string &path) { return Sideset::create_from_file(sfem::Communicator::wrap(MPI_COMM_WORLD), path.c_str()); })
            .def("create",
                 [](const std::shared_ptr<Buffer<element_idx_t>> &parent, const std::shared_ptr<Buffer<int16_t>> &lfi) {
                     return Sideset::create(sfem::Communicator::wrap(MPI_COMM_WORLD), parent, lfi);
                 })
            .def_static("create_from_selector",
                        [](const std::shared_ptr<Mesh> &mesh, nb::callable selector) {
                            auto cpp_selector = [selector](geom_t x, geom_t y, geom_t z) -> bool {
                                return nb::cast<bool>(selector(x, y, z));
                            };
                            return Sideset::create_from_selector(mesh, cpp_selector);
                        })
            .def("size", &Sideset::size)
            .def("node_indices",
                 [](const std::shared_ptr<Sideset> &sideset, const std::shared_ptr<FunctionSpace> &fs) {
                     return create_nodeset_from_sideset(fs, sideset);
                 })
            .def("parent", &Sideset::parent)
            .def("lfi", &Sideset::lfi);

    // Add Input class binding
    nb::class_<Input>(m, "Input");

    // Add YAMLNoIndent class bindings
    nb::class_<YAMLNoIndent>(m, "YAMLNoIndent")
            .def_static("create_from_file", &YAMLNoIndent::create_from_file)
            .def("parse", [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &input) { return yaml->parse(input); })
            .def("get",
                 [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, ptrdiff_t &val) { return yaml->get(key, val); })
            .def("get", [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, int &val) { return yaml->get(key, val); })
            .def("get",
                 [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, double &val) { return yaml->get(key, val); })
            .def("get",
                 [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, std::string &val) {
                     return yaml->get(key, val);
                 })
            .def("require",
                 [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, ptrdiff_t &val) {
                     return yaml->require(key, val);
                 })
            .def("require",
                 [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, int &val) { return yaml->require(key, val); })
            .def("require",
                 [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, double &val) { return yaml->require(key, val); })
            .def("require",
                 [](std::shared_ptr<YAMLNoIndent> &yaml, const std::string &key, std::string &val) {
                     return yaml->require(key, val);
                 })
            .def("key_exists", &YAMLNoIndent::key_exists);

    // Add create_ssmgc function bindings
    m.def("create_ssmgc",
          [](const std::shared_ptr<Function> &f, const std::shared_ptr<ContactConditions> &contact_conds)
                  -> std::shared_ptr<SSMGC_t> { return sfem::create_ssmgc(f, contact_conds, nullptr); });

    m.def("create_ssmgc",
          [](const std::shared_ptr<Function>          &f,
             const std::shared_ptr<ContactConditions> &contact_conds,
             const std::shared_ptr<Input> &in) -> std::shared_ptr<SSMGC_t> { return sfem::create_ssmgc(f, contact_conds, in); });

    // Add create_dirichlet_conditions function binding
    m.def("create_dirichlet_conditions",
          [](const std::shared_ptr<FunctionSpace> &space,
             nb::list                              conditions_list,
             const enum ExecutionSpace             es) -> std::shared_ptr<Constraint> {
              std::vector<DirichletConditions::Condition> conditions;
              for (size_t i = 0; i < conditions_list.size(); i++) {
                  conditions.push_back(nb::cast<DirichletConditions::Condition>(conditions_list[i]));
              }
              return sfem::create_dirichlet_conditions(space, conditions, es);
          });

    // Add create_shifted_penalty function bindings
    m.def("create_shifted_penalty",
          [](const std::shared_ptr<Function> &f, const std::shared_ptr<ContactConditions> &contact_conds)
                  -> std::shared_ptr<ShiftedPenalty_t> { return sfem::create_shifted_penalty(f, contact_conds, nullptr); });

    m.def("create_shifted_penalty",
          [](const std::shared_ptr<Function>          &f,
             const std::shared_ptr<ContactConditions> &contact_conds,
             const std::shared_ptr<Input>             &in) -> std::shared_ptr<ShiftedPenalty_t> {
              return sfem::create_shifted_penalty(f, contact_conds, in);
          });

    // Add BLAS_Tpl class binding
    nb::class_<BLAS_Tpl<real_t>>(m, "BLAS_Tpl")
            .def("good", &BLAS_Tpl<real_t>::good)
            .def("zeros",
                 [](BLAS_Tpl<real_t> &blas, std::shared_ptr<sfem::Buffer<real_t>> x) { blas.zeros(x->size(), x->data()); })
            .def("values", [](BLAS_Tpl<real_t> &blas, const real_t value, std::shared_ptr<sfem::Buffer<real_t>> x) {
                blas.values(x->size(), value, x->data());
            });

    // Add blas function binding
    m.def("blas", [](const enum ExecutionSpace es) -> BLAS_Tpl<real_t> {
        BLAS_Tpl<real_t> blas;
        OpenMP_BLAS<real_t>::build_blas(blas);
        return blas;
    });

    nb::class_<Mesh>(m, "Mesh")  //
            .def(nb::init<>())
            .def("read", &Mesh::read)
            .def("write", &Mesh::write)
            .def("n_nodes", &Mesh::n_nodes)
            .def("n_elements", &Mesh::n_elements)
            .def("convert_to_macro_element_mesh", &Mesh::convert_to_macro_element_mesh)
            .def("spatial_dimension", &Mesh::spatial_dimension);

    m.def("mesh_connectivity_from_file", [](const char *folder) -> std::shared_ptr<IdxBuffer2D> {
        return sfem::mesh_connectivity_from_file(sfem::Communicator::world(), folder);
    });

    nb::class_<IdxBuffer2D>(m, "IdxBuffer2D");
    nb::class_<sfem::Buffer<int>>(m, "Buffer<int>")
            .def(nb::init<>())
            .def("data", [](sfem::Buffer<int> &b) { return b.data(); })
            .def("size", &sfem::Buffer<int>::size);
    nb::class_<sfem::Buffer<long>>(m, "Buffer<long>")
            .def(nb::init<>())
            .def("data", [](sfem::Buffer<long> &b) { return b.data(); })
            .def("size", &sfem::Buffer<long>::size);
    nb::class_<sfem::Buffer<float>>(m, "Buffer<float>")
            .def(nb::init<>())
            .def("data", [](sfem::Buffer<float> &b) { return b.data(); })
            .def("size", &sfem::Buffer<float>::size);
    nb::class_<sfem::Buffer<double>>(m, "Buffer<double>")
            .def(nb::init<>())
            .def("data", [](sfem::Buffer<double> &b) { return b.data(); })
            .def("size", &sfem::Buffer<double>::size);

    m.def("len", [](const std::shared_ptr<sfem::Buffer<int>> &b) -> size_t { return b->size(); });

    m.def("len", [](const std::shared_ptr<sfem::Buffer<double>> &b) -> size_t { return b->size(); });

    m.def("create_real_buffer",
          [](const ptrdiff_t n) -> std::shared_ptr<sfem::Buffer<real_t>> { return sfem::create_host_buffer<real_t>(n); });

    m.def("numpy_view", [](const std::shared_ptr<sfem::Buffer<int>> &b) -> nb::ndarray<nb::numpy, int> {
        return nb::ndarray<nb::numpy, int>(b->data(), {(size_t)b->size()}, nb::handle());
    });

    m.def("numpy_view", [](const std::shared_ptr<sfem::Buffer<long>> &b) -> nb::ndarray<nb::numpy, long> {
        return nb::ndarray<nb::numpy, long>(b->data(), {(size_t)b->size()}, nb::handle());
    });

    m.def("numpy_view", [](const std::shared_ptr<sfem::Buffer<double>> &b) -> nb::ndarray<nb::numpy, double> {
        return nb::ndarray<nb::numpy, double>(b->data(), {(size_t)b->size()}, nb::handle());
    });

    m.def("numpy_view", [](const std::shared_ptr<sfem::Buffer<float>> &b) -> nb::ndarray<nb::numpy, float> {
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
          [](const char                           *elem_type_name,
             std::shared_ptr<sfem::Buffer<idx_t>>  idx,
             std::shared_ptr<sfem::Buffer<geom_t>> p) -> std::shared_ptr<Mesh> {
              size_t        n            = idx->size();
              enum ElemType element_type = type_from_string(elem_type_name);

              int       nnxe              = p->size() / n;  // Assuming p is a flattened array
              ptrdiff_t nelements         = n;
              int       spatial_dimension = 3;  // Assuming 3D
              ptrdiff_t nnodes            = p->size() / spatial_dimension;

              auto elements = sfem::create_host_buffer<idx_t>(nnxe, nelements);
              auto points   = sfem::create_host_buffer<geom_t>(spatial_dimension, nnodes);

              auto d_elements = elements->data();
              auto d_points   = points->data();

              for (int d = 0; d < nnxe; d++) {
                  for (ptrdiff_t i = 0; i < nelements; i++) {
                      d_elements[d][i] = idx->data()[d * nelements + i];
                  }
              }

              for (int d = 0; d < spatial_dimension; d++) {
                  for (ptrdiff_t i = 0; i < nnodes; i++) {
                      d_points[d][i] = p->data()[d * nnodes + i];
                  }
              }

              // Transfer ownership to mesh
              return std::make_shared<Mesh>(sfem::Communicator::world(), spatial_dimension, element_type, nelements, elements, nnodes, points);
          });

    m.def("points", [](std::shared_ptr<Mesh> &mesh, int coord) -> nb::ndarray<nb::numpy, const geom_t> {
        return nb::ndarray<nb::numpy, const geom_t>(mesh->points(coord), {(size_t)mesh->n_nodes()}, nb::handle());
    });

    nb::class_<FunctionSpace>(m, "FunctionSpace")
            .def(nb::init<std::shared_ptr<Mesh>>())
            .def(nb::init<std::shared_ptr<Mesh>, const int>())
            .def("derefine", &FunctionSpace::derefine)
            .def("mesh", &FunctionSpace::mesh_ptr)
            .def("n_dofs", &FunctionSpace::n_dofs)
            .def("block_size", &FunctionSpace::block_size)
            .def("promote_to_semi_structured", &FunctionSpace::promote_to_semi_structured)
            .def(
                    "semi_structured_mesh",
                    [](FunctionSpace &self) -> SemiStructuredMesh * { return &self.semi_structured_mesh(); },
                    nb::rv_policy::reference);

    m.def("create_derefined_crs_graph", [](const std::shared_ptr<FunctionSpace> &space) -> std::shared_ptr<CRSGraph> {
        return sfem::create_derefined_crs_graph(*space);
    });

    m.def("create_edge_idx", [](const std::shared_ptr<CRSGraph> &crs_graph) -> std::shared_ptr<Buffer<idx_t>> {
        return sfem::create_edge_idx(*crs_graph);
    });

    m.def("create_hierarchical_restriction", &sfem::create_hierarchical_restriction);
    m.def("create_hierarchical_prolongation", &sfem::create_hierarchical_prolongation);

    nb::class_<Constraint>(m, "Constraint");
    nb::class_<Op>(m, "Op").def("initialize", &Op::initialize);
    m.def("create_op", [](const std::shared_ptr<FunctionSpace> &space, const char *name, nb::handle es_handle = nb::handle()) {
        if (!es_handle.is_valid()) {
            return Factory::create_op(space, name);
        } else {
            auto es = nb::cast<ExecutionSpace>(es_handle);
            return sfem::create_op(space, name, es);
        }
    });
    m.def("create_boundary_op", &Factory::create_boundary_op);

    nb::class_<Output>(m, "Output")
            .def("set_output_dir", &Output::set_output_dir)
            .def("enable_AoS_to_SoA", &Output::enable_AoS_to_SoA)
            .def("write",
                 [](std::shared_ptr<Output> &out, const char *name, std::shared_ptr<sfem::Buffer<real_t>> x) {
                     out->write(name, x->data());
                 })
            .def("write_time_step", &Output::write_time_step);

    m.def("write_time_step",
          [](std::shared_ptr<Output> &out, const char *name, const real_t t, std::shared_ptr<sfem::Buffer<real_t>> x) {
              out->write_time_step(name, t, x->data());
          });

    m.def("write", [](std::shared_ptr<Output> &out, const char *name, std::shared_ptr<sfem::Buffer<real_t>> x) {
        out->write(name, x->data());
    });

    m.def("set_field",
          [](std::shared_ptr<Op> &op, const char *name, const int component, const std::shared_ptr<Buffer<real_t>> &v) {
              op->set_field(name, v, component);
          });

    m.def("hessian_diag", [](std::shared_ptr<Op> &op, std::shared_ptr<Buffer<real_t>> x, std::shared_ptr<Buffer<real_t>> d) {
        op->hessian_diag(x->data(), d->data());
    });

    m.def("hessian_diag",
          [](std::shared_ptr<Function> &fun, std::shared_ptr<Buffer<real_t>> x, std::shared_ptr<Buffer<real_t>> h) {
              fun->hessian_diag(x->data(), h->data());
          });

    m.def("apply",
          [](std::shared_ptr<Function>            &fun,
             std::shared_ptr<sfem::Buffer<real_t>> x,
             std::shared_ptr<sfem::Buffer<real_t>> h,
             std::shared_ptr<sfem::Buffer<real_t>> y) { fun->apply(x->data(), h->data(), y->data()); });

    m.def("hessian_crs",
          [](std::shared_ptr<Function>              fun,
             std::shared_ptr<sfem::Buffer<real_t>>  x,
             std::shared_ptr<sfem::Buffer<count_t>> rowptr,
             std::shared_ptr<sfem::Buffer<idx_t>>   colidx,
             std::shared_ptr<sfem::Buffer<real_t>>  values) {
              fun->hessian_crs(x->data(), rowptr->data(), colidx->data(), values->data());
          });

    m.def("crs_spmv",
          [](std::shared_ptr<sfem::Buffer<count_t>> rowptr,
             std::shared_ptr<sfem::Buffer<idx_t>>   colidx,
             std::shared_ptr<sfem::Buffer<real_t>>  values) -> std::shared_ptr<Operator_t> {
              return sfem::h_crs_spmv(rowptr->size() - 1, rowptr->size() - 1, rowptr, colidx, values, real_t(0));
          });

    m.def("hessian_crs",
          [](std::shared_ptr<sfem::Buffer<count_t>> rowptr,
             std::shared_ptr<sfem::Buffer<idx_t>>   colidx,
             std::shared_ptr<sfem::Buffer<real_t>>  values) -> std::shared_ptr<Operator_t> {
              return sfem::h_crs_spmv(rowptr->size() - 1, rowptr->size() - 1, rowptr, colidx, values, real_t(0));
          });

    nb::class_<CRSGraph>(m, "CRSGraph")
            .def("n_nodes", &CRSGraph::n_nodes)
            .def("nnz", &CRSGraph::nnz)
            .def("rowptr", &CRSGraph::rowptr)
            .def("colidx", &CRSGraph::colidx);

    nb::class_<Function>(m, "Function")
            .def(nb::init<std::shared_ptr<FunctionSpace>>())
            .def("add_operator", &Function::add_operator)
            .def("add_constraint", &Function::add_constraint)
            .def("space", &Function::space)
            .def("add_dirichlet_conditions", &Function::add_dirichlet_conditions)
            .def("set_output_dir", &Function::set_output_dir)
            .def("output", &Function::output)
            .def("crs_graph", &Function::crs_graph)
            .def("apply_constraints",
                 [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x) {
                     fun->apply_constraints(x->data());
                 })
            .def("apply_zero_constraints",
                 [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x) {
                     fun->apply_zero_constraints(x->data());
                 })
            .def("gradient",
                 [](std::shared_ptr<Function>            &fun,
                    std::shared_ptr<sfem::Buffer<real_t>> x,
                    std::shared_ptr<sfem::Buffer<real_t>> g) { fun->gradient(x->data(), g->data()); })
            .def("hessian_diag",
                 [](std::shared_ptr<Function>            &fun,
                    std::shared_ptr<sfem::Buffer<real_t>> x,
                    std::shared_ptr<sfem::Buffer<real_t>> h) { fun->hessian_diag(x->data(), h->data()); })
            .def("report_solution", [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x) {
                fun->report_solution(x->data());
            });

    m.def("diag", [](std::shared_ptr<sfem::Buffer<real_t>> d) -> std::shared_ptr<Operator_t> {
        auto op = std::make_shared<LambdaOperator<real_t>>(
                d->size(),
                d->size(),
                [=](const real_t *const x, real_t *const y) {
                    ptrdiff_t     n  = d->size();
                    const real_t *d_ = d->data();

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        y[i] += d_[i] * x[i];
                    }
                },
                EXECUTION_SPACE_HOST);

        return op;
    });

    m.def("value", [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x) -> real_t {
        real_t value = 0;
        fun->value(x->data(), &value);
        return value;
    });

    m.def("gradient",
          [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x, std::shared_ptr<sfem::Buffer<real_t>> y) {
              fun->gradient(x->data(), y->data());
          });

    m.def("apply_constraints",
          [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x) { fun->apply_constraints(x->data()); });

    m.def("apply_zero_constraints", [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x) {
        fun->apply_zero_constraints(x->data());
    });

    m.def("gradient",
          [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x, std::shared_ptr<sfem::Buffer<real_t>> g) {
              fun->gradient(x->data(), g->data());
          });

    m.def("copy_constrained_dofs",
          [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x, std::shared_ptr<sfem::Buffer<real_t>> g) {
              fun->copy_constrained_dofs(x->data(), g->data());
          });

    m.def("constraints_gradient",
          [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x, std::shared_ptr<sfem::Buffer<real_t>> g) {
              fun->constraints_gradient(x->data(), g->data());
          });

    m.def("report_solution",
          [](std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> x) { fun->report_solution(x->data()); });

    nb::class_<DirichletConditions, Constraint>(m, "DirichletConditions")
            .def(nb::init<std::shared_ptr<FunctionSpace>>())
            .def("apply",
                 [](std::shared_ptr<DirichletConditions> &dc, std::shared_ptr<sfem::Buffer<real_t>> x) {
                     return dc->apply(x->data());
                 })
            .def("add_condition",
                 [](std::shared_ptr<DirichletConditions> &dc,
                    const ptrdiff_t                       local_size,
                    const ptrdiff_t                       global_size,
                    std::shared_ptr<sfem::Buffer<idx_t>>  idx,
                    const int                             component,
                    const real_t value) { dc->add_condition(local_size, global_size, idx->data(), component, value); });

    // Add DirichletConditions::Condition binding
    nb::class_<DirichletConditions::Condition>(m, "DirichletCondition")
            .def(nb::init<>())
            .def_rw("sidesets", &DirichletConditions::Condition::sidesets)
            .def_rw("nodeset", &DirichletConditions::Condition::nodeset)
            .def_rw("values", &DirichletConditions::Condition::values)
            .def_rw("value", &DirichletConditions::Condition::value)
            .def_rw("component", &DirichletConditions::Condition::component);

    nb::class_<AxisAlignedContactConditions, Constraint>(m, "AxisAlignedContactConditions");

    nb::class_<ContactConditions>(m, "ContactConditions")
            .def(nb::init<std::shared_ptr<FunctionSpace>>())
            .def("n_constrained_dofs", &ContactConditions::n_constrained_dofs)
            .def("linear_constraints_op", &ContactConditions::linear_constraints_op)
            .def("linear_constraints_op_transpose", &ContactConditions::linear_constraints_op_transpose)
            .def("init", &ContactConditions::init)
            .def("update",
                 [](std::shared_ptr<ContactConditions> &cc, std::shared_ptr<sfem::Buffer<real_t>> x) { cc->update(x->data()); })
            .def("signed_distance_for_mesh_viz",
                 [](const std::shared_ptr<ContactConditions> &cc,
                    std::shared_ptr<sfem::Buffer<real_t>>     x,
                    std::shared_ptr<sfem::Buffer<real_t>>     gap) { cc->signed_distance_for_mesh_viz(x->data(), gap->data()); })
            .def("full_apply_boundary_mass_inverse",
                 [](std::shared_ptr<ContactConditions>   &cc,
                    std::shared_ptr<sfem::Buffer<real_t>> x,
                    std::shared_ptr<sfem::Buffer<real_t>> y) { cc->full_apply_boundary_mass_inverse(x->data(), y->data()); })
            .def_static("create",
                        [](const std::shared_ptr<FunctionSpace> &fs,
                           const std::shared_ptr<Grid<geom_t>>  &sdf,
                           const std::vector<std::shared_ptr<Sideset>> &sidesets,
                           const enum ExecutionSpace             es) { return ContactConditions::create(fs, sdf, sidesets, es); });

    m.def("signed_distance_for_mesh_viz",
          [](const std::shared_ptr<ContactConditions> &cc,
             std::shared_ptr<sfem::Buffer<real_t>>     x,
             std::shared_ptr<sfem::Buffer<real_t>>     gap) { cc->signed_distance_for_mesh_viz(x->data(), gap->data()); });

    m.def("signed_distance",
          [](const std::shared_ptr<ContactConditions> &cc,
             std::shared_ptr<sfem::Buffer<real_t>>     x,
             std::shared_ptr<sfem::Buffer<real_t>>     y) { cc->signed_distance(x->data(), y->data()); });

    m.def("contact_conditions_from_file",
          [](const std::shared_ptr<FunctionSpace> &space, const char *path) -> std::shared_ptr<ContactConditions> {
              return ContactConditions::create_from_file(space, path, sfem::EXECUTION_SPACE_HOST);
          });

    nb::class_<NeumannConditions, Op>(m, "NeumannConditions").def(nb::init<std::shared_ptr<FunctionSpace>>());

    m.def("apply_value", [](std::shared_ptr<DirichletConditions> &dc, real_t value, std::shared_ptr<sfem::Buffer<real_t>> y) {
        dc->apply_value(value, y->data());
    });

    nb::class_<Operator_t>(m, "Operator")
            .def("__add__",
                 [](const std::shared_ptr<Operator_t> &l, const std::shared_ptr<Operator_t> &r) {
                     assert(l->cols() == r->rows());
                     return l + r;
                 })
            .def("__mul__", [](const std::shared_ptr<Operator_t> &l, const std::shared_ptr<Operator_t> &r) {
                assert(l->cols() == r->rows());

                auto temp = sfem::create_buffer<real_t>(l->rows(), l->execution_space());

                return sfem::make_op<real_t>(
                        l->rows(),
                        r->cols(),
                        [=](const real_t *const x, real_t *const y) {
                            auto      data = temp->data();
                            ptrdiff_t n    = l->rows();
#pragma omp parallel for
                            for (ptrdiff_t i = 0; i < n; i++) {
                                data[i] = 0;
                            }
                            r->apply(x, data);
                            l->apply(data, y);
                        },
                        l->execution_space());
            });

    // Add SSMGC class bindings after Operator class
    nb::class_<SSMGC_t, Operator_t>(m, "SSMGC")
            .def("create", &SSMGC_t::create)
            .def("apply",
                 [](std::shared_ptr<SSMGC_t>             &solver,
                    std::shared_ptr<sfem::Buffer<real_t>> x,
                    std::shared_ptr<sfem::Buffer<real_t>> y) { solver->apply(x->data(), y->data()); });

    m.def("make_op",
          [](const std::shared_ptr<Function> &fun, std::shared_ptr<sfem::Buffer<real_t>> u) -> std::shared_ptr<Operator_t> {
              return sfem::make_op<real_t>(
                      u->size(),
                      u->size(),
                      [=](const real_t *const x, real_t *const y) {
                          memset(y, 0, u->size() * sizeof(real_t));
                          fun->apply(u->data(), x, y);
                      },
                      fun->execution_space());
          });

    m.def("apply",
          [](const std::shared_ptr<Operator_t>    &op,
             std::shared_ptr<sfem::Buffer<real_t>> x,
             std::shared_ptr<sfem::Buffer<real_t>> y) { op->apply(x->data(), y->data()); });

    nb::class_<MatrixFreeLinearSolver_t, Operator_t>(m, "MatrixFreeLinearSolver");

    nb::class_<ConjugateGradient_t, MatrixFreeLinearSolver_t>(m, "ConjugateGradient")
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
             std::shared_ptr<sfem::Buffer<real_t>> x,
             std::shared_ptr<sfem::Buffer<real_t>> y) { cg->apply(x->data(), y->data()); });

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
          [](std::shared_ptr<Chebyshev3_t>        &op,
             std::shared_ptr<sfem::Buffer<real_t>> x,
             std::shared_ptr<sfem::Buffer<real_t>> y) { op->apply(x->data(), y->data()); });

    m.def("apply",
          [](std::shared_ptr<BiCGStab_t> &op, std::shared_ptr<sfem::Buffer<real_t>> x, std::shared_ptr<sfem::Buffer<real_t>> y) {
              op->apply(x->data(), y->data());
          });

    nb::class_<MPRGP_t>(m, "MPRGP")
            .def(nb::init<>())
            .def("default_init", &MPRGP_t::default_init)
            .def("set_op", &MPRGP_t::set_op)
            .def("set_max_it", &MPRGP_t::set_max_it)
            .def("set_verbose", &MPRGP_t::set_verbose)
            .def("set_rtol", &MPRGP_t::set_rtol)
            .def("set_atol", &MPRGP_t::set_atol);

    m.def("set_upper_bound", [](std::shared_ptr<MPRGP_t> &op, std::shared_ptr<sfem::Buffer<real_t>> &x) {
        op->set_upper_bound(sfem::Buffer<real_t>::wrap(x->size(), x->data()));
    });

    m.def("set_upper_bound", [](std::shared_ptr<ShiftedPenalty_t> &op, std::shared_ptr<sfem::Buffer<real_t>> &x) {
        op->set_upper_bound(sfem::Buffer<real_t>::wrap(x->size(), x->data()));
    });

    m.def("apply",
          [](std::shared_ptr<MPRGP_t> &op, std::shared_ptr<sfem::Buffer<real_t>> x, std::shared_ptr<sfem::Buffer<real_t>> y) {
              op->apply(x->data(), y->data());
          });

    nb::class_<Multigrid_t>(m, "Multigrid")
            .def(nb::init<>())
            .def("default_init", &Multigrid_t::default_init)
            .def("set_max_it", &Multigrid_t::set_max_it)
            .def("set_atol", &Multigrid_t::set_atol)
            .def("set_max_it", &Multigrid_t::set_max_it)
            .def("add_level", &Multigrid_t::add_level);

    m.def("apply",
          [](std::shared_ptr<Multigrid_t> &op, std::shared_ptr<sfem::Buffer<real_t>> x, std::shared_ptr<sfem::Buffer<real_t>> y) {
              op->apply(x->data(), y->data());
          });

    nb::class_<ShiftedPenalty_t>(m, "ShiftedPenalty")
            .def(nb::init<>())
            .def("default_init", &ShiftedPenalty_t::default_init)
            .def("set_max_it", &ShiftedPenalty_t::set_max_it)
            .def("set_atol", &ShiftedPenalty_t::set_atol)
            .def("set_max_it", &ShiftedPenalty_t::set_max_it)
            .def("set_penalty_param", &ShiftedPenalty_t::set_penalty_param)
            .def("set_linear_solver", &ShiftedPenalty_t::set_linear_solver)
            .def("set_upper_bound", &ShiftedPenalty_t::set_upper_bound)
            .def("set_lower_bound", &ShiftedPenalty_t::set_lower_bound)
            .def("set_constraints_op", &ShiftedPenalty_t::set_constraints_op)
            .def("set_max_inner_it", &ShiftedPenalty_t::set_max_inner_it)
            .def("enable_steepest_descent", &ShiftedPenalty_t::enable_steepest_descent)
            .def("set_damping", &ShiftedPenalty_t::set_damping)
            .def("set_op", &ShiftedPenalty_t::set_op);

    m.def("apply",
          [](std::shared_ptr<ShiftedPenalty_t>    &op,
             std::shared_ptr<sfem::Buffer<real_t>> x,
             std::shared_ptr<sfem::Buffer<real_t>> y) { op->apply(x->data(), y->data()); });

    nb::class_<ShiftedPenaltyMultigrid_t, Operator_t>(m, "ShiftedPenaltyMultigrid")
            .def(nb::init<>())
            .def("set_max_it", &ShiftedPenaltyMultigrid_t::set_max_it)
            .def("set_upper_bound", &ShiftedPenaltyMultigrid_t::set_upper_bound)
            .def("set_lower_bound", &ShiftedPenaltyMultigrid_t::set_lower_bound)
            .def("set_constraints_op", &ShiftedPenaltyMultigrid_t::set_constraints_op);

    m.def(
            "create_spmg", [](const std::shared_ptr<Function> &f, const enum ExecutionSpace es) -> auto{
                auto spmg = sfem::create_ssmg<ShiftedPenaltyMultigrid_t>(f, es);
                return spmg;
            });

    m.def("set_upper_bound", [](std::shared_ptr<ShiftedPenaltyMultigrid_t> &op, std::shared_ptr<sfem::Buffer<real_t>> &x) {
        op->set_upper_bound(sfem::Buffer<real_t>::wrap(x->size(), x->data()));
    });

    // Add create_hex8_cube function binding
    m.def("create_hex8_cube",
          [](const int    nx   = 1,
             const int    ny   = 1,
             const int    nz   = 1,
             const geom_t xmin = 0,
             const geom_t ymin = 0,
             const geom_t zmin = 0,
             const geom_t xmax = 1,
             const geom_t ymax = 1,
             const geom_t zmax = 1) {
              return Mesh::create_hex8_cube(sfem::Communicator::world(), nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax);
          });

    // Add create_sdf function binding
    m.def("create_sdf",
          [](const int    nx,
             const int    ny,
             const int    nz,
             const geom_t xmin,
             const geom_t ymin,
             const geom_t zmin,
             const geom_t xmax,
             const geom_t ymax,
             const geom_t zmax,
             nb::callable sdf_func) {
              auto cpp_sdf_func = [sdf_func](geom_t x, geom_t y, geom_t z) -> geom_t {
                  return nb::cast<geom_t>(sdf_func(x, y, z));
              };
              return sfem::create_sdf(sfem::Communicator::world(), nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax, cpp_sdf_func);
          });

    // Add SemiStructuredMesh class and export_as_standard binding
    nb::class_<SemiStructuredMesh>(m, "SemiStructuredMesh")
            .def("export_as_standard", &SemiStructuredMesh::export_as_standard)
            .def("apply_hierarchical_renumbering", &SemiStructuredMesh::apply_hierarchical_renumbering)
            .def("level", &SemiStructuredMesh::level)
            .def("__copy__", [](const SemiStructuredMesh &) { throw std::runtime_error("Copy not allowed"); })
            .def("__deepcopy__",
                 [](const SemiStructuredMesh &, nb::handle) { throw std::runtime_error("Deepcopy not allowed"); });

    // Expose the C++ types as Python dtypes
    try {
        nb::module_ numpy = nb::module_::import_("numpy");
        m.attr("real_t")  = numpy.attr(dtype_REAL_T);
        m.attr("idx_t")   = numpy.attr(dtype_IDX_T);
    } catch (const std::exception &) {
        // If numpy is not available, use native nanobind dtypes
        m.attr("real_t") = nb::dtype<real_t>();
        m.attr("idx_t")  = nb::dtype<idx_t>();
    }
}
