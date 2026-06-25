<!-- ENERGY BASED MATERIALS -->

The element type is used twice in function and variable names (e.g., tet10_tet10_), use it once if standard Galerkin. For Petrov-Galerkin you can specify both trial and test.


The mesh-level kernel versions where we pass all the fields in SoA (e.g., generated_neohookean_ogden_tet10_tet10_objective_soa[_impl], ) are not needed anymore. Keep only the versions where we pass the fields as in SFEM (e.g., ux,uy,uz)


Diagnostic print function such as print_generated_rate should be part of the kernel_diagnostic header and specialized calls in the generated mesh-level kernel files (with the least amount of parameter necessary)


Create a script codegen_perf.py to run and extract performance metrics with llvm-mca, also generate the assembly listings for AVX512 architectures to be put in dedicated folders


Generate the cpp OOP wrapper inheriting from sfem::Op, see sfem_NeoHookeanOgden.hpp for one example, an call the generated kernels accordingly


@materials Create the material for The Holzapfel-Gasser-Ogden (HGO) strain energy function workflow (see neoohokean_ogden.py for reference)


<!-- RESIDUAL BASED MATERIALS -->

The residual code generator injects depencies in kernels that are not actually needed (e.g., see the hessian action passing the old/previous timestep quantities). The symbolic framework must make sure that no unneeded quantities are passed to the kernels.

<!-- BOTH  -->

Organize the example materials (e.g., neohookean and twophaseflow) in python/codegen/framework/materials and make them as clean as possible (like user intened code), the infrastructure code should be abstracted away and put it in the API module sfem.gen.


Create a script that I can run to generate and install the materials from python/codegen/framework/materials  including their C++ wrapper in frontend/ops/generated/ 

In framework create folder tests (where all the test should be moved)

<!-- SMESH -->

Note that PROTEUS_HEX27 has a cartesian node ordering, different from HEX27, add HEX27 in smesh since it is needed