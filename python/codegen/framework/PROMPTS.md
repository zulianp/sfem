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

For the affine geometry: when the kernel allows it try to move geometry related computations to preprocessing, for instance see FFF in SFEM laplace operators. This uses the properties of dot products, tensor contractions, etc... to perform operand rearrangement

Create a script that I can run to generate and (re-)install all the materials from python/codegen/framework/materials  including their C++ wrapper in frontend/ops/generated/ 

In framework create folder tests (where all the test should be moved)

Lets revisit naming convetions (if incomplete stop and let me know what is missing)

Naming conventions for kernels 
`<material_name>_<elem_type>_<a|i>_<objective|gradient|hessian_apply|...>[_block_<var_name>][other_qualifiers]`

elem_type is one if the formulation is standard Galerkin, <trial>_<test> if Petrov-Galerkin
a := affine
i := isoparametric

naming conversion for micro-kernels (local)
`<material_name>_<family>_d<dim>_<objective|gradient|hessian_apply|...>[_block_<var_name>][other_qualifiers]`

dim := 1|2|3
family := simplex|tensor_product

add generation for PROTEUS discretization (smesh element types)

<!-- SMESH -->


<!-- SFEM -->

The generated Op subclass (e.g., GeneratedNeoHookeanOgden) should allow to choose (independently) for affine version of the objective, gradient, and hessian action
The generated Op subclass should also include the create_from_yaml function (see sfem_LinearElasticity.hpp) to include  the model parameters