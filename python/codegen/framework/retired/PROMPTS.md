<!-- ENERGY BASED MATERIALS -->

The element type is used twice in function and variable names (e.g., tet10_tet10_), use it once if standard Galerkin. For Petrov-Galerkin you can specify both trial and test.


The mesh-level kernel versions where we pass all the fields in SoA (e.g., generated_neohookean_ogden_tet10_tet10_objective_soa[_impl], ) are not needed anymore. Keep only the versions where we pass the fields as in SFEM (e.g., ux,uy,uz)


Diagnostic print function such as print_generated_rate should be part of the kernel_diagnostic header and specialized calls in the generated mesh-level kernel files (with the least amount of parameter necessary)


Create a script codegen_perf.py to run and extract performance metrics with llvm-mca, also generate the assembly listings for AVX512 architectures to be put in dedicated folders


Generate the cpp OOP wrapper inheriting from sfem::Op, see sfem_NeoHookeanOgden.hpp for one example, an call the generated kernels accordingly


@materials Create the material for The Holzapfel-Gasser-Ogden (HGO) strain energy function workflow (see neoohokean_ogden.py for reference)


The code gen framework is too fragmented and there are too many redundancies. The design should be revisited as follows:
- Symbolic Layer: 
    - Expressions and abstractions should have classes and fully compatible with  SymPy
    - The asthetic style should follow UFL (see paper Unified Form Language: A Domain-Specific Language for Weak Formulations of Partial Differential Equations), although we can inject extra qualifiers (e.g., for hyperelasticity to inform the code generator)
    - The user can create systems of equations starting from merit/energy, residual/gradient (wording different but implementation is exactly the same in the layer below)
    - Input: user UFL-style spec
    - Output: EquationSystem complete with 0-, 1-, 2- forms (automated)
- Form Manipulation Layer
    - Forms are manipulated based on automated policies: 
        - Mesh-level kernel
            1. Jacobian computation (for iso-parametric use sum-factorisation) or routing (for affine Jacobin adjugate and determinant are passed as input, 1 per element)
        - Local kernel
            1. Values, Gradients, ... are computed (reference). Use (for iso-parametric use sum-factorisation)
            2. Transformations using the Adjugate are performed
            3. Material computations using energy/merit, gradient/residual, hessian/jacobian, the style is the left-operand style (already in-place)
            4. If requied apply test-based contactions (for iso-parametric use sum-factorisation)
    - Input: Equation system with Standardized form collection (with qualifiers)
    - Ouput: Generation plan according to steps
- Code generation Layer (input unified system of equations expression graphs)
    - The code generation is general and works for any combination of equations with a specific plan
    - In case of block-systems (multphyiscs) the code generated is both monolithic and the separate blocks
    - Input: plan + platforms (OpenMP, CUDA, etc..)
    - Output: generated code kernels and OOP wrappers

    
Lets create classes for each of the following 


The Jacobian computed from the points per quadrature point for tensor-product elements must use sum factorization

<!-- RESIDUAL BASED MATERIALS -->

<!-- BOTH  -->

Unify abstraction for energy and residual based code generation, note that the pipelines ar paralleled as follows

User input | energy   | residual (variational)
0-Form     | energy   | merit function
1-Form     | gradient | residual
2-From     | hessian  | jacobian

- Remove code duplication in code generator, organize stateful parts into proper classes
- Create separate files form FEM specifics (fem.py) where basis functions, gradients, are specified
- Create dedicated classes in separate files for the target platform (e.g., OpenMP and CUDA)



We still have seprate paths for the hyperleasiticty and two-phase flow. The framework should always be aware that there might be coupled physics in any of the expressions passed by the user, the unified framework should be able to deal with this, un order to do this make this lets implement Poro-hyperelasticity for the unified code-generation framework.
Do the necessary redesigns and refactor.



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





energy.energy, should be energy.add_energy as we may want to add multiple energies, e.g., for contact and friction. Also the signature must allow to have a tuple of differentiating variables matching the fields, (e.g., variables=(F, ) in neo hookean)



Constants in the kernels and micro-kernles need to be casted to scalar_t to avoid unwanted implicit conversions