The action of the kernels must be tested against a simple harcoded python version (the results must coincide). Use a reference element as an input as well as a deformed one, specialize the test for neohookean and test the operator with zero displacement and shear displacement. Tests include TRI3, TET4, and HEX8 versions of the kernels.


generate_neohookean_ogden_files.py must generate the code for HEX27 as well, the files should be prefixed based on the element type. The local versions must work for multiple element types, the distinction there should be betweem 1D, 2D, 3D and tensor-product families.


Refactor the code generation so that SfemKernelDiagnostics has its own file and it is included in the kernel files
also use namespace sfem::codegen as a namespace and rename SfemKernelDiagnostics to KernelDiagnostics


Maintain a minimal high-level code-gen framework usage example of an incompressible mooney-rivlin material in python/codegen/framework/docs/ the example includes a python file mooney_rivlin.py, mooney_rivlin.md with the exaplanation, mooney_rivlin.sh to run the code-gen and compile the code to a shared object.


The log function is missing from the ExpressionCost? Make sure to include also trigonometric functions


pow(x, y) substituion with sepcialized inline function pow_y(x) (e.g., `pow_2(x) { return x*x}`)  is missing (pow is no efficient with CUDA). Such functions should be available in a header included in the kernel files


The code now assumes that the elements are affine (as there is one jacobian per element), this is good and we should keep it. At the same time we should support iso-parametric elements where we pass the x,y,z coordinates and evaluate geometric jacobian related quantities per quadrature point. This should not affect local functions, as these quantities can be precomputed outside.


The deformation gradient does not have to be a forced intermediate evaluation, let sympy figure it out


scalar_t should be a template parameter of the local kernels, accumulator_t is not necessary just use scalar_t there as well


The mesh level loops should reflect the API used in SFEM where we can pass the coefficient vectors and have gather/scatter phases around the assembly. The vectorized buffers are constructed on the fly before calling the local micro-kernels.
Specialize for affine elements (one Jacobian per element) and iso-parametric elements (jacobians are computed on the fly from coordinates).


Set up a benchmark using the sfem/smesh Mesh to generate the mesh and report the throughputs and dof rates and solve a basic problem with Newton/CG 



