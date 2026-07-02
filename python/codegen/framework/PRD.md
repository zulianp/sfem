# Goals


1. The user writes either the system energy or residual
2. The code generator detects patterns (see current content of codegen: e.g., gpu_linear_elasticity_op.py) and uses intermediate symbols to store optimal intermediate evluations
3. Evalutates the expression graph for quantities that should be evaluated or be outside loop scopes (e.g., trial loop vs test loop, quadrature loop or mesh wide loop)
4. Keeps track of the arithmetic intensity and has euristics for register pressure
5. Reference shape functions, gradients, are passed as arrays, the generate code is generic for the dimension (specialized for 1D, 2D, 3D) so that any element can be used with the same math/materials, element related sizes are passed to the kernel as template parameters 
6. For tensor product elements, specializes the codes to use sum-factorization and matrix-units (e.g., tensor-cores, matrix-cores, SME)
7. The code generator targets and specializes for
	- OpenMP
	- CUDA/Hip
	- AVX512
	- ARM SVE and SME
8. It generates
	- Matrix free kernels for
		- Hessian/Jacobian application
		- Gradient application
		- Standard mesh format (see SFEM)
		- Packed mesh format (see SFEM, in particular PackedLaplacian for nearly-optimal implementation style. Data-layouts can still be improved at set-up to fit directly the computational layout). Implement the two pass scheme (as well as the one pass) described in /Users/patrickzulian/Desktop/cloud/owncloud_USI/zulian/scientific_collaborator/papers/packedop_paper/main.tex
		- Generate variants for per thread per-warp optimized set-ups and executation
		- Patch based (overall and with index for specific nodes)
		- Element base (as it is already)
	- Matrix assembly (see how it is done now in SFEM)
		- CRS
		- BSR
		- DIA
		- COO
		- Patch-based assembly with index for specific nodes
	- Objective/Energy evaluation (see value_steps in SFEM), when available, merit function otherwise
	- All the generated kernels will have FLOP counting and arithmetic intensity functions that are used to generate performance analyses autmatically
9. Specializations for hyperelasticity (see sr_hyperleasticity.py and neohookean_partial_assembly.py)
10. Clean and usable software design
11. Generated kernels are in procedural style, with OOP wrapper (as it is now SFEM)
12. Use SoA (priority), AoS, and AoSoSoA 

It is a code-generator, the runs are done by compiling the kernels and running them within the SFEM library

# References

- SFEM for reproducing (and improving) current kernel generation outputs
- HOG: https://i10git.cs.fau.de/hyteg/hog for hybrid grids loops
- ExaStencils: https://github.com/lssfau/ExaStencils for loop manipulation

# Dependencies

- SymPy (for symbolic manipulation)
- NetworkX (for graph manipulation)
-

# Milestones and Tasks

## Guidelines
- Every step expand the NeoHookean Ogden test as a testing ground
- Prioritize the notation objective, gradient, hessian for (NeoHookean Ogden)

## M1: Symbolic front-end and expression graph

- Accept energy, residual, gradient, Jacobian/Hessian-action, and merit-function expressions from SymPy.
- Build a dependency graph with NetworkX for symbolic expressions, intermediates, loops, element data, and kernel outputs.
- Detect repeated subexpressions and material/geometric patterns using existing generators such as `gpu_linear_elasticity_op.py`.
- Emit explicit intermediate symbols for reused values, prioritizing straight-line code that is easy for compilers to vectorize.
- Track expression cost in FLOPs, loads, stores, temporary count, and estimated register pressure.

## M2: Loop scope and data-layout model

- Represent mesh-wide, element, patch, quadrature, trial, test, vector-lane, warp, and thread scopes.
- Hoist invariant expressions to the widest valid scope.
- Model SoA as the default data layout and support AoS and AoSoA as explicit variants.
- Pass reference shape values and gradients as linear arrays.
- Specialize generated code for 1D, 2D, and 3D while keeping element sizes as kernel template parameters.

## M3: CPU back ends

- Generate procedural OpenMP kernels with OOP wrappers matching the current SFEM style.
- Generate AVX512-oriented kernels with unit-stride memory access, branch-free hot loops, and compiler-friendly temporaries.
- Generate ARM SVE and SME variants with vector-length-aware loops and matrix-unit paths where applicable.
- Provide matrix-free kernels for Hessian/Jacobian application and gradient application on standard and packed mesh formats.
- Generate element, patch, per-thread, and per-warp variants where the execution model benefits from them.

## M4: GPU back ends

- Generate CUDA and HIP kernels for standard mesh and packed mesh formats.
- Generate per-thread and per-warp matrix-free variants for Hessian/Jacobian application, gradient application, and patch kernels.
- Specialize tensor-product elements with sum-factorization.
- Use tensor cores, matrix cores, or equivalent matrix units for tensor-product microkernels when precision and shape permit.
- Keep data movement explicit and expose arithmetic intensity for roofline analysis.

## M5: Assembly and objective kernels

- Generate matrix assembly for CRS, BSR, DIA, and COO formats.
- Generate patch-based assembly with optional node-index filtering.
- Generate objective/energy evaluation kernels when an energy is available.
- Generate merit-function kernels when energy evaluation is unavailable.
- Provide FLOP counting and arithmetic-intensity functions for every generated kernel.

## M6: Hyperelasticity support

- Reproduce and improve the current hyperelasticity generators, including `sr_hyperleasticity.py` and `neohookean_partial_assembly.py`.
- Support residual, Jacobian-action, energy, and objective paths for hyperelastic materials.
- Reuse geometric and material intermediates across quadrature, trial, and test loops.
- Validate generated kernels against existing SFEM hyperelasticity workflows.

## M7: Integration, validation, and performance reports

- Compile generated kernels inside the SFEM build and run them through existing SFEM execution paths.
- Compare generated outputs against current hand-written or existing generated kernels.
- Add performance reports with FLOPs, bytes, arithmetic intensity, and achieved throughput.
- Benchmark standard mesh, packed mesh, patch, and tensor-product variants.
- Keep the framework API small, procedural at the generated-code level, and wrapped with SFEM-style OOP interfaces.


## M8: NASA Wall-Mounted Hump
- Implement incompressible Navier Stokes material with Strang scheme using the UFL-like API
- Implement the mesh generator for the Wall-Mounted Hump as Mesh::create_wall_mounted_hump in SMESH, with high-order surface PROTEUS elements
- Implement the executable to set-up and run the simulation