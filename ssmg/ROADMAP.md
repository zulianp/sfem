# Title (WIP): A GPU Accelerated Shifted--Penalty Multigrid for Contact in Elasticity

SPMG =  Shifted--Penalty Multigrid

## Abstract (structure of work)

## Problem and motivation

- [ ] Unilateral contact
- [ ] Rough surfaces ?
- [ ] CAD -> IVPs
- [ ] ~1G dof size problem (or greater) to capture high-freq details

## Method
 - [ ] Nonlinear Hybrid MG (GMG+AMG)
 - [ ] Shfted penalty for constraints
 - [ ] Nonlinear smoothing

## Discretization
 - [ ] 3D Q1-FEM
 - [ ] Semi-structured discretization for memory efficiency (low order FEM on GPU)
 - [ ] SDF for obstacle ?

## Numerical experiments
 - [ ] Toy verification(/validation) problems
   - [ ] Scalar problem verification (constrained reaction diffusion?)
   - [ ] Vector problem verification (linear elasticity)
 - [ ] Complex scenarios
   - [ ] Rough surfaces
   - [ ] Complex CAD geometries

## Convergence and Performance measurements
 - [ ] !! TTS ??
 - [ ] AMG Perf of matrix formats in cuSPARSE and Matrix-free
 - [ ] AMG vs HGAMG
 - [ ] Hyper-parameters?

## Conclusions
   - [ ] Will it run?

## References
 - [ ] https://arxiv.org/pdf/1002.1859
 - [ ] https://developer.nvidia.com/amgx
 - [ ] https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.5748
 - [ ] https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6680 (AMG, Multi-body contact: 23.9 MDOF on 480 MPI ranks, 2 seconds per linear solver iterations)
 - [ ] ...


# TODO

## Conceptual obstacles

   - [ ] **Problem!** Piecewise constant coarse space is not working for larger AMG fine resolutions in combination with the SPMG and is working poorly with standard AMG (~0.97 convergence rate).
   - [ ] **Solution?**
      1) Do not aggregate dofs on the contact boundary in the AMG (graded or not). No grading: penalty contribution will still be passed down with the GMG restriction but not the AMG one.
      2) Other coarse spaces (Smoothed aggregation?)
      3) GMG with coarse grid solver PCG-AMG (fully linear solving the penalized system)

## Plan-B

   - [ ] Full GMG or variant of SPMG (Simple problem was working with aggressive coarsening cf: ~4K)
   - [ ] Desperation mode: Monotone GMG/AMG

## Milestones

MS := Milestone
() slot for achieved milesone MM/YY (e.g., 02/2025)

### Austen

**MS 1: ()**
   - [x] Baseline piecewise constant prolongation
   - [ ] Try adding some smoothing steps on the near-null to see if it improves things
   - [ ] Try SSMG with MG preconditioned CG on the inner iteration
   - [ ] Adaptive coarsening implementation

**MS 2: ()**

   - [ ] AMGx integration
   - [ ] Implement a more sophisticated interpolation ('classical' modification, smoothed aggregation, and energy minimization)
      a. Will require SPMM implementation and / or 'triple product' i.e. Pt A P
   - [ ] Proof-reading and helping to finalize Paper 1

**MS 3: ()**

   - [ ] Adapt the AMG portion to work for elasticity (or general vector problems, respecting the block size on aggregations and adapt interpolator for block matrices)
   - [ ] Help drafting P2

**MS 4: ()**

   - [ ] Port the AMG to work on GPU using cuSPARSE, probably leave assembly on CPU though
   - [ ] Perf. analysis of different matrix formats
   - [ ] Compare AMG perf with AMGx
   - [ ] Drafting P3

**MS Optional:**

   - [ ] Hypre AMG integration?

###  Patrick

**MS 1: ()**

   - [x] Two-level method matrix-free SPGMG on GPU (basic)
   - [x] Baseline obstacle problem with SDF using Shifted-Penalty (reduced size of constrained dofs and quantities, normal field)
   - [x] GMG: prolongation/restriction on CPU for SSHEX8, and SSQUAD4 (for contact boundary)
   - [x] Basic tracing facilities for timing different parts of the code (CPU only)
   - [x] GMG: Hierarchical indexing for no redundancy multilevel discretization
   - [x] SSHEX8-Mesh online generation from HEX8
   - [x] Multilevel GMG on CPU
   - [x] Contact stresses post-processor for SSHEX8
   - [ ] Provide boundary surface mask for adaptive coarsening
   - [x] Boundary mass-matrix
   - [x] MG with Support non-axis aligned normal fields for contact
   - [ ] Chebyshev smoother
   - [X] Export facilities for Paper 1
   - [ ] Nonlinear obstacle contact loop (Optional)

**MS 2: ()**

   - [ ] Drafting/Finalizing P1
   - [ ] Nonlinear obstacle contact loop (if not done in MS 1)

**MS 3: ()**

   - [ ] GPU porting of all missing discretization routines (Operators, Algebra, SDF sampling?,)
   - [ ] Contact discretization is performed on CPU, handling of GPU-GPU two-way transfers (SDF, normals, ...)
   - [X] GPU: ShiftableBlockSymJacobi
   - [ ] GPU: constraints_mask (minimal)
   - [X] GPU: hessian_block_diag_sym for linear elasticity (HEX8 and SSHEX8)
   - [X] GPU: sshex8_restrict and sshex8_prolongate
   - [X] GPU: ssquad4_restrict and ssquad4_prolongate
   - [ ] GPU: SparseBlockVector

**MS 4: ()**

   - [ ] Optimize memory for shifted penalty multigrid
   - [ ] Optimizing operator applications for elasticity to compete with Laplacian perf
   
**MS (optional):**

   - [ ] (KPZ)[https://arxiv.org/pdf/1002.1859] which NVIDIA uses is a possible alternative
   - [ ] Explore high-order HEXA
   - [ ] Explore sum factorization algorithm for HEX8


### Hardik

**MS 1: ()**

   - [x] MatLab implementation of SPGMG in 2D (scalar problem)
   - [ ] Update strategy of Lagrange multiplier
   - [ ] MatLab implementations for Paper 1 (P1)
   - [ ] Drafting P1

**MS 2:**
   - [ ] Finalizing P1

**MS 3:**

   - [ ] Conceptualizing verification experiments for P2/P3
   - [ ] Help drafting P2

**MS 4:**
   - [ ] Proof-reading P3

### Gabriele

   - [ ] Collects and presents the convergence numbers of the MATLAB-based problems for P1 
   - [ ] Total number of iterations, inner iterations, number of linearizations, total number of smoothing steps, local energy norm, convergence rate (energy?), Penetration norm, difference from reference solution
   - [ ] 2D scalar problems, obstacle problem with source term and obstacle
         1. Create 2D meshes using Trelis scripting (with increasing resolution, max resolution 100K), Square 
         3. Compare with monotone MG
   - [ ] 3D problems
         1. One sphere problem
         2. Multi-sphere problem
         3. Complex surface problem
   - [ ] Show, contact stresses 


### Nice to have (When there is time or requests)

- [ ] YML config for parameters
- [ ] Python front-end

### Finalizing the implementation (MS 5, Everyone)

- [ ] Tracing and tuning
- [ ] Vertical solution (redesign of some parts of the code)
- [ ] Algorithmic Optimizations and parameter tuning

## Paper 1: Shifted--Penalty Multigrid for Variational Inequalities

- [ ] Authorship: Hardik, Austen, Rolf, Panayot, Patrick
- [ ] GMG for solving A x = b, B * x <= g, A in R^{n x n}, B in R^{m x n}
- [ ] Details of optimization algorithm
- [ ] Fully geometric MG
- [ ] Examples in 2D and 3D including constrained Poisson problem, linear elasticity
- [ ] Matlab Implementation
- [ ] Convergence of the method on toy problems (no Perf)
- [ ] Experiements  
   - [ ] Elasticity 1) 2D Axis-aligned contact Matlab
   - [x] Elasticity 2) SDF cube vs half-sphere (s) (stresses), correct stress verification
   - [x] Elasticity 3) Complex showcase (displacement)
- [ ] Solver convergence: number of iterations, residuals
   - [x] Energy norm `|| . ||A`  of correction, convergence rates x is the solution `|| xkp1 - xk ||A / || xk - xkm1||A`
   - [x] Statistics: Number of outer iterations / cycles / smoothing (we keep track of norms per cycle per outer iteration)

- [ ] Post-processor 
   - [ ] Cauchy  stress for HEX8/SSHEX8
   - [ ] Contact stress for HEX8/SSHEX8 (lambda = M^-1 (b - A x))

If rejected content goes to **P2**

## Paper 2: A GPU Accelerated Shifted--Penalty Multigrid for Contact in Elasticity

 - [ ] Authorship: Patrick, Austen, Hardik, Panayot, Rolf
 - [ ] Full paper as described in the abstract above with the best working set-up for the HSPMG

## Paper 3: Either SPAMG or Performance-based paper

- [ ] Authorship: Austen, Patrick, Hardik, Rolf, Panayot
- [ ] Options:
   1) SPAMG
   2) Comparing matrix-based (different formats) and matrix-free operator in the context of **P2** (Either Conference Proceeding or CCF Transactions on High Performance Computing):

## Maybes

- [ ] If Hypre AMG transfer ops work involve also Rui Peng Li?
