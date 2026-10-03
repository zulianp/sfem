# Design of the CVFEM spike code

The code must be lean and clean. Performance first.

The code must be structured as follows withing the `src` foloder


- `kernels` header only self-contained code with templated types and localized macros (no library dependencies allowed, exceptions for CUDA, OpenMP or other wrappers). The style is purely imperative without abstractions, the signatures of the functions are lean-and-mean only arguments that are acually used are passed. No user level option flags are propgated down here (like now), they are handled outside in the front-end
Kernels are organized by mesh format in separate folders, within each folder we have `affine`, `iosparametric` and `axis_aligned` (new placeholder) kernels logically separated (now they are mixed in with enum and booleans). For the matrix-free kenrels only the SIMD version is kept, the rest is moved to subpar. The threading model for atomics free kernels is abstract outside the function and what is passed from outside is a range: `typedef struct { ptrdiff_t begin, end; } range;` (this will allow to use other thread libraries other than OpenMP). The compiler must add vectorization check flags (if the lane loop does not vectorize). cuda version is inside each layout folder within a `cuda` subfolders.
	- `packed`
	- `colored`
	- `standard`
	- `microkernels` avoid branching in micro-kernels `if constexpr` is allowed


- `frontend`
	- `op`

- `cases`
- `drivers` files with the main function user facing
- `tests` organized in subfolders per type of test


.. additional structure where it fits