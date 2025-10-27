# SFEM AI Coding Instructions

## Project Architecture

SFEM (Simple FEM) is a high-performance C/C++ finite element library with Python bindings for scientific computing, mesh processing, and resampling operations. The architecture follows a modular design with clear separation of concerns.

### Core Components
- **`base/`** - Core type definitions (`real_t`, `idx_t`, `geom_t`) and base functionality
- **`operators/`** - Element-specific FEM operators (tet4, tet10, hex8, tri3, etc.)
- **`algebra/`** - Linear algebra kernels (SpMV, CG, preconditioners) with OpenMP/CUDA variants
- **`mesh/`** - Mesh I/O, refinement, and topology operations
- **`resampling/`** - Field interpolation between different grids/meshes (CPU/CUDA)
- **`drivers/`** - Command-line executables for specific operations
- **`python/`** - Python bindings and utilities for visualization/workflows

### Build System Architecture
- CMake-based with modular configuration in `cmake/` directory
- Conditional compilation for CUDA, OpenMP, Python bindings
- Driver executables automatically generated from `drivers/` directory
- Key options: `SFEM_ENABLE_CUDA`, `SFEM_ENABLE_OPENMP`, `SFEM_ENABLE_RESAMPLING`

## Development Patterns

### Data Layout and Types
```c
// Primary data types (sfem_base.h)
typedef float geom_t;    // Geometry coordinates
typedef int idx_t;       // Mesh indices  
typedef double real_t;   // Field values
typedef long count_t;    // Large counts
```

### Memory Management
- Raw arrays with explicit ownership (no automatic memory management)
- MPI-aware data structures throughout
- CUDA/OpenMP memory spaces handled explicitly via `mem_space` field

### Element Type System
```c
// Common element types
enum ElemType { TET4=4, TET10=10, HEX8=8, QUAD4=104, TRI3=3, ... }
```

### File I/O Conventions
- **Raw binary format**: Primary data format (`.raw` files)
- **Mesh folders**: Collections of raw files (`x.raw`, `y.raw`, `z.raw`, `i0.raw`, etc.)
- **Python utilities**: `raw_to_db.py`, `raw_to_xdmf.py` for visualization conversion

## Critical Workflows

### Environment Setup
```bash
# Standard workflow initialization (from any script)
source $SCRIPTPATH/../../build/sfem_config.sh
export PATH=$SCRIPTPATH/../../build/:$PATH
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
```

### Build Process
```bash
# Basic build
mkdir build && cd build
cmake .. -DSFEM_ENABLE_OPENMP=ON -DSFEM_ENABLE_CUDA=ON
make -j

# With Python bindings
cmake .. -DSFEM_ENABLE_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
```

### Driver Usage Pattern
```bash
# Most drivers follow: driver input_mesh_or_grid [params] output
grid_to_mesh $sizes $origins $scaling $sdf $mesh $field TET4 CUDA
raw_to_db.py $mesh output.vtk --point_data=$field --point_data_type=float32
```

## Code Conventions
- for if and while statements anfter the closing parenthesis always add a comment line with "END if (description)" or "END while (description)" or similar.

- For the functions always add a comment line at the end with "END Function: function_name" where function_name is the name of the function.

- The function returns the values by using  the macro RETURN_FROM_FUNCTION(ret_val), that prints debug info and returns the value.

- The header of each function body must be in the following format:
```c
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// function_name
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
function_name
```



### Naming Patterns
- **Functions**: `snake_case` with element prefix (`tet4_assemble_laplacian`)
- **Files**: Match function names (`tet4_laplacian.c`)
- **Environment vars**: `SFEM_` prefix for runtime configuration
- **Drivers**: Imperative names (`assemble`, `refine`, `skin`)

### Error Handling
- Return 0 for success, non-zero for failure
- `MPI_Abort()` for fatal errors in parallel code
- Extensive use of `assert()` for debug builds

### Performance Considerations
- Critical loops use `SFEM_RESTRICT` keyword
- CUDA kernels in separate `.cu` files with `__global__` functions
- OpenMP parallelization via `#pragma omp parallel for`
- Environment variables control optimization (`SFEM_CLUSTER_SIZE`, `SFEM_VEC_SIZE`)

## Integration Points

### MPI Communication
- All data structures MPI-aware from ground up
- Use `MPI_COMM_WORLD` unless explicitly passed different communicator
- Parallel I/O via custom `array_create_from_file()` functions

### CUDA Integration
- Separate kernel files in `cuda/` subdirectories
- Memory space tracking via `mesh.mem_space` field
- Conditional compilation with `SFEM_ENABLE_CUDA`

### Python Interop
- Raw binary files as primary exchange format
- Python utilities in `python/sfem/` for pre/post-processing
- NumPy-compatible data layouts

## Testing and Debugging

### Environment Variables
```bash
export SFEM_LOG_LEVEL=5           # Enable verbose logging
export CUDA_LAUNCH_BLOCKING=1     # Synchronous CUDA calls
export SFEM_HANDLE_DIRICHLET=0    # Disable boundary condition handling
export SFEM_READ_FP32=1           # Force 32-bit precision reading
```

### Common Debug Patterns
- Use `printf()` with rank checks: `if (!rank) printf(...)`
- Timing via `MPI_Wtime()` for performance analysis
- File existence checks before operations

### Validation Workflows
- Compare CUDA vs CPU implementations with `fdiff.py`
- Use `raw_to_db.py` for visual inspection of field data
- Mesh quality checks via `volumes`, `extract_sharp_edges`

When implementing new features, follow the existing modular structure, maintain MPI awareness throughout, and provide both CPU and CUDA variants for performance-critical operations.