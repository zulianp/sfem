# SFEM - Semi-structured Finite Element Method
## Product Requirements Document (PRD)

### Project Overview
SFEM is a high-performance computing framework for finite element analysis, featuring a mixed C++/C/CUDA architecture optimized for large-scale problems (>600M DOFs). The project combines object-oriented C++ frontend with performance-critical procedural C/CUDA backend.

### Current Architecture
- **Frontend**: C++ with OOP patterns (classes, templates, RAII)
- **Backend**: Procedural C and CUDA for performance-critical code
- **Data-oriented design**: Structure of Arrays (SoA) over Array of Structures (AoS)
- **Memory management**: Manual allocation with clear ownership semantics
- **Python bindings**: Basic C++ to Python interface via `sfem.cpp`
- **MPI support**: Optional MPI dependency with proper abstractions
  - MPI functionalities organized in dedicated folders (`mpi/`, `parallel/`)
  - Explicit MPI dependencies hidden behind simple abstractions
  - Backend functions have compile-time fallbacks for non-MPI builds
  - Frontend provides proper abstractions to handle communicators

---

## Milestone 1: Mesh Refactoring
### Objective
Refactor the mesh system to support multi-block meshes with heterogeneous properties and element types, enabling more complex and realistic simulations. The refactoring follows the principle of minimal information passing to algorithms and moves memory management to the C++ frontend.

### Current State
- Single-block mesh representation
- Homogeneous element types per mesh
- Limited support for different material properties
- Basic mesh generation and manipulation
- Algorithms depend on `mesh_t` struct

### Requirements

#### 1.1 Multi-Block Mesh Architecture
- **1.1.1** Support for multiple mesh blocks within a single simulation
- **1.1.2** Each block can have different:
  - Element types (hex8, tet4, quad4, tri3, etc.)
  - Material properties
  - Boundary conditions
- **1.1.3** Block connectivity and interface handling
- **1.1.4** Efficient block-to-block data transfer

#### 1.2 Algorithm Interface Refactoring
- **1.2.1** Remove dependency on `mesh_t` struct in all algorithms
- **1.2.2** Pass only minimal required information to algorithms:
  - Element connectivity arrays (`idx_t** elements`)
  - Node coordinates (`geom_t** points`)
  - Element counts (`ptrdiff_t n_elements`, `ptrdiff_t n_nodes`)
  - Element type information
- **1.2.3** Maintain algorithm performance with minimal data access
- **1.2.4** Ensure algorithms work with both single-block and multi-block data
- **1.2.5** Memory allocation should be done outside functions whenever possible. Hence helper methods to identify sizes will be required.

#### 1.3 Memory Management Migration
- **1.3.1** Move all memory allocation/deallocation to C++ frontend
- **1.3.2** Use RAII patterns for mesh data management
- **1.3.3** Implement smart pointers for automatic cleanup
- **1.3.4** Provide clear ownership semantics for mesh data
- **1.3.5** Ensure thread-safe memory management for parallel operations

#### 1.4 Block Management System
- **1.4.1** Block identification and indexing
- **1.4.2** Block property management (material, boundary conditions, etc.)
- **1.4.3** Block-level operations (refinement, coarsening, partitioning)
- **1.4.4** Block metadata and attributes

#### 1.5 Interface Handling
- **1.5.1** Automatic interface detection between blocks
- **1.5.2** Interface mesh generation and management
- **1.5.3** Interface constraint handling
- **1.5.4** Interface data exchange protocols

#### 1.6 Performance Requirements
- **1.6.1** Maintain performance for large-scale problems (>2 Billion DOFs)
- **1.6.2** Efficient memory layout for multi-block structures
- **1.6.3** Parallel block processing capabilities
- **1.6.4** Minimal overhead for single-block cases
- **1.6.5** Cache-friendly data access patterns

#### File Organization
- `frontend/sfem_MeshBlock.hpp/cpp` - RAII-based mesh data management using sfem_Buffer
- `frontend/sfem_Mesh.hpp/cpp` - Multi-block container using shared_ptr
- `mesh/block_connectivity.h/c` - Block connectivity (no memory management)
- `mesh/interface_mesh.h/c` - Interface handling (no memory management)
- `mesh/block_operations.h/c` - Block-level operations (minimal data interface)

### Migration Strategy

#### Phase 1: Algorithm Refactoring
1. **Audit all algorithms** to identify `mesh_t` dependencies
2. **Extract minimal data requirements** for each algorithm
3. **Refactor algorithm signatures** to accept individual parameters, only C types
4. **Update all algorithm calls** throughout the codebase
5. **Maintain backward compatibility** during transition

#### Phase 2: Memory Management Migration
1. **Create C++ classes** using existing sfem_Buffer patterns (frontend only)
2. **Implement shared_ptr patterns** for automatic cleanup (following existing codebase)
3. **Migrate memory allocation** from C to C++ frontend using sfem::create_host_buffer
4. **Update Python bindings** to use new C++ classes
5. **Remove C memory management** functions

#### Phase 3: Multi-Block Implementation
1. **Implement multi-block container** in C++ frontend using sfem_Buffer
2. **Add block management functionality**
3. **Implement interface handling**
4. **Performance optimization** and testing
5. **Integration with existing solvers**

### Success Criteria
- [ ] All algorithms refactored to use minimal information interface
- [ ] No `mesh_t` struct dependencies in algorithm implementations
- [ ] Memory management fully migrated to C++ frontend with RAII
- [ ] Support for at least 4 different element types across blocks
- [ ] Interface handling between different element types
- [ ] Performance variation within 2% of single-block implementation
- [ ] Memory usage optimization for large multi-block meshes
- [ ] Comprehensive test suite for multi-block scenarios
- [ ] Thread-safe memory management for parallel operations

---

## Milestone 2: Python Frontend
### Objective
Create a comprehensive, pythonic Python frontend that exposes all C++ frontend functionalities, enabling rapid prototyping and easier integration with Python-based workflows.

### Current State
- Basic C++ to Python bindings via `sfem.cpp`
- Limited functionality exposed
- Non-pythonic interface design
- Minimal documentation and examples

### Requirements

#### 2.1 Core Python API Design
- **2.1.1** Pythonic interface following PEP 8 conventions
- **2.1.2** Object-oriented design with proper Python classes
- **2.1.3** Context managers for resource management
- **2.1.4** Exception handling with meaningful Python exceptions
- **2.1.5** Type hints for better IDE support

#### 2.2 Mesh Operations
- **2.2.1** Multi-block mesh creation and manipulation
- **2.2.2** Mesh generation utilities (structured, unstructured)
- **2.2.3** Mesh I/O (using the raw data and meta.yaml)
- **2.2.4** Mesh visualization and inspection

#### 2.3 Finite Element Operations
- **2.3.1** Function space creation and management
- **2.3.2** Element assembly and operator creation
- **2.3.3** Boundary condition application
- **2.3.4** Linear and nonlinear solver interfaces
- **2.3.5** Contact condition handling

#### 2.4 Solver Integration
- **2.4.1** Linear solver configuration and execution
- **2.4.2** Nonlinear solver with convergence monitoring
- **2.4.3** Time integration schemes
- **2.4.4** Solution post-processing and analysis
- **2.4.5** Performance monitoring and profiling

#### 2.5 Integration and Interoperability
- **2.5.1** NumPy array compatibility
- **2.5.2** SciPy sparse matrix integration
- **2.5.3** Matplotlib visualization support
- **2.5.5** Integration with popular Python scientific libraries

### Technical Specifications

#### Python Package Structure
```
pysfem/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── mesh.py          # Mesh classes
│   ├── function_space.py # Function space
│   ├── operators.py     # FE operators
│   └── solvers.py       # Solver interfaces
├── utils/
│   ├── __init__.py
│   ├── io.py           # I/O utilities
│   ├── visualization.py # Plotting utilities
│   └── examples.py     # Example problems
└── tests/
    ├── __init__.py
    ├── test_mesh.py
    ├── test_operators.py
    └── test_solvers.py
```

pysfem should become sfem

#### MPI Architecture Requirements
- **Optional Dependency**: MPI support must be completely optional
- **Folder Organization**: All MPI-related code must be organized in dedicated folders:
  - `mpi/` - Core MPI abstractions and utilities
  - `parallel/` - Parallel algorithms and data structures
- **Abstraction Layer**: Explicit MPI dependencies must be hidden behind simple abstractions:
  - `Communicator` class for MPI_Comm handling
  - Current abstractions will have adapt their implementation internally
- **Compile-time Fallbacks**: Backend functions must have compile-time fallbacks:
  - `#ifdef SFEM_ENABLE_MPI` guards for MPI-specific code (including matrix.io functionalities)
  - Serial implementations for non-MPI builds
  - No runtime MPI dependency checks
- **Frontend Abstractions**: C++ frontend must provide proper communicator abstractions:
  - `sfem::Communicator` class wrapping MPI_Comm
  - Default communicator for serial execution
  - Python bindings for communicator management
- **Python Interface**: Python frontend must handle communicators properly:
  - `sfem.Communicator` class for Python users
  - Automatic fallback to serial execution when MPI not available
  - Seamless integration with mpi4py when available

#### API Design Examples
```python
# Mesh creation
mesh = sfem.Mesh()
block1 = mesh.add_block(element_type=sfem.HEX8, n_elements=(10, 10, 10))
block2 = mesh.add_block(element_type=sfem.TET4, material=steel)

# Function space
V = sfem.VectorFunctionSpace(mesh, order=1)
Q = sfem.ScalarFunctionSpace(mesh, order=1)

# Operators
K = sfem.assemble_elasticity(V, material=steel)
M = sfem.assemble_mass(V)

# Solvers
solver = sfem.LinearSolver(K, method="amg")
u = solver.solve(f)
```

### Success Criteria
- [ ] Complete C++ frontend functionality exposed
- [ ] Pythonic API following PEP 8 and Python best practices
- [ ] Comprehensive documentation with examples
- [ ] NumPy/SciPy integration working seamlessly
- [ ] Jupyter notebook tutorials for common use cases
- [ ] Performance within 5% of direct C++ usage
- [ ] Test coverage >90% for Python API

---

## Implementation Timeline

### Phase 1: Mesh Refactoring (Months 1-3)
- **Month 1**: Core multi-block data structures and basic operations
- **Month 2**: Interface handling and connectivity
- **Month 3**: Performance optimization and testing

### Phase 2: Python Frontend (Months 4-6)
- **Month 4**: Core Python API design and basic bindings
- **Month 5**: Advanced functionality and integration
- **Month 6**: Documentation, examples, and testing

### Phase 3: Integration and Polish (Months 7-8)
- **Month 7**: Integration testing and performance validation
- **Month 8**: Documentation completion and release preparation

---

## Risk Assessment

### Technical Risks
1. **Performance degradation** with multi-block meshes
   - *Mitigation*: Early performance testing and optimization
2. **Interface complexity** between different element types
   - *Mitigation*: Prototype interface handling early
3. **Python binding complexity** for advanced C++ features
   - *Mitigation*: Use modern binding tools (nanobind)

### Resource Risks
1. **Development time** for comprehensive Python API
   - *Mitigation*: Prioritize core functionality first
2. **Testing complexity** for multi-block scenarios
   - *Mitigation*: Automated test generation

---

## Success Metrics

### Technical Metrics
- Performance: <10% overhead for multi-block meshes
- Memory usage: Efficient multi-block memory layout
- API coverage: 100% of C++ frontend functionality exposed
- Test coverage: >90% for both C++ and Python code

### User Experience Metrics
- Python API usability: Intuitive, pythonic interface
- Documentation quality: Comprehensive examples and tutorials
- Integration ease: Seamless NumPy/SciPy interoperability
- Learning curve: New users productive within 1 hour

---

## Dependencies and Prerequisites

### External Dependencies
- **nanobind**: Modern C++ to Python binding library
- **NumPy**: Numerical array support
- **SciPy**: Sparse matrix and scientific computing
- **Matplotlib**: Visualization capabilities
- **MPI**: Optional parallel computing support
  - Must be organized in dedicated folders (`parallel/`)
  - Explicit MPI dependencies hidden behind abstractions
  - Compile-time fallbacks for non-MPI builds
  - Frontend communicator abstractions for Python interface

### Internal Dependencies
- **Milestone 1 completion**: Python frontend depends on multi-block mesh
- **C++ frontend stability**: Python API should be built on stable C++ API
- **Testing infrastructure**: Comprehensive test suite for validation

---

## Future Considerations

### Post-Milestone Enhancements
1. **Parallel computing**: MPI support in Python frontend
2. **GPU acceleration**: CUDA operations exposed to Python
3. **Advanced solvers**: More sophisticated solver algorithms


### Long-term Vision
- **Cloud deployment**: Containerized SFEM for cloud computing
- **Web interface**: Browser-based SFEM frontend
- **Real-time simulation**: Interactive simulation capabilities
- **Industry integration**: CAD/CAE software integration

## Examples

### Example 1: Extending Existing Mesh Class for Multi-block Support

```cpp
// Extend existing Mesh class with multi-block capabilities
class Mesh {
    // ... existing members ...
private:
    // Add multi-block support to existing Mesh::Impl
    // Points are shared across all blocks - only one buffer
    std::shared_ptr<sfem::Buffer<geom_t>> points;  // Single points buffer
    
    // Element connectivity per block
    std::vector<std::shared_ptr<sfem::Buffer<idx_t>>> block_elements;
    std::vector<enum ElemType> block_element_types;
    std::vector<MeshBlockProperties> block_properties;
    
public:
    // ... existing methods ...
    
    // New methods for multi-block support
    ptrdiff_t n_blocks() const { return block_elements.size(); }
    
    ptrdiff_t get_block_size(int block_idx) const {
        return block_elements[block_idx]->size();
    }
    
    idx_t* get_block_elements(int block_idx) const {
        return block_elements[block_idx]->data();
    }
    
    // Points are shared - same buffer for all blocks
    geom_t* get_points() const {
        return points->data();
    }
    
    ptrdiff_t n_points() const {
        return points->size() / 3; // 3D coordinates
    }
    
    enum ElemType get_block_element_type(int block_idx) const {
        return block_element_types[block_idx];
    }
    
    // Create multi-block mesh from existing single-block mesh
    static std::shared_ptr<Mesh> create_multi_block(
        const std::shared_ptr<sfem::Buffer<geom_t>>& points,
        const std::vector<std::shared_ptr<sfem::Buffer<idx_t>>>& elements,
        const std::vector<enum ElemType>& element_types,
        const std::vector<MeshBlockProperties>& properties,
        MPI_Comm comm) {
        
        auto mesh = std::make_shared<Mesh>();
        mesh->impl_->points = points;  // Single points buffer
        mesh->impl_->block_elements = elements;
        mesh->impl_->block_element_types = element_types;
        mesh->impl_->block_properties = properties;
        mesh->impl_->comm = comm;
        
        // Set primary block as main mesh data (for backward compatibility)
        if (!elements.empty()) {
            mesh->impl_->elements = elements[0];
            mesh->impl_->element_type = element_types[0];
        }
        
        return mesh;
    }
};
```

### Example 2: Extending Existing Op Classes for Multi-block Support

```cpp
// Extend existing Laplacian class with multi-block support
class Laplacian final : public Op {
    // ... existing members ...
    
public:
    // ... existing methods ...
    
    // Override existing methods to support multi-block
    int apply(const real_t* const x, const real_t* const h, real_t* const out) override {
        SFEM_TRACE_SCOPE("Laplacian::apply");
        
        auto mesh = space->mesh_ptr();
        
        // Check if this is a multi-block mesh
        if (mesh->n_blocks() > 1) {
            return apply_multi_block(h, out);
        }
        
        // Use existing single-block implementation
        double tick = MPI_Wtime();
        
        int err = laplacian_apply(
            element_type, 
            mesh->n_elements(), 
            mesh->n_nodes(), 
            mesh->elements()->data(), 
            mesh->points()->data(), 
            h, out);
            
        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }
    
    int hessian_crs(const real_t* const x,
                    const count_t* const rowptr,
                    const idx_t* const colidx,
                    real_t* const values) override {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs");
        
        auto mesh = space->mesh_ptr();
        
        // Check if this is a multi-block mesh
        if (mesh->n_blocks() > 1) {
            return hessian_crs_multi_block(rowptr, colidx, values);
        }
        
        // Use existing single-block implementation
        auto graph = space->dof_to_dof_graph();
        
        return laplacian_crs(element_type,
                             mesh->n_elements(),
                             mesh->n_nodes(),
                             mesh->elements()->data(),
                             mesh->points()->data(),
                             graph->rowptr()->data(),
                             graph->colidx()->data(),
                             values);
    }
    
private:
    // New private methods for multi-block support
    int apply_multi_block(const real_t* const h, real_t* const out) {
        auto mesh = space->mesh_ptr();
        const ptrdiff_t n_blocks = mesh->n_blocks();
        const ptrdiff_t n_points = mesh->n_points();
        
        std::vector<ptrdiff_t> block_sizes(n_blocks);
        std::vector<idx_t*> block_elements(n_blocks);
        std::vector<enum ElemType> element_types(n_blocks);
        
        // Extract minimal data for C backend
        for (ptrdiff_t b = 0; b < n_blocks; b++) {
            block_sizes[b] = mesh->get_block_size(b);
            block_elements[b] = mesh->get_block_elements(b);
            element_types[b] = mesh->get_block_element_type(b);
        }
        
        double tick = MPI_Wtime();
        
        int err = multi_block_laplacian_apply(
            n_blocks,
            block_sizes.data(),
            block_elements.data(),
            mesh->get_points(),  // Single points buffer
            n_points,
            element_types.data(),
            h, out);
            
        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }
    
    int hessian_crs_multi_block(const count_t* const rowptr,
                                const idx_t* const colidx,
                                real_t* const values) {
        auto mesh = space->mesh_ptr();
        const ptrdiff_t n_blocks = mesh->n_blocks();
        const ptrdiff_t n_points = mesh->n_points();
        
        std::vector<ptrdiff_t> block_sizes(n_blocks);
        std::vector<idx_t*> block_elements(n_blocks);
        std::vector<enum ElemType> element_types(n_blocks);
        
        // Extract minimal data for C backend
        for (ptrdiff_t b = 0; b < n_blocks; b++) {
            block_sizes[b] = mesh->get_block_size(b);
            block_elements[b] = mesh->get_block_elements(b);
            element_types[b] = mesh->get_block_element_type(b);
        }
        
        return multi_block_laplacian_crs(
            n_blocks,
            block_sizes.data(),
            block_elements.data(),
            mesh->get_points(),  // Single points buffer
            n_points,
            element_types.data(),
            rowptr, colidx, values);
    }
};
```


### Example 3: Extending FunctionSpace for Multi-block Support

```cpp
// Extend existing FunctionSpace class
class FunctionSpace {
    // ... existing members ...
    
public:
    // ... existing methods ...
    
    // New methods for multi-block support
    bool is_multi_block() const {
        return impl_->mesh->n_blocks() > 1;
    }
    
    ptrdiff_t n_blocks() const {
        return impl_->mesh->n_blocks();
    }
    
    // Override existing methods to handle multi-block
    int create_vector(ptrdiff_t* nlocal, ptrdiff_t* nglobal, real_t** values) {
        if (is_multi_block()) {
            return create_multi_block_vector(nlocal, nglobal, values);
        }
        
        // Use existing single-block implementation
        return create_single_block_vector(nlocal, nglobal, values);
    }
    
private:
    // New private methods
    int create_multi_block_vector(ptrdiff_t* nlocal, ptrdiff_t* nglobal, real_t** values) {
        auto mesh = impl_->mesh;
        const ptrdiff_t n_blocks = mesh->n_blocks();
        
        // Calculate total DOFs across all blocks
        ptrdiff_t total_dofs = 0;
        for (ptrdiff_t b = 0; b < n_blocks; b++) {
            total_dofs += mesh->get_block_size(b) * block_size;
        }
        
        *nlocal = total_dofs;
        *nglobal = total_dofs; // Simplified for single MPI rank
        
        // Allocate vector
        *values = (real_t*)calloc((size_t)total_dofs, sizeof(real_t));
        if (!*values) return SFEM_FAILURE;
        
        return SFEM_SUCCESS;
    }
    
    int create_single_block_vector(ptrdiff_t* nlocal, ptrdiff_t* nglobal, real_t** values) {
        // Existing implementation
        *nlocal = impl_->nlocal;
        *nglobal = impl_->nglobal;
        *values = (real_t*)calloc((size_t)impl_->nlocal, sizeof(real_t));
        if (!*values) return SFEM_FAILURE;
        return SFEM_SUCCESS;
    }
};
```

### Example 4: Python Frontend Using Extended Classes

```python
# Python Frontend: Using extended existing classes
import sfem

# Create shared points buffer
points = sfem.Buffer.create_from_array(all_points, dtype=sfem.geom_t)

# Create individual block element connectivities
block1_elements = sfem.Buffer.create_from_array(elements1, dtype=sfem.idx_t)
block1_props = sfem.MeshBlockProperties(
    element_type=sfem.ElemType.HEX8,
    material_id=1,
    boundary_conditions={"left": "dirichlet", "right": "neumann"}
)

block2_elements = sfem.Buffer.create_from_array(elements2, dtype=sfem.idx_t)
block2_props = sfem.MeshBlockProperties(
    element_type=sfem.ElemType.TET4,
    material_id=2,
    boundary_conditions={"top": "dirichlet"}
)

# Create multi-block mesh using extended Mesh class
# Note: points are shared across all blocks
mesh = sfem.Mesh.create_multi_block(
    points=points,  # Single points buffer
    elements=[block1_elements, block2_elements],
    element_types=[sfem.ElemType.HEX8, sfem.ElemType.TET4],
    properties=[block1_props, block2_props],
    comm=MPI.COMM_WORLD
)

# Create function space (automatically handles multi-block)
space = sfem.FunctionSpace(mesh, block_size=1)

# Create operator (automatically handles multi-block)
laplacian = sfem.Laplacian.create(space)  # Uses existing Laplacian class

# Create function and solve
function = sfem.Function(space)
function.add_operator(laplacian)

# Add boundary conditions
dirichlet = sfem.DirichletBC(space)
dirichlet.add_boundary("left", 0.0)  # Fix left boundary to zero
dirichlet.add_boundary("right", 1.0) # Fix right boundary to one
function.add_constraint(dirichlet)

# Solve
solver = sfem.LinearSolver()
solution = solver.solve(function)

```

These examples demonstrate:
- **Element connectivity only**: Multi-block support is for element connectivity, not points
- **Shared points**: Single points buffer shared across all blocks
- **Extending existing classes**: Adding new methods to `Mesh`, `FunctionSpace`, and `Op` classes
- **Backward compatibility**: Existing single-block functionality remains unchanged
- **Minimal data passing**: C++ frontend extracts only necessary data for C backend algorithms
- **Memory safety**: Using `std::shared_ptr<sfem::Buffer<T>>` for automatic memory management
- **Performance**: Direct pointer access to data for maximum efficiency
- **Consistent patterns**: Following existing codebase architecture and style 

---

### New Operator: NeoHookeanSmithActiveStrainPacked
- Adds HEX8 Smith Neo-Hookean active-strain variant with partial assembly backend.
- Frontend op: `NeoHookeanSmithActiveStrainPacked` (packed, multi-block aware).
- Parameters per block (with defaults, overridable via `Parameters`):
  - `mu` (shear modulus)
  - `lambda` (Lamé first parameter)
  - `lmda` (Smith model parameter)
- Active strain input `Fa`:
  - AoS (3x3) per element, contiguous with stride `Fa_stride`
  - Provided via `set_active_strain_global(Fa_aos, stride)` or `set_active_strain_in_block(name, Fa_aos, stride)`
- Exposed methods: `value`, `value_steps`, `gradient`, `hessian_diag`, `apply` (via partial assembly), `hessian_bsr`, `hessian_bcrs_sym` (API parity with Ogden).
- Backend C entry points: `hex8_neohookean_smith_active_strain_*` mirroring Ogden active-strain with extra `lmda` argument.