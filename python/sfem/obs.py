#!/usr/bin/env python3
"""
Python equivalent of obs.exe.cpp - Obstacle Problem Solver

This script demonstrates how to solve obstacle problems using SFEM Python bindings.
It replicates the functionality of the C++ obs.exe.cpp driver.
"""

import sys
import os
import numpy as np
import pysfem as sfem

def solve_obstacle_problem(mesh_path, sdf_path, dirichlet_path, contact_boundary_path, output_path):
    """
    Solve obstacle problem using SFEM Python bindings.
    
    Args:
        mesh_path: Path to the mesh file
        sdf_path: Path to the signed distance function file
        dirichlet_path: Path to Dirichlet boundary conditions
        contact_boundary_path: Path to contact boundary conditions
        output_path: Output directory path
    """
    sfem.init()
    
    try:
        # Read environment variables (Python equivalent)
        element_refine_level = int(os.environ.get('SFEM_ELEMENT_REFINE_LEVEL', '2'))
        operator_type = os.environ.get('SFEM_OPERATOR', 'LinearElasticity')
        execution_space = os.environ.get('SFEM_EXECUTION_SPACE', 'EXECUTION_SPACE_HOST')
        use_spmg = int(os.environ.get('SFEM_USE_SPMG', '1'))
        ssmgc_yaml = os.environ.get('SFEM_SSMGC_YAML', None)
        
        # Validate parameters
        if element_refine_level <= 1:
            raise ValueError("SFEM_ELEMENT_REFINE_LEVEL must be > 1")
        
        # Convert execution space string to enum
        if execution_space == 'EXECUTION_SPACE_DEVICE':
            es = sfem.ExecutionSpace.EXECUTION_SPACE_DEVICE
        else:
            es = sfem.ExecutionSpace.EXECUTION_SPACE_HOST
        
        print(f"Creating mesh from: {mesh_path}")
        # Create mesh and function space
        mesh = sfem.Mesh.read(mesh_path)
        block_size = mesh.spatial_dimension()
        fs = sfem.FunctionSpace(mesh, block_size)
        
        print(f"Promoting to semi-structured with level {element_refine_level}")
        # Promote to semi-structured mesh
        fs.promote_to_semi_structured(element_refine_level)
        fs.semi_structured_mesh().apply_hierarchical_renumbering()
        
        # Create function and operator
        print(f"Creating {operator_type} operator")
        f = sfem.Function(fs)
        op = sfem.create_op(fs, operator_type, es)
        op.initialize()
        f.add_operator(op)
        
        print(f"Loading SDF from: {sdf_path}")
        # Load SDF and contact boundary
        sdf = sfem.Grid.create_from_file(sfem.MPI_COMM_WORLD, sdf_path)
        contact_boundary = sfem.Sideset.create_from_file(sfem.MPI_COMM_WORLD, contact_boundary_path)
        contact_conds = sfem.ContactConditions.create(fs, sdf, contact_boundary, es)
        
        # Create solution vectors
        ndofs = fs.n_dofs()
        x = sfem.create_real_buffer(ndofs)
        rhs = sfem.create_real_buffer(ndofs)
        gap = sfem.create_real_buffer(ndofs)
        
        # Apply constraints
        f.apply_constraints(rhs)
        f.apply_constraints(x)
        
        print("Solving obstacle problem...")
        # Solve using SPMG or shifted penalty
        if use_spmg:
            print("Using SPMG solver")
            if ssmgc_yaml:
                # Load YAML configuration
                input_config = sfem.YAMLNoIndent.create_from_file(ssmgc_yaml)
                solver = sfem.create_ssmgc(f, contact_conds, input_config)
            else:
                solver = sfem.create_ssmgc(f, contact_conds)
        else:
            print("Using shifted penalty solver")
            solver = sfem.create_shifted_penalty(f, contact_conds)
        
        # Solve the system
        solver.apply(rhs, x)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Writing results to: {output_path}")
        # Write mesh files
        fs.mesh().write(f"{output_path}/coarse_mesh")
        fs.semi_structured_mesh().export_as_standard(f"{output_path}/mesh")
        
        # Setup output
        out = f.output()
        out.set_output_dir(f"{output_path}/out")
        out.enable_AoS_to_SoA(True)
        
        # Write solution data
        out.write("rhs", rhs)
        out.write("disp", x)
        
        # Additional output for host execution
        if es != sfem.ExecutionSpace.EXECUTION_SPACE_DEVICE:
            print("Computing additional output quantities...")
            
            # Update contact conditions and compute gap
            contact_conds.update(x)
            contact_conds.signed_distance_for_mesh_viz(x, gap)
            out.write("gap", gap)
            
            # Compute contact stress
            blas = sfem.blas(es)
            blas.zeros(rhs)
            f.gradient(x, rhs)
            
            blas.zeros(x)
            contact_conds.full_apply_boundary_mass_inverse(rhs, x)
            out.write("contact_stress", x)
        
        print("Obstacle problem solved successfully!")
        return True
        
    except Exception as e:
        print(f"Error solving obstacle problem: {e}")
        return False
    
    finally:
        # Finalize SFEM
        sfem.finalize()

# --- Test Problem Builders (from sfem_MGSDFContactTest.cpp) ---
def build_cuboid_sphere_contact(base_resolution=2, element_refine_level=2, es=None):
    # Mesh parameters
    y_top = 0.05
    resolution_ratio = 20
    if es is None:
        es = sfem.ExecutionSpace.EXECUTION_SPACE_HOST
    # Create mesh
    m = sfem.create_hex8_cube(
        base_resolution * resolution_ratio,
        base_resolution * 1,
        base_resolution * resolution_ratio,
        0, 0, 0, 1, y_top, 1)
    
    block_size = m.spatial_dimension()
    fs = sfem.FunctionSpace(m, block_size)
    fs.promote_to_semi_structured(element_refine_level)
    fs.semi_structured_mesh().apply_hierarchical_renumbering()
    f = sfem.Function(fs)
    op = sfem.create_op(fs, "LinearElasticity", es)
    op.initialize()
    f.add_operator(op)
    # Dirichlet BCs (left/right faces)
    left_right_selector = lambda x, y, z: abs(x) < 1e-8 or abs(x - 1) < 1e-8
    left_right = sfem.Sideset.create_from_selector(m, left_right_selector)

    conds = []
    for i, val in enumerate([0, -0.1, 0]):
        cond = sfem.DirichletCondition()
        cond.sideset = left_right
        cond.component = i
        cond.value = val
        print(f"Dirichlet condition {i}: component={cond.component}, value={cond.value}")
        conds.append(cond)
    dirichlet = sfem.create_dirichlet_conditions(fs, conds, es)
    print(f"Created Dirichlet conditions with {len(conds)} conditions")
    
    f.add_constraint(dirichlet)

    # Contact SDF (half-sphere)
    n = base_resolution * fs.semi_structured_mesh().level()
    def sdf_func(x, y, z):
        cx, cy, cz = 0.5, -0.5, 0.5
        radius = 0.5
        dx, dy, dz = cx - x, cy - y, cz - z
        return radius - np.sqrt(dx*dx + dy*dy + dz*dz)
    sdf = sfem.create_sdf(
        n * resolution_ratio * 2,
        n * 1 * 2,
        n * resolution_ratio * 2,
        -0.1, -0.2, -0.1, 1.1, y_top * 0.5, 1.1,
        sdf_func)
    # Contact boundary (bottom face)
    bottom_selector = lambda x, y, z: y > -1e-5 and y < 1e-5
    bottom_ss = sfem.Sideset.create_from_selector(m, bottom_selector)
    contact_conds = sfem.ContactConditions.create(fs, sdf, bottom_ss, es)
    return fs, f, contact_conds

def build_cuboid_highfreq_contact(base_resolution=2, element_refine_level=2, es=None):
    y_top = 0.05
    resolution_ratio = 20
    if es is None:
        es = sfem.ExecutionSpace.EXECUTION_SPACE_HOST
    m = sfem.create_hex8_cube(
        base_resolution * resolution_ratio,
        base_resolution * 1,
        base_resolution * resolution_ratio,
        0, 0, 0, 1, y_top, 1)
    block_size = m.spatial_dimension()
    fs = sfem.FunctionSpace(m, block_size)
    fs.promote_to_semi_structured(element_refine_level)
    fs.semi_structured_mesh().apply_hierarchical_renumbering()
    f = sfem.Function(fs)
    op = sfem.create_op(fs, "LinearElasticity", es)
    op.initialize()
    f.add_operator(op)
    left_right_selector = lambda x, y, z: abs(x) < 1e-8 or abs(x - 1) < 1e-8
    left_right = sfem.Sideset.create_from_selector(m, left_right_selector)
    print(f"Created sideset with {left_right.size()} sides")

    conds = []
    for i, val in enumerate([0, -0.1, 0]):
        cond = sfem.DirichletCondition()
        cond.sideset = left_right
        cond.component = i
        cond.value = val
        print(f"Dirichlet condition {i}: component={cond.component}, value={cond.value}")
        conds.append(cond)
    dirichlet = sfem.create_dirichlet_conditions(fs, conds, es)
    print(f"Created Dirichlet conditions with {len(conds)} conditions")
    
    f.add_constraint(dirichlet)

    n = base_resolution * fs.semi_structured_mesh().level()
    def sdf_func(x, y, z):
        cx = 0.6 * (1 - (x - .5) * (x - .5))
        cz = 0.6 * (1 - (z - .5) * (z - .5))
        fx = 0.1 * np.cos(cx * np.pi * 8) * cx * cx + 0.02 * np.cos(cx * np.pi * 16)
        fz = 0.1 * np.cos(cz * np.pi * 8) * cz * cz + 0.02 * np.cos(cx * np.pi * 16)
        fx += 0.005 * np.cos(cx * np.pi * 32)
        fz += 0.005 * np.cos(cz * np.pi * 32)
        fx += 0.0025 * np.cos(cx * np.pi * 64)
        fz += 0.0025 * np.cos(cz * np.pi * 64)
        fx += 0.001 * np.cos(np.pi + cx * np.pi * 128)
        fz += 0.001 * np.cos(np.pi + cz * np.pi * 128)
        fx += 0.001 * np.cos(cx * np.pi * 256)
        fz += 0.001 * np.cos(cz * np.pi * 256)
        fx += 0.001 * np.cos(cx * np.pi * 512)
        fz += 0.001 * np.cos(cz * np.pi * 512)
        obstacle = -0.1 - fx - fz
        return obstacle - y
    sdf = sfem.create_sdf(
        n * resolution_ratio * 2,
        n * 1 * 2,
        n * resolution_ratio * 2,
        0.1, -0.2, 0.1, 0.9, y_top * 0.5, 0.9,
        sdf_func)
    bottom_selector = lambda x, y, z: y > -1e-5 and y < 1e-5
    bottom_ss = sfem.Sideset.create_from_selector(m, bottom_selector)
    contact_conds = sfem.ContactConditions.create(fs, sdf, bottom_ss, es)
    return fs, f, contact_conds

def build_cuboid_multisphere_contact(base_resolution=2, element_refine_level=2, es=None, n_spheres=2):
    y_top = 0.05
    resolution_ratio = 20
    if es is None:
        es = sfem.ExecutionSpace.EXECUTION_SPACE_HOST
    m = sfem.create_hex8_cube(
        base_resolution * resolution_ratio,
        base_resolution * 1,
        base_resolution * resolution_ratio,
        0, 0, 0, 1, y_top, 1)
    block_size = m.spatial_dimension()
    fs = sfem.FunctionSpace(m, block_size)
    fs.promote_to_semi_structured(element_refine_level)
    fs.semi_structured_mesh().apply_hierarchical_renumbering()
    f = sfem.Function(fs)
    op = sfem.create_op(fs, "LinearElasticity", es)
    op.initialize()
    f.add_operator(op)
    left_right_selector = lambda x, y, z: abs(x) < 1e-8 or abs(x - 1) < 1e-8
    left_right = sfem.Sideset.create_from_selector(m, left_right_selector)
    print(f"Created sideset with {left_right.size()} sides")
    
    conds = []
    for i, val in enumerate([0, -0.1, 0]):
        cond = sfem.DirichletCondition()
        cond.sideset = left_right
        cond.component = i
        cond.value = val
        print(f"Dirichlet condition {i}: component={cond.component}, value={cond.value}")
        conds.append(cond)
    dirichlet = sfem.create_dirichlet_conditions(fs, conds, es)
    print(f"Created Dirichlet conditions with {len(conds)} conditions")
    
    f.add_constraint(dirichlet)

    n = base_resolution * fs.semi_structured_mesh().level()
    def sdf_func(x, y, z):
        dd = 1e6
        hx = 1. / (n_spheres + 1)
        hz = 1. / (n_spheres + 1)
        for i in range(n_spheres):
            for j in range(n_spheres):
                cx, cy, cz = hx + i * hx, -0.1, hz + j * hz
                radius = 1. / (8 + n_spheres)
                dx, dy, dz = cx - x, cy - y, cz - z
                ddij = radius - np.sqrt(dx*dx + dy*dy + dz*dz)
                if abs(ddij) < abs(dd):
                    dd = ddij
        return dd
    sdf = sfem.create_sdf(
        n * 5 * 2,
        n * 1 * 2,
        n * 5 * 2,
        -0.1, -0.2, -0.1, 1.1, y_top * 0.5, 1.1,
        sdf_func)
        
    bottom_selector = lambda x, y, z: y > -1e-5 and y < 1e-5
    bottom_ss = sfem.Sideset.create_from_selector(m, bottom_selector)
    contact_conds = sfem.ContactConditions.create(fs, sdf, bottom_ss, es)
    return fs, f, contact_conds

# --- Test Problem Solver ---
def solve_test_problem(problem="sphere", base_resolution=2, element_refine_level=2, es=None, n_spheres=2, output_path="test_contact"):
    sfem.init()
    if problem == "sphere":
        fs, f, contact_conds = build_cuboid_sphere_contact(base_resolution, element_refine_level, es)
    elif problem == "hifreq":
        fs, f, contact_conds = build_cuboid_highfreq_contact(base_resolution, element_refine_level, es)
    elif problem == "multisphere":
        fs, f, contact_conds = build_cuboid_multisphere_contact(base_resolution, element_refine_level, es, n_spheres)
    else:
        raise ValueError(f"Unknown test problem: {problem}")
    ndofs = fs.n_dofs()
    x = sfem.create_real_buffer(ndofs)
    rhs = sfem.create_real_buffer(ndofs)
    gap = sfem.create_real_buffer(ndofs)
        
    # Initialize contact conditions
    contact_conds.init()
    
    # Apply constraints to solution vector
    f.apply_constraints(x)
    f.apply_constraints(rhs)
        
    use_spmg = int(os.environ.get('SFEM_USE_SPMG', '1'))
    ssmgc_yaml = os.environ.get('SFEM_SSMGC_YAML', None)
    
    if use_spmg:
        if ssmgc_yaml:
            input_config = sfem.YAMLNoIndent.create_from_file(ssmgc_yaml)
            solver = sfem.create_ssmgc(f, contact_conds, input_config)
        else:
            solver = sfem.create_ssmgc(f, contact_conds)
    else:
        solver = sfem.create_shifted_penalty(f, contact_conds)

    solver.apply(rhs, x)
    os.makedirs(output_path, exist_ok=True)
    fs.mesh().write(f"{output_path}/coarse_mesh")
    fs.semi_structured_mesh().export_as_standard(f"{output_path}/mesh")
    out = f.output()
    out.set_output_dir(f"{output_path}/out")
    out.enable_AoS_to_SoA(True)
    out.write("rhs", rhs)
    out.write("disp", x)
    
    if es is None or es != sfem.ExecutionSpace.EXECUTION_SPACE_DEVICE:
        contact_conds.update(x)
        contact_conds.signed_distance_for_mesh_viz(x, gap)
        out.write("gap", gap)
        blas = sfem.blas(sfem.ExecutionSpace.EXECUTION_SPACE_HOST)
        blas.zeros(rhs)
        f.gradient(x, rhs)
        blas.zeros(x)
        contact_conds.full_apply_boundary_mass_inverse(rhs, x)
        out.write("contact_stress", x)
    print(f"Test problem '{problem}' solved and output written to {output_path}")
    sfem.finalize()
    return True

# --- Main Entrypoint ---
def main():
    if len(sys.argv) == 1:
        # No arguments: run default test problem (sphere)
        print("No arguments provided. Running default 'sphere' test problem...")
        solve_test_problem(problem="sphere")

        sys.exit(0)
    elif len(sys.argv) == 2:
        # One argument: use as test problem name
        problem = sys.argv[1]
        solve_test_problem(problem=problem)
        sys.exit(0)
    elif len(sys.argv) == 6:
        mesh_path = sys.argv[1]
        sdf_path = sys.argv[2]
        dirichlet_path = sys.argv[3]
        contact_boundary_path = sys.argv[4]
        output_path = sys.argv[5]
        for path in [mesh_path, sdf_path, dirichlet_path, contact_boundary_path]:
            if not os.path.exists(path):
                print(f"Error: Input file not found: {path}")
                sys.exit(1)
        success = solve_obstacle_problem(mesh_path, sdf_path, dirichlet_path, contact_boundary_path, output_path)
        if success:
            print("✅ Obstacle problem completed successfully!")
            sys.exit(0)
        else:
            print("❌ Obstacle problem failed!")
            sys.exit(1)
    else:
        print("Usage: python obs.py [sphere|hifreq|multisphere] OR python obs.py <mesh> <sdf> <dirichlet_conditions> <contact_boundary> <output>")
        sys.exit(1)

if __name__ == "__main__":
    main() 
    