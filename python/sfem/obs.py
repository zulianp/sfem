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
    
    # Initialize SFEM
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
        f.apply_constraints(rhs.data())
        f.apply_constraints(x.data())
        
        print("Solving obstacle problem...")
        # Solve using SPMG or shifted penalty
        if use_spmg:
            print("Using SPMG solver")
            if ssmgc_yaml:
                # Load YAML configuration
                input_config = sfem.YAMLNoIndent.create_from_file(ssmgc_yaml)
                solver = sfem.create_ssmgc(f, contact_conds, input_config)
            else:
                solver = sfem.create_ssmgc(f, contact_conds, None)
        else:
            print("Using shifted penalty solver")
            solver = sfem.create_shifted_penalty(f, contact_conds, None)
        
        # Solve the system
        solver.apply(rhs.data(), x.data())
        
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
        out.write("rhs", rhs.data())
        out.write("disp", x.data())
        
        # Additional output for host execution
        if es != sfem.ExecutionSpace.EXECUTION_SPACE_DEVICE:
            print("Computing additional output quantities...")
            
            # Update contact conditions and compute gap
            contact_conds.update(x.data())
            contact_conds.signed_distance_for_mesh_viz(x.data(), gap.data())
            out.write("gap", gap.data())
            
            # Compute contact stress
            blas = sfem.blas(es)
            blas.zeros(rhs.size(), rhs.data())
            f.gradient(x.data(), rhs.data())
            
            blas.zeros(x.size(), x.data())
            contact_conds.full_apply_boundary_mass_inverse(rhs.data(), x.data())
            out.write("contact_stress", x.data())
        
        print("Obstacle problem solved successfully!")
        return True
        
    except Exception as e:
        print(f"Error solving obstacle problem: {e}")
        return False
    
    finally:
        # Finalize SFEM
        sfem.finalize()

def main():
    """Main function - equivalent to main() in obs.exe.cpp"""
    if len(sys.argv) != 6:
        print("Usage: python obs.py <mesh> <sdf> <dirichlet_conditions> <contact_boundary> <output>")
        print("Example: python obs.py data/mesh.vtk data/sdf.vtk data/dirichlet.txt data/contact.txt results/")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    sdf_path = sys.argv[2]
    dirichlet_path = sys.argv[3]
    contact_boundary_path = sys.argv[4]
    output_path = sys.argv[5]
    
    # Validate input files exist
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

if __name__ == "__main__":
    main() 
    