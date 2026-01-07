#include <math.h>
#include <stdio.h>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_bsr_SpMV.hpp"
#include "sfem_test.h"


static void compute_contact_lower_bound(
    real_t* lower_bound,
    const real_t* displacement,
    geom_t * const * ref_coords,  // Reference coordinates: ref_coords[dim][node]
    const ptrdiff_t* contact_nodes,
    const ptrdiff_t n_contact_nodes,
    const real_t contact_plane,   // ABSOLUTE coordinate of contact plane (e.g., z=0)
    const int contact_dir,
    const int block_size,
    const ptrdiff_t ndofs,
    const real_t default_lb = -1000.0)
{
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        lower_bound[i] = default_lb;
    }
    for (ptrdiff_t i = 0; i < n_contact_nodes; i++) {
        const ptrdiff_t node_idx = contact_nodes[i];
        const ptrdiff_t dof_idx = node_idx * block_size + contact_dir;
        const real_t current_disp = displacement[dof_idx];
        const real_t ref_coord = ref_coords[contact_dir][node_idx];
        // Constraint: ref_coord + disp + delta >= contact_plane
        // => delta >= contact_plane - ref_coord - disp
        lower_bound[dof_idx] = contact_plane - ref_coord - current_disp;
    }
}

// Stepped obstacle: contact plane varies with y coordinate in steps
// Creates a staircase-like obstacle that works with box constraints
// Uses REFERENCE coordinates to keep constraints stable throughout simulation
static void compute_contact_lower_bound_stepped(
    real_t* lower_bound,
    const real_t* displacement,
    geom_t * const * ref_coords,  // Reference coordinates: ref_coords[dim][node]
    const ptrdiff_t* contact_nodes,
    const ptrdiff_t n_contact_nodes,
    const real_t base_plane,      // Base contact plane (at y=0)
    const real_t step_height,     // Height of each step in y direction
    const real_t step_depth,      // Depth change per step in x direction (negative = deeper)
    const int contact_dir,        // Direction of contact constraint (0=x)
    const int slope_dir,          // Direction for steps (1=y)
    const int block_size,
    const ptrdiff_t ndofs,
    const real_t default_lb = -1000.0)
{
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        lower_bound[i] = default_lb;
    }
    for (ptrdiff_t i = 0; i < n_contact_nodes; i++) {
        const ptrdiff_t node_idx = contact_nodes[i];
        const ptrdiff_t dof_idx = node_idx * block_size + contact_dir;
        
        // Use REFERENCE coordinate (stable constraint throughout simulation)
        const real_t ref_slope_coord = ref_coords[slope_dir][node_idx] +displacement[node_idx * block_size + slope_dir];
        
        // Compute which step this node is on based on reference y position
        int step_num = (int)(ref_slope_coord / step_height);
        if (step_num < 0) step_num = 0;  // Clamp to first step
        
        // Compute local contact plane position for this step
        const real_t local_plane = base_plane + step_num * step_depth;
        
        const real_t current_disp = displacement[dof_idx];
        lower_bound[dof_idx] = local_plane - current_disp;
    }
}

// Triangular obstacle: contact plane varies with y coordinate
// Creates a V-shaped or ramp obstacle
static void compute_contact_lower_bound_triangular(
    real_t* lower_bound,
    const real_t* displacement,
    geom_t * const * ref_coords,  // Reference coordinates: ref_coords[dim][node]
    const ptrdiff_t* contact_nodes,
    const ptrdiff_t n_contact_nodes,
    const real_t base_plane,      // Base contact plane (at y=0)
    const real_t slope,           // Slope: how much plane moves per unit y
    const int contact_dir,        // Direction of contact constraint (0=x)
    const int slope_dir,          // Direction for slope calculation (1=y)
    const int block_size,
    const ptrdiff_t ndofs,
    const real_t default_lb = -1000.0)
{
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        lower_bound[i] = default_lb;
    }
    for (ptrdiff_t i = 0; i < n_contact_nodes; i++) {
        const ptrdiff_t node_idx = contact_nodes[i];
        const ptrdiff_t dof_idx = node_idx * block_size + contact_dir;
        
        // Get reference coordinate in slope direction (coords stored as [dim][node])
        const real_t ref_slope_coord = ref_coords[slope_dir][node_idx]+displacement[node_idx * block_size + slope_dir];
        
        // Compute local contact plane position: plane = base + slope * y
        const real_t local_plane = base_plane + slope * ref_slope_coord;
        
        const real_t current_disp = displacement[dof_idx];
        lower_bound[dof_idx] = local_plane - current_disp;
    }
}

// Hemisphere obstacle: contact with a hemispherical surface
// Hemisphere equation: (x-cx)² + (y-cy)² + (z-cz)² = R²
static void compute_contact_lower_bound_hemisphere(
    real_t* lower_bound,
    const real_t* displacement,
    geom_t * const * ref_coords,
    const ptrdiff_t* contact_nodes,
    const ptrdiff_t n_contact_nodes,
    const real_t center_x,
    const real_t center_y,
    const real_t center_z,
    const real_t radius,
    const int contact_dir,        // Direction of constraint (0=x)
    const int block_size,
    const ptrdiff_t ndofs,
    const real_t default_lb = -1000.0)
{
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        lower_bound[i] = default_lb;
    }
    
    // Determine the two perpendicular directions
    int h1, h2;
    if (contact_dir == 0) { h1 = 1; h2 = 2; }       // contact in x, perpendicular: y,z
    else if (contact_dir == 1) { h1 = 0; h2 = 2; }  // contact in y, perpendicular: x,z
    else { h1 = 0; h2 = 1; }                         // contact in z, perpendicular: x,y
    
    real_t center[3] = {center_x, center_y, center_z};
    
    static bool first_call = true;
    
    for (ptrdiff_t i = 0; i < n_contact_nodes; i++) {
        const ptrdiff_t node_idx = contact_nodes[i];
        const ptrdiff_t dof_idx = node_idx * block_size + contact_dir;
        
        // Current position in perpendicular directions (reference + displacement)
        real_t cur_h1 = ref_coords[h1][node_idx] + displacement[node_idx * block_size + h1];
        real_t cur_h2 = ref_coords[h2][node_idx] + displacement[node_idx * block_size + h2];
        
        // Distance squared from center in perpendicular plane
        real_t dh1 = cur_h1 - center[h1];
        real_t dh2 = cur_h2 - center[h2];
        real_t dist_h_sq = dh1*dh1 + dh2*dh2;
        
        // Hemisphere surface position in contact direction
        real_t surface_coord;
        if (dist_h_sq < radius * radius) {
            // Inside hemisphere projection: surface = center + sqrt(R² - d²)
            surface_coord = center[contact_dir] + sqrt(radius*radius - dist_h_sq);
        } else {
            // Outside hemisphere: use flat plane at hemisphere center
            surface_coord = center[contact_dir];
        }
        
        // Same pattern as triangular/stepped: lower_bound = surface - current_disp
        const real_t current_disp = displacement[dof_idx];
        lower_bound[dof_idx] = surface_coord - current_disp;
        
        // Debug output for first call
        if (first_call && i < 5) {
            printf("  DEBUG hemisphere node %ld: ref_y=%.3f ref_z=%.3f dh1=%.3f dh2=%.3f dist=%.3f surface=%.3f disp=%.3f lb=%.3f\n",
                   (long)node_idx, ref_coords[h1][node_idx], ref_coords[h2][node_idx], 
                   dh1, dh2, sqrt(dist_h_sq), surface_coord, current_disp, lower_bound[dof_idx]);
        }
    }
    first_call = false;
}


std::shared_ptr<sfem::Output> create_output(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto fs = f->space();

    sfem::create_directory(output_dir.c_str());
    auto output = f->output();
    output->enable_AoS_to_SoA(fs->block_size() > 1);
    output->set_output_dir(output_dir.c_str());

    if (fs->has_semi_structured_mesh()) {
        fs->semi_structured_mesh().export_as_standard(output_dir.c_str());
    } else {
        fs->mesh_ptr()->write(output_dir.c_str());
    }
    return output;
}

// Export sloped obstacle geometry as a simple quad mesh (triangular/plane with slope)
void export_obstacle_mesh(const std::string &output_dir,
                          real_t base_plane,
                          real_t slope,
                          int slope_dir,
                          real_t y_min, real_t y_max,
                          real_t z_min, real_t z_max) {
    std::string obstacle_dir = output_dir + "/obstacle";
    sfem::create_directory(obstacle_dir.c_str());

    // Create 4 corner points of the obstacle plane
    // The plane equation: x = base_plane + slope * y (if slope_dir=1)
    std::vector<float> x_coords(4), y_coords(4), z_coords(4);

    if (slope_dir == 1) {  // slope in y direction
        // Point 0: (y_min, z_min)
        x_coords[0] = base_plane + slope * y_min;
        y_coords[0] = y_min;
        z_coords[0] = z_min;

        // Point 1: (y_max, z_min)
        x_coords[1] = base_plane + slope * y_max;
        y_coords[1] = y_max;
        z_coords[1] = z_min;

        // Point 2: (y_max, z_max)
        x_coords[2] = base_plane + slope * y_max;
        y_coords[2] = y_max;
        z_coords[2] = z_max;

        // Point 3: (y_min, z_max)
        x_coords[3] = base_plane + slope * y_min;
        y_coords[3] = y_min;
        z_coords[3] = z_max;
    } else {  // slope in z direction
        // Point 0: (y_min, z_min)
        x_coords[0] = base_plane + slope * z_min;
        y_coords[0] = y_min;
        z_coords[0] = z_min;

        // Point 1: (y_max, z_min)
        x_coords[1] = base_plane + slope * z_min;
        y_coords[1] = y_max;
        z_coords[1] = z_min;

        // Point 2: (y_max, z_max)
        x_coords[2] = base_plane + slope * z_max;
        y_coords[2] = y_max;
        z_coords[2] = z_max;

        // Point 3: (y_min, z_max)
        x_coords[3] = base_plane + slope * z_max;
        y_coords[3] = y_min;
        z_coords[3] = z_max;
    }

    // Write coordinates
    FILE* fx = fopen((obstacle_dir + "/x.raw").c_str(), "wb");
    FILE* fy = fopen((obstacle_dir + "/y.raw").c_str(), "wb");
    FILE* fz = fopen((obstacle_dir + "/z.raw").c_str(), "wb");
    fwrite(x_coords.data(), sizeof(float), 4, fx);
    fwrite(y_coords.data(), sizeof(float), 4, fy);
    fwrite(z_coords.data(), sizeof(float), 4, fz);
    fclose(fx);
    fclose(fy);
    fclose(fz);

    // Write quad element (as two triangles: 0-1-2 and 0-2-3)
    std::vector<int32_t> i0 = {0, 0};
    std::vector<int32_t> i1 = {1, 2};
    std::vector<int32_t> i2 = {2, 3};

    FILE* fi0 = fopen((obstacle_dir + "/i0.raw").c_str(), "wb");
    FILE* fi1 = fopen((obstacle_dir + "/i1.raw").c_str(), "wb");
    FILE* fi2 = fopen((obstacle_dir + "/i2.raw").c_str(), "wb");
    fwrite(i0.data(), sizeof(int32_t), 2, fi0);
    fwrite(i1.data(), sizeof(int32_t), 2, fi1);
    fwrite(i2.data(), sizeof(int32_t), 2, fi2);
    fclose(fi0);
    fclose(fi1);
    fclose(fi2);

    printf("Obstacle mesh exported to: %s\n", obstacle_dir.c_str());
    printf("  Corners: (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f)\n",
           x_coords[0], y_coords[0], z_coords[0],
           x_coords[1], y_coords[1], z_coords[1],
           x_coords[2], y_coords[2], z_coords[2],
           x_coords[3], y_coords[3], z_coords[3]);
}

// Export flat obstacle geometry as a simple quad mesh for a given contact direction
void export_flat_obstacle_mesh(const std::string &output_dir,
                               real_t contact_plane,
                               int contact_dir,
                               real_t a_min, real_t a_max,
                               real_t b_min, real_t b_max) {
    std::string obstacle_dir = output_dir + "/obstacle";
    sfem::create_directory(obstacle_dir.c_str());
    
    int h1, h2;
    if (contact_dir == 0) { h1 = 1; h2 = 2; }       // x = const, span y,z
    else if (contact_dir == 1) { h1 = 0; h2 = 2; }  // y = const, span x,z
    else { h1 = 0; h2 = 1; }                         // z = const, span x,y

    // Create 4 corner points of the obstacle plane
    std::vector<float> x_coords(4), y_coords(4), z_coords(4);

    float coords[4][3] = {
        {0.f, 0.f, 0.f},
        {0.f, 0.f, 0.f},
        {0.f, 0.f, 0.f},
        {0.f, 0.f, 0.f},
    };

    const float a_vals[2] = {static_cast<float>(a_min), static_cast<float>(a_max)};
    const float b_vals[2] = {static_cast<float>(b_min), static_cast<float>(b_max)};

    coords[0][contact_dir] = static_cast<float>(contact_plane);
    coords[0][h1] = a_vals[0];
    coords[0][h2] = b_vals[0];

    coords[1][contact_dir] = static_cast<float>(contact_plane);
    coords[1][h1] = a_vals[1];
    coords[1][h2] = b_vals[0];

    coords[2][contact_dir] = static_cast<float>(contact_plane);
    coords[2][h1] = a_vals[1];
    coords[2][h2] = b_vals[1];

    coords[3][contact_dir] = static_cast<float>(contact_plane);
    coords[3][h1] = a_vals[0];
    coords[3][h2] = b_vals[1];

    for (int i = 0; i < 4; ++i) {
        x_coords[i] = coords[i][0];
        y_coords[i] = coords[i][1];
        z_coords[i] = coords[i][2];
    }
    
    // Write coordinates
    FILE* fx = fopen((obstacle_dir + "/x.raw").c_str(), "wb");
    FILE* fy = fopen((obstacle_dir + "/y.raw").c_str(), "wb");
    FILE* fz = fopen((obstacle_dir + "/z.raw").c_str(), "wb");
    fwrite(x_coords.data(), sizeof(float), 4, fx);
    fwrite(y_coords.data(), sizeof(float), 4, fy);
    fwrite(z_coords.data(), sizeof(float), 4, fz);
    fclose(fx);
    fclose(fy);
    fclose(fz);
    
    // Write quad element (as two triangles: 0-1-2 and 0-2-3)
    std::vector<int32_t> i0 = {0, 0};
    std::vector<int32_t> i1 = {1, 2};
    std::vector<int32_t> i2 = {2, 3};
    
    FILE* fi0 = fopen((obstacle_dir + "/i0.raw").c_str(), "wb");
    FILE* fi1 = fopen((obstacle_dir + "/i1.raw").c_str(), "wb");
    FILE* fi2 = fopen((obstacle_dir + "/i2.raw").c_str(), "wb");
    fwrite(i0.data(), sizeof(int32_t), 2, fi0);
    fwrite(i1.data(), sizeof(int32_t), 2, fi1);
    fwrite(i2.data(), sizeof(int32_t), 2, fi2);
    fclose(fi0);
    fclose(fi1);
    fclose(fi2);
    
    printf("Obstacle mesh exported to: %s\n", obstacle_dir.c_str());
    printf("  Corners: (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f)\n",
           x_coords[0], y_coords[0], z_coords[0],
           x_coords[1], y_coords[1], z_coords[1],
           x_coords[2], y_coords[2], z_coords[2],
           x_coords[3], y_coords[3], z_coords[3]);
}

// Export stepped obstacle geometry as multiple quads
void export_stepped_obstacle_mesh(const std::string &output_dir,
                                  real_t base_plane,
                                  real_t step_height,
                                  real_t step_depth,
                                  int n_steps,
                                  real_t z_min, real_t z_max) {
    std::string obstacle_dir = output_dir + "/obstacle";
    sfem::create_directory(obstacle_dir.c_str());
    
    // Each step has 4 corners (forms a horizontal surface)
    // Plus vertical risers between steps
    int n_horizontal_quads = n_steps;
    int n_vertical_quads = n_steps - 1;
    int n_quads = n_horizontal_quads + n_vertical_quads;
    int n_points = 4 * n_quads;
    
    std::vector<float> x_coords(n_points), y_coords(n_points), z_coords(n_points);
    
    int pt_idx = 0;
    // Create horizontal surfaces for each step
    for (int s = 0; s < n_steps; s++) {
        real_t y0 = s * step_height;
        real_t y1 = (s + 1) * step_height;
        real_t x_plane = base_plane + s * step_depth;
        
        // 4 corners of horizontal surface
        x_coords[pt_idx] = x_plane; y_coords[pt_idx] = y0; z_coords[pt_idx] = z_min; pt_idx++;
        x_coords[pt_idx] = x_plane; y_coords[pt_idx] = y1; z_coords[pt_idx] = z_min; pt_idx++;
        x_coords[pt_idx] = x_plane; y_coords[pt_idx] = y1; z_coords[pt_idx] = z_max; pt_idx++;
        x_coords[pt_idx] = x_plane; y_coords[pt_idx] = y0; z_coords[pt_idx] = z_max; pt_idx++;
    }
    
    // Create vertical risers between steps
    for (int s = 0; s < n_steps - 1; s++) {
        real_t y = (s + 1) * step_height;
        real_t x0 = base_plane + s * step_depth;
        real_t x1 = base_plane + (s + 1) * step_depth;
        
        // 4 corners of vertical riser
        x_coords[pt_idx] = x0; y_coords[pt_idx] = y; z_coords[pt_idx] = z_min; pt_idx++;
        x_coords[pt_idx] = x1; y_coords[pt_idx] = y; z_coords[pt_idx] = z_min; pt_idx++;
        x_coords[pt_idx] = x1; y_coords[pt_idx] = y; z_coords[pt_idx] = z_max; pt_idx++;
        x_coords[pt_idx] = x0; y_coords[pt_idx] = y; z_coords[pt_idx] = z_max; pt_idx++;
    }
    
    // Write coordinates
    FILE* fx = fopen((obstacle_dir + "/x.raw").c_str(), "wb");
    FILE* fy = fopen((obstacle_dir + "/y.raw").c_str(), "wb");
    FILE* fz = fopen((obstacle_dir + "/z.raw").c_str(), "wb");
    fwrite(x_coords.data(), sizeof(float), n_points, fx);
    fwrite(y_coords.data(), sizeof(float), n_points, fy);
    fwrite(z_coords.data(), sizeof(float), n_points, fz);
    fclose(fx);
    fclose(fy);
    fclose(fz);
    
    // Write triangles (each quad = 2 triangles)
    std::vector<int32_t> i0(2 * n_quads), i1(2 * n_quads), i2(2 * n_quads);
    for (int q = 0; q < n_quads; q++) {
        int base = q * 4;
        // Triangle 1: 0-1-2
        i0[2*q] = base; i1[2*q] = base+1; i2[2*q] = base+2;
        // Triangle 2: 0-2-3
        i0[2*q+1] = base; i1[2*q+1] = base+2; i2[2*q+1] = base+3;
    }
    
    FILE* fi0 = fopen((obstacle_dir + "/i0.raw").c_str(), "wb");
    FILE* fi1 = fopen((obstacle_dir + "/i1.raw").c_str(), "wb");
    FILE* fi2 = fopen((obstacle_dir + "/i2.raw").c_str(), "wb");
    fwrite(i0.data(), sizeof(int32_t), 2 * n_quads, fi0);
    fwrite(i1.data(), sizeof(int32_t), 2 * n_quads, fi1);
    fwrite(i2.data(), sizeof(int32_t), 2 * n_quads, fi2);
    fclose(fi0);
    fclose(fi1);
    fclose(fi2);
    
    printf("Stepped obstacle mesh exported to: %s\n", obstacle_dir.c_str());
    printf("  %d steps, step_height=%.3f, step_depth=%.3f\n", n_steps, step_height, step_depth);
}

// Export hemisphere obstacle geometry
void export_hemisphere_obstacle_mesh(const std::string &output_dir,
                                     real_t center_x, real_t center_y, real_t center_z,
                                     real_t radius,
                                     int contact_dir,
                                     int n_segments = 16) {
    std::string obstacle_dir = output_dir + "/obstacle";
    sfem::create_directory(obstacle_dir.c_str());
    
    int h1, h2;
    if (contact_dir == 0) { h1 = 1; h2 = 2; }
    else if (contact_dir == 1) { h1 = 0; h2 = 2; }
    else { h1 = 0; h2 = 1; }

    real_t center[3] = {center_x, center_y, center_z};

    // Generate hemisphere mesh using spherical coordinates
    int n_rings = n_segments / 2;  // Number of latitude rings
    int n_points = 1 + n_rings * n_segments;  // Tip point + ring points
    int n_triangles = n_segments + (n_rings - 1) * n_segments * 2;  // Tip cap + body
    
    std::vector<float> x_coords(n_points), y_coords(n_points), z_coords(n_points);
    
    // Tip point of hemisphere (facing +contact_dir)
    {
        float coords[3] = {static_cast<float>(center_x),
                           static_cast<float>(center_y),
                           static_cast<float>(center_z)};
        coords[contact_dir] = static_cast<float>(center[contact_dir] + radius);
        x_coords[0] = coords[0];
        y_coords[0] = coords[1];
        z_coords[0] = coords[2];
    }
    
    int pt_idx = 1;
    for (int ring = 1; ring <= n_rings; ring++) {
        real_t phi = M_PI / 2.0 * ring / n_rings;  // 0 to pi/2 (tip to equator)
        real_t r_ring = radius * sin(phi);
        real_t c_ring = center[contact_dir] + radius * cos(phi);
        
        for (int seg = 0; seg < n_segments; seg++) {
            real_t theta = 2.0 * M_PI * seg / n_segments;
            float coords[3] = {static_cast<float>(center_x),
                               static_cast<float>(center_y),
                               static_cast<float>(center_z)};
            coords[contact_dir] = static_cast<float>(c_ring);
            coords[h1] = static_cast<float>(center[h1] + r_ring * cos(theta));
            coords[h2] = static_cast<float>(center[h2] + r_ring * sin(theta));
            x_coords[pt_idx] = coords[0];
            y_coords[pt_idx] = coords[1];
            z_coords[pt_idx] = coords[2];
            pt_idx++;
        }
    }
    
    // Write coordinates
    FILE* fx = fopen((obstacle_dir + "/x.raw").c_str(), "wb");
    FILE* fy = fopen((obstacle_dir + "/y.raw").c_str(), "wb");
    FILE* fz = fopen((obstacle_dir + "/z.raw").c_str(), "wb");
    fwrite(x_coords.data(), sizeof(float), n_points, fx);
    fwrite(y_coords.data(), sizeof(float), n_points, fy);
    fwrite(z_coords.data(), sizeof(float), n_points, fz);
    fclose(fx);
    fclose(fy);
    fclose(fz);
    
    // Generate triangles
    std::vector<int32_t> i0(n_triangles), i1(n_triangles), i2(n_triangles);
    int tri_idx = 0;
    
    // Top cap triangles (connect to top point)
    for (int seg = 0; seg < n_segments; seg++) {
        i0[tri_idx] = 0;  // Top point
        i1[tri_idx] = 1 + seg;
        i2[tri_idx] = 1 + (seg + 1) % n_segments;
        tri_idx++;
    }
    
    // Body triangles (between rings)
    for (int ring = 0; ring < n_rings - 1; ring++) {
        int ring_start = 1 + ring * n_segments;
        int next_ring_start = 1 + (ring + 1) * n_segments;
        
        for (int seg = 0; seg < n_segments; seg++) {
            int next_seg = (seg + 1) % n_segments;
            
            // First triangle
            i0[tri_idx] = ring_start + seg;
            i1[tri_idx] = next_ring_start + seg;
            i2[tri_idx] = ring_start + next_seg;
            tri_idx++;
            
            // Second triangle
            i0[tri_idx] = ring_start + next_seg;
            i1[tri_idx] = next_ring_start + seg;
            i2[tri_idx] = next_ring_start + next_seg;
            tri_idx++;
        }
    }
    
    FILE* fi0 = fopen((obstacle_dir + "/i0.raw").c_str(), "wb");
    FILE* fi1 = fopen((obstacle_dir + "/i1.raw").c_str(), "wb");
    FILE* fi2 = fopen((obstacle_dir + "/i2.raw").c_str(), "wb");
    fwrite(i0.data(), sizeof(int32_t), n_triangles, fi0);
    fwrite(i1.data(), sizeof(int32_t), n_triangles, fi1);
    fwrite(i2.data(), sizeof(int32_t), n_triangles, fi2);
    fclose(fi0);
    fclose(fi1);
    fclose(fi2);
    
    printf("Hemisphere obstacle mesh exported to: %s\n", obstacle_dir.c_str());
    printf("  Center: (%.3f, %.3f, %.3f), Radius: %.3f, contact_dir=%d\n",
           center_x, center_y, center_z, radius, contact_dir);
}

int test_mooney_rivlin_gravity() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    int SFEM_BASE_RESOLUTION = 10;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    std::shared_ptr<sfem::Mesh> mesh;
    const char *mesh_path = getenv("SFEM_MESH");
    if (mesh_path && mesh_path[0] != '\0') {
        printf("Loading mesh from: %s\n", mesh_path);
        mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), mesh_path);
    } else {
        mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm),
                                             SFEM_BASE_RESOLUTION,
                                             SFEM_BASE_RESOLUTION,
                                             SFEM_BASE_RESOLUTION,  // Grid
                                             0,
                                             0,
                                             0,  // Origin
                                             2,
                                             1,
                                             1  // Dimensions (cube)
        );
    }

    auto fs = sfem::FunctionSpace::create(mesh, 3);  // 3D displacement
    auto f  = sfem::Function::create(fs);

    // Operator
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);

    // LumpedMass
    auto mass_op = sfem::create_op(fs, "LumpedMass", es);
    mass_op->initialize();

    // Material Parameters from environment
    real_t SFEM_C10          = 1.0;
    real_t SFEM_C01          = 0.5;
    real_t SFEM_BULK_MODULUS = 100.0;
    real_t SFEM_DT           = 0.1;
    SFEM_READ_ENV(SFEM_C10, atof);
    SFEM_READ_ENV(SFEM_C01, atof);
    SFEM_READ_ENV(SFEM_BULK_MODULUS, atof);
    SFEM_READ_ENV(SFEM_DT, atof);

    op->set_C10(SFEM_C10);
    op->set_C01(SFEM_C01);
    op->set_K(SFEM_BULK_MODULUS);

    real_t dt = SFEM_DT;
    op->set_dt(dt);


    int SFEM_ENABLE_CONTACT = false;
    SFEM_READ_ENV(SFEM_ENABLE_CONTACT, atoi);

    // WLF temperature shift parameters (set BEFORE Prony terms to avoid redundant calculations)
    int SFEM_USE_WLF = 1;
    real_t SFEM_WLF_C1 = 16.6253;
    real_t SFEM_WLF_C2 = 47.4781;
    real_t SFEM_WLF_T_REF = -54.29;
    real_t SFEM_TEMPERATURE = 20.0;
    SFEM_READ_ENV(SFEM_USE_WLF, atoi);
    SFEM_READ_ENV(SFEM_WLF_C1, atof);
    SFEM_READ_ENV(SFEM_WLF_C2, atof);
    SFEM_READ_ENV(SFEM_WLF_T_REF, atof);
    SFEM_READ_ENV(SFEM_TEMPERATURE, atof);
    
    if (SFEM_USE_WLF) {
        // Set WLF params and enable BEFORE setting Prony terms
        op->set_wlf_params(SFEM_WLF_C1, SFEM_WLF_C2, SFEM_WLF_T_REF);
        op->set_temperature(SFEM_TEMPERATURE);
        op->enable_wlf(true);
        printf("WLF enabled: C1=%.4f, C2=%.4f, T_ref=%.2f, T=%.2f\n",
               SFEM_WLF_C1, SFEM_WLF_C2, SFEM_WLF_T_REF, SFEM_TEMPERATURE);
        fflush(stdout);
    }

    // Prony series parameters from environment
    // Format: comma-separated values, e.g., "0.15,0.15,0.10,0.05"
    // Default: 4 terms with sum(g) = 0.45, g_inf = 0.55
    std::vector<real_t> g_prony   = {0.15, 0.15, 0.10, 0.05};
    std::vector<real_t> tau_prony = {0.1, 1.0, 10.0, 100.0};
    
    // Read from environment if provided
    const char* env_g   = getenv("SFEM_PRONY_G");
    const char* env_tau = getenv("SFEM_PRONY_TAU");
    
    if (env_g && env_tau) {
        g_prony.clear();
        tau_prony.clear();
        
        // Parse comma-separated g values
        std::string g_str(env_g);
        std::stringstream g_ss(g_str);
        std::string token;
        while (std::getline(g_ss, token, ',')) {
            g_prony.push_back(std::stod(token));
        }
        
        // Parse comma-separated tau values
        std::string tau_str(env_tau);
        std::stringstream tau_ss(tau_str);
        while (std::getline(tau_ss, token, ',')) {
            tau_prony.push_back(std::stod(token));
        }
        
        if (g_prony.size() != tau_prony.size()) {
            printf("Error: SFEM_PRONY_G and SFEM_PRONY_TAU must have the same number of terms!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("Prony series from environment: %zu terms\n", g_prony.size());
        for (size_t i = 0; i < g_prony.size(); ++i) {
            printf("  term %zu: g=%.6f, tau=%.6f\n", i+1, g_prony[i], tau_prony[i]);
        }
    }
    
    // set_prony_terms will call compute_prony_coefficients() with WLF already enabled
    op->set_prony_terms((int)g_prony.size(), g_prony.data(), tau_prony.data());

    op->initialize();
    op->initialize_history();
    f->add_operator(op);

    // Read contact direction early (needed for BC selection)
    int SFEM_CONTACT_DIR = 2;          // Contact direction: 0=x, 1=y, 2=z
    SFEM_READ_ENV(SFEM_CONTACT_DIR, atoi);

    // BC - Sidesets (use mesh bounds so custom meshes work)
    auto coords = mesh->points();
    geom_t* const* coords_ptr = coords->data();
    real_t min_c[3] = {std::numeric_limits<real_t>::max(),
                       std::numeric_limits<real_t>::max(),
                       std::numeric_limits<real_t>::max()};
    real_t max_c[3] = {std::numeric_limits<real_t>::lowest(),
                       std::numeric_limits<real_t>::lowest(),
                       std::numeric_limits<real_t>::lowest()};
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const real_t v = coords_ptr[d][i];
            if (v < min_c[d]) min_c[d] = v;
            if (v > max_c[d]) max_c[d] = v;
        }
    }
    real_t eps_c[3];
    for (int d = 0; d < 3; ++d) {
        const real_t range = max_c[d] - min_c[d];
        // Use a larger epsilon for complex meshes (like lion) that may have non-planar boundaries
        // 5% of the range should capture most of the "bottom" region
        eps_c[d] = (range > 0) ? range * 0.05 : 1e-3;
    }
    printf("Mesh bounds: x=[%.4f, %.4f], y=[%.4f, %.4f], z=[%.4f, %.4f]\n",
           min_c[0], max_c[0], min_c[1], max_c[1], min_c[2], max_c[2]);
    printf("Sideset epsilon: eps_x=%.6f, eps_y=%.6f, eps_z=%.6f\n", eps_c[0], eps_c[1], eps_c[2]);

    auto make_sideset = [&](int dir, bool is_min) {
        const real_t target = is_min ? min_c[dir] : max_c[dir];
        const real_t eps = eps_c[dir];
        if (dir == 0) {
            return sfem::Sideset::create_from_selector(
                mesh, [=](const geom_t x, const geom_t, const geom_t) -> bool {
                    return fabs(x - target) <= eps;
                });
        } else if (dir == 1) {
            return sfem::Sideset::create_from_selector(
                mesh, [=](const geom_t, const geom_t y, const geom_t) -> bool {
                    return fabs(y - target) <= eps;
                });
        } else {
            return sfem::Sideset::create_from_selector(
                mesh, [=](const geom_t, const geom_t, const geom_t z) -> bool {
                    return fabs(z - target) <= eps;
                });
        }
    };

    auto min_sideset_x = make_sideset(0, true);
    auto max_sideset_x = make_sideset(0, false);
    auto min_sideset_y = make_sideset(1, true);
    auto max_sideset_y = make_sideset(1, false);
    auto min_sideset_z = make_sideset(2, true);
    auto max_sideset_z = make_sideset(2, false);

    auto contact_sideset = (SFEM_CONTACT_DIR == 0) ? min_sideset_x
                           : (SFEM_CONTACT_DIR == 1) ? min_sideset_y
                                                     : min_sideset_z;
    auto opposite_sideset = (SFEM_CONTACT_DIR == 0) ? max_sideset_x
                           : (SFEM_CONTACT_DIR == 1) ? max_sideset_y
                                                     : max_sideset_z;

    // **Boundary conditions based on contact direction**
    if (!SFEM_ENABLE_CONTACT) {
        // No contact: fix opposite side completely
        sfem::DirichletConditions::Condition right_bc_x{.sidesets = opposite_sideset, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition right_bc_y{.sidesets = opposite_sideset, .value = 0, .component = 1};
        sfem::DirichletConditions::Condition right_bc_z{.sidesets = opposite_sideset, .value = 0, .component = 2};
        auto                                 conds = sfem::create_dirichlet_conditions(fs, {right_bc_x, right_bc_y, right_bc_z}, es);
        f->add_constraint(conds);
    } else if (SFEM_CONTACT_DIR == 1) {
        // Y-direction contact (vertical drop): fix max-y surface in x and z only (allow vertical motion)
        // This prevents lateral drift while allowing the object to fall
        sfem::DirichletConditions::Condition top_bc_x{.sidesets = max_sideset_y, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition top_bc_z{.sidesets = max_sideset_y, .value = 0, .component = 2};
        auto                                 conds = sfem::create_dirichlet_conditions(fs, {top_bc_x, top_bc_z}, es);
        f->add_constraint(conds);
        printf("Y-direction contact: fixed top surface in x,z directions\n");
    } else if (SFEM_CONTACT_DIR == 2) {
        // Z-direction contact: fix max-z surface in x and y only (allow vertical motion)
        sfem::DirichletConditions::Condition top_bc_x{.sidesets = max_sideset_z, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition top_bc_y{.sidesets = max_sideset_z, .value = 0, .component = 1};
        auto                                 conds = sfem::create_dirichlet_conditions(fs, {top_bc_x, top_bc_y}, es);
        f->add_constraint(conds);
        printf("Z-direction contact: fixed max-z surface in x,y directions\n");
    }
    // For x-direction contact (CONTACT_DIR=0), no BC is needed - object is free to move in -x

    // **GRAVITY as body force instead of Neumann surface force**
    real_t SFEM_GRAVITY = 9.81;  // Positive value, direction handled below
    SFEM_READ_ENV(SFEM_GRAVITY, atof);

    const ptrdiff_t ndofs   = fs->n_dofs();
    auto            x       = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs     = sfem::create_buffer<real_t>(ndofs, es);
    auto            delta_x = sfem::create_buffer<real_t>(ndofs, es);
    auto            diag    = sfem::create_buffer<real_t>(ndofs, es);

    // Lumped mass vector (diagonal of M)
    real_t SFEM_DENSITY = 1000.0;
    SFEM_READ_ENV(SFEM_DENSITY, atof);
    real_t density   = SFEM_DENSITY;
    auto   mass_diag = sfem::create_buffer<real_t>(ndofs, es);
    mass_op->hessian_diag(nullptr, mass_diag->data());
    // Scale mass by density
    auto blas = sfem::blas<real_t>(es);
    blas->scal(ndofs, density, mass_diag->data());
    f->set_value_to_constrained_dofs(1.0, mass_diag->data());  // Set 1 for BC nodes

    // **Gravity force vector** (negative direction of SFEM_CONTACT_DIR)
    // This replaces the Neumann surface force
    auto f_gravity_neg = sfem::create_buffer<real_t>(ndofs, es);
    blas->zeros(ndofs, f_gravity_neg->data());
    
    // Gravity direction: -SFEM_CONTACT_DIR (e.g., -x if CONTACT_DIR=0, -y if CONTACT_DIR=1)
    // F_gravity = m * g in -CONTACT_DIR direction = -m*g in +CONTACT_DIR direction
    // For gradient convention (we add -gradient to RHS), we need to store -F_gravity
    // So we store -(-m*g) = +m*g when gravity is in +CONTACT_DIR direction
    // But for hemisphere test, gravity should be in -CONTACT_DIR (downward)
    // So the sign needs to be NEGATIVE for the gravity to pull DOWN
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    for (ptrdiff_t node = 0; node < n_nodes; ++node) {
        // For gravity in -CONTACT_DIR direction: F_gravity[CONTACT_DIR] = -m*g
        // Stored as -F_gravity = +m*g (POSITIVE value causes motion in -CONTACT_DIR)
        // Wait, let me check the original code logic for x-direction:
        // Original: pushed in -x, contact on left (x=0 plane), worked correctly
        // So positive f_gravity_neg causes motion in -CONTACT_DIR
        f_gravity_neg->data()[node * 3 + SFEM_CONTACT_DIR] = mass_diag->data()[node * 3 + SFEM_CONTACT_DIR] * SFEM_GRAVITY;
    }
    // Apply BC to gravity force (zero at constrained DOFs)
    f->set_value_to_constrained_dofs(0.0, f_gravity_neg->data());

    // Output setup
    bool SFEM_ENABLE_OUTPUT = false;
    SFEM_READ_ENV(SFEM_ENABLE_OUTPUT, atoi);
    auto output = create_output(f, "test_mooney_rivlin_gravity");

    // Newmark state
    auto v      = sfem::create_buffer<real_t>(ndofs, es);
    auto a      = sfem::create_buffer<real_t>(ndofs, es);
    auto u_pred = sfem::create_buffer<real_t>(ndofs, es);  // Predicted displacement
    auto v_pred = sfem::create_buffer<real_t>(ndofs, es);  // Predicted velocity

    blas->zeros(ndofs, x->data());
    blas->zeros(ndofs, v->data());
    blas->zeros(ndofs, a->data());

    // Newmark parameters (implicit, unconditionally stable)
    real_t beta_nm  = 0.25;
    real_t gamma_nm = 0.5;
    real_t c0       = 1.0 / (beta_nm * dt * dt);  // Coefficient for M in effective stiffness

    // Matrix assembly buffers (BSR format: 3x3 blocks)
    auto            graph      = fs->node_to_node_graph();
    const int       block_size = 3;
    auto            values     = sfem::create_buffer<real_t>(graph->nnz() * block_size * block_size, es);

    // Linear Solver Wrapper (BSR SpMV)
    auto linear_op_apply = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [=](const real_t *const in, real_t *const out) {
                sfem::bsr_spmv<count_t, idx_t, real_t>(n_nodes,
                                                       n_nodes,
                                                       block_size,
                                                       graph->rowptr()->data(),
                                                       graph->colidx()->data(),
                                                       values->data(),
                                                       0.0,
                                                       in,
                                                       out);
            },
            es);

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
    auto                                                  jacobi = sfem::create_shiftable_jacobi(diag, es);
    sfem::SharedBuffer<real_t>                            lower_bound;

    // Contact parameters (LEFT side contact, same as original)
    real_t SFEM_CONTACT_PLANE = -0.1;  // Position of contact plane (left of x=0)
    // Note: SFEM_CONTACT_DIR is declared earlier (before gravity setup)
    SFEM_READ_ENV(SFEM_CONTACT_PLANE, atof);

    // Obstacle type: 0=flat, 1=triangular (NOT RECOMMENDED), 2=stepped, 4=hemisphere
    int SFEM_OBSTACLE_TYPE = 4;
    real_t SFEM_OBSTACLE_SLOPE = 0.2;  // For triangular: slope per unit y
    int SFEM_SLOPE_DIR = 1;            // Direction for slope (1=y, 2=z)
    
    // Stepped obstacle parameters
    real_t SFEM_STEP_HEIGHT = 0.25;    // Height of each step
    real_t SFEM_STEP_DEPTH = -0.05;    // Depth change per step (negative = deeper into -x)
    
    SFEM_READ_ENV(SFEM_OBSTACLE_TYPE, atoi);
    SFEM_READ_ENV(SFEM_OBSTACLE_SLOPE, atof);
    SFEM_READ_ENV(SFEM_SLOPE_DIR, atoi);
    SFEM_READ_ENV(SFEM_STEP_HEIGHT, atof);
    SFEM_READ_ENV(SFEM_STEP_DEPTH, atof);
    
    // Hemisphere obstacle parameters (type=4)
    real_t SFEM_HEMISPHERE_CENTER_X = 1.0;   // Default: center of mesh in x
    real_t SFEM_HEMISPHERE_CENTER_Y = -0.3;  // Below the mesh
    real_t SFEM_HEMISPHERE_CENTER_Z = 0.5;   // Center of mesh in z
    real_t SFEM_HEMISPHERE_RADIUS = 0.5;
    SFEM_READ_ENV(SFEM_HEMISPHERE_CENTER_X, atof);
    SFEM_READ_ENV(SFEM_HEMISPHERE_CENTER_Y, atof);
    SFEM_READ_ENV(SFEM_HEMISPHERE_CENTER_Z, atof);
    SFEM_READ_ENV(SFEM_HEMISPHERE_RADIUS, atof);

    // Get reference coordinates for obstacle
    // coords already retrieved above for bounds
    
    // Store contact node indices
    std::vector<ptrdiff_t> contact_node_indices;

    if (!SFEM_ENABLE_CONTACT) {
        auto cg = sfem::create_cg<real_t>(linear_op_apply, es);
        cg->set_n_dofs(ndofs);
        cg->set_max_it(2000);
        cg->set_rtol(1e-5);
        cg->verbose = false;

        // Preconditioner (Jacobi)
        cg->set_preconditioner_op(jacobi);

        solver = cg;
    } else {
        auto mprgp = sfem::create_mprgp(linear_op_apply, es);
        lower_bound = sfem::create_buffer<real_t>(ndofs, es);

        // For flat plane (0), stepped (2), triangular (1): use ALL nodes for contact detection
        // Each node has its own constraint based on its reference coordinate
        // For hemisphere (4): use only contact boundary nodes
        if (SFEM_OBSTACLE_TYPE == 0 || SFEM_OBSTACLE_TYPE == 1 || SFEM_OBSTACLE_TYPE == 2) {
            contact_node_indices.resize(n_nodes);
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                contact_node_indices[i] = i;
            }
            printf("Obstacle (type=%d): using ALL %ld nodes for contact detection\n", 
                   SFEM_OBSTACLE_TYPE, (long)n_nodes);
        } else {
            // Hemisphere (4): only use contact boundary nodes
            auto lnodes = sfem::create_nodeset_from_sideset(fs, contact_sideset[0]);
            const ptrdiff_t nbnodes = lnodes->size();
            contact_node_indices.resize(nbnodes);
            for (ptrdiff_t i = 0; i < nbnodes; i++) {
                contact_node_indices[i] = lnodes->data()[i];
            }
            printf("Hemisphere obstacle (type=%d): using %ld boundary nodes for contact detection\n", 
                   SFEM_OBSTACLE_TYPE, (long)nbnodes);
        }

        // Initial lower bound computation (displacement = 0 at start)
        geom_t* const* coords_ptr = coords->data();
        if (SFEM_OBSTACLE_TYPE == 2) {
            // Stepped obstacle
            compute_contact_lower_bound_stepped(
                lower_bound->data(),
                x->data(),
                coords_ptr,
                contact_node_indices.data(),
                (ptrdiff_t)contact_node_indices.size(),
                SFEM_CONTACT_PLANE,
                SFEM_STEP_HEIGHT,
                SFEM_STEP_DEPTH,
                SFEM_CONTACT_DIR,
                SFEM_SLOPE_DIR,
                block_size,
                ndofs);
        } else if (SFEM_OBSTACLE_TYPE == 1) {
            // Triangular obstacle (warning: may have penetration issues!)
            compute_contact_lower_bound_triangular(
                lower_bound->data(),
                x->data(),
                coords_ptr,
                contact_node_indices.data(),
                (ptrdiff_t)contact_node_indices.size(),
                SFEM_CONTACT_PLANE,
                SFEM_OBSTACLE_SLOPE,
                SFEM_CONTACT_DIR,
                SFEM_SLOPE_DIR,
                block_size,
                ndofs);
        } else if (SFEM_OBSTACLE_TYPE == 4) {
            // Hemisphere obstacle
            compute_contact_lower_bound_hemisphere(
                lower_bound->data(),
                x->data(),
                coords_ptr,
                contact_node_indices.data(),
                (ptrdiff_t)contact_node_indices.size(),
                SFEM_HEMISPHERE_CENTER_X,
                SFEM_HEMISPHERE_CENTER_Y,
                SFEM_HEMISPHERE_CENTER_Z,
                SFEM_HEMISPHERE_RADIUS,
                SFEM_CONTACT_DIR,
                block_size,
                ndofs);
        } else {
            // Flat plane - use absolute coordinate (0)
            compute_contact_lower_bound(
                lower_bound->data(),
                x->data(),
                coords_ptr,
                contact_node_indices.data(),
                (ptrdiff_t)contact_node_indices.size(),
                0.0,  // Contact plane at z=0 (absolute coordinate)
                SFEM_CONTACT_DIR,
                block_size,
                ndofs);
        }

        mprgp->verbose = false;
        mprgp->set_lower_bound(lower_bound);
        mprgp->set_preconditioner_op(jacobi);  // Must set preconditioner!
        mprgp->set_max_it(2000);
        mprgp->set_rtol(1e-5);

        solver = mprgp;
    }

    // 5. Time Loop with Full Newmark Integration
    real_t SFEM_T = 8.0;
    SFEM_READ_ENV(SFEM_T, atof);
    real_t t           = 0;
    size_t steps       = 0;
    size_t export_freq = 1;

    auto f_int        = sfem::create_buffer<real_t>(ndofs, es);
    auto inertia_term = sfem::create_buffer<real_t>(ndofs, es);

    printf("===== Mooney-Rivlin Gravity Contact Test =====\n");
    printf("Mesh: %ld nodes, %ld DOFs\n", (long)n_nodes, (long)ndofs);
    printf("Newmark parameters: beta=%.2f, gamma=%.2f, c0=%.2e, dt=%.3f, density=%.1f\n", beta_nm, gamma_nm, c0, dt, density);
    printf("Time: T=%.2f, dt=%.3f\n", SFEM_T, dt);
    printf("Gravity: %.2f m/s^2 in direction %d\n", SFEM_GRAVITY, -SFEM_CONTACT_DIR);
    printf("Contact: enabled=%d, obstacle_type=%d\n", SFEM_ENABLE_CONTACT, SFEM_OBSTACLE_TYPE);
    if (SFEM_OBSTACLE_TYPE == 1) {
        printf("Triangular obstacle (MPRGP): slope=%.3f in direction %d\n", SFEM_OBSTACLE_SLOPE, SFEM_SLOPE_DIR);
        printf("  WARNING: May have penetration issues with box constraints!\n");
        printf("  Obstacle equation: x_contact = %.2f + %.3f * coord[%d]\n", 
               SFEM_CONTACT_PLANE, SFEM_OBSTACLE_SLOPE, SFEM_SLOPE_DIR);
    } else if (SFEM_OBSTACLE_TYPE == 2) {
        printf("Stepped obstacle (MPRGP): step_height=%.3f, step_depth=%.3f\n", SFEM_STEP_HEIGHT, SFEM_STEP_DEPTH);
        printf("  Obstacle equation: x_contact = %.2f + floor(coord[%d]/%.3f) * %.3f\n", 
               SFEM_CONTACT_PLANE, SFEM_SLOPE_DIR, SFEM_STEP_HEIGHT, SFEM_STEP_DEPTH);
    } else if (SFEM_OBSTACLE_TYPE == 4) {
        printf("Hemisphere obstacle (MPRGP): center=(%.3f, %.3f, %.3f), radius=%.3f\n",
               SFEM_HEMISPHERE_CENTER_X, SFEM_HEMISPHERE_CENTER_Y, SFEM_HEMISPHERE_CENTER_Z, SFEM_HEMISPHERE_RADIUS);
    }

    // Export obstacle mesh for visualization
    if (SFEM_ENABLE_OUTPUT && SFEM_ENABLE_CONTACT) {
        // Get mesh bounds for all axes
        real_t min_c[3] = {std::numeric_limits<real_t>::max(),
                           std::numeric_limits<real_t>::max(),
                           std::numeric_limits<real_t>::max()};
        real_t max_c[3] = {std::numeric_limits<real_t>::lowest(),
                           std::numeric_limits<real_t>::lowest(),
                           std::numeric_limits<real_t>::lowest()};
        geom_t* const* coords_ptr = coords->data();
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            for (int d = 0; d < 3; ++d) {
                const real_t v = coords_ptr[d][i];
                if (v < min_c[d]) min_c[d] = v;
                if (v > max_c[d]) max_c[d] = v;
            }
        }
        
        // Export obstacle geometry based on type
        if (SFEM_OBSTACLE_TYPE == 2) {
            // Stepped obstacle
            const real_t y_min = min_c[1];
            const real_t y_max = max_c[1];
            const real_t z_min = min_c[2];
            const real_t z_max = max_c[2];
            int n_steps = (int)(y_max / SFEM_STEP_HEIGHT) + 1;
            export_stepped_obstacle_mesh("test_mooney_rivlin_gravity",
                                         SFEM_CONTACT_PLANE,
                                         SFEM_STEP_HEIGHT,
                                         SFEM_STEP_DEPTH,
                                         n_steps,
                                         z_min, z_max);
        } else if (SFEM_OBSTACLE_TYPE == 4) {
            // Hemisphere obstacle
            export_hemisphere_obstacle_mesh("test_mooney_rivlin_gravity",
                                            SFEM_HEMISPHERE_CENTER_X,
                                            SFEM_HEMISPHERE_CENTER_Y,
                                            SFEM_HEMISPHERE_CENTER_Z,
                                            SFEM_HEMISPHERE_RADIUS,
                                            SFEM_CONTACT_DIR);
        } else if (SFEM_OBSTACLE_TYPE == 1) {
            // Triangular obstacle
            const real_t y_min = min_c[1];
            const real_t y_max = max_c[1];
            const real_t z_min = min_c[2];
            const real_t z_max = max_c[2];
            export_obstacle_mesh("test_mooney_rivlin_gravity",
                                SFEM_CONTACT_PLANE,
                                SFEM_OBSTACLE_SLOPE,
                                SFEM_SLOPE_DIR,
                                y_min, y_max,
                                z_min, z_max);
        } else {
            // Flat obstacle (type 0)
            // SFEM_CONTACT_PLANE is a displacement constraint, actual plane position is:
            // min_c[contact_dir] + SFEM_CONTACT_PLANE
            const real_t actual_plane_pos = min_c[SFEM_CONTACT_DIR] + SFEM_CONTACT_PLANE;
            int h1, h2;
            if (SFEM_CONTACT_DIR == 0) { h1 = 1; h2 = 2; }
            else if (SFEM_CONTACT_DIR == 1) { h1 = 0; h2 = 2; }
            else { h1 = 0; h2 = 1; }
            export_flat_obstacle_mesh("test_mooney_rivlin_gravity",
                                      actual_plane_pos,
                                      SFEM_CONTACT_DIR,
                                      min_c[h1], max_c[h1],
                                      min_c[h2], max_c[h2]);
        }
    }

    // Output
    if (SFEM_ENABLE_OUTPUT) {
        output->write_time_step("disp", t, x->data());
        output->write_time_step("velocity", t, v->data());
        output->write_time_step("acceleration", t, a->data());
        output->log_time(t);
    }

    // Time counting
    double total_hessian_time = 0;

    while (t < SFEM_T) {
        printf("Step %zu: t=%.3f\n", steps, t);

        // Newmark Prediction Step
        // u_pred = u_n + dt*v_n + (0.5-beta)*dt^2*a_n
        // v_pred = v_n + (1-gamma)*dt*a_n
        blas->copy(ndofs, x->data(), u_pred->data());
        blas->axpy(ndofs, dt, v->data(), u_pred->data());
        blas->axpy(ndofs, (0.5 - beta_nm) * dt * dt, a->data(), u_pred->data());

        blas->copy(ndofs, v->data(), v_pred->data());
        blas->axpy(ndofs, (1.0 - gamma_nm) * dt, a->data(), v_pred->data());

        // Use prediction as initial guess for Newton
        blas->copy(ndofs, u_pred->data(), x->data());

        // Compute lower_bound based on current displacement (before solve)
        if (SFEM_ENABLE_CONTACT && !contact_node_indices.empty()) {
            geom_t* const* coords_ptr = coords->data();
            if (SFEM_OBSTACLE_TYPE == 2) {
                // Stepped obstacle
                compute_contact_lower_bound_stepped(
                    lower_bound->data(),
                    x->data(),
                    coords_ptr,
                    contact_node_indices.data(),
                    (ptrdiff_t)contact_node_indices.size(),
                    SFEM_CONTACT_PLANE,
                    SFEM_STEP_HEIGHT,
                    SFEM_STEP_DEPTH,
                    SFEM_CONTACT_DIR,
                    SFEM_SLOPE_DIR,
                    block_size,
                    ndofs);
            } else if (SFEM_OBSTACLE_TYPE == 4) {
                // Hemisphere obstacle
                compute_contact_lower_bound_hemisphere(
                    lower_bound->data(),
                    x->data(),
                    coords_ptr,
                    contact_node_indices.data(),
                    (ptrdiff_t)contact_node_indices.size(),
                    SFEM_HEMISPHERE_CENTER_X,
                    SFEM_HEMISPHERE_CENTER_Y,
                    SFEM_HEMISPHERE_CENTER_Z,
                    SFEM_HEMISPHERE_RADIUS,
                    SFEM_CONTACT_DIR,
                    block_size,
                    ndofs);
            } else if (SFEM_OBSTACLE_TYPE == 1) {
                // Triangular obstacle
                compute_contact_lower_bound_triangular(
                    lower_bound->data(),
                    x->data(),
                    coords_ptr,
                    contact_node_indices.data(),
                    (ptrdiff_t)contact_node_indices.size(),
                    SFEM_CONTACT_PLANE,
                    SFEM_OBSTACLE_SLOPE,
                    SFEM_CONTACT_DIR,
                    SFEM_SLOPE_DIR,
                    block_size,
                    ndofs);
            } else {
                // Flat obstacle (type 0) - use absolute coordinate (0)
                compute_contact_lower_bound(
                    lower_bound->data(),
                    x->data(),
                    coords_ptr,
                    contact_node_indices.data(),
                    (ptrdiff_t)contact_node_indices.size(),
                    0.0,  // Contact plane at z=0 (absolute coordinate)
                    SFEM_CONTACT_DIR,
                    block_size,
                    ndofs);
            }
        }

        // Newton Loop with Inertia
        for (int iter = 0; iter < 20; ++iter) {
            // Residual: R = M*a + F_int(u) - F_ext + F_contact
            // where a = c0*(u - u_pred) from Newmark relation
            blas->zeros(ndofs, rhs->data());

            // Inertia term: c0 * M * (u - u_pred)
            blas->zeros(ndofs, inertia_term->data());
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                inertia_term->data()[i] = c0 * mass_diag->data()[i] * (x->data()[i] - u_pred->data()[i]);
            }
            real_t inertia_norm = blas->norm2(ndofs, inertia_term->data());
            blas->axpy(ndofs, 1.0, inertia_term->data(), rhs->data());

            // F_int(x)
            blas->zeros(ndofs, f_int->data());
            op->gradient(x->data(), f_int->data());
            real_t f_int_norm = blas->norm2(ndofs, f_int->data());
            blas->axpy(ndofs, 1.0, f_int->data(), rhs->data());

            // F_ext (Gravity body force) - same convention as Neumann gradient
            real_t f_grav_norm = blas->norm2(ndofs, f_gravity_neg->data());
            blas->axpy(ndofs, 1.0, f_gravity_neg->data(), rhs->data());

            // Apply BC to residual
            f->set_value_to_constrained_dofs(0.0, rhs->data());
            real_t r_norm = blas->norm2(ndofs, rhs->data());
            real_t x_norm = blas->norm2(ndofs, x->data());

            printf("  Iter %d: |R|=%e |u|=%e |M*a|=%e |F_int|=%e |F_grav|=%e\n", 
                   iter, r_norm, x_norm, inertia_norm, f_int_norm, f_grav_norm);
            if (r_norm < 1e-8) break;

            // ===== Tangent Stiffness: K_eff = K_tan + c0*M =====
            blas->zeros(values->size(), values->data());
            auto t_start = std::chrono::high_resolution_clock::now();
            op->hessian_bsr(x->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data());
            auto t_end = std::chrono::high_resolution_clock::now();
            total_hessian_time += std::chrono::duration<double>(t_end - t_start).count();

            auto      rowptr = graph->rowptr()->data();
            auto      colidx = graph->colidx()->data();
            auto      vals   = values->data();
            const int bs2    = block_size * block_size;

            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                for (count_t k = rowptr[node]; k < rowptr[node + 1]; ++k) {
                    if (colidx[k] == (idx_t)node) {
                        for (int d = 0; d < block_size; ++d) {
                            ptrdiff_t dof_idx = node * block_size + d;
                            vals[k * bs2 + d * block_size + d] += c0 * mass_diag->data()[dof_idx];
                        }
                        break;
                    }
                }
            }

            // Extract diagonal for Jacobi preconditioner
            blas->zeros(ndofs, diag->data());
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                for (count_t k = rowptr[node]; k < rowptr[node + 1]; ++k) {
                    if (colidx[k] == (idx_t)node) {
                        for (int d = 0; d < block_size; ++d) {
                            ptrdiff_t dof_idx     = node * block_size + d;
                            diag->data()[dof_idx] = vals[k * bs2 + d * block_size + d];
                        }
                        break;
                    }
                }
            }
            
            jacobi->set_diag(diag);

            blas->scal(ndofs, -1.0, rhs->data());
            blas->zeros(ndofs, delta_x->data());
            solver->apply(rhs->data(), delta_x->data());

            // u = u + dx
            blas->axpy(ndofs, 1.0, delta_x->data(), x->data());

            if (SFEM_ENABLE_CONTACT) {
                blas->axpy(ndofs, -1, delta_x->data(), lower_bound->data());
            }
        }

        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            a->data()[i] = c0 * (x->data()[i] - u_pred->data()[i]);
            v->data()[i] = v_pred->data()[i] + gamma_nm * dt * a->data()[i];
        }

        // Update history
        op->update_history(x->data());

        t += dt;
        steps++;

        // Output
        if (SFEM_ENABLE_OUTPUT && steps % export_freq == 0) {
            output->write_time_step("disp", t, x->data());
            output->write_time_step("velocity", t, v->data());
            output->write_time_step("acceleration", t, a->data());
            output->log_time(t);
        }
    }

    printf("===== Test Completed =====\n");
    printf("Total Hessian assembly time: %.3f s\n", total_hessian_time);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_mooney_rivlin_gravity);
    SFEM_UNIT_TEST_FINALIZE();
}
