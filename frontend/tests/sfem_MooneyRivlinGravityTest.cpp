#include <math.h>
#include <stdio.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_Function.hpp"
#include "sfem_mprgp.hpp"
#include "sfem_prony_test_material.hpp"
#include "sfem_test.hpp"
#include "smesh_grid.hpp"
#include "smesh_sideset.hpp"

#include "sfem_ContactStabilisedNewmarkScheme.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_NodeObstacleContact.hpp"

// The obstacle as a signed distance function, positive inside: a shape cut off by a box wide
// enough for the mesh to fall onto it and spread over it, the cut's edges rounded with `edge`.
// It is sampled on an SDF grid over the box and a margin (see obstacle_grid); the grid is the
// obstacle the contact sees. The body approaches the obstacle along -dir.
//
// SFEM_OBSTACLE_TYPE:
//   0: a plane at SFEM_CONTACT_PLANE along dir
//   1: a plane sloping by SFEM_OBSTACLE_SLOPE along SFEM_SLOPE_DIR, highest under the mesh at
//      SFEM_CONTACT_PLANE
//   2: a staircase descending along SFEM_SLOPE_DIR from SFEM_CONTACT_PLANE under the mesh,
//      treads SFEM_STEP_HEIGHT wide, each SFEM_STEP_DEPTH (< 0) below the last
//   4: a hemisphere on a floor; by default centred under the mesh, its top at SFEM_CONTACT_PLANE
//
// A contact node is held along one normal, so the normal must not jump and the curvature must
// stay bounded: the obstacles' concave creases (the hemisphere and the floor, a riser and the
// tread below) and convex edges (a step's nose) are rounded with radius SFEM_OBSTACLE_FILLET.
// Elsewhere each is an exact signed distance outside the obstacle, where contact is decided.
struct Obstacle {
    std::function<real_t(const real_t, const real_t, const real_t)> sdf;
    real_t                                                          lo[3], hi[3];  // the grid's box
    real_t                                                          fillet;
};

// The union of two obstacles, positive inside, rounded within k of a crease (polynomial smooth
// maximum; the plain maximum for k = 0).
static real_t rounded_union(const real_t a, const real_t b, const real_t k) {
    const real_t blend = k > 0 ? std::max(k - std::abs(a - b), real_t(0)) / k : 0;
    return std::max(a, b) + blend * blend * k / 4;
}

static Obstacle create_obstacle(const real_t* const min_c, const real_t* const max_c, const int dir) {
    const int h1 = dir == 0 ? 1 : 0;
    const int h2 = dir == 2 ? 1 : 2;

    int    SFEM_OBSTACLE_TYPE  = 4;
    real_t SFEM_CONTACT_PLANE  = 0;
    real_t SFEM_OBSTACLE_SLOPE = 0.2;
    int    SFEM_SLOPE_DIR      = h1;
    real_t SFEM_STEP_HEIGHT    = 0.25;
    real_t SFEM_STEP_DEPTH     = -0.05;
    SFEM_READ_ENV(SFEM_OBSTACLE_TYPE, atoi);
    SFEM_READ_ENV(SFEM_CONTACT_PLANE, atof);
    SFEM_READ_ENV(SFEM_OBSTACLE_SLOPE, atof);
    SFEM_READ_ENV(SFEM_SLOPE_DIR, atoi);
    SFEM_READ_ENV(SFEM_STEP_HEIGHT, atof);
    SFEM_READ_ENV(SFEM_STEP_DEPTH, atof);

    real_t SFEM_HEMISPHERE_RADIUS = 0.25 * std::min(max_c[h1] - min_c[h1], max_c[h2] - min_c[h2]);
    SFEM_READ_ENV(SFEM_HEMISPHERE_RADIUS, atof);
    real_t center[3];
    center[h1]                      = (min_c[h1] + max_c[h1]) / 2;
    center[h2]                      = (min_c[h2] + max_c[h2]) / 2;
    center[dir]                     = SFEM_CONTACT_PLANE - SFEM_HEMISPHERE_RADIUS;
    real_t SFEM_HEMISPHERE_CENTER_X = center[0];
    real_t SFEM_HEMISPHERE_CENTER_Y = center[1];
    real_t SFEM_HEMISPHERE_CENTER_Z = center[2];
    SFEM_READ_ENV(SFEM_HEMISPHERE_CENTER_X, atof);
    SFEM_READ_ENV(SFEM_HEMISPHERE_CENTER_Y, atof);
    SFEM_READ_ENV(SFEM_HEMISPHERE_CENTER_Z, atof);

    if ((SFEM_OBSTACLE_TYPE == 1 || SFEM_OBSTACLE_TYPE == 2) &&
        (SFEM_SLOPE_DIR == dir || SFEM_SLOPE_DIR < 0 || SFEM_SLOPE_DIR > 2)) {
        SFEM_ERROR("SFEM_SLOPE_DIR=%d must be a tangential axis of SFEM_CONTACT_DIR=%d\n", SFEM_SLOPE_DIR, dir);
    }
    if (SFEM_OBSTACLE_TYPE == 2 && !(SFEM_STEP_DEPTH < 0 && SFEM_STEP_HEIGHT > 0)) {
        SFEM_ERROR("SFEM_STEP_DEPTH=%g must be negative and SFEM_STEP_HEIGHT=%g positive\n", SFEM_STEP_DEPTH, SFEM_STEP_HEIGHT);
    }

    // A step is rounded over its whole depth: the interpolating B-spline does not smooth, and a
    // tighter nose resolved by two grid cells left Newton unable to slide the body over it.
    real_t SFEM_OBSTACLE_FILLET =
            SFEM_OBSTACLE_TYPE == 2 ? std::min(SFEM_STEP_HEIGHT, -SFEM_STEP_DEPTH) : SFEM_HEMISPHERE_RADIUS / 2;
    SFEM_READ_ENV(SFEM_OBSTACLE_FILLET, atof);

    const int    s      = SFEM_SLOPE_DIR;
    const real_t plane  = SFEM_CONTACT_PLANE;
    const real_t fillet = SFEM_OBSTACLE_FILLET;

    std::function<real_t(const real_t, const real_t, const real_t)> sdf;
    real_t                                                          lowest = plane;
    switch (SFEM_OBSTACLE_TYPE) {
        case 0: {
            printf("Obstacle: plane at %g along axis %d\n", plane, dir);
            sdf = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                const real_t p[3] = {x, y, z};
                return plane - p[dir];
            };
            break;
        }
        case 1: {
            const real_t slope = SFEM_OBSTACLE_SLOPE;
            const real_t top   = slope > 0 ? max_c[s] : min_c[s];
            printf("Obstacle: plane sloping by %g along axis %d, at %g along axis %d where %g\n", slope, s, plane, dir, top);
            const real_t scale = 1 / std::sqrt(1 + slope * slope);
            sdf                = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                const real_t p[3] = {x, y, z};
                return (plane + slope * (p[s] - top) - p[dir]) * scale;
            };
            lowest = plane - std::abs(slope) * (max_c[s] - min_c[s]);
            break;
        }
        case 2: {
            const real_t width = SFEM_STEP_HEIGHT;
            const real_t drop  = SFEM_STEP_DEPTH;
            const real_t start = min_c[s];
            const int    steps = (int)std::ceil((max_c[s] - min_c[s]) / width) + 1;
            printf("Obstacle: %d steps descending along axis %d from %g, %g wide, %g apart, fillet %g\n",
                   steps,
                   s,
                   plane,
                   width,
                   drop,
                   fillet);
            // Under the stairs is the union of the quarter-spaces {p_s <= start + (k + 1) w,
            // p_dir <= plane + k drop}; the last one runs on, the floor beyond. Each is inset by
            // the fillet and offset back out, which rounds its nose and keeps its faces.
            sdf = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                const real_t p[3] = {x, y, z};
                real_t       d    = plane + (steps - 1) * drop - p[dir];
                for (int k = 0; k + 1 < steps; ++k) {
                    const real_t a = start + (k + 1) * width - p[s] - fillet;
                    const real_t b = plane + k * drop - p[dir] - fillet;
                    d = rounded_union(d, ((a < 0 && b < 0) ? -std::sqrt(a * a + b * b) : std::min(a, b)) + fillet, fillet);
                }
                return d;
            };
            lowest = plane + (steps - 1) * drop;
            break;
        }
        case 4: {
            const real_t c[3] = {SFEM_HEMISPHERE_CENTER_X, SFEM_HEMISPHERE_CENTER_Y, SFEM_HEMISPHERE_CENTER_Z};
            const real_t r    = SFEM_HEMISPHERE_RADIUS;
            printf("Obstacle: hemisphere centre (%g, %g, %g), radius %g, on a floor at %g along axis %d, fillet %g\n",
                   c[0],
                   c[1],
                   c[2],
                   r,
                   c[dir],
                   dir,
                   fillet);
            sdf = [=](const real_t x, const real_t y, const real_t z) -> real_t {
                const real_t p[3]   = {x, y, z};
                const real_t sphere = r - std::sqrt((x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]) + (z - c[2]) * (z - c[2]));
                return rounded_union(sphere, c[dir] - p[dir], fillet);
            };
            lowest = c[dir];
            break;
        }
        default: {
            SFEM_ERROR("Unknown SFEM_OBSTACLE_TYPE=%d (0, 1, 2 or 4)\n", SFEM_OBSTACLE_TYPE);
        }
    }

    // Where the box cuts the obstacle a body slides over the edge with its face, and node-to-
    // obstacle contact cannot keep a sharp edge from entering a face between nodes: the edge is
    // rounded on the scale of the body, at least an eighth of its footprint.
    const real_t edge = std::max(fillet, std::min(max_c[h1] - min_c[h1], max_c[h2] - min_c[h2]) / 8);

    real_t extent = 0;
    for (int d = 0; d < 3; ++d) {
        extent = std::max(extent, max_c[d] - min_c[d]);
    }
    real_t lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
        lo[d] = min_c[d] - extent / 2;
        hi[d] = max_c[d] + extent / 2;
    }
    lo[dir] = std::min(lowest, min_c[dir]) - extent / 2;

    Obstacle obstacle;
    obstacle.fillet = fillet;
    obstacle.sdf    = [=](const real_t x, const real_t y, const real_t z) -> real_t {
        // The box's signed distance, positive inside; the intersection is the smaller field.
        const real_t p[3]    = {x, y, z};
        real_t       outside = 0, inside = std::numeric_limits<real_t>::max(), beyond = -std::numeric_limits<real_t>::max();
        for (int d = 0; d < 3; ++d) {
            const real_t a = std::max(lo[d] - p[d], p[d] - hi[d]);
            beyond         = std::max(beyond, a);
            inside         = std::min(inside, -a);
            if (a > 0) outside += a * a;
        }
        const real_t box = beyond <= 0 ? inside : -std::sqrt(outside);
        return -rounded_union(-sdf(x, y, z), -box, edge);
    };
    // The grid reaches past the box, so the cut and its rounding are on it.
    for (int d = 0; d < 3; ++d) {
        obstacle.lo[d] = lo[d] - 2 * edge;
        obstacle.hi[d] = hi[d] + 2 * edge;
    }
    return obstacle;
}

// The obstacle on an SDF grid over its box, SFEM_OBSTACLE_GRID cells along the longest side, or
// finer: at most half the fillet apart, so the grid resolves the rounding whose curvature the
// contact tangent uses. This grid is the obstacle the contact sees, and the one written for
// visualisation.
static std::shared_ptr<smesh::Grid<geom_t>> obstacle_grid(const std::shared_ptr<sfem::Communicator>& comm,
                                                          const Obstacle&                            obstacle) {
    int SFEM_OBSTACLE_GRID = 128;
    SFEM_READ_ENV(SFEM_OBSTACLE_GRID, atoi);

    real_t extent = 0;
    for (int d = 0; d < 3; ++d) {
        extent = std::max(extent, obstacle.hi[d] - obstacle.lo[d]);
    }
    const real_t h =
            obstacle.fillet > 0 ? std::min(extent / SFEM_OBSTACLE_GRID, obstacle.fillet / 2) : extent / SFEM_OBSTACLE_GRID;

    // And 8 cells more on every side: the B-spline's prefilter extends each grid line beyond its
    // ends (sfem_sdf_sampling.cpp), which pulls the curvature towards zero within a few cells of
    // them (by 0.268^k at k cells), so the obstacle's box keeps clear of that layer.
    real_t    lo[3], hi[3];
    ptrdiff_t n[3];
    for (int d = 0; d < 3; ++d) {
        lo[d] = obstacle.lo[d] - 8 * h;
        hi[d] = obstacle.hi[d] + 8 * h;
        n[d]  = (ptrdiff_t)std::ceil((hi[d] - lo[d]) / h) + 1;
    }
    return smesh::create_sdf(
            comm,
            n[0],
            n[1],
            n[2],
            lo[0],
            lo[1],
            lo[2],
            hi[0],
            hi[1],
            hi[2],
            [&obstacle](const geom_t x, const geom_t y, const geom_t z) -> geom_t { return obstacle.sdf(x, y, z); });
}

std::shared_ptr<sfem::Output> create_output(const std::shared_ptr<sfem::Function>& f, const smesh::Path& output_dir) {
    auto fs = f->space();

    smesh::create_directory(output_dir);
    auto output = f->output();
    output->enable_AoS_to_SoA(fs->block_size() > 1);
    output->set_output_dir(output_dir);

    if (fs->has_semi_structured_mesh()) {
        smesh::semistructured_export_as_standard(fs->mesh_ptr(), output_dir);
    } else {
        fs->mesh_ptr()->write(smesh::Path(output_dir));
    }
    return output;
}

// A block falling under gravity, without contact (hanging from a clamped side) or onto an
// obstacle. `output_dir` is where SFEM_ENABLE_OUTPUT writes.
static int run_mooney_rivlin_gravity(const bool contact, const char* const output_dir) {
    auto es = sfem::EXECUTION_SPACE_HOST;

    int SFEM_BASE_RESOLUTION = 2;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    std::shared_ptr<sfem::Mesh> mesh;
    const char*                 mesh_path = getenv("SFEM_MESH");
    if (mesh_path && mesh_path[0] != '\0') {
        printf("Loading mesh from: %s\n", mesh_path);
        mesh = sfem::Mesh::create_from_file(sfem::Communicator::world(), smesh::Path(mesh_path));
    } else {
        // Default cube mesh with configurable initial gap
        real_t SFEM_INIT_GAP = 0.05;  // Default gap above contact plane
        SFEM_READ_ENV(SFEM_INIT_GAP, atof);

        printf("Creating default cube mesh with SFEM_INIT_GAP=%.4f\n", SFEM_INIT_GAP);
        mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::world(),
                                            SFEM_BASE_RESOLUTION,
                                            SFEM_BASE_RESOLUTION,
                                            SFEM_BASE_RESOLUTION,  // Grid
                                            0,
                                            0,
                                            SFEM_INIT_GAP,  // Origin z with gap
                                            2,
                                            1,
                                            1  // Dimensions (cube)
        );
    }

    auto fs = sfem::FunctionSpace::create(mesh, 3);  // 3D displacement
    auto f  = sfem::Function::create(fs);

    // Material Parameters from environment
    real_t SFEM_C10          = 1.0;
    real_t SFEM_C01          = 0.5;
    real_t SFEM_BULK_MODULUS = 100.0;
    real_t SFEM_DT           = 0.1;
    SFEM_READ_ENV(SFEM_C10, atof);
    SFEM_READ_ENV(SFEM_C01, atof);
    SFEM_READ_ENV(SFEM_BULK_MODULUS, atof);
    SFEM_READ_ENV(SFEM_DT, atof);

    sfem::test::PronyMooneyRivlin material;
    material.C10 = SFEM_C10;
    material.C01 = SFEM_C01;
    material.K   = SFEM_BULK_MODULUS;

    real_t dt = SFEM_DT;

    // WLF temperature shift parameters (set BEFORE Prony terms to avoid redundant calculations)
    int    SFEM_USE_WLF     = 1;
    real_t SFEM_WLF_C1      = 16.6253;
    real_t SFEM_WLF_C2      = 47.4781;
    real_t SFEM_WLF_T_REF   = -54.29;
    real_t SFEM_TEMPERATURE = 20.0;
    SFEM_READ_ENV(SFEM_USE_WLF, atoi);
    SFEM_READ_ENV(SFEM_WLF_C1, atof);
    SFEM_READ_ENV(SFEM_WLF_C2, atof);
    SFEM_READ_ENV(SFEM_WLF_T_REF, atof);
    SFEM_READ_ENV(SFEM_TEMPERATURE, atof);

    if (SFEM_USE_WLF) {
        material.wlf         = true;
        material.wlf_C1      = SFEM_WLF_C1;
        material.wlf_C2      = SFEM_WLF_C2;
        material.wlf_T_ref   = SFEM_WLF_T_REF;
        material.temperature = SFEM_TEMPERATURE;
        printf("WLF enabled: C1=%.4f, C2=%.4f, T_ref=%.2f, T=%.2f\n", SFEM_WLF_C1, SFEM_WLF_C2, SFEM_WLF_T_REF, SFEM_TEMPERATURE);
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
        std::string       g_str(env_g);
        std::stringstream g_ss(g_str);
        std::string       token;
        while (std::getline(g_ss, token, ',')) {
            g_prony.push_back(std::stod(token));
        }

        // Parse comma-separated tau values
        std::string       tau_str(env_tau);
        std::stringstream tau_ss(tau_str);
        while (std::getline(tau_ss, token, ',')) {
            tau_prony.push_back(std::stod(token));
        }

        if (g_prony.size() != tau_prony.size()) {
            printf("Error: SFEM_PRONY_G and SFEM_PRONY_TAU must have the same number of terms!\n");
            exit(EXIT_FAILURE);
        }

        printf("Prony series from environment: %zu terms\n", g_prony.size());
        for (size_t i = 0; i < g_prony.size(); ++i) {
            printf("  term %zu: g=%.6f, tau=%.6f\n", i + 1, g_prony[i], tau_prony[i]);
        }
    }

    material.g   = g_prony;
    material.tau = tau_prony;
    auto op      = sfem::test::make_prony_mooney_rivlin(fs, material);
    SFEM_TEST_ASSERT(op != nullptr);
    f->add_operator(op);

    // Read contact direction early (needed for BC selection)
    int SFEM_CONTACT_DIR = 2;  // Contact direction: 0=x, 1=y, 2=z
    SFEM_READ_ENV(SFEM_CONTACT_DIR, atoi);

    // BC - Sidesets (use mesh bounds so custom meshes work)
    auto           coords     = mesh->points();
    geom_t* const* coords_ptr = coords->data();
    real_t         min_c[3]   = {
            std::numeric_limits<real_t>::max(), std::numeric_limits<real_t>::max(), std::numeric_limits<real_t>::max()};
    real_t max_c[3] = {
            std::numeric_limits<real_t>::lowest(), std::numeric_limits<real_t>::lowest(), std::numeric_limits<real_t>::lowest()};
#pragma omp parallel for reduction(min : min_c[:3]) reduction(max : max_c[:3])
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
           min_c[0],
           max_c[0],
           min_c[1],
           max_c[1],
           min_c[2],
           max_c[2]);
    printf("Sideset epsilon: eps_x=%.6f, eps_y=%.6f, eps_z=%.6f\n", eps_c[0], eps_c[1], eps_c[2]);

    auto make_sideset = [&](int dir, bool is_min) {
        const real_t target = is_min ? min_c[dir] : max_c[dir];
        const real_t eps    = eps_c[dir];
        if (dir == 0) {
            return sfem::Sideset::create_from_selector(
                    mesh, [=](const geom_t x, const geom_t, const geom_t) -> bool { return fabs(x - target) <= eps; });
        } else if (dir == 1) {
            return sfem::Sideset::create_from_selector(
                    mesh, [=](const geom_t, const geom_t y, const geom_t) -> bool { return fabs(y - target) <= eps; });
        } else {
            return sfem::Sideset::create_from_selector(
                    mesh, [=](const geom_t, const geom_t, const geom_t z) -> bool { return fabs(z - target) <= eps; });
        }
    };

    auto min_sideset_x = make_sideset(0, true);
    auto max_sideset_x = make_sideset(0, false);
    auto min_sideset_y = make_sideset(1, true);
    auto max_sideset_y = make_sideset(1, false);
    auto min_sideset_z = make_sideset(2, true);
    auto max_sideset_z = make_sideset(2, false);

    auto contact_sideset  = (SFEM_CONTACT_DIR == 0) ? min_sideset_x : (SFEM_CONTACT_DIR == 1) ? min_sideset_y : min_sideset_z;
    auto opposite_sideset = (SFEM_CONTACT_DIR == 0) ? max_sideset_x : (SFEM_CONTACT_DIR == 1) ? max_sideset_y : max_sideset_z;

    // **Boundary conditions based on contact direction**
    // SFEM_FIX_SIDE: 0=fix min side, 1=fix max side (default for backward compatibility)
    int SFEM_FIX_SIDE = 1;
    SFEM_READ_ENV(SFEM_FIX_SIDE, atoi);

    if (!contact) {
        // No contact: fix one side completely based on SFEM_FIX_SIDE
        auto fixed_sideset = (SFEM_FIX_SIDE == 0) ? contact_sideset : opposite_sideset;
        printf("Fixing %s side in %s direction (all DOFs)\n",
               SFEM_FIX_SIDE == 0 ? "min" : "max",
               SFEM_CONTACT_DIR == 0 ? "x" : (SFEM_CONTACT_DIR == 1 ? "y" : "z"));
        sfem::DirichletConditions::Condition bc_x{.sidesets = fixed_sideset, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition bc_y{.sidesets = fixed_sideset, .value = 0, .component = 1};
        sfem::DirichletConditions::Condition bc_z{.sidesets = fixed_sideset, .value = 0, .component = 2};
        auto                                 conds = sfem::create_dirichlet_conditions(fs, {bc_x, bc_y, bc_z}, es);
        f->add_constraint(conds);
    } else {
        // With contact the body is free: the obstacle is all that holds it, and c0 M keeps the
        // Newton tangent definite on its rigid motions.
        printf("Contact: no Dirichlet conditions, the body is free\n");
    }

    // **GRAVITY as body force instead of Neumann surface force**
    real_t SFEM_GRAVITY = 9.81;  // Positive value, direction handled below
    SFEM_READ_ENV(SFEM_GRAVITY, atof);

    // Gravity direction: independent of contact direction
    // Default: same as CONTACT_DIR (for backward compatibility)
    // Set SFEM_GRAVITY_DIR to override (0=x, 1=y, 2=z)
    int SFEM_GRAVITY_DIR = SFEM_CONTACT_DIR;
    SFEM_READ_ENV(SFEM_GRAVITY_DIR, atoi);

    const ptrdiff_t ndofs   = fs->n_dofs();
    auto            x       = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs     = sfem::create_buffer<real_t>(ndofs, es);
    auto            delta_x = sfem::create_buffer<real_t>(ndofs, es);
    auto            diag    = sfem::create_buffer<real_t>(ndofs, es);

    real_t SFEM_DENSITY = 1000.0;
    SFEM_READ_ENV(SFEM_DENSITY, atof);
    real_t density = SFEM_DENSITY;
    auto   blas    = sfem::blas<real_t>(es);

    // Newmark (implicit, unconditionally stable), held by the material: its inertia
    // c0 M (x - u_hat), c0 = 1/(beta dt^2) and M the lumped mass, is in the material's gradient
    // and Hessians. With contact the material holds the contact-stabilised variant instead.
    auto newmark = std::make_shared<sfem::NewmarkScheme>(fs, es);
    newmark->set_beta(0.25);
    newmark->set_gamma(0.5);
    newmark->set_density(density);
    SFEM_TEST_ASSERT(newmark->initialize() == SFEM_SUCCESS);
    std::shared_ptr<sfem::TimeScheme> scheme = newmark;

    // **Gravity as an operator the Function owns**
    //
    // This used to be a vector built here by hand -- the lumped mass diagonal
    // times `rho * g` -- and `axpy`'d onto the residual below.  Built outside
    // the `Function`, it was invisible to `Function::value`, so the merit was
    // `1/2*||R_interior||^2`: not zero at the solution being sought, and wrong
    // for any line search along it.
    //
    // The operator computes the consistent load `int(rho * g . v)`.  For a
    // constant `g` that is the same vector as the lumped construction, exactly:
    // row-sum lumping gives `M_ii = sum_j int(phi_i phi_j) = int(phi_i)` by
    // partition of unity, so `rho * M_ii * g` is `int(rho * g * phi_i)`.
    // `test_body_force_matches_the_lumped_construction` pins that, which is why
    // this trajectory does not move.
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    printf("Gravity: %.4f in -%s direction\n", SFEM_GRAVITY, SFEM_GRAVITY_DIR == 0 ? "x" : (SFEM_GRAVITY_DIR == 1 ? "y" : "z"));
    auto gravity_op = sfem::create_op(fs, "GeneratedBodyForce", es);
    SFEM_TEST_ASSERT(gravity_op != nullptr);
    for (auto &block : fs->mesh_ptr()->blocks()) {
        gravity_op->set_value_in_block(block->name(), "density", density);
        for (int component = 0; component < 3; ++component) {
            gravity_op->set_value_in_block(
                    block->name(),
                    component == 0 ? "g0" : (component == 1 ? "g1" : "g2"),
                    component == SFEM_GRAVITY_DIR ? -SFEM_GRAVITY : 0.0);
        }
    }
    f->add_operator(gravity_op);

    // Output setup
    bool SFEM_ENABLE_OUTPUT = false;
    SFEM_READ_ENV(SFEM_ENABLE_OUTPUT, atoi);
    auto output = create_output(f, smesh::Path(output_dir));

    blas->zeros(ndofs, x->data());

    // Matrix assembly buffers (BSR format: 3x3 blocks)
    auto      graph      = fs->node_to_node_graph();
    const int block_size = 3;
    auto      values     = sfem::create_buffer<real_t>(graph->nnz() * block_size * block_size, es);

    // Linear Solver Wrapper (BSR SpMV)
    auto linear_op_apply = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [=](const real_t* const in, real_t* const out) {
                sfem::bsr_spmv<count_t, idx_t, real_t>(n_nodes,
                                                       n_nodes,
                                                       block_size,
                                                       graph->rowptr()->data(),
                                                       graph->colidx()->data(),
                                                       values->data(),
                                                       real_t(0.0),
                                                       in,
                                                       out);
            },
            es);

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
    auto                                                  jacobi = sfem::create_shiftable_jacobi(diag, es);

    // Contact: every node of the body's surface (it is free, any face can touch), the gap and
    // normals sampled at every Newton iterate, and the linearised step solved as a QP.
    Obstacle                             obstacle_field;
    std::shared_ptr<smesh::Grid<geom_t>> obstacle_sdf;
    std::shared_ptr<sfem::NodeObstacleContact>            obstacle;
    std::shared_ptr<sfem::ContactStabilisedNewmarkScheme> stabilised;
    std::shared_ptr<sfem::MPRGP<real_t>> contact_qp;
    sfem::SharedBuffer<real_t>           upper_bound, qp_rhs, projected, gap_viz, contact_force, inertia_stiffness, x_prev;
    real_t                               gap_tol = 0, max_inertia_stiffness = 0;
    bool                                 touched = false;  // the obstacle pushed back in some step

    if (!contact) {
        auto cg = sfem::create_cg<real_t>(linear_op_apply, es);
        cg->set_n_dofs(ndofs);
        cg->set_max_it(2000);
        cg->set_rtol(1e-5);
        cg->verbose = false;

        // Preconditioner (Jacobi)
        cg->set_preconditioner_op(jacobi);

        solver = cg;
    } else {
        real_t extent = 0;
        for (int d = 0; d < 3; ++d) {
            extent = std::max(extent, max_c[d] - min_c[d]);
        }

        obstacle_field = create_obstacle(min_c, max_c, SFEM_CONTACT_DIR);
        obstacle_sdf   = obstacle_grid(mesh->comm(), obstacle_field);
        obstacle       = std::make_shared<sfem::NodeObstacleContact>(obstacle_sdf,
                                                         mesh->comm(),
                                                         smesh::create_nodeset_from_sidesets(mesh, smesh::skin_sidesets(mesh)),
                                                         coords,
                                                         ndofs,
                                                         SFEM_CONTACT_DIR);
        SFEM_TEST_ASSERT(obstacle->size() > 0);
        printf("Contact: %ld nodes, the body's surface\n", (long)obstacle->size());
        stabilised = std::make_shared<sfem::ContactStabilisedNewmarkScheme>(newmark, obstacle);
        scheme     = stabilised;

        projected     = sfem::create_buffer<real_t>(ndofs, es);
        gap_viz       = sfem::create_buffer<real_t>(ndofs, es);
        contact_force = sfem::create_buffer<real_t>(ndofs, es);

        // Contact is resolved to this fraction of the mesh's size.
        gap_tol = 1e-10 * extent;

        inertia_stiffness = sfem::create_buffer<real_t>(ndofs, es);
        x_prev            = sfem::create_buffer<real_t>(ndofs, es);

        // The step in the contact nodes' frames, Q^T K Q dx' = -Q^T R with dx'_n <= gap_i,
        // by MPRGP: the constraints hold exactly at every iterate.
        auto framed    = sfem::create_buffer<real_t>(ndofs, es);
        auto framed_op = sfem::make_op<real_t>(
                ndofs,
                ndofs,
                [=](const real_t* const in, real_t* const out) {
                    blas->copy(ndofs, in, framed->data());
                    obstacle->from_frame(framed->data());
                    linear_op_apply->apply(framed->data(), out);
                    obstacle->to_frame(out);
                },
                es);

        upper_bound            = sfem::create_buffer<real_t>(ndofs, es);
        qp_rhs                 = sfem::create_buffer<real_t>(ndofs, es);
        contact_qp             = sfem::create_mprgp(framed_op, es);
        contact_qp->verbose    = false;
        contact_qp->check_each = std::numeric_limits<int>::max();
        contact_qp->set_max_it(20000);
        contact_qp->set_rtol(1e-10);
        contact_qp->set_upper_bound(upper_bound);
    }

    op->set_time_scheme(scheme);

    // 5. Time Loop with Full Newmark Integration
    real_t SFEM_T = 0.2;
    SFEM_READ_ENV(SFEM_T, atof);
    real_t t     = 0;
    size_t steps = 0;

    // Output frequency: export every N steps (default 1 = every step)
    int SFEM_EXPORT_FREQ = 1;
    SFEM_READ_ENV(SFEM_EXPORT_FREQ, atoi);
    size_t export_freq = SFEM_EXPORT_FREQ;
    if (export_freq > 1) {
        printf("Output frequency: every %zu steps\n", export_freq);
    }

    auto f_int        = sfem::create_buffer<real_t>(ndofs, es);
    auto inertia_term = sfem::create_buffer<real_t>(ndofs, es);

    printf("===== Mooney-Rivlin Gravity Contact Test =====\n");
    printf("Mesh: %ld nodes, %ld DOFs\n", (long)n_nodes, (long)ndofs);
    printf("Newmark parameters: beta=%.2f, gamma=%.2f, dt=%.3f, density=%.1f\n",
           newmark->weight("beta"),
           newmark->weight("gamma"),
           dt,
           density);
    printf("Time: T=%.2f, dt=%.3f\n", SFEM_T, dt);
    printf("Gravity: %.2f m/s^2 in direction %d\n", SFEM_GRAVITY, -SFEM_CONTACT_DIR);
    printf("Contact: %s\n", contact ? "enabled" : "disabled");

    // The obstacle's SDF grid, for visualisation: contour sdf/sdf.xdmf at 0.
    if (SFEM_ENABLE_OUTPUT && contact) {
        obstacle_sdf->to_file(smesh::Path(output_dir) / "sdf");
    }

    // Output
    if (SFEM_ENABLE_OUTPUT) {
        output->write_time_step("disp", t, x->data());
        output->write_time_step("velocity", t, newmark->velocity()->data());
        output->write_time_step("acceleration", t, newmark->acceleration()->data());
        if (contact) {
            obstacle->gap_field(gap_viz->data());
            blas->zeros(ndofs, contact_force->data());
            output->write_time_step("gap", t, gap_viz->data());
            output->write_time_step("contact_force", t, contact_force->data());
        }
        output->log_time(t);
    }

    // Time counting
    double total_hessian_time = 0;

    while (t < SFEM_T) {
        printf("Step %zu: t=%.3f\n", steps, t);

        // Newmark Prediction Step: the scheme's predictor, shift and history for the step.
        scheme->begin_step(t, dt);
        const real_t c0 = scheme->weight("alpha_a");

        if (contact) {
            // u_n + dt v_n (ContactStabilisedNewmarkScheme::start).
            stabilised->start(x->data());

            // The scale of each node's equation, c0 M: what turns a reaction into a displacement.
            blas->copy(ndofs, newmark->mass()->data(), inertia_stiffness->data());
            blas->scal(ndofs, c0, inertia_stiffness->data());
            const real_t* const k     = inertia_stiffness->data();
            const idx_t* const  nodes = obstacle->nodes()->data();
            max_inertia_stiffness     = 0;
#pragma omp parallel for reduction(max : max_inertia_stiffness)
            for (ptrdiff_t i = 0; i < obstacle->size(); ++i) {
                max_inertia_stiffness = std::max(max_inertia_stiffness, k[(ptrdiff_t)nodes[i] * 3]);
            }
        } else {
            // Use prediction as initial guess for Newton
            blas->copy(ndofs, newmark->predictor()->data(), x->data());
        }

        // The Dirichlet values are imposed, not predicted: Newton never moves a constrained
        // dof, so whatever the predictor put there would stay.
        f->apply_constraints(x->data());

        // The step being solved ends at t + dt; its history is fixed here.
        op->begin_step(t + dt, dt);

        // Newton Loop with Inertia
        //
        // The residual, its norms and, with contact, the KKT measures at the current x.
        struct Evaluation {
            real_t r_norm, kkt, penetration, inertia_norm, f_int_norm;
        };
        auto evaluate = [&]() -> Evaluation {
            Evaluation e;
            // Residual: R = M*a + F_int(u) - F_ext + F_contact
            // where a = c0*(u - u_hat) from Newmark relation
            //
            // F_int(x) and the gravity load together: `Function::gradient` runs
            // every operator it owns, which is the point of the body force
            // being one. The inertia comes with the material, which holds the scheme.
            blas->zeros(ndofs, rhs->data());
            f->gradient(x->data(), rhs->data());

            // Inertia term: c0 * M * (u - u_hat), and what is left of R without it
            blas->zeros(ndofs, inertia_term->data());
            scheme->inertia_op()->gradient(x->data(), inertia_term->data());
            e.inertia_norm = blas->norm2(ndofs, inertia_term->data());
            blas->zaxpby(ndofs, 1, rhs->data(), -1, inertia_term->data(), f_int->data());
            e.f_int_norm = blas->norm2(ndofs, f_int->data());

            // Apply BC to residual
            f->set_value_to_constrained_dofs(0.0, rhs->data());
            e.r_norm = blas->norm2(ndofs, rhs->data());

            // With contact the residual does not vanish at the solution: where the obstacle is
            // touched it is the obstacle's reaction. The step converges on what remains.
            e.kkt         = e.r_norm;
            e.penetration = 0;
            if (contact) {
                obstacle->update(x->data());
                e.kkt = obstacle->kkt_residual(rhs->data(), inertia_stiffness->data(), projected->data(), &e.penetration);
            }
            return e;
        };

        bool       step_converged = false;
        real_t     kkt_tol        = 0;
        Evaluation e              = evaluate();
        if (contact && !std::isfinite(e.kkt)) {
            // On a fine mesh dt v can invert elements away from the obstacle right after an
            // impact; u_n, converged and admissible, is a start that always evaluates.
            blas->copy(ndofs, newmark->state()->data(), x->data());
            e = evaluate();
        }
        for (int iter = 0; iter < 20; ++iter) {
            if (iter == 0) {
                // Relative to the forces in balance: with contact the residual's reaction part
                // can far exceed what remains of it, and sets the rounding floor.
                kkt_tol = std::max(real_t(1e-8), real_t(1e-9) * std::max(e.kkt, e.r_norm));
            }

            printf("  Iter %d: |R|=%e |KKT|=%e penetration=%e |u|=%e |M*a|=%e |F_int+F_grav|=%e\n",
                   iter,
                   e.r_norm,
                   e.kkt,
                   e.penetration,
                   blas->norm2(ndofs, x->data()),
                   e.inertia_norm,
                   e.f_int_norm);
            if (!std::isfinite(e.kkt)) break;
            if (e.kkt < kkt_tol && e.penetration <= gap_tol) {
                step_converged = true;
                break;
            }

            // ===== Tangent Stiffness: K_eff = K_tan + c0*M =====
            blas->zeros(values->size(), values->data());
            auto t_start = std::chrono::high_resolution_clock::now();
            f->hessian_bsr(x->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data());
            auto t_end = std::chrono::high_resolution_clock::now();
            total_hessian_time += std::chrono::duration<double>(t_end - t_start).count();

            auto      rowptr = graph->rowptr()->data();
            auto      colidx = graph->colidx()->data();
            auto      vals   = values->data();
            const int bs2    = block_size * block_size;

            // MPRGP's expansion step needs an upper bound on the largest eigenvalue; the
            // largest row sum bounds it, and the frames do not change it.
            auto max_row_sum = [&]() {
                real_t bound = 0;
#pragma omp parallel for reduction(max : bound)
                for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                    for (int r = 0; r < block_size; ++r) {
                        real_t row_sum = 0;
                        for (count_t k = rowptr[node]; k < rowptr[node + 1]; ++k) {
                            for (int c = 0; c < block_size; ++c) {
                                row_sum += std::abs(vals[k * bs2 + r * block_size + c]);
                            }
                        }
                        bound = std::max(bound, row_sum);
                    }
                }
                return bound;
            };

            if (contact) {
                obstacle->add_curvature(x->data(), rhs->data(), inertia_stiffness->data(), false, 1, rowptr, colidx, vals);
                contact_qp->set_atol(real_t(0.1) * kkt_tol);
                obstacle->upper_bound(contact_qp->infty, upper_bound->data());
            } else {
                // Extract diagonal for Jacobi preconditioner
                real_t* const dg = diag->data();
#pragma omp parallel for
                for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                    const real_t* const B = sfem::bsr_diagonal_block(node, rowptr, colidx, block_size, vals);
                    for (int d = 0; d < block_size; ++d) {
                        dg[node * block_size + d] = B[d * block_size + d];
                    }
                }

                jacobi->set_diag(diag);
            }

            if (contact) {
                // -R in the frames; rhs keeps R for a second attempt.
                auto solve_step = [&]() {
                    contact_qp->set_max_eig(max_row_sum());
                    blas->copy(ndofs, rhs->data(), qp_rhs->data());
                    blas->scal(ndofs, -1.0, qp_rhs->data());
                    obstacle->to_frame(qp_rhs->data());
                    blas->zeros(ndofs, delta_x->data());
                    return contact_qp->apply(qp_rhs->data(), delta_x->data());
                };
                if (solve_step() != SFEM_SUCCESS) {
                    // MPRGP stops at negative curvature. Away from the solution the residual
                    // overstates the reactions and the curvature term can make the tangent
                    // indefinite: solve again with it capped.
                    obstacle->add_curvature(x->data(), rhs->data(), inertia_stiffness->data(), false, -1, rowptr, colidx, vals);
                    obstacle->add_curvature(x->data(), rhs->data(), inertia_stiffness->data(), true, 1, rowptr, colidx, vals);

                    // The material's tangent can be indefinite too (a hyperelastic block under
                    // compression, on a mesh fine enough that c0 M no longer outweighs it): shift
                    // its diagonal by a growing fraction of itself until the step solves. A
                    // shifted step still descends, and a stable solution needs no shift.
                    real_t shift = 0;
                    while (solve_step() != SFEM_SUCCESS) {
                        if (shift >= 100) {
                            printf("  contact QP did not reach its tolerance\n");
                            break;
                        }
                        real_t* const dg = diag->data();
                        if (shift == 0) {
#pragma omp parallel for
                            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                                const real_t* const B = sfem::bsr_diagonal_block(node, rowptr, colidx, block_size, vals);
                                for (int d = 0; d < block_size; ++d) {
                                    dg[node * block_size + d] = B[d * block_size + d];
                                }
                            }
                        }
                        const real_t next = shift == 0 ? real_t(1e-2) : 10 * shift;
#pragma omp parallel for
                        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                            real_t* const B = sfem::bsr_diagonal_block(node, rowptr, colidx, block_size, vals);
                            for (int d = 0; d < block_size; ++d) {
                                B[d * block_size + d] += (next - shift) * std::abs(dg[node * block_size + d]);
                            }
                        }
                        shift = next;
                        printf("  tangent shifted by %g of its diagonal\n", shift);
                    }
                }
                obstacle->from_frame(delta_x->data());
                // MPRGP's recurrences leave rounding in the identity rows of the constrained dofs.
                f->apply_zero_constraints(delta_x->data());
            } else {
                blas->scal(ndofs, -1.0, rhs->data());
                blas->zeros(ndofs, delta_x->data());
                if (solver->apply(rhs->data(), delta_x->data()) != SFEM_SUCCESS) {
                    printf("  linear step did not reach its tolerance\n");
                }
            }

            // u = u + dx (with optional damping for stability)
            // For fine meshes, damping < 1 may be needed
            static real_t SFEM_NEWTON_DAMPING = 1.0;
            static bool   damping_read        = false;
            if (!damping_read) {
                SFEM_READ_ENV(SFEM_NEWTON_DAMPING, atof);
                damping_read = true;
                if (SFEM_NEWTON_DAMPING < 1.0) {
                    printf("Using Newton damping = %.3f\n", SFEM_NEWTON_DAMPING);
                }
            }
            real_t damping = SFEM_NEWTON_DAMPING;

            if (contact) {
                // Backtracking on the KKT residual, penetration counted as a force. Where the
                // obstacle's normal turns quickly (a crease, a corner) the full step can cycle
                // between two contact states.
                const real_t merit0 = e.kkt + max_inertia_stiffness * e.penetration;
                blas->copy(ndofs, x->data(), x_prev->data());
                // A trial that cannot be evaluated (an inverted element) is never accepted: the
                // step keeps halving, and if none evaluates the step fails below.
                real_t alpha = 1;
                for (;;) {
                    blas->copy(ndofs, x_prev->data(), x->data());
                    blas->axpy(ndofs, alpha * damping, delta_x->data(), x->data());
                    e                  = evaluate();
                    const real_t merit = e.kkt + max_inertia_stiffness * e.penetration;
                    if (std::isfinite(merit) && (merit <= (1 - real_t(1e-4) * alpha) * merit0 || alpha <= real_t(1) / 64)) break;
                    if (alpha <= real_t(1e-6)) break;
                    alpha /= 2;
                }
                if (alpha < 1) {
                    printf("  step length %g\n", alpha);
                }
            } else {
                blas->axpy(ndofs, damping, delta_x->data(), x->data());
                e = evaluate();
            }
        }

        if (!step_converged) {
            fprintf(stderr, "Step %zu (t=%g) did not converge\n", steps, double(t));
            return SFEM_TEST_FAILURE;
        }

        if (contact) {
            // The obstacle's reaction in the converged step: whether it pushed back.
            obstacle->reaction(contact_force->data());
            touched = touched || blas->norm2(ndofs, contact_force->data()) > 0;
        }

        // Update history
        op->advance(x->data());
        scheme->advance(x->data());

        t += dt;
        steps++;

        // Output
        if (SFEM_ENABLE_OUTPUT && steps % export_freq == 0) {
            output->write_time_step("disp", t, x->data());
            output->write_time_step("velocity", t, newmark->velocity()->data());
            output->write_time_step("acceleration", t, newmark->acceleration()->data());
            if (contact) {
                // The residual of the converged step, less what remains of it, is the obstacle's
                // reaction in the step's scaling, where the material force enters at both ends of
                // the step with dt^2/4 and the contact force once: the force is half of it.
                obstacle->gap_field(gap_viz->data());
                blas->scal(ndofs, real_t(0.5), contact_force->data());
                output->write_time_step("gap", t, gap_viz->data());
                output->write_time_step("contact_force", t, contact_force->data());
            }
            output->log_time(t);
        }
    }

    // A contact run in which nothing touched the obstacle has not tested contact.
    if (contact && !touched) {
        fprintf(stderr, "The body never touched the obstacle\n");
        return SFEM_TEST_FAILURE;
    }

    printf("===== Test Completed =====\n");
    printf("Total Hessian assembly time: %.3f s\n", total_hessian_time);
    return SFEM_TEST_SUCCESS;
}

int test_mooney_rivlin_gravity() { return run_mooney_rivlin_gravity(false, "test_mooney_rivlin_gravity"); }

int test_mooney_rivlin_gravity_contact() { return run_mooney_rivlin_gravity(true, "test_mooney_rivlin_gravity_contact"); }

int main(int argc, char* argv[]) {
    // Line by line: the solvers' warnings go to stderr, and a log taking both must not have them
    // inside a progress line.
    setvbuf(stdout, nullptr, _IOLBF, 0);
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_mooney_rivlin_gravity);
    SFEM_RUN_TEST(test_mooney_rivlin_gravity_contact);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
