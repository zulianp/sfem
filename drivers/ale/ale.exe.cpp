#include "sfem_API.hpp"

int         fvm_quad4_volumes(const ptrdiff_t             nelements,
                              idx_t **SFEM_RESTRICT       elements,
                              geom_t **SFEM_RESTRICT      points,
                              real_t *const SFEM_RESTRICT volumes) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; e++) {
        idx_t i0 = elements[0][e];
        idx_t i1 = elements[1][e];
        idx_t i3 = elements[3][e];

        real_t J[2 * 2] = {points[0][i1] - points[0][i0],
                           points[0][i3] - points[0][i0],
                           points[1][i1] - points[1][i0],
                           points[1][i3] - points[1][i0]};

        real_t v   = J[0] * J[3] - J[1] * J[2];
        volumes[e] = fabs(v);
    }

    return SFEM_SUCCESS;
}

static SFEM_INLINE void fvm_quad4_surface_vectors(const geom_t x0,
                                                  const geom_t x1,
                                                  const geom_t x2,
                                                  const geom_t x3,
                                                  const geom_t y0,
                                                  const geom_t y1,
                                                  const geom_t y2,
                                                  const geom_t y3,
                                                  // Output (len 2)
                                                  scalar_t *const SFEM_RESTRICT sf0,
                                                  scalar_t *const SFEM_RESTRICT sf1,
                                                  scalar_t *const SFEM_RESTRICT sf2,
                                                  scalar_t *const SFEM_RESTRICT sf3) {
    {
        scalar_t ux = x1 - x0;
        scalar_t uy = y1 - y0;
        sf0[0]      = -uy;
        sf0[1]      = ux;
    }

    {
        scalar_t ux = x2 - x1;
        scalar_t uy = y2 - y1;
        sf1[0]      = -uy;
        sf1[1]      = ux;
    }

    {
        scalar_t ux = x3 - x2;
        scalar_t uy = y3 - y2;
        sf2[0]      = -uy;
        sf2[1]      = ux;
    }

    {
        scalar_t ux = x0 - x3;
        scalar_t uy = y0 - y3;
        sf3[0]      = -uy;
        sf3[1]      = ux;
    }
}

static SFEM_INLINE scalar_t fvm_quad4_convection_flux()
{
    return 0;
}

static SFEM_INLINE scalar_t fvm_quad4_face_interp(const scalar_t volume,
                                                  const scalar_t val,
                                                  const scalar_t volume_neigh,
                                                  const scalar_t val_neigh) {
    return (volume * val + volume_neigh * val_neigh) / (volume + volume_neigh);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    std::string output_folder = "ale_fvm_out";
    sfem::create_directory(output_folder.c_str());


    // 1) Initialize fields, grids and other paramerers

    ptrdiff_t nx   = 2;
    int       L    = 4;
    int       H    = 1;
    auto      mesh = sfem::Mesh::create_quad4_square(comm, L * nx, H * nx, 0, 0, L, H);

    auto sideset_inlet = sfem::Sideset::create_from_selector(
            mesh, [L](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > -1e-5 && x < 1e-5; });

    auto sideset_outlet = sfem::Sideset::create_from_selector(
            mesh,
            [L](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > (L - 1e-5) && x < (L + 1e-5); });

    auto sideset_wall =
            sfem::Sideset::create_from_selector(mesh, [H](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool {
                return (y > -1e-5 && y < 1e-5) || (y > (H - 1e-5) && y < (H + 1e-5));
            });

    mesh->write((output_folder + "/mesh").c_str());
    auto fs = sfem::FunctionSpace::create(mesh, 1);

    element_idx_t *table_{nullptr};
    create_element_adj_table(mesh->n_elements(), mesh->n_nodes(), mesh->element_type(), mesh->elements()->data(), &table_);

    const int ns    = elem_num_sides(mesh->element_type());
    auto      table = sfem::manage_host_buffer<element_idx_t>(mesh->n_elements() * ns, table_);


    // Timestepping loop
    	// 2) Put F_f = F^n_f (first guess)


	    // Loop untill F^*_f is converged
	    	// 3) The momentum equation without pressure term (Eq. 9) is solved including BCs with fixed F^*


	    	// 4) Find F_{0f} from auxiliary velocities


	    	// 5) Mass flux at cell face Eq. 16



    return MPI_Finalize();
}
