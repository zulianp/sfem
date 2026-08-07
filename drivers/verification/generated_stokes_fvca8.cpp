#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Function.hpp"
#include "generated/sfem_generated_ops_registration.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"
#include "smesh_output.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/stat.h>

namespace {

enum CaseKind { BERCOVIER_ENGELMAN_2D = 0, TAYLOR_GREEN_3D = 1 };

struct CaseData {
    CaseKind kind;
    int      dim;
    const char *name;
};

static constexpr double PI = 3.141592653589793238462643383279502884;

bool case_from_name(const char *const name, CaseData &out) {
    if (!std::strcmp(name, "bercovier_engelman_2d")) {
        out = {BERCOVIER_ENGELMAN_2D, 2, "bercovier_engelman_2d"};
        return true;
    }

    if (!std::strcmp(name, "taylor_green_3d")) {
        out = {TAYLOR_GREEN_3D, 3, "taylor_green_3d"};
        return true;
    }

    return false;
}

void eval_case(const CaseKind kind,
               const double   x,
               const double   y,
               const double   z,
               real_t *const  u,
               real_t *const  p,
               real_t *const  force) {
    u[0] = u[1] = u[2] = 0;
    force[0] = force[1] = force[2] = 0;

    if (kind == BERCOVIER_ENGELMAN_2D) {
        const double xm1 = x - 1.0;
        const double ym1 = y - 1.0;
        const double x2  = x * x;
        const double y2  = y * y;

        const double u01 = 256.0 * x2 * xm1 * xm1 * y * ym1 * (2.0 * y - 1.0);
        const double u10 = 256.0 * y2 * ym1 * ym1 * x * xm1 * (2.0 * x - 1.0);
        u[0]             = real_t(u01);
        u[1]             = real_t(-u10);
        *p               = real_t((x - 0.5) * (y - 0.5));

        const double core_xy = x2 * xm1 * xm1 * (12.0 * y - 6.0) +
                               y * ym1 * (2.0 * y - 1.0) * (12.0 * x2 - 12.0 * x + 2.0);
        const double core_yx = y2 * ym1 * ym1 * (12.0 * x - 6.0) +
                               x * xm1 * (2.0 * x - 1.0) * (12.0 * y2 - 12.0 * y + 2.0);
        force[0] = real_t(-256.0 * core_xy + (y - 0.5));
        force[1] = real_t(256.0 * core_yx + (x - 0.5));
        return;
    }

    const double sx = std::sin(2.0 * PI * x);
    const double sy = std::sin(2.0 * PI * y);
    const double sz = std::sin(2.0 * PI * z);
    const double cx = std::cos(2.0 * PI * x);
    const double cy = std::cos(2.0 * PI * y);
    const double cz = std::cos(2.0 * PI * z);

    u[0] = real_t(2.0 * cx * sy * sz);
    u[1] = real_t(-sx * cy * sz);
    u[2] = real_t(-sx * sy * cz);
    *p   = real_t(6.0 * PI * sx * sy * sz);

    force[0] = real_t(36.0 * PI * PI * cx * sy * sz);
}

bool is_boundary_node(const geom_t *const *const points, const int dim, const ptrdiff_t i) {
    constexpr double eps = 1e-7;
    const double     x   = points[0][i];
    const double     y   = points[1][i];
    if (x < eps || x > 1.0 - eps || y < eps || y > 1.0 - eps) {
        return true;
    }

    if (dim == 3) {
        const double z = points[2][i];
        return z < eps || z > 1.0 - eps;
    }

    return false;
}

void tri6_shape(const double l0, const double l1, const double l2, double *const phi) {
    phi[0] = l0 * (2.0 * l0 - 1.0);
    phi[1] = l1 * (2.0 * l1 - 1.0);
    phi[2] = l2 * (2.0 * l2 - 1.0);
    phi[3] = 4.0 * l0 * l1;
    phi[4] = 4.0 * l1 * l2;
    phi[5] = 4.0 * l0 * l2;
}

void tet10_shape(const double l0, const double l1, const double l2, const double l3, double *const phi) {
    phi[0] = l0 * (2.0 * l0 - 1.0);
    phi[1] = l1 * (2.0 * l1 - 1.0);
    phi[2] = l2 * (2.0 * l2 - 1.0);
    phi[3] = l3 * (2.0 * l3 - 1.0);
    phi[4] = 4.0 * l0 * l1;
    phi[5] = 4.0 * l1 * l2;
    phi[6] = 4.0 * l0 * l2;
    phi[7] = 4.0 * l0 * l3;
    phi[8] = 4.0 * l1 * l3;
    phi[9] = 4.0 * l2 * l3;
}

void assemble_rhs_2d(const CaseKind kind, const std::shared_ptr<sfem::Mesh> &mesh, real_t *const rhs) {
    static constexpr double gx[5] = {
            0.046910077030668, 0.230765344947158, 0.5, 0.769234655052842, 0.953089922969332};
    static constexpr double gw[5] = {
            0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095};

    constexpr int block_size = 3;
    auto          elems      = mesh->elements(0)->data();
    auto          points     = mesh->points()->data();
    const auto    nelements  = mesh->n_elements();

    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const idx_t  ev0 = elems[0][e];
        const idx_t  ev1 = elems[1][e];
        const idx_t  ev2 = elems[2][e];
        const double x0  = points[0][ev0];
        const double y0  = points[1][ev0];
        const double x1  = points[0][ev1];
        const double y1  = points[1][ev1];
        const double x2  = points[0][ev2];
        const double y2  = points[1][ev2];

        const double j00  = x1 - x0;
        const double j01  = x2 - x0;
        const double j10  = y1 - y0;
        const double j11  = y2 - y0;
        const double detJ = std::fabs(j00 * j11 - j01 * j10);

        real_t fval[3], uval[3], pval;
        double phi[6];
        for (int ia = 0; ia < 5; ++ia) {
            const double a      = gx[ia];
            const double one_ma = 1.0 - a;
            for (int ib = 0; ib < 5; ++ib) {
                const double b  = gx[ib];
                const double l1 = a;
                const double l2 = one_ma * b;
                const double l0 = 1.0 - l1 - l2;
                const double x  = l0 * x0 + l1 * x1 + l2 * x2;
                const double y  = l0 * y0 + l1 * y1 + l2 * y2;
                const double w  = gw[ia] * gw[ib] * one_ma * detJ;

                eval_case(kind, x, y, 0.0, uval, &pval, fval);
                tri6_shape(l0, l1, l2, phi);

                for (int a_node = 0; a_node < 6; ++a_node) {
                    const idx_t  node  = elems[a_node][e];
                    const double scale = w * phi[a_node];
                    rhs[(ptrdiff_t)node * block_size + 0] += real_t(scale * fval[0]);
                    rhs[(ptrdiff_t)node * block_size + 1] += real_t(scale * fval[1]);
                }
            }
        }
    }
}

void assemble_rhs_3d(const CaseKind kind, const std::shared_ptr<sfem::Mesh> &mesh, real_t *const rhs) {
    static constexpr double gx[5] = {
            0.046910077030668, 0.230765344947158, 0.5, 0.769234655052842, 0.953089922969332};
    static constexpr double gw[5] = {
            0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095};

    constexpr int block_size = 4;
    auto          elems      = mesh->elements(0)->data();
    auto          points     = mesh->points()->data();
    const auto    nelements  = mesh->n_elements();

    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const idx_t ev0 = elems[0][e];
        const idx_t ev1 = elems[1][e];
        const idx_t ev2 = elems[2][e];
        const idx_t ev3 = elems[3][e];

        const double x0 = points[0][ev0], y0 = points[1][ev0], z0 = points[2][ev0];
        const double x1 = points[0][ev1], y1 = points[1][ev1], z1 = points[2][ev1];
        const double x2 = points[0][ev2], y2 = points[1][ev2], z2 = points[2][ev2];
        const double x3 = points[0][ev3], y3 = points[1][ev3], z3 = points[2][ev3];

        const double a00 = x1 - x0, a01 = x2 - x0, a02 = x3 - x0;
        const double a10 = y1 - y0, a11 = y2 - y0, a12 = y3 - y0;
        const double a20 = z1 - z0, a21 = z2 - z0, a22 = z3 - z0;
        const double detJ = std::fabs(a00 * (a11 * a22 - a12 * a21) -
                                      a01 * (a10 * a22 - a12 * a20) +
                                      a02 * (a10 * a21 - a11 * a20));

        real_t fval[3], uval[3], pval;
        double phi[10];
        for (int ia = 0; ia < 5; ++ia) {
            const double r      = gx[ia];
            const double one_mr = 1.0 - r;
            for (int ib = 0; ib < 5; ++ib) {
                const double b      = gx[ib];
                const double s      = one_mr * b;
                const double one_mb = 1.0 - b;
                for (int ic = 0; ic < 5; ++ic) {
                    const double c  = gx[ic];
                    const double t  = one_mr * one_mb * c;
                    const double l1 = r;
                    const double l2 = s;
                    const double l3 = t;
                    const double l0 = 1.0 - l1 - l2 - l3;
                    const double x  = l0 * x0 + l1 * x1 + l2 * x2 + l3 * x3;
                    const double y  = l0 * y0 + l1 * y1 + l2 * y2 + l3 * y3;
                    const double z  = l0 * z0 + l1 * z1 + l2 * z2 + l3 * z3;
                    const double w  = gw[ia] * gw[ib] * gw[ic] * one_mr * one_mr * one_mb * detJ;

                    eval_case(kind, x, y, z, uval, &pval, fval);
                    tet10_shape(l0, l1, l2, l3, phi);

                    for (int a_node = 0; a_node < 10; ++a_node) {
                        const idx_t  node  = elems[a_node][e];
                        const double scale = w * phi[a_node];
                        rhs[(ptrdiff_t)node * block_size + 0] += real_t(scale * fval[0]);
                        rhs[(ptrdiff_t)node * block_size + 1] += real_t(scale * fval[1]);
                        rhs[(ptrdiff_t)node * block_size + 2] += real_t(scale * fval[2]);
                    }
                }
            }
        }
    }
}

void hex27_shape(const double a, const double b, const double c, double *const phi) {
    static constexpr int hex27_to_cartesian[27] = {
            0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15, 10, 14, 16, 12, 4, 22, 13,
    };
    const double lx[3] = {(2.0 * a - 1.0) * (a - 1.0), 4.0 * a * (1.0 - a), a * (2.0 * a - 1.0)};
    const double ly[3] = {(2.0 * b - 1.0) * (b - 1.0), 4.0 * b * (1.0 - b), b * (2.0 * b - 1.0)};
    const double lz[3] = {(2.0 * c - 1.0) * (c - 1.0), 4.0 * c * (1.0 - c), c * (2.0 * c - 1.0)};
    for (int node = 0; node < 27; ++node) {
        const int cart = hex27_to_cartesian[node];
        const int ix   = cart % 3;
        const int iy   = (cart / 3) % 3;
        const int iz   = cart / 9;
        phi[node]      = lx[ix] * ly[iy] * lz[iz];
    }
}

void assemble_rhs_3d_hex27(const CaseKind kind, const std::shared_ptr<sfem::Mesh> &mesh, real_t *const rhs) {
    static constexpr double gx[5] = {
            0.046910077030668, 0.230765344947158, 0.5, 0.769234655052842, 0.953089922969332};
    static constexpr double gw[5] = {
            0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095};

    constexpr int block_size = 4;
    auto          elems      = mesh->elements(0)->data();
    auto          points     = mesh->points()->data();
    const auto    nelements  = mesh->n_elements();

    for (ptrdiff_t e = 0; e < nelements; ++e) {
        double xmin = points[0][elems[0][e]], xmax = xmin;
        double ymin = points[1][elems[0][e]], ymax = ymin;
        double zmin = points[2][elems[0][e]], zmax = zmin;
        for (int i = 1; i < 8; ++i) {
            const idx_t node = elems[i][e];
            xmin = std::fmin(xmin, double(points[0][node]));
            xmax = std::fmax(xmax, double(points[0][node]));
            ymin = std::fmin(ymin, double(points[1][node]));
            ymax = std::fmax(ymax, double(points[1][node]));
            zmin = std::fmin(zmin, double(points[2][node]));
            zmax = std::fmax(zmax, double(points[2][node]));
        }
        const double hx = xmax - xmin;
        const double hy = ymax - ymin;
        const double hz = zmax - zmin;
        const double detJ = hx * hy * hz;

        real_t fval[3], uval[3], pval;
        double phi[27];
        for (int ia = 0; ia < 5; ++ia) {
            const double a = gx[ia];
            const double x = xmin + hx * a;
            for (int ib = 0; ib < 5; ++ib) {
                const double b = gx[ib];
                const double y = ymin + hy * b;
                for (int ic = 0; ic < 5; ++ic) {
                    const double c = gx[ic];
                    const double z = zmin + hz * c;
                    const double w = gw[ia] * gw[ib] * gw[ic] * detJ;

                    eval_case(kind, x, y, z, uval, &pval, fval);
                    hex27_shape(a, b, c, phi);

                    for (int a_node = 0; a_node < 27; ++a_node) {
                        const idx_t  node  = elems[a_node][e];
                        const double scale = w * phi[a_node];
                        rhs[(ptrdiff_t)node * block_size + 0] += real_t(scale * fval[0]);
                        rhs[(ptrdiff_t)node * block_size + 1] += real_t(scale * fval[1]);
                        rhs[(ptrdiff_t)node * block_size + 2] += real_t(scale * fval[2]);
                    }
                }
            }
        }
    }
}

int write_outputs(const CaseData &case_data,
                  const std::shared_ptr<sfem::Mesh> &mesh,
                  const char *const out_dir,
                  const real_t *const x) {
    mkdir(out_dir, 0777);

    auto output = smesh::Output::create(mesh, smesh::Path(out_dir));
    auto points = mesh->points()->data();
    const ptrdiff_t nnodes = mesh->n_nodes();
    if (output->write_nodal("x", smesh::TypeToEnum<geom_t>::value(), points[0])) return SFEM_FAILURE;
    if (output->write_nodal("y", smesh::TypeToEnum<geom_t>::value(), points[1])) return SFEM_FAILURE;
    if (case_data.dim == 3 && output->write_nodal("z", smesh::TypeToEnum<geom_t>::value(), points[2])) return SFEM_FAILURE;

    const int block_size = case_data.dim + 1;
    auto      component  = sfem::create_host_buffer<real_t>(nnodes);
    for (int d = 0; d < case_data.dim; ++d) {
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            component->data()[i] = x[i * block_size + d];
        }

        char name[32];
        std::snprintf(name, sizeof(name), "u%d", d);
        if (output->write_nodal(name, smesh::TypeToEnum<real_t>::value(), component->data())) return SFEM_FAILURE;
    }

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        component->data()[i] = x[i * block_size + case_data.dim];
    }

    return output->write_nodal("p", smesh::TypeToEnum<real_t>::value(), component->data());
}

void compute_errors(const CaseData &case_data,
                    const unsigned char *const pressure_active,
                    const std::shared_ptr<sfem::Mesh> &mesh,
                    const real_t *const x,
                    double &velocity_l2,
                    double &pressure_l2) {
    auto points = mesh->points()->data();
    const ptrdiff_t nnodes = mesh->n_nodes();
    const int       bs     = case_data.dim + 1;
    double          vu     = 0;
    double          pp     = 0;

    real_t exact_u[3], exact_p, force[3];
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        const double z = case_data.dim == 3 ? double(points[2][i]) : 0.0;
        eval_case(case_data.kind, points[0][i], points[1][i], z, exact_u, &exact_p, force);
        for (int d = 0; d < case_data.dim; ++d) {
            const double diff = double(x[i * bs + d] - exact_u[d]);
            vu += diff * diff;
        }
    }

    ptrdiff_t n_pressure = 0;
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        if (!pressure_active[i]) continue;
        const double z = case_data.dim == 3 ? double(points[2][i]) : 0.0;
        eval_case(case_data.kind, points[0][i], points[1][i], z, exact_u, &exact_p, force);
        const double diff = double(x[i * bs + case_data.dim] - exact_p);
        pp += diff * diff;
        ++n_pressure;
    }

    velocity_l2 = std::sqrt(vu / double(nnodes * case_data.dim));
    pressure_l2 = std::sqrt(pp / double(n_pressure));
}

void fill_exact_state(const CaseData &case_data,
                      const std::shared_ptr<sfem::Mesh> &mesh,
                      real_t *const x) {
    auto points = mesh->points()->data();
    const ptrdiff_t nnodes = mesh->n_nodes();
    const int       bs     = case_data.dim + 1;
    real_t exact_u[3], exact_p, force[3];
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        const double z = case_data.dim == 3 ? double(points[2][i]) : 0.0;
        eval_case(case_data.kind, points[0][i], points[1][i], z, exact_u, &exact_p, force);
        for (int d = 0; d < case_data.dim; ++d) {
            x[i * bs + d] = exact_u[d];
        }
        x[i * bs + case_data.dim] = exact_p;
    }
}

int dense_generated_solve(const std::shared_ptr<sfem::Operator<real_t>> &op,
                          const real_t *const rhs,
                          real_t *const       x) {
    const ptrdiff_t n = op->rows();
    real_t *const   A = (real_t *)std::calloc((size_t)n * (size_t)n, sizeof(real_t));
    real_t *const   e = (real_t *)std::calloc((size_t)n, sizeof(real_t));
    real_t *const   y = (real_t *)std::calloc((size_t)n, sizeof(real_t));
    if (!A || !e || !y) {
        std::free(A);
        std::free(e);
        std::free(y);
        return SFEM_FAILURE;
    }

    for (ptrdiff_t col = 0; col < n; ++col) {
        e[col] = 1;
        std::memset(y, 0, sizeof(real_t) * (size_t)n);
        op->apply(e, y);
        e[col] = 0;
        for (ptrdiff_t row = 0; row < n; ++row) {
            A[row * n + col] = y[row];
        }
    }

    for (ptrdiff_t i = 0; i < n; ++i) {
        x[i] = rhs[i];
    }

    int info = SFEM_SUCCESS;
    for (ptrdiff_t k = 0; k < n; ++k) {
        ptrdiff_t pivot = k;
        double    maxv  = std::fabs(double(A[k * n + k]));
        for (ptrdiff_t i = k + 1; i < n; ++i) {
            const double v = std::fabs(double(A[i * n + k]));
            if (v > maxv) {
                maxv  = v;
                pivot = i;
            }
        }

        if (maxv < 1e-14) {
            info = SFEM_FAILURE;
            break;
        }

        if (pivot != k) {
            for (ptrdiff_t j = k; j < n; ++j) {
                const real_t tmp     = A[k * n + j];
                A[k * n + j]         = A[pivot * n + j];
                A[pivot * n + j]     = tmp;
            }

            const real_t rhs_tmp = x[k];
            x[k]                 = x[pivot];
            x[pivot]             = rhs_tmp;
        }

        const real_t diag = A[k * n + k];
        for (ptrdiff_t i = k + 1; i < n; ++i) {
            const real_t factor = A[i * n + k] / diag;
            A[i * n + k] = 0;
            for (ptrdiff_t j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }

            x[i] -= factor * x[k];
        }
    }

    if (info == SFEM_SUCCESS) {
        for (ptrdiff_t i = n; i-- > 0;) {
            real_t sum = x[i];
            for (ptrdiff_t j = i + 1; j < n; ++j) {
                sum -= A[i * n + j] * x[j];
            }

            x[i] = sum / A[i * n + i];
        }
    }

    std::free(A);
    std::free(e);
    std::free(y);
    return info;
}

}  // namespace

int main(int argc, char **argv) {
    auto ctx  = sfem::initialize(argc, argv);
    auto comm = ctx->communicator();

    if (comm->size() != 1) {
        if (comm->rank() == 0) {
            std::fprintf(stderr, "generated_stokes_fvca8 currently supports serial runs only\n");
        }
        return SFEM_FAILURE;
    }

    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <bercovier_engelman_2d|taylor_green_3d> <resolution> <output-dir>\n", argv[0]);
        return SFEM_FAILURE;
    }

    CaseData case_data;
    if (!case_from_name(argv[1], case_data)) {
        std::fprintf(stderr, "unsupported case: %s\n", argv[1]);
        return SFEM_FAILURE;
    }

    const int resolution = std::atoi(argv[2]);
    if (resolution < 1) {
        std::fprintf(stderr, "resolution must be positive\n");
        return SFEM_FAILURE;
    }

    sfem::register_generated_ops();

    std::shared_ptr<sfem::Mesh> mesh;
    if (case_data.dim == 2) {
        mesh = sfem::Mesh::create_square(comm, smesh::TRI6, resolution, resolution, 0, 0, 1, 1);
    } else {
        const bool use_tets = smesh::Env::read<bool>("SFEM_FVCA8_USE_TETS", false);
        if (use_tets) {
            mesh = sfem::Mesh::create_cube(comm, smesh::TET10, resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
        } else {
            mesh = sfem::Mesh::create_cube(comm, smesh::HEX27, resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
        }
    }

    auto fs = sfem::FunctionSpace::create(mesh, case_data.dim + 1);

    auto op = sfem::create_op(fs, "GeneratedStokes", sfem::EXECUTION_SPACE_HOST);
    if (!op) {
        std::fprintf(stderr, "failed to create GeneratedStokes\n");
        return SFEM_FAILURE;
    }

    auto dirichlet = std::make_shared<sfem::DirichletConditions>(fs);
    auto points    = mesh->points()->data();
    const ptrdiff_t nnodes = mesh->n_nodes();
    auto pressure_active = sfem::create_host_buffer<unsigned char>(nnodes);
    const int n_pressure_nodes_per_element = mesh->element_type(0) == smesh::HEX27 ? 8 : (case_data.dim + 1);
    auto elems = mesh->elements(0)->data();
    for (ptrdiff_t e = 0; e < mesh->n_elements(); ++e) {
        for (int i = 0; i < n_pressure_nodes_per_element; ++i) {
            pressure_active->data()[elems[i][e]] = 1;
        }
    }

    ptrdiff_t n_boundary = 0;
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        n_boundary += is_boundary_node(points, case_data.dim, i) ? 1 : 0;
    }

    for (int d = 0; d < case_data.dim; ++d) {
        idx_t  *nodes  = (idx_t *)std::malloc(sizeof(idx_t) * n_boundary);
        real_t *values = (real_t *)std::malloc(sizeof(real_t) * n_boundary);
        ptrdiff_t offset = 0;
        real_t exact_u[3], exact_p, force[3];
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            if (!is_boundary_node(points, case_data.dim, i)) continue;
            const double z = case_data.dim == 3 ? double(points[2][i]) : 0.0;
            eval_case(case_data.kind, points[0][i], points[1][i], z, exact_u, &exact_p, force);
            nodes[offset]  = idx_t(i);
            values[offset] = exact_u[d];
            ++offset;
        }

        dirichlet->add_condition(n_boundary, n_boundary, nodes, d, values);
    }

    ptrdiff_t n_pressure_inactive = 0;
    idx_t     pressure_pin        = 0;
    bool      found_pressure_pin  = false;
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        if (pressure_active->data()[i]) {
            if (!found_pressure_pin) {
                pressure_pin       = idx_t(i);
                found_pressure_pin = true;
            }
        } else {
            ++n_pressure_inactive;
        }
    }

    const ptrdiff_t n_pressure_constraints = n_pressure_inactive + 1;
    idx_t          *p_nodes                = (idx_t *)std::malloc(sizeof(idx_t) * n_pressure_constraints);
    real_t         *p_values               = (real_t *)std::malloc(sizeof(real_t) * n_pressure_constraints);
    real_t          exact_u[3], exact_p, force[3];
    eval_case(case_data.kind, points[0][pressure_pin], points[1][pressure_pin], case_data.dim == 3 ? double(points[2][pressure_pin]) : 0.0, exact_u, &exact_p, force);
    p_nodes[0]  = pressure_pin;
    p_values[0] = exact_p;
    ptrdiff_t out = 1;
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        if (pressure_active->data()[i]) continue;
        const double    z   = case_data.dim == 3 ? double(points[2][i]) : 0.0;
        eval_case(case_data.kind, points[0][i], points[1][i], z, exact_u, &exact_p, force);
        p_nodes[out]  = idx_t(i);
        p_values[out] = exact_p;
        ++out;
    }
    dirichlet->add_condition(n_pressure_constraints, n_pressure_constraints, p_nodes, case_data.dim, p_values);

    auto f = sfem::Function::create(fs);
    f->add_constraint(dirichlet);
    f->add_operator(op);

    auto x   = sfem::create_host_buffer<real_t>(fs->n_dofs());
    auto rhs = sfem::create_host_buffer<real_t>(fs->n_dofs());

    if (case_data.dim == 2) {
        assemble_rhs_2d(case_data.kind, mesh, rhs->data());
    } else if (mesh->element_type(0) == smesh::HEX27) {
        assemble_rhs_3d_hex27(case_data.kind, mesh, rhs->data());
    } else {
        assemble_rhs_3d(case_data.kind, mesh, rhs->data());
    }

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    auto linear_op       = sfem::make_linear_op(f);
    const bool use_dense = smesh::Env::read<bool>("SFEM_DENSE_SOLVE", fs->n_dofs() <= 2048);

    int info       = SFEM_FAILURE;
    int iterations = 0;
    const double tick = MPI_Wtime();
    if (use_dense) {
        info = dense_generated_solve(linear_op, rhs->data(), x->data());
    } else {
        auto solver     = sfem::create_bcgs<real_t>(linear_op, sfem::EXECUTION_SPACE_HOST);
        solver->verbose = smesh::Env::read<bool>("SFEM_VERBOSE", false);
        solver->set_max_it(smesh::Env::read<int>("SFEM_MAX_IT", 20000));
        solver->set_atol(smesh::Env::read<real_t>("SFEM_ATOL", 1e-10));
        info       = solver->apply(rhs->data(), x->data());
        iterations = solver->iterations();
    }
    const double solve_time = MPI_Wtime() - tick;

    auto residual = sfem::create_host_buffer<real_t>(fs->n_dofs());
    std::memset(residual->data(), 0, sizeof(real_t) * (size_t)fs->n_dofs());
    linear_op->apply(x->data(), residual->data());
    double r2 = 0;
    for (ptrdiff_t i = 0; i < fs->n_dofs(); ++i) {
        const double diff = double(residual->data()[i] - rhs->data()[i]);
        r2 += diff * diff;
    }

    auto exact_state = sfem::create_host_buffer<real_t>(fs->n_dofs());
    auto exact_residual = sfem::create_host_buffer<real_t>(fs->n_dofs());
    fill_exact_state(case_data, mesh, exact_state->data());
    f->apply_constraints(exact_state->data());
    std::memset(exact_residual->data(), 0, sizeof(real_t) * (size_t)fs->n_dofs());
    linear_op->apply(exact_state->data(), exact_residual->data());
    double exact_r2 = 0;
    for (ptrdiff_t i = 0; i < fs->n_dofs(); ++i) {
        const double diff = double(exact_residual->data()[i] - rhs->data()[i]);
        exact_r2 += diff * diff;
    }

    double velocity_l2 = 0, pressure_l2 = 0;
    compute_errors(case_data, pressure_active->data(), mesh, x->data(), velocity_l2, pressure_l2);

    const char *const out_dir = argv[3];
    mkdir(out_dir, 0777);
    mesh->write(smesh::Path(std::string(out_dir) + "/mesh"));
    if (write_outputs(case_data, mesh, out_dir, x->data())) {
        return SFEM_FAILURE;
    }

    const std::string summary_path = std::string(out_dir) + "/summary.csv";
    FILE *summary = std::fopen(summary_path.c_str(), "w");
    if (summary) {
        std::fprintf(summary, "case,resolution,nodes,elements,dofs,solver_info,iterations,solve_seconds,residual_l2,exact_residual_l2,velocity_l2,pressure_l2\n");
        std::fprintf(summary,
                     "%s,%d,%ld,%ld,%ld,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     case_data.name,
                     resolution,
                     long(nnodes),
                     long(mesh->n_elements()),
                     long(fs->n_dofs()),
                     info,
                     iterations,
                     solve_time,
                     std::sqrt(r2),
                     std::sqrt(exact_r2),
                     velocity_l2,
                     pressure_l2);
        std::fclose(summary);
    }

    std::printf("case %s resolution %d dofs %ld info %d iterations %d residual_l2 %.6e exact_residual_l2 %.6e velocity_l2 %.6e pressure_l2 %.6e solve_seconds %.6e\n",
                case_data.name,
                resolution,
                long(fs->n_dofs()),
                info,
                iterations,
                std::sqrt(r2),
                std::sqrt(exact_r2),
                velocity_l2,
                pressure_l2,
                solve_time);

    return info == SFEM_SUCCESS ? SFEM_SUCCESS : SFEM_FAILURE;
}
