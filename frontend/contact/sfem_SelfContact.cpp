#include "sfem_SelfContact.hpp"

#include "bvh/bvh.hpp"
#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_OpFactory.hpp"
#include "smesh_sort.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef SFEM_ENABLE_YAML
#include <ryml.hpp>
#endif

// #define VIZ_DEBUG

#ifdef VIZ_DEBUG
#include "/Users/patrickzulian/Desktop/code/sviz/src/sviz_monitor_client.hpp"
#endif

namespace sfem {

    namespace {

        struct BiorthogonalQuad4Weights {
            // real_t values[16] = {
            //         // phi 0
            //         4.0,
            //         -2.0,
            //         1.0,
            //         -2.0,
            //         // phi 1
            //         -2.0,
            //         4.0,
            //         -2.0,
            //         1.0,
            //         // phi 2
            //         1.0,
            //         -2.0,
            //         4.0,
            //         -2.0,
            //         // phi 3
            //         -2.0,
            //         1.0,
            //         -2.0,
            //         4.0};

            real_t values[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        };
        std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> create_contact_graph(const smesh::SharedBuffer<idx_t*>& elements,
                                                                              const smesh::SharedBuffer<idx_t>&  element_idx) {
            const ptrdiff_t npoints = element_idx->size();
            const int       nxe     = elements->extent(0);
            auto            rowptr  = sfem::create_host_buffer<count_t>(npoints + 1);
            auto            colidx  = sfem::create_host_buffer<idx_t>(npoints * nxe);

            auto elements_data    = elements->data();
            auto rowptr_data      = rowptr->data();
            auto colidx_data      = colidx->data();
            auto element_idx_data = element_idx->data();

            for (ptrdiff_t i = 0; i < npoints; i++) {
                ptrdiff_t offset   = element_idx_data[i] == -1 ? 0 : nxe;
                rowptr_data[i + 1] = offset + rowptr_data[i];
            }

            for (ptrdiff_t i = 0; i < npoints; i++) {
                const idx_t e = element_idx_data[i];
                if (e == -1) continue;
                for (int j = 0; j < nxe; j++) {
                    colidx_data[rowptr_data[i] + j] = elements_data[j][e];
                }
            }

            return std::make_shared<smesh::CRSGraph<count_t, idx_t>>(rowptr, colidx);
        }
        void assemble_coupling_operator(const smesh::ElemType                  element_type,
                                        const smesh::SharedBuffer<idx_t*>&     elements,
                                        const smesh::SharedBuffer<idx_t>&      element_idx,
                                        const smesh::SharedBuffer<real_t>&     s,
                                        const smesh::SharedBuffer<real_t>&     t,
                                        const smesh::CRSGraph<count_t, idx_t>& graph,
                                        const smesh::SharedBuffer<real_t>&     values) {
            const ptrdiff_t n   = element_idx->size();
            const int       nxe = elements->extent(0);

            SMESH_ASSERT(n == s->size());
            SMESH_ASSERT(n == t->size());

            auto rowptr = graph.rowptr()->data();
            auto vals   = values->data();
            auto s_data = s->data();
            auto t_data = t->data();
            auto e_data = element_idx->data();

            if (element_type == smesh::TRISHELL3) {
                SMESH_ASSERT(nxe == 3);
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    if (e_data[i] == -1) continue;

                    const count_t row_offset = rowptr[i];
                    SMESH_ASSERT(rowptr[i + 1] - row_offset == nxe);

                    const real_t si = s_data[i];
                    const real_t ti = t_data[i];

                    vals[row_offset + 0] = 1 - si - ti;
                    vals[row_offset + 1] = si;
                    vals[row_offset + 2] = ti;
                }
            } else if (element_type == smesh::QUADSHELL4) {
                SMESH_ASSERT(nxe == 4);
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    if (e_data[i] == -1) continue;

                    const count_t row_offset = rowptr[i];
                    SMESH_ASSERT(rowptr[i + 1] - row_offset == nxe);

                    const real_t si          = s_data[i];
                    const real_t ti          = t_data[i];
                    const real_t one_minus_s = 1 - si;
                    const real_t one_minus_t = 1 - ti;

                    vals[row_offset + 0] = one_minus_s * one_minus_t;
                    vals[row_offset + 1] = si * one_minus_t;
                    vals[row_offset + 2] = si * ti;
                    vals[row_offset + 3] = one_minus_s * ti;
                }
            } else {
                SFEM_ERROR("assemble_coupling_operator not implemented for element type %d\n", element_type);
            }
        }

        void displace_points(const std::shared_ptr<smesh::Mesh>&     surface,
                             const std::shared_ptr<Buffer<real_t>>&  displacement,
                             const std::shared_ptr<Buffer<real_t*>>& inout) {
            auto p = inout->data();
            auto u = displacement->data();
            auto m = surface->node_mapping()->data();

            const ptrdiff_t n   = surface->node_mapping()->size();
            const int       dim = surface->spatial_dimension();

            for (int d = 0; d < dim; d++) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    p[d][i] += u[m[i] * dim + d];
                }
            }
        }

        class ContactNodeToSurface final : public Contact {
        public:
            ContactNodeToSurface(const std::shared_ptr<FunctionSpace>& space,
                                 const std::shared_ptr<smesh::Mesh>&   surface,
                                 const real_t                          margin,
                                 const real_t                          search_radius_sqr,
                                 const ExecutionSpace                  es)
                : space_(space),
                  surface_(surface),
                  margin_(margin),
                  search_radius_sqr_(search_radius_sqr),
                  es_(es),
                  dim_(surface->spatial_dimension()),
                  npoints_(surface->n_nodes()),
                  nselements_(surface->n_elements()),
                  trace_space_(std::make_shared<FunctionSpace>(surface_, 1)),
                  mass_vector_(create_host_buffer<real_t>(trace_space_->n_dofs())),
                  surface_elements_(surface->block(0)->elements()),
                  surface_element_type_(surface->block(0)->element_type()),
                  closest_points_(sfem::create_buffer<real_t>(dim_, npoints_, es_)),
                  closest_s_(sfem::create_buffer<real_t>(npoints_, es_)),
                  closest_t_(sfem::create_buffer<real_t>(npoints_, es_)),
                  distances_(sfem::create_buffer<real_t>(npoints_, es_)),
                  closest_triangles_(sfem::create_buffer<idx_t>(npoints_, es_)),
                  distances_whole_(sfem::create_buffer<real_t>(space_->n_dofs(), es_)),
                  directors_(sfem::create_buffer<real_t>(space_->n_dofs(), es_)),
                  normals_(sfem::create_buffer<real_t>(dim_, npoints_, es_)),
                  frozen_displacement_(sfem::create_buffer<real_t>(space_->n_dofs(), es_)) {
                assemble_mass_vector();
            }

            void recompute(const std::shared_ptr<Buffer<real_t>>& displacement) override {
                SFEM_TRACE_SCOPE("ContactNodeToSurface::recompute");

                auto blas = sfem::blas<real_t>(es_);

                p1_ = smesh::astype<real_t>(surface_->points());
                displace_points(surface_, displacement, p1_);

                if (surface_element_type_ == smesh::TRISHELL3) {
                    ssdf::closest_within_radius_local_bvh(npoints_,
                                                          p1_->data()[0],
                                                          p1_->data()[1],
                                                          p1_->data()[2],
                                                          nselements_,
                                                          surface_elements_->data()[0],
                                                          surface_elements_->data()[1],
                                                          surface_elements_->data()[2],
                                                          npoints_,
                                                          p1_->data()[0],
                                                          p1_->data()[1],
                                                          p1_->data()[2],
                                                          0,
                                                          &search_radius_sqr_,
                                                          closest_triangles_->data(),
                                                          distances_->data(),
                                                          closest_s_->data(),
                                                          closest_t_->data(),
                                                          true);
                } else if (surface_element_type_ == smesh::QUADSHELL4) {
                    ssdf::closest_within_radius_quads_local_bvh(npoints_,
                                                                p1_->data()[0],
                                                                p1_->data()[1],
                                                                p1_->data()[2],
                                                                nselements_,
                                                                surface_elements_->data()[0],
                                                                surface_elements_->data()[1],
                                                                surface_elements_->data()[2],
                                                                surface_elements_->data()[3],
                                                                npoints_,
                                                                p1_->data()[0],
                                                                p1_->data()[1],
                                                                p1_->data()[2],
                                                                0,
                                                                &search_radius_sqr_,
                                                                closest_triangles_->data(),
                                                                distances_->data(),
                                                                closest_s_->data(),
                                                                closest_t_->data(),
                                                                true);
                } else {
                    SFEM_ERROR("Closest point search not implemented for element type %d\n", surface_element_type_);
                }

                blas->values(space_->n_dofs(), 0, distances_whole_->data());
                blas->values(space_->n_dofs(), 0, directors_->data());
                for (int d = 0; d < dim_; ++d) {
                    blas->values(npoints_, 0, normals_->data()[d]);
                }

                auto node_mapping           = surface_->node_mapping()->data();
                auto directors_data         = directors_->data();
                auto distances_whole_data   = distances_whole_->data();
                auto distances_data         = distances_->data();
                auto closest_points_data    = closest_points_->data();
                auto p1_data                = p1_->data();
                auto closest_triangles_data = closest_triangles_->data();
                auto closest_s_data         = closest_s_->data();
                auto closest_t_data         = closest_t_->data();
                auto normals_data           = normals_->data();
                auto surface_elements_data  = surface_elements_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < npoints_; i++) {
                    const idx_t elem = closest_triangles_data[i];
                    if (elem == -1) {
                        distances_data[i] = std::sqrt(distances_data[i]);
                        continue;
                    }

                    const real_t s = closest_s_data[i];
                    const real_t t = closest_t_data[i];
                    real_t       cx, cy, cz;
                    real_t       tnx, tny, tnz;

                    if (surface_element_type_ == smesh::TRISHELL3) {
                        const idx_t e0 = surface_elements_data[0][elem];
                        const idx_t e1 = surface_elements_data[1][elem];
                        const idx_t e2 = surface_elements_data[2][elem];

                        const real_t r = 1 - s - t;

                        cx = r * p1_data[0][e0] + s * p1_data[0][e1] + t * p1_data[0][e2];
                        cy = r * p1_data[1][e0] + s * p1_data[1][e1] + t * p1_data[1][e2];
                        cz = r * p1_data[2][e0] + s * p1_data[2][e1] + t * p1_data[2][e2];

                        const real_t v0x = p1_data[0][e1] - p1_data[0][e0];
                        const real_t v0y = p1_data[1][e1] - p1_data[1][e0];
                        const real_t v0z = p1_data[2][e1] - p1_data[2][e0];

                        const real_t v1x = p1_data[0][e2] - p1_data[0][e0];
                        const real_t v1y = p1_data[1][e2] - p1_data[1][e0];
                        const real_t v1z = p1_data[2][e2] - p1_data[2][e0];

                        tnx = v0y * v1z - v0z * v1y;
                        tny = v0z * v1x - v0x * v1z;
                        tnz = v0x * v1y - v0y * v1x;
                    } else {
                        const idx_t e0 = surface_elements_data[0][elem];
                        const idx_t e1 = surface_elements_data[1][elem];
                        const idx_t e2 = surface_elements_data[2][elem];
                        const idx_t e3 = surface_elements_data[3][elem];

                        const real_t one_minus_s = 1 - s;
                        const real_t one_minus_t = 1 - t;
                        const real_t w0          = one_minus_s * one_minus_t;
                        const real_t w1          = s * one_minus_t;
                        const real_t w2          = s * t;
                        const real_t w3          = one_minus_s * t;

                        cx = w0 * p1_data[0][e0] + w1 * p1_data[0][e1] + w2 * p1_data[0][e2] + w3 * p1_data[0][e3];
                        cy = w0 * p1_data[1][e0] + w1 * p1_data[1][e1] + w2 * p1_data[1][e2] + w3 * p1_data[1][e3];
                        cz = w0 * p1_data[2][e0] + w1 * p1_data[2][e1] + w2 * p1_data[2][e2] + w3 * p1_data[2][e3];

                        const real_t dsx =
                                one_minus_t * (p1_data[0][e1] - p1_data[0][e0]) + t * (p1_data[0][e2] - p1_data[0][e3]);
                        const real_t dsy =
                                one_minus_t * (p1_data[1][e1] - p1_data[1][e0]) + t * (p1_data[1][e2] - p1_data[1][e3]);
                        const real_t dsz =
                                one_minus_t * (p1_data[2][e1] - p1_data[2][e0]) + t * (p1_data[2][e2] - p1_data[2][e3]);

                        const real_t dtx =
                                one_minus_s * (p1_data[0][e3] - p1_data[0][e0]) + s * (p1_data[0][e2] - p1_data[0][e1]);
                        const real_t dty =
                                one_minus_s * (p1_data[1][e3] - p1_data[1][e0]) + s * (p1_data[1][e2] - p1_data[1][e1]);
                        const real_t dtz =
                                one_minus_s * (p1_data[2][e3] - p1_data[2][e0]) + s * (p1_data[2][e2] - p1_data[2][e1]);

                        tnx = dsy * dtz - dsz * dty;
                        tny = dsz * dtx - dsx * dtz;
                        tnz = dsx * dty - dsy * dtx;
                    }

                    closest_points_data[0][i] = cx;
                    closest_points_data[1][i] = cy;
                    closest_points_data[2][i] = cz;

                    const real_t tnn = std::sqrt(tnx * tnx + tny * tny + tnz * tnz);

                    tnx /= tnn;
                    tny /= tnn;
                    tnz /= tnn;

                    const real_t dx = p1_data[0][i] - cx;
                    const real_t dy = p1_data[1][i] - cy;
                    const real_t dz = p1_data[2][i] - cz;
                    const real_t dn = std::sqrt(dx * dx + dy * dy + dz * dz);

                    real_t nx = 0, ny = 0, nz = 0;
                    if (dn > 0) {
                        nx = dx / dn;
                        ny = dy / dn;
                        nz = dz / dn;
                    } else {
                        nx = tnx;
                        ny = tny;
                        nz = tnz;
                    }

                    const real_t cos_angle = nx * tnx + ny * tny + nz * tnz;
                    if (std::abs(cos_angle) < 1e-6) {
                        closest_triangles_data[i] = -1;
                        continue;
                    }

                    const real_t    signed_dist = dx * nx + dy * ny + dz * nz - margin_;
                    const ptrdiff_t dof         = (ptrdiff_t)node_mapping[i] * dim_;

                    distances_data[i]         = signed_dist;
                    distances_whole_data[dof] = signed_dist;
                    directors_data[dof + 0]   = -signed_dist * nx;
                    directors_data[dof + 1]   = -signed_dist * ny;
                    directors_data[dof + 2]   = -signed_dist * nz;
                    normals_data[0][i]        = -nx;
                    normals_data[1][i]        = -ny;
                    normals_data[2][i]        = -nz;
                }

                graph_  = create_contact_graph(surface_elements_, closest_triangles_);
                values_ = sfem::create_buffer<real_t>(graph_->nnz(), es_);

                assemble_coupling_operator(
                        surface_element_type_, surface_elements_, closest_triangles_, closest_s_, closest_t_, *graph_, values_);
                blas->copy(space_->n_dofs(), displacement->data(), frozen_displacement_->data());
            }

            const std::shared_ptr<smesh::CRSGraph<count_t, idx_t>>& graph() const override { return graph_; }
            smesh::SharedBuffer<real_t>&                            values() override { return values_; }
            smesh::SharedBuffer<real_t>&                            mass_vector() override { return mass_vector_; }
            smesh::SharedBuffer<real_t*>&                           normals() override { return normals_; }
            smesh::SharedBuffer<real_t>&                            distances() override { return distances_; }
            smesh::SharedBuffer<real_t>&       frozen_displacement() override { return frozen_displacement_; }
            const smesh::SharedBuffer<real_t>& distances_whole() const override { return distances_whole_; }
            const smesh::SharedBuffer<real_t>& directors() const override { return directors_; }

        private:
            void assemble_mass_vector() {
                auto bop = sfem::Factory::create_op(trace_space_, "Mass");
                bop->initialize();

                auto ones = create_host_buffer<real_t>(trace_space_->n_dofs());
                sfem::blas<real_t>(EXECUTION_SPACE_HOST)->values(trace_space_->n_dofs(), 1, ones->data());
                bop->apply(nullptr, ones->data(), mass_vector_->data());
            }

            std::shared_ptr<FunctionSpace>                   space_;
            std::shared_ptr<smesh::Mesh>                     surface_;
            real_t                                           margin_;
            real_t                                           search_radius_sqr_;
            ExecutionSpace                                   es_;
            int                                              dim_;
            ptrdiff_t                                        npoints_;
            ptrdiff_t                                        nselements_;
            std::shared_ptr<FunctionSpace>                   trace_space_;
            smesh::SharedBuffer<real_t>                      mass_vector_;
            smesh::SharedBuffer<idx_t*>                      surface_elements_;
            smesh::ElemType                                  surface_element_type_;
            smesh::SharedBuffer<real_t*>                     p1_;
            smesh::SharedBuffer<real_t*>                     closest_points_;
            smesh::SharedBuffer<real_t>                      closest_s_;
            smesh::SharedBuffer<real_t>                      closest_t_;
            smesh::SharedBuffer<real_t>                      distances_;
            smesh::SharedBuffer<idx_t>                       closest_triangles_;
            smesh::SharedBuffer<real_t>                      distances_whole_;
            smesh::SharedBuffer<real_t>                      directors_;
            smesh::SharedBuffer<real_t*>                     normals_;
            smesh::SharedBuffer<real_t>                      frozen_displacement_;
            std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> graph_;
            smesh::SharedBuffer<real_t>                      values_;
        };

        namespace {

            constexpr int MORTAR_TRI4_NQP = 6;
            // Capacity of the clipped intersection polygon. Two convex quads intersect in at most
            // 8 vertices in exact arithmetic; the extra room absorbs floating-point degeneracies and
            // bounds the Sutherland-Hodgman output (writes past this are dropped, see clipping loop).
            constexpr int MORTAR_MAX_POLY     = 16;
            constexpr int MORTAR_MAX_TRIS     = MORTAR_MAX_POLY - 2;
            constexpr int MORTAR_MAX_QP       = MORTAR_MAX_TRIS * MORTAR_TRI4_NQP;
            constexpr int MORTAR_NEWTON_ITERS = 5;

            // Per-pair output layout in the values buffer: M(4x4) = 16 reals.
            // D (slave-slave) is diagonal by construction of the dual basis and is not stored.
            constexpr int MORTAR_PAIR_STRIDE = 16;

            // Maximum number of nodes per surface element handled here (QUADSHELL4).
            constexpr int MORTAR_MAX_NXE = 4;

            // Position of key in a sorted row (lower-bound index). Key is assumed present.
            SFEM_INLINE idx_t mortar_find_col(const idx_t key, const idx_t* const SFEM_RESTRICT row, const int lenrow) {
                int lo = 0;
                int hi = lenrow;
                while (lo < hi) {
                    const int mid = (lo + hi) / 2;
                    if (row[mid] < key) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                return lo;
            }

            // Find the column offset of each target in a sorted row at once (mirrors tet4_find_cols).
            // For short rows the branchless count (#entries < target == its index) is used; longer rows
            // fall back to binary search.
            SFEM_INLINE void mortar_find_cols(const idx_t* const SFEM_RESTRICT targets,
                                              const int                        ntargets,
                                              const idx_t* const SFEM_RESTRICT row,
                                              const int                        lenrow,
                                              idx_t* const SFEM_RESTRICT       ks) {
                if (lenrow > 32) {
                    for (int d = 0; d < ntargets; ++d) {
                        ks[d] = mortar_find_col(targets[d], row, lenrow);
                    }
                } else {
                    for (int d = 0; d < ntargets; ++d) {
                        ks[d] = 0;
                    }
                    for (int i = 0; i < lenrow; ++i) {
                        for (int d = 0; d < ntargets; ++d) {
                            ks[d] += row[i] < targets[d];
                        }
                    }
                }
            }

            // QUADSHELL4 bilinear shape functions (node order: w0=(1-s)(1-t), w1=s(1-t), w2=st, w3=(1-s)t).
            SFEM_INLINE void quad4_shape(const real_t s, const real_t t, real_t* const SFEM_RESTRICT phi) {
                const real_t one_minus_s = real_t(1) - s;
                const real_t one_minus_t = real_t(1) - t;
                phi[0]                   = one_minus_s * one_minus_t;
                phi[1]                   = s * one_minus_t;
                phi[2]                   = s * t;
                phi[3]                   = one_minus_s * t;
            }

            // TODO replace it with mass vector implemenetation is SFEM

            // Lumped nodal areas of a (3D shell) bilinear quad: out[a] = integral over the full element of phi_a dA.
            // By biorthogonality of the dual basis this equals the proper dual-mortar diagonal D_aa, which is strictly
            // positive (phi_a >= 0, dA > 0) and hence a safe nodal mass / projection normalizer. 2x2 Gauss is exact
            // for phi_a * |x_s x x_t| on a bilinear quad.
            SFEM_INLINE void quad4_nodal_areas(const real_t* const SFEM_RESTRICT X,
                                               const real_t* const SFEM_RESTRICT Y,
                                               const real_t* const SFEM_RESTRICT Z,
                                               real_t* const SFEM_RESTRICT       out_area) {
                const real_t half_offset = real_t(0.5) / std::sqrt(real_t(3));
                const real_t gp[2]       = {real_t(0.5) - half_offset, real_t(0.5) + half_offset};

                out_area[0] = out_area[1] = out_area[2] = out_area[3] = real_t(0);

                for (int is = 0; is < 2; ++is) {
                    for (int it = 0; it < 2; ++it) {
                        const real_t s = gp[is];
                        const real_t t = gp[it];

                        real_t phi[4];
                        quad4_shape(s, t, phi);

                        const real_t dphi_ds[4] = {-(real_t(1) - t), (real_t(1) - t), t, -t};
                        const real_t dphi_dt[4] = {-(real_t(1) - s), -s, s, (real_t(1) - s)};

                        real_t xs[3] = {0, 0, 0};
                        real_t xt[3] = {0, 0, 0};
                        for (int c = 0; c < 4; ++c) {
                            xs[0] += dphi_ds[c] * X[c];
                            xs[1] += dphi_ds[c] * Y[c];
                            xs[2] += dphi_ds[c] * Z[c];
                            xt[0] += dphi_dt[c] * X[c];
                            xt[1] += dphi_dt[c] * Y[c];
                            xt[2] += dphi_dt[c] * Z[c];
                        }

                        const real_t cx   = xs[1] * xt[2] - xs[2] * xt[1];
                        const real_t cy   = xs[2] * xt[0] - xs[0] * xt[2];
                        const real_t cz   = xs[0] * xt[1] - xs[1] * xt[0];
                        const real_t detJ = std::sqrt(cx * cx + cy * cy + cz * cz);

                        // 2x2 Gauss weight on [0,1]^2 is 0.25 per point.
                        const real_t wj = real_t(0.25) * detJ;
                        for (int a = 0; a < 4; ++a) {
                            out_area[a] += wj * phi[a];
                        }
                    }
                }
            }

            // Degree-4 Strang triangle rule (6 points, weights sum to 1).
            static const real_t mortar_tri4_l1[MORTAR_TRI4_NQP] = {0.445948174109818469204897801127010,
                                                                   0.445948174109818469204897801127010,
                                                                   0.108103018168070960085165291402836,
                                                                   0.091576213509771705118396573460019,
                                                                   0.816847572980327727169962891363187,
                                                                   0.091576213509771705118396573460019};
            static const real_t mortar_tri4_l2[MORTAR_TRI4_NQP] = {0.108103018168070960085165291402836,
                                                                   0.445948174109818469204897801127010,
                                                                   0.445948174109818469204897801127010,
                                                                   0.816847572980327727169962891363187,
                                                                   0.091576213509771705118396573460019,
                                                                   0.091576213509771705118396573460019};
            static const real_t mortar_tri4_qw[MORTAR_TRI4_NQP] = {0.223381589678011143442532970975561,
                                                                   0.223381589678011143442532970975561,
                                                                   0.223381589678011143442532970975561,
                                                                   0.109951743655322763443642875337266,
                                                                   0.109951743655322763443642875337266,
                                                                   0.109951743655322763443642875337266};

            SFEM_INLINE void invert_quad4_plane(const real_t* SFEM_RESTRICT px,
                                                const real_t* SFEM_RESTRICT py,
                                                const real_t                target_x,
                                                const real_t                target_y,
                                                real_t* const               out_s,
                                                real_t* const               out_t) {
                real_t s = real_t(0.5);
                real_t t = real_t(0.5);

                for (int iter = 0; iter < MORTAR_NEWTON_ITERS; iter++) {
                    const real_t one_minus_s = real_t(1) - s;
                    const real_t one_minus_t = real_t(1) - t;
                    const real_t w0          = one_minus_s * one_minus_t;
                    const real_t w1          = s * one_minus_t;
                    const real_t w2          = s * t;
                    const real_t w3          = one_minus_s * t;

                    const real_t map_x = w0 * px[0] + w1 * px[1] + w2 * px[2] + w3 * px[3];
                    const real_t map_y = w0 * py[0] + w1 * py[1] + w2 * py[2] + w3 * py[3];

                    const real_t rx = map_x - target_x;
                    const real_t ry = map_y - target_y;

                    const real_t dsx = one_minus_t * (px[1] - px[0]) + t * (px[2] - px[3]);
                    const real_t dsy = one_minus_t * (py[1] - py[0]) + t * (py[2] - py[3]);
                    const real_t dtx = one_minus_s * (px[3] - px[0]) + s * (px[2] - px[1]);
                    const real_t dty = one_minus_s * (py[3] - py[0]) + s * (py[2] - py[1]);

                    const real_t det     = dsx * dty - dsy * dtx;
                    const real_t inv_det = real_t(1) / (det + (det >= 0 ? real_t(1e-30) : real_t(-1e-30)));

                    s -= (dty * rx - dtx * ry) * inv_det;
                    t -= (-dsy * rx + dsx * ry) * inv_det;
                }

                *out_s = s;
                *out_t = t;
            }

        }  // namespace

        void assemble_mortar_matrices(const smesh::ElemType          element_type,
                                      const SharedBuffer<idx_t*>&    elements,
                                      const SharedBuffer<real_t*>&   points,
                                      const SharedBuffer<ptrdiff_t>& pc_ptr,
                                      const SharedBuffer<idx_t>&     pc_idx,
                                      const SharedBuffer<real_t>&    values,
                                      const SharedBuffer<real_t>&    weighted_normals,
                                      const SharedBuffer<real_t>&    weighted_gap,
                                      const SharedBuffer<real_t>&    weighted_distance,
                                      const SharedBuffer<real_t>&    distance_weight,
                                      const SharedBuffer<real_t>&    mass_vector,
                                      const SharedBuffer<mask_t>&    is_valid,
                                      const real_t                   max_gap) {
            auto ptr  = pc_ptr->data();
            auto idx  = pc_idx->data();
            auto vals = values->data();
            auto x    = points->data()[0];
            auto y    = points->data()[1];
            auto z    = points->data()[2];

            auto i0 = elements->data()[0];
            auto i1 = elements->data()[1];
            auto i2 = elements->data()[2];
            auto i3 = elements->data()[3];

            const ptrdiff_t nselements = elements->extent(1);
            const ptrdiff_t nspoints   = points->extent(1);
            const int       nxe        = elements->extent(0);
            (void)nspoints;

            SMESH_ASSERT(nxe == elem_num_nodes(element_type));

            auto ed  = elements->data();
            auto pd  = points->data();
            auto ivd = is_valid->data();

            // Per slave-node mortar quantities (indexed by surface point id).
            //   weighted_normals: 3 interleaved components per node [node*3 + d] (caller normalizes).
            //   weighted_gap:     dual-basis gap used by the weak contact equation.
            //   weighted_distance: primal-basis physical gap used for exported distance/directors.
            auto wnorm        = weighted_normals->data();
            auto wgap         = weighted_gap->data();
            auto wdist        = weighted_distance->data();
            auto wdist_weight = distance_weight->data();

            // blas->zeros(mass_vector->size(), mass_vector->data());
            auto m = mass_vector->data();
            {
                ptrdiff_t n = mass_vector->size();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    m[i] = real_t(0);
                }
            }

            if (element_type == smesh::QUADSHELL4) {
                BiorthogonalQuad4Weights weights;

#ifdef VIZ_DEBUG

                {
                    sviz::Message msg("assemble_mortar_matrices (surface)");
                    msg.quad_mesh_soa(sviz::view(x, nspoints),
                                      sviz::view(y, nspoints),
                                      sviz::view(z, nspoints),
                                      sviz::view(i0, nselements),
                                      sviz::view(i1, nselements),
                                      sviz::view(i2, nselements),
                                      sviz::view(i3, nselements));

                    try {
                        sviz::Client().send(msg);
                    } catch (const std::exception& e) {
                        printf("Error sending message to sviz: %s\n", e.what());
                    }
                }
                sviz::Message msg("assemble_mortar_matrices (gaps)");
#endif  // VIZ_DEBUG

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < nselements; i++) {
                    const idx_t     av[4]       = {i0[i], i1[i], i2[i], i3[i]};
                    const ptrdiff_t ncandidates = ptr[i + 1] - ptr[i];
                    const auto*     candidates  = &idx[ptr[i]];

                    const real_t ax[4] = {x[av[0]], x[av[1]], x[av[2]], x[av[3]]};
                    const real_t ay[4] = {y[av[0]], y[av[1]], y[av[2]], y[av[3]]};
                    const real_t az[4] = {z[av[0]], z[av[1]], z[av[2]], z[av[3]]};

                    const real_t asx = real_t(0.5) * (ax[1] + ax[2] - ax[0] - ax[3]);
                    const real_t asy = real_t(0.5) * (ay[1] + ay[2] - ay[0] - ay[3]);
                    const real_t asz = real_t(0.5) * (az[1] + az[2] - az[0] - az[3]);
                    const real_t atx = real_t(0.5) * (ax[2] + ax[3] - ax[0] - ax[1]);
                    const real_t aty = real_t(0.5) * (ay[2] + ay[3] - ay[0] - ay[1]);
                    const real_t atz = real_t(0.5) * (az[2] + az[3] - az[0] - az[1]);

                    real_t anormal[3]  = {asy * atz - asz * aty, asz * atx - asx * atz, asx * aty - asy * atx};
                    real_t anormal_len = std::sqrt(anormal[0] * anormal[0] + anormal[1] * anormal[1] + anormal[2] * anormal[2]);
                    anormal[0] /= anormal_len;
                    anormal[1] /= anormal_len;
                    anormal[2] /= anormal_len;

                    // Pick the most stable coordinate plane for building the first unit tangent.
                    // This avoids subtractive cancellation when the normal is close to an axis.
                    const bool   use_xy_axis = std::abs(anormal[0]) > std::abs(anormal[2]);
                    const real_t tangent0_inv_len =
                            use_xy_axis ? real_t(1) / std::sqrt(anormal[0] * anormal[0] + anormal[1] * anormal[1])
                                        : real_t(1) / std::sqrt(anormal[1] * anormal[1] + anormal[2] * anormal[2]);

                    // tangent0 is perpendicular to anormal and normalized by tangent0_inv_len.
                    const real_t tangent0[3] = {use_xy_axis ? -anormal[1] * tangent0_inv_len : 0,
                                                use_xy_axis ? anormal[0] * tangent0_inv_len : -anormal[2] * tangent0_inv_len,
                                                use_xy_axis ? 0 : anormal[1] * tangent0_inv_len};

                    // tangent1 completes the orthonormal basis of the plane: tangent1 = anormal x tangent0.
                    const real_t tangent1[3] = {anormal[1] * tangent0[2] - anormal[2] * tangent0[1],
                                                anormal[2] * tangent0[0] - anormal[0] * tangent0[2],
                                                anormal[0] * tangent0[1] - anormal[1] * tangent0[0]};

                    // Project each 3D point onto the tangent basis. The two dot products are the local 2D plane coordinates.
                    const auto project_to_normal_plane = [&tangent0, &tangent1](const real_t* const px,
                                                                                const real_t* const py,
                                                                                const real_t* const pz,
                                                                                real_t* const       out_x,
                                                                                real_t* const       out_y) {
                        for (int k = 0; k < 4; k++) {
                            // Dot point k with each tangent axis to obtain its projected x/y coordinates.
                            out_x[k] = px[k] * tangent0[0] + py[k] * tangent0[1] + pz[k] * tangent0[2];
                            out_y[k] = px[k] * tangent1[0] + py[k] * tangent1[1] + pz[k] * tangent1[2];
                        }
                    };

                    // Project the active quad once; candidate quads reuse the same plane basis below.
                    real_t a_projected_x[4];
                    real_t a_projected_y[4];
                    project_to_normal_plane(ax, ay, az, a_projected_x, a_projected_y);

                    for (ptrdiff_t j = 0; j < ncandidates; j++) {
                        const idx_t candidate = candidates[j];
                        const idx_t bv[4]     = {i0[candidate], i1[candidate], i2[candidate], i3[candidate]};

                        const real_t bx[4] = {x[bv[0]], x[bv[1]], x[bv[2]], x[bv[3]]};
                        const real_t by[4] = {y[bv[0]], y[bv[1]], y[bv[2]], y[bv[3]]};
                        const real_t bz[4] = {z[bv[0]], z[bv[1]], z[bv[2]], z[bv[3]]};

                        const real_t bsx = real_t(0.5) * (bx[1] + bx[2] - bx[0] - bx[3]);
                        const real_t bsy = real_t(0.5) * (by[1] + by[2] - by[0] - by[3]);
                        const real_t bsz = real_t(0.5) * (bz[1] + bz[2] - bz[0] - bz[3]);
                        const real_t btx = real_t(0.5) * (bx[2] + bx[3] - bx[0] - bx[1]);
                        const real_t bty = real_t(0.5) * (by[2] + by[3] - by[0] - by[1]);
                        const real_t btz = real_t(0.5) * (bz[2] + bz[3] - bz[0] - bz[1]);

                        real_t       bnormal[3] = {bsy * btz - bsz * bty, bsz * btx - bsx * btz, bsx * bty - bsy * btx};
                        const real_t bnormal_len =
                                std::sqrt(bnormal[0] * bnormal[0] + bnormal[1] * bnormal[1] + bnormal[2] * bnormal[2]);
                        bnormal[0] /= bnormal_len;
                        bnormal[1] /= bnormal_len;
                        bnormal[2] /= bnormal_len;

                        const real_t face_dot = anormal[0] * bnormal[0] + anormal[1] * bnormal[1] + anormal[2] * bnormal[2];
                        if (face_dot > real_t(-0.5)) {
                            ivd[ptr[i] + j] = 0;
                            continue;
                        }

                        const real_t outer_normal[3] = {anormal[0], anormal[1], anormal[2]};

                        real_t b_projected_x[4];
                        real_t b_projected_y[4];

                        // Project the candidate quad onto the active quad plane.
                        project_to_normal_plane(bx, by, bz, b_projected_x, b_projected_y);

                        real_t poly_x[MORTAR_MAX_POLY] = {a_projected_x[0], a_projected_x[1], a_projected_x[2], a_projected_x[3]};
                        real_t poly_y[MORTAR_MAX_POLY] = {a_projected_y[0], a_projected_y[1], a_projected_y[2], a_projected_y[3]};
                        int    poly_n                  = 4;

                        const real_t b_area2 = (b_projected_x[0] * b_projected_y[1] - b_projected_y[0] * b_projected_x[1]) +
                                               (b_projected_x[1] * b_projected_y[2] - b_projected_y[1] * b_projected_x[2]) +
                                               (b_projected_x[2] * b_projected_y[3] - b_projected_y[2] * b_projected_x[3]) +
                                               (b_projected_x[3] * b_projected_y[0] - b_projected_y[3] * b_projected_x[0]);
                        const real_t clip_sign = b_area2 >= 0 ? real_t(1) : real_t(-1);

                        for (int edge = 0; edge < 4 && poly_n > 0; edge++) {
                            const int    next_edge = (edge + 1) & 3;
                            const real_t ex0       = b_projected_x[edge];
                            const real_t ey0       = b_projected_y[edge];
                            const real_t edx       = b_projected_x[next_edge] - ex0;
                            const real_t edy       = b_projected_y[next_edge] - ey0;

                            real_t clipped_x[MORTAR_MAX_POLY];
                            real_t clipped_y[MORTAR_MAX_POLY];
                            int    clipped_n = 0;

                            real_t sx     = poly_x[poly_n - 1];
                            real_t sy     = poly_y[poly_n - 1];
                            real_t s_dist = clip_sign * (edx * (sy - ey0) - edy * (sx - ex0));
                            bool   s_in   = s_dist >= 0;

                            for (int p = 0; p < poly_n; p++) {
                                const real_t ex     = poly_x[p];
                                const real_t ey     = poly_y[p];
                                const real_t e_dist = clip_sign * (edx * (ey - ey0) - edy * (ex - ex0));
                                const bool   e_in   = e_dist >= 0;

                                // Guard every push: degenerate/diverged geometry can otherwise inflate the
                                // Sutherland-Hodgman output past the buffer capacity.
                                if (e_in != s_in && clipped_n < MORTAR_MAX_POLY) {
                                    const real_t alpha   = s_dist / (s_dist - e_dist);
                                    clipped_x[clipped_n] = sx + alpha * (ex - sx);
                                    clipped_y[clipped_n] = sy + alpha * (ey - sy);
                                    clipped_n++;
                                }

                                if (e_in && clipped_n < MORTAR_MAX_POLY) {
                                    clipped_x[clipped_n] = ex;
                                    clipped_y[clipped_n] = ey;
                                    clipped_n++;
                                }

                                sx     = ex;
                                sy     = ey;
                                s_dist = e_dist;
                                s_in   = e_in;
                            }

                            poly_n = clipped_n;
                            for (int p = 0; p < poly_n; p++) {
                                poly_x[p] = clipped_x[p];
                                poly_y[p] = clipped_y[p];
                            }
                        }

                        if (poly_n < 3) {
                            ivd[ptr[i] + j] = 0;
                            continue;
                        }

                        ivd[ptr[i] + j] = 1;

                        real_t sa[MORTAR_MAX_QP];
                        real_t ta[MORTAR_MAX_QP];
                        real_t sb[MORTAR_MAX_QP];
                        real_t tb[MORTAR_MAX_QP];
                        real_t wq[MORTAR_MAX_QP];
                        int    nqp = 0;

                        const int ntris = poly_n - 2;

                        for (int tri = 0; tri < ntris; tri++) {
                            const real_t v0x = poly_x[0];
                            const real_t v0y = poly_y[0];
                            const real_t v1x = poly_x[tri + 1];
                            const real_t v1y = poly_y[tri + 1];
                            const real_t v2x = poly_x[tri + 2];
                            const real_t v2y = poly_y[tri + 2];

                            const real_t e1x        = v1x - v0x;
                            const real_t e1y        = v1y - v0y;
                            const real_t e2x        = v2x - v0x;
                            const real_t e2y        = v2y - v0y;
                            const real_t tri_area2  = e1x * e2y - e1y * e2x;
                            const real_t area_scale = std::abs(tri_area2);

                            if (area_scale <= real_t(0)) {
                                continue;
                            }

#pragma omp simd
                            for (int q = 0; q < MORTAR_TRI4_NQP; q++) {
                                const real_t l1   = mortar_tri4_l1[q];
                                const real_t l2   = mortar_tri4_l2[q];
                                const real_t qp_x = v0x + l1 * e1x + l2 * e2x;
                                const real_t qp_y = v0y + l1 * e1y + l2 * e2y;
                                // area_scale = |e1 x e2| = 2 * triangle_area; the 0.5 maps it to the physical area.
                                const real_t w = mortar_tri4_qw[q] * area_scale * real_t(0.5);

                                real_t qs = 0;
                                real_t qt = 0;
                                invert_quad4_plane(a_projected_x, a_projected_y, qp_x, qp_y, &qs, &qt);

                                real_t rs = 0;
                                real_t rt = 0;
                                invert_quad4_plane(b_projected_x, b_projected_y, qp_x, qp_y, &rs, &rt);

                                const int idx = nqp + q;
                                sa[idx]       = qs;
                                ta[idx]       = qt;
                                sb[idx]       = rs;
                                tb[idx]       = rt;
                                wq[idx]       = w;
                            }

                            nqp += MORTAR_TRI4_NQP;
                        }

                        // Accumulate the local biorthogonal mortar block M (slave-master) over the intersection quadrature.
                        //   M[a][b] = sum_q wq * psi_a(slave) * phi_b(master)
                        // where psi_a = sum_c W[a][c] * phi_c(slave) is the dual (biorthogonal) slave basis.
                        // The slave-slave block D is diagonal by construction of the dual basis and is not stored.
                        //
                        // Also accumulate the per slave-node mortar quantities (scattered with atomics since
                        // neighbouring slave elements share nodes):
                        //   weighted normal: n_bar_a = sum_q wq * phi_a(slave) * n   (n = slave element normal)
                        //   weighted gap:    g_bar_a = sum_q wq * psi_a(slave) * (x_master - x_slave) . n
                        real_t m_block[16] = {0};

                        bool hit = false;
                        for (int q = 0; q < nqp; q++) {
                            real_t phi_a[4];
                            real_t phi_b[4];
                            quad4_shape(sa[q], ta[q], phi_a);
                            quad4_shape(sb[q], tb[q], phi_b);

                            real_t psi_a[4];
                            for (int a = 0; a < 4; a++) {
                                const real_t* const w_row = &weights.values[a * 4];
                                psi_a[a] = w_row[0] * phi_a[0] + w_row[1] * phi_a[1] + w_row[2] * phi_a[2] + w_row[3] * phi_a[3];
                            }

                            const real_t w = wq[q];
                            for (int a = 0; a < 4; a++) {
                                const real_t w_psi = w * psi_a[a];
                                for (int b = 0; b < 4; b++) {
                                    m_block[a * 4 + b] += w_psi * phi_b[b];
                                }
                            }

                            // Physical slave and master points at this quadrature point.
                            real_t xs[3] = {0, 0, 0};
                            real_t xm[3] = {0, 0, 0};
                            for (int c = 0; c < 4; c++) {
                                xs[0] += phi_a[c] * ax[c];
                                xs[1] += phi_a[c] * ay[c];
                                xs[2] += phi_a[c] * az[c];
                                xm[0] += phi_b[c] * bx[c];
                                xm[1] += phi_b[c] * by[c];
                                xm[2] += phi_b[c] * bz[c];
                            }

                            for (int a = 0; a < 4; a++) {
                                const ptrdiff_t node  = av[a];
                                const real_t    w_phi = w * phi_a[a];

#pragma omp atomic update
                                m[node] += w_phi;
                            }

                            const real_t gap = (xm[0] - xs[0]) * outer_normal[0] + (xm[1] - xs[1]) * outer_normal[1] +
                                               (xm[2] - xs[2]) * outer_normal[2];

                            if (gap * gap > max_gap * max_gap) {
                                continue;
                            }

                            hit = true;

#ifdef VIZ_DEBUG
                            // sviz::Message msg("normal");
                            real_t dx = outer_normal[0] * gap;
                            real_t dy = outer_normal[1] * gap;
                            real_t dz = outer_normal[2] * gap;

                            msg.set_vector_scale(1).quivers_soa(sviz::view(xs, 1),
                                                                sviz::view(xs + 1, 1),
                                                                sviz::view(xs + 2, 1),
                                                                sviz::view(&dx, 1),
                                                                sviz::view(&dy, 1),
                                                                sviz::view(&dz, 1));

                            // sviz::Client().send(msg);

#endif  // VIZ_DEBUG

                            for (int a = 0; a < 4; a++) {
                                const ptrdiff_t node  = av[a];
                                const real_t    w_phi = w * phi_a[a];
                                const real_t    w_psi = w * psi_a[a];

#pragma omp atomic update
                                wnorm[node * 3 + 0] += w_phi * outer_normal[0];
#pragma omp atomic update
                                wnorm[node * 3 + 1] += w_phi * outer_normal[1];
#pragma omp atomic update
                                wnorm[node * 3 + 2] += w_phi * outer_normal[2];
#pragma omp atomic update
                                wgap[node] += w_psi * gap;
#pragma omp atomic update
                                wdist[node] += w_phi * gap;
#pragma omp atomic update
                                wdist_weight[node] += w_phi;
                            }
                        }

                        if (!hit) {
                            ivd[ptr[i] + j] = false;
                            continue;
                        }

                        real_t* const pair_out = &vals[(ptr[i] + j) * MORTAR_PAIR_STRIDE];
                        for (int a = 0; a < 16; a++) {
                            pair_out[a] = m_block[a];
                        }
                    }
                }

#ifdef VIZ_DEBUG
                try {
                    sviz::Client().send(msg);
                } catch (const std::exception& e) {
                    printf("Error sending message to sviz: %s\n", e.what());
                }
#endif  // VIZ_DEBUG

            } else if (element_type == smesh::TRISHELL3) {
                // TODO
            } else {
                SFEM_ERROR("assemble_mortar_matrices not implemented for element type %d\n", element_type);
            }
        }

        void mortar_elemental_matrices_to_crs(const smesh::ElemType                             element_type,
                                              const ptrdiff_t                                   n_nodes,
                                              const SharedBuffer<idx_t*>&                       elements,
                                              const SharedBuffer<ptrdiff_t>&                    pc_ptr,
                                              const SharedBuffer<idx_t>&                        pc_idx,
                                              const SharedBuffer<real_t>&                       values,
                                              const SharedBuffer<mask_t>&                       is_valid,
                                              std::shared_ptr<smesh::CRSGraph<count_t, idx_t>>& crs_graph,
                                              SharedBuffer<real_t>&                             crs_values) {
            if (element_type != smesh::QUADSHELL4 && element_type != smesh::TRISHELL3) {
                SFEM_ERROR("mortar_elemental_matrices_to_crs not implemented for element type %d\n", element_type);
                return;
            }

            const int       nxe        = elem_num_nodes(element_type);
            const int       block      = nxe * nxe;
            const ptrdiff_t n_elements = pc_ptr->size() - 1;

            auto ptr   = pc_ptr->data();
            auto pidx  = pc_idx->data();
            auto ivd   = is_valid->data();
            auto ed    = elements->data();
            auto mvals = values->data();

            // 1) Node-to-(master element) connectivity for slave nodes, built from the valid contact pairs only.
            //    For each valid pair (slave element e, master element pidx[k]) every slave node of e couples to
            //    that master element.
            std::vector<count_t> n2e_ptr(n_nodes + 1, 0);
            for (ptrdiff_t e = 0; e < n_elements; ++e) {
                for (ptrdiff_t k = ptr[e]; k < ptr[e + 1]; ++k) {
                    if (!ivd[k]) continue;
                    for (int a = 0; a < nxe; ++a) {
                        ++n2e_ptr[ed[a][e] + 1];
                    }
                }
            }
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                n2e_ptr[i + 1] += n2e_ptr[i];
            }

            std::vector<idx_t>   n2e_idx(n2e_ptr[n_nodes]);
            std::vector<count_t> book(n_nodes, 0);
            for (ptrdiff_t e = 0; e < n_elements; ++e) {
                for (ptrdiff_t k = ptr[e]; k < ptr[e + 1]; ++k) {
                    if (!ivd[k]) continue;
                    const idx_t master = pidx[k];
                    for (int a = 0; a < nxe; ++a) {
                        const idx_t node                      = ed[a][e];
                        n2e_idx[n2e_ptr[node] + book[node]++] = master;
                    }
                }
            }

            // 2) CRS graph: slave-node row -> unique master-node columns (union of the coupled master elements' nodes).
            //    Gather the row's master nodes into a fixed stack buffer, then sort_and_unique (as in smesh_graph.impl.hpp).
            static constexpr int MORTAR_ROW_BUF = 4096;

            const auto gather_row = [&](const ptrdiff_t node, idx_t* const buf) -> idx_t {
                idx_t nneighs = 0;
                for (count_t e = n2e_ptr[node]; e < n2e_ptr[node + 1]; ++e) {
                    const idx_t master = n2e_idx[e];
                    for (int b = 0; b < nxe; ++b) {
                        assert(nneighs < MORTAR_ROW_BUF);
                        buf[nneighs++] = ed[b][master];
                    }
                }
                return static_cast<idx_t>(smesh::sort_and_unique(buf, static_cast<size_t>(nneighs)));
            };

            auto rowptr = sfem::create_host_buffer<count_t>(n_nodes + 1);
            auto rp     = rowptr->data();
            rp[0]       = 0;

#pragma omp parallel
            {
                idx_t buf[MORTAR_ROW_BUF];
#pragma omp for
                for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                    rp[node + 1] = static_cast<count_t>(gather_row(node, buf));
                }
            }

            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                rp[node + 1] += rp[node];
            }

            const ptrdiff_t nnz    = rp[n_nodes];
            auto            colidx = sfem::create_host_buffer<idx_t>(nnz);
            auto            cidx   = colidx->data();

#pragma omp parallel
            {
                idx_t buf[MORTAR_ROW_BUF];
#pragma omp for
                for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                    const idx_t nneighs = gather_row(node, buf);
                    for (idx_t i = 0; i < nneighs; ++i) {
                        cidx[rp[node] + i] = buf[i];
                    }
                }
            }

            crs_graph  = std::make_shared<smesh::CRSGraph<count_t, idx_t>>(rowptr, colidx);
            crs_values = sfem::create_host_buffer<real_t>(nnz);
            auto cvals = crs_values->data();

#pragma omp parallel for
            for (ptrdiff_t k = 0; k < nnz; ++k) {
                cvals[k] = 0;
            }

            // 3) Scatter-add the local M blocks into the global CRS coupling values.
            //    The master nodes are the same for all slave rows of a pair, so resolve their column
            //    offsets per row with mortar_find_cols (mirrors tet4_local_to_global).
#pragma omp parallel for
            for (ptrdiff_t e = 0; e < n_elements; ++e) {
                for (ptrdiff_t k = ptr[e]; k < ptr[e + 1]; ++k) {
                    if (!ivd[k]) continue;
                    const idx_t         master = pidx[k];
                    const real_t* const m      = &mvals[k * block];

                    idx_t targets[MORTAR_MAX_NXE];
                    for (int b = 0; b < nxe; ++b) {
                        targets[b] = ed[b][master];
                    }

                    idx_t ks[MORTAR_MAX_NXE];
                    for (int a = 0; a < nxe; ++a) {
                        const idx_t   row_node = ed[a][e];
                        const count_t rbegin   = rp[row_node];
                        const idx_t*  row      = &cidx[rbegin];
                        const int     lenrow   = static_cast<int>(rp[row_node + 1] - rbegin);

                        mortar_find_cols(targets, nxe, row, lenrow, ks);

                        real_t* const       rowvalues = &cvals[rbegin];
                        const real_t* const m_row     = &m[a * nxe];

                        for (int b = 0; b < nxe; ++b) {
                            assert(ks[b] >= 0 && ks[b] < lenrow);
#pragma omp atomic update
                            rowvalues[ks[b]] += m_row[b];
                        }
                    }
                }
            }
        }

        void sum_diag(const std::shared_ptr<smesh::CRSGraph<count_t, idx_t>>& graph,
                      const SharedBuffer<real_t>&                             values,
                      const SharedBuffer<real_t>&                             diag) {
            auto rowptr = graph->rowptr()->data();
            auto colidx = graph->colidx()->data();
            auto vals   = values->data();
            auto d      = diag->data();

            const ptrdiff_t n = graph->rowptr()->size() - 1;
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                const count_t lenrow = rowptr[i + 1] - rowptr[i];
                const real_t* values = &vals[rowptr[i]];

                real_t sum = 0;
                for (count_t j = 0; j < lenrow; j++) {
                    sum += values[j];
                }
                d[i] = sum;
            }
        }

        void sum_postprocess_weighted_quantities(const std::shared_ptr<smesh::CRSGraph<count_t, idx_t>>& graph,
                                                 const SharedBuffer<real_t>&                             values,
                                                 const SharedBuffer<real_t>&                             weighted_normals,
                                                 const SharedBuffer<real_t>&                             weighted_gap,
                                                 const SharedBuffer<real_t>&                             diag) {
            auto rowptr = graph->rowptr()->data();
            auto colidx = graph->colidx()->data();
            auto vals   = values->data();
            auto d      = diag->data();
            auto wg     = weighted_gap->data();
            auto wn     = weighted_normals->data();

            const ptrdiff_t n = graph->rowptr()->size() - 1;

            // Normalize the coupling weights row-wise by the mortar diagonal: M -> D^{-1} M.
            // The diagonal d (== D) is left intact so it can be reused as the slave mass.
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                if (d[i] == 0) continue;

                const count_t begin  = rowptr[i];
                const count_t lenrow = rowptr[i + 1] - begin;
                const real_t  inv_d  = real_t(1) / d[i];
                for (count_t j = 0; j < lenrow; j++) {
                    vals[begin + j] *= inv_d;
                }
            }

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                if (d[i] == 0) continue;
                wg[i] /= d[i];
            }

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                // if (d[i] == 0) continue;

                real_t len_v = 0;
                for (int d = 0; d < 3; d++) {
                    len_v += wn[i * 3 + d] * wn[i * 3 + d];
                }

                if (len_v == 0) {
                    continue;
                }

                len_v = sqrt(len_v);
                for (int d = 0; d < 3; d++) {
                    wn[i * 3 + d] /= len_v;
                }
            }
        }

        class ContactMortar final : public Contact {
        public:
            ContactMortar(const std::shared_ptr<FunctionSpace>& space,
                          const std::shared_ptr<smesh::Mesh>&   surface,
                          const real_t                          margin,
                          const real_t                          search_radius_sqr,
                          const ExecutionSpace                  es)
                : space_(space),
                  surface_(surface),
                  margin_(margin),
                  search_radius_sqr_(search_radius_sqr),
                  es_(es),
                  dim_(surface->spatial_dimension()),
                  npoints_(surface->n_nodes()),
                  nselements_(surface->n_elements()),
                  surface_elements_(surface->block(0)->elements()),
                  surface_element_type_(surface->block(0)->element_type()),
                  mass_vector_(sfem::create_buffer<real_t>(surface->n_nodes(), es)),
                  normals_(sfem::create_buffer<real_t>(surface->spatial_dimension(), surface->n_nodes(), es)),
                  distances_(sfem::create_buffer<real_t>(surface->n_nodes(), es)),
                  distances_whole_(sfem::create_buffer<real_t>(space->n_dofs(), es)),
                  directors_(sfem::create_buffer<real_t>(space->n_dofs(), es)),
                  frozen_displacement_(sfem::create_buffer<real_t>(space->n_dofs(), es)) {}

            void assemble_mass_vector(const smesh::ElemType        element_type,
                                      const SharedBuffer<idx_t*>&  elements,
                                      const SharedBuffer<real_t*>& current_points,
                                      const SharedBuffer<real_t>&  mass_vector) {
                auto deformed = std::make_shared<smesh::Mesh>(
                        surface_->comm(), element_type, elements, smesh::astype<geom_t>(current_points));

                auto trace_space = std::make_shared<FunctionSpace>(deformed, 1);
                auto bop         = sfem::Factory::create_op(trace_space, "Mass");
                bop->initialize();

                auto ones = create_host_buffer<real_t>(trace_space->n_dofs());
                sfem::blas<real_t>(EXECUTION_SPACE_HOST)->values(trace_space->n_dofs(), 1, ones->data());
                bop->apply(nullptr, ones->data(), mass_vector->data());
            }

            void recompute(const std::shared_ptr<Buffer<real_t>>& displacement) override {
                SFEM_TRACE_SCOPE("ContactMortar::recompute");

                if (surface_element_type_ != smesh::QUADSHELL4) {
                    SFEM_ERROR("ContactMortar is only implemented for QUADSHELL4 (got %d)\n", surface_element_type_);
                    return;
                }

                auto blas = sfem::blas<real_t>(es_);

                // 1) Current (displaced) surface configuration.
                p1_ = smesh::astype<real_t>(surface_->points());
                displace_points(surface_, displacement, p1_);

                // 2) Broad-phase: candidate master faces per slave face.
                auto         pc_ptr      = create_host_buffer<ptrdiff_t>(nselements_ + 1);
                auto         pc_ptr_data = pc_ptr->data();
                idx_t*       raw_pc_idx  = nullptr;
                const real_t extrusion   = std::sqrt(search_radius_sqr_) + margin_;
                printf("extrusion: %g\n", extrusion);

                const int err = ssdf::potential_contact_quads_bvh<real_t, real_t, idx_t, idx_t>(nselements_,
                                                                                                surface_elements_->data()[0],
                                                                                                surface_elements_->data()[1],
                                                                                                surface_elements_->data()[2],
                                                                                                surface_elements_->data()[3],
                                                                                                npoints_,
                                                                                                p1_->data()[0],
                                                                                                p1_->data()[1],
                                                                                                p1_->data()[2],
                                                                                                extrusion,
                                                                                                pc_ptr_data,
                                                                                                &raw_pc_idx);

                if (err != 0) {
                    SFEM_ERROR("potential_contact_quads_bvh failed (%d)\n", err);
                    return;
                }

                const ptrdiff_t npairs = pc_ptr_data[nselements_];
                printf("npairs: %ld\n", (long)npairs);

                // Adopt the BVH-owned (malloc'd) index array into a managed buffer, then release it.
                auto pc_idx = create_host_buffer<idx_t>(std::max<ptrdiff_t>(npairs, 1));
                if (npairs > 0 && raw_pc_idx) {
                    std::memcpy(pc_idx->data(), raw_pc_idx, static_cast<size_t>(npairs) * sizeof(idx_t));
                }
                std::free(raw_pc_idx);

                // 3) Per-pair mortar M blocks + per-slave-node weighted normals/gap.
                //    The gap is accumulated directly into the persistent distances buffer.
                auto is_valid     = create_host_buffer<mask_t>(std::max<ptrdiff_t>(npairs, 1));
                auto pair_values  = create_host_buffer<real_t>(std::max<ptrdiff_t>(npairs, 1) * MORTAR_PAIR_STRIDE);
                auto wnormals     = create_host_buffer<real_t>((ptrdiff_t)npoints_ * 3);
                auto wdistance    = create_host_buffer<real_t>(npoints_);
                auto wdist_weight = create_host_buffer<real_t>(npoints_);
                distances_        = sfem::create_buffer<real_t>(npoints_, es_);

                blas->values((ptrdiff_t)npoints_ * 3, 0, wnormals->data());
                blas->values(npoints_, 0, wdistance->data());
                blas->values(npoints_, 0, wdist_weight->data());
                blas->values(npoints_, 0, distances_->data());

                {
                    auto iv = is_valid->data();
#pragma omp parallel for
                    for (ptrdiff_t k = 0; k < npairs; ++k) {
                        iv[k] = 1;
                    }
                }

                assemble_mortar_matrices(surface_element_type_,
                                         surface_elements_,
                                         p1_,
                                         pc_ptr,
                                         pc_idx,
                                         pair_values,
                                         wnormals,
                                         distances_,
                                         wdistance,
                                         wdist_weight,
                                         mass_vector_,
                                         is_valid,
                                         std::sqrt(search_radius_sqr_));

                // 4) Assemble the global slave->master coupling (mortar M) into CRS.
                mortar_elemental_matrices_to_crs(surface_element_type_,
                                                 npoints_,
                                                 surface_elements_,
                                                 pc_ptr,
                                                 pc_idx,
                                                 pair_values,
                                                 is_valid,
                                                 graph_,
                                                 values_);

                // assemble_mass_vector(surface_element_type_, surface_elements_, p1_, mass_vector_);

                // 5) Proper dual-mortar diagonal D: integrate the slave shape functions over the FULL slave
                //    element (not the partial overlap). By biorthogonality this is the dual diagonal D_aa, and it
                //    is strictly positive, so it is a safe nodal mass and projection normalizer. Integrating over
                //    the partial overlap instead (e.g. sum_diag of the dual M) can go negative and destabilizes
                //    the contact iteration. A node accumulates the contribution of every contacting slave element
                //    it belongs to.
                //                 mass_vector_ = sfem::create_buffer<real_t>(npoints_, es_);
                // {
                //                     auto       mass = mass_vector_->data();
                //                     const auto ptr  = pc_ptr->data();
                //                     const auto iv   = is_valid->data();
                //                     const auto ed   = surface_elements_->data();
                //                     const auto px   = p1_->data()[0];
                //                     const auto py   = p1_->data()[1];
                //                     const auto pz   = p1_->data()[2];

                //                     // #pragma omp parallel for
                //                     //                     for (ptrdiff_t i = 0; i < npoints_; ++i) {
                //                     //                         mass[i] = 0;
                //                     //                     }

                // #pragma omp parallel for
                //                     for (ptrdiff_t e = 0; e < nselements_; ++e) {
                //                         bool in_contact = false;
                //                         for (ptrdiff_t k = ptr[e]; k < ptr[e + 1]; ++k) {
                //                             if (iv[k]) {
                //                                 in_contact = true;
                //                                 break;
                //                             }
                //                         }
                //                         if (!in_contact) {
                //                             continue;
                //                         }

                //                         const idx_t  v[4] = {ed[0][e], ed[1][e], ed[2][e], ed[3][e]};
                //                         const real_t X[4] = {px[v[0]], px[v[1]], px[v[2]], px[v[3]]};
                //                         const real_t Y[4] = {py[v[0]], py[v[1]], py[v[2]], py[v[3]]};
                //                         const real_t Z[4] = {pz[v[0]], pz[v[1]], pz[v[2]], pz[v[3]]};

                //                         real_t area[4];
                //                         quad4_nodal_areas(X, Y, Z, area);

                //                         for (int a = 0; a < 4; ++a) {
                // #pragma omp atomic update
                //                             mass[v[a]] += area[a];
                //                         }
                //                     }
                //                 }

                // 6) Normalize: values -> D^{-1} M, gap -> D^{-1} gap, normals -> unit.
                sum_postprocess_weighted_quantities(graph_, values_, wnormals, distances_, mass_vector_);

                // 7) Convert interleaved weighted normals to SoA and fill the per-dof output fields.
                auto       nrm                 = normals_->data();
                const auto wn                  = wnormals->data();
                const auto d                   = mass_vector_->data();
                const auto gap                 = distances_->data();
                const auto physical_gap        = wdistance->data();
                const auto physical_gap_weight = wdist_weight->data();
                auto       dw                  = distances_whole_->data();
                auto       dir                 = directors_->data();
                const auto nm                  = surface_->node_mapping()->data();

                blas->values(space_->n_dofs(), 0, distances_whole_->data());
                blas->values(space_->n_dofs(), 0, directors_->data());

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < npoints_; ++i) {
                    for (int c = 0; c < dim_; ++c) {
                        nrm[c][i] = wn[i * 3 + c];
                    }

                    if (d[i] == 0) {
                        continue;
                    }

                    const ptrdiff_t dof       = (ptrdiff_t)nm[i] * dim_;
                    const real_t director_gap = physical_gap_weight[i] != 0 ? physical_gap[i] / physical_gap_weight[i] : gap[i];
                    dw[dof]                   = director_gap;
                    for (int c = 0; c < dim_; ++c) {
                        dir[dof + c] = director_gap * nrm[c][i];
                    }
                }

                blas->copy(space_->n_dofs(), displacement->data(), frozen_displacement_->data());
            }

            const std::shared_ptr<smesh::CRSGraph<count_t, idx_t>>& graph() const override { return graph_; }
            smesh::SharedBuffer<real_t>&                            values() override { return values_; }
            smesh::SharedBuffer<real_t>&                            mass_vector() override { return mass_vector_; }
            smesh::SharedBuffer<real_t*>&                           normals() override { return normals_; }
            smesh::SharedBuffer<real_t>&                            distances() override { return distances_; }
            smesh::SharedBuffer<real_t>&       frozen_displacement() override { return frozen_displacement_; }
            const smesh::SharedBuffer<real_t>& distances_whole() const override { return distances_whole_; }
            const smesh::SharedBuffer<real_t>& directors() const override { return directors_; }

        private:
            std::shared_ptr<FunctionSpace> space_;
            std::shared_ptr<smesh::Mesh>   surface_;
            real_t                         margin_;
            real_t                         search_radius_sqr_;
            ExecutionSpace                 es_;
            int                            dim_;
            ptrdiff_t                      npoints_;
            ptrdiff_t                      nselements_;
            smesh::SharedBuffer<idx_t*>    surface_elements_;
            smesh::ElemType                surface_element_type_;

            smesh::SharedBuffer<real_t*>                     p1_;
            std::shared_ptr<smesh::CRSGraph<count_t, idx_t>> graph_;
            smesh::SharedBuffer<real_t>                      values_;
            smesh::SharedBuffer<real_t>                      mass_vector_;
            smesh::SharedBuffer<real_t*>                     normals_;
            smesh::SharedBuffer<real_t>                      distances_;
            smesh::SharedBuffer<real_t>                      distances_whole_;
            smesh::SharedBuffer<real_t>                      directors_;
            smesh::SharedBuffer<real_t>                      frozen_displacement_;
        };

    }  // namespace

    // Select the contact strategy at runtime. SFEM_CONTACT = "nts" (default) | "mortar".
    std::shared_ptr<Contact> create_contact(const std::shared_ptr<FunctionSpace>& space,
                                            const std::shared_ptr<smesh::Mesh>&   surface,
                                            const real_t                          margin,
                                            const real_t                          search_radius_sqr,
                                            const ExecutionSpace                  es) {
        const char* const sel    = std::getenv("SFEM_CONTACT");
        const std::string method = sel ? sel : "mortar";

        if (method == "nts") {
            printf("[Contact] strategy: nts (SFEM_CONTACT=nts)\n");
            return std::make_shared<ContactNodeToSurface>(space, surface, margin, search_radius_sqr, es);
        } else {
            printf("[Contact] strategy: mortar (SFEM_CONTACT=mortar)\n");
            return std::make_shared<ContactMortar>(space, surface, margin, search_radius_sqr, es);
        }
    }

#ifdef SFEM_ENABLE_YAML
    std::shared_ptr<Contact> create_contact(const std::shared_ptr<FunctionSpace>& space,
                                            const std::shared_ptr<smesh::Mesh>&   surface,
                                            const ryml::ConstNodeRef&             node,
                                            ExecutionSpace                        es) {
        real_t margin        = 0;
        real_t search_radius = 1e-4;

        if (node["margin"].readable()) {
            node["margin"] >> margin;
        }
        if (node["search_radius"].readable()) {
            node["search_radius"] >> search_radius;
        }

        return create_contact(space, surface, margin, search_radius * search_radius, es);
    }
#endif

    using domain_t = smesh::u32;

    class MultiBodyContact : public Contact {
    public:
        MultiBodyContact(const std::shared_ptr<FunctionSpace>&      space,
                         const std::shared_ptr<smesh::Mesh>&        surface,
                         std::vector<SharedBuffer<domain_t>>        tags,
                         std::vector<std::pair<domain_t, domain_t>> pairings,
                         real_t                                     margin,
                         real_t                                     search_radius_sqr,
                         ExecutionSpace                             es);

        //  TODO:
        // Identify collision pairs between surface elements with the specified tag pairings
        // Then assemble using the mortar method and the functions defined in the file just the same way as the ContactMortar
        // class Info: tags are associated with the surface mesh, one tag per element.
        // If elements have the same tag (i.e., they are part of the same domain) do not consider them as collision pairs.
        // Implement also the create_domain_tags function as descibed in the stub.
    };

    SharedBuffer<domain_t> create_domain_tags(const std::shared_ptr<smesh::Mesh>& surface) {
        // TODO: Implement this. Identify surfaces of unconnected bodies and assign a unique tag to each surface element
        // use n2e to traverse the mesh and assign a unique tag to eavery element group. Initialize tags with 0 (not considered)
        // And start tagging from 1. Use breadth to search untagged elements. Faces connected to the same node should have the
        // same tag. Go through the n2e graph for each node (this should guarantee that all elements are tagged). If nodes have no
        // incidente elements, ignore them.
        return SharedBuffer<domain_t>();
    }

    std::shared_ptr<Contact> create_mulitbody_contact(const std::shared_ptr<FunctionSpace>& space,
                                                      const std::shared_ptr<smesh::Mesh>&   surface,
                                                      real_t                                margin,
                                                      real_t                                search_radius_sqr,
                                                      ExecutionSpace                        es) {
        // TODO: Implement this
        return nullptr;
    }

}  // namespace sfem
