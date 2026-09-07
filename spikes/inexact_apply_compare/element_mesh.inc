// The element meshes shared by the split benchmark and the warp sweep.
//
// One definition, because the two programs must agree on the mesh exactly:
// a difference in the TET10 edge order alone would scramble the interpolated
// field and make every projection measurement meaningless.
//
// Select with -DELEMENT_TET4, -DELEMENT_HEX8 or -DELEMENT_TET10.
#pragma once
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

#if defined(ELEMENT_HEX8)
#define NXE 8
#define ELEMENT_NAME "HEX8"
#define SIZES {8, 16, 24, 32, 40}
#elif defined(ELEMENT_TET10)
#define NXE 10
#define ELEMENT_NAME "TET10"
#define SIZES {4, 8, 12, 16, 20}
#else
#define NXE 4
#define ELEMENT_NAME "TET4"
#define SIZES {8, 16, 24, 32, 40}
#endif

struct Mesh {
    ptrdiff_t nelements = 0, nnodes = 0;
    std::vector<std::vector<idx_t>> ev;
    std::vector<idx_t *> evp;
    std::vector<std::vector<geom_t>> adj;
    std::vector<geom_t> det;
    std::vector<double> px, py, pz;  // node positions, for seeding the fields
};

#if defined(ELEMENT_HEX8)
static Mesh build(int n) {
    Mesh m;
    const int nn = n + 1;
    const double h = 1.0 / n;
    m.nnodes = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n;
    m.ev.assign(NXE, std::vector<idx_t>(m.nelements));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i, ++e) {
        m.ev[0][e]=nid(i,j,k);     m.ev[1][e]=nid(i+1,j,k);     m.ev[2][e]=nid(i+1,j+1,k); m.ev[3][e]=nid(i,j+1,k);
        m.ev[4][e]=nid(i,j,k+1);   m.ev[5][e]=nid(i+1,j,k+1);   m.ev[6][e]=nid(i+1,j+1,k+1); m.ev[7][e]=nid(i,j+1,k+1);
    }
    // A uniform grid of cubes: the map is affine, so the adjugate is constant.
    m.adj.assign(9, std::vector<geom_t>(m.nelements, 0));
    m.det.assign(m.nelements, (geom_t)(h*h*h));
    for (ptrdiff_t i = 0; i < m.nelements; ++i) { m.adj[0][i]=(geom_t)(h*h); m.adj[4][i]=(geom_t)(h*h); m.adj[8][i]=(geom_t)(h*h); }
    m.px.resize(m.nnodes); m.py.resize(m.nnodes); m.pz.resize(m.nnodes);
    for (int k = 0; k < nn; ++k) for (int j = 0; j < nn; ++j) for (int i = 0; i < nn; ++i) {
        const ptrdiff_t v = nid(i,j,k);
        m.px[v] = i*h; m.py[v] = j*h; m.pz[v] = k*h;
    }
    for (auto &r : m.ev) m.evp.push_back(r.data());
    return m;
}
#else
// TET4 and TET10 share the Freudenthal split of a cube grid.  TET10 adds the
// six edge midpoints per tetrahedron, in SFEM's edge order
// (0,1) (1,2) (0,2) (0,3) (1,3) (2,3).  The edges are straight, so the geometry
// is still affine and the adjugate is still the tetrahedron's -- what makes
// TET10 a genuine test of the projection is that its reference gradients vary
// over the element, not that its geometry is curved.
static Mesh build(int n) {
    Mesh m;
    const int nn = n + 1;
    const double h = 1.0 / n;
    const ptrdiff_t nvert = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n * 6;
    m.ev.assign(NXE, std::vector<idx_t>(m.nelements));
    std::vector<std::vector<geom_t>> pts(3, std::vector<geom_t>(nvert));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    for (int k = 0; k < nn; ++k) for (int j = 0; j < nn; ++j) for (int i = 0; i < nn; ++i) {
        pts[0][nid(i,j,k)] = (geom_t)(i * h); pts[1][nid(i,j,k)] = (geom_t)(j * h); pts[2][nid(i,j,k)] = (geom_t)(k * h);
    }
    static const int tets[6][4] = {{0,1,3,7},{0,1,5,7},{0,4,5,7},{0,4,6,7},{0,2,6,7},{0,2,3,7}};
    m.adj.assign(9, std::vector<geom_t>(m.nelements, 0));
    m.det.assign(m.nelements, 0);
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
        for (int t = 0; t < 6; ++t, ++e) {
            idx_t v[4];
            for (int c = 0; c < 4; ++c) {
                const int b = tets[t][c];
                v[c] = nid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
            }
            double J[9];
            for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
                J[d*3+c] = (double)pts[d][v[c+1]] - (double)pts[d][v[0]];
            double D = J[0]*(J[4]*J[8]-J[5]*J[7]) - J[1]*(J[3]*J[8]-J[5]*J[6]) + J[2]*(J[3]*J[7]-J[4]*J[6]);
            if (D < 0) { std::swap(v[1], v[2]);
                for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
                    J[d*3+c] = (double)pts[d][v[c+1]] - (double)pts[d][v[0]];
                D = -D; }
            for (int c = 0; c < 4; ++c) m.ev[c][e] = v[c];
            m.adj[0][e]=(geom_t)(J[4]*J[8]-J[5]*J[7]); m.adj[1][e]=(geom_t)(J[2]*J[7]-J[1]*J[8]); m.adj[2][e]=(geom_t)(J[1]*J[5]-J[2]*J[4]);
            m.adj[3][e]=(geom_t)(J[5]*J[6]-J[3]*J[8]); m.adj[4][e]=(geom_t)(J[0]*J[8]-J[2]*J[6]); m.adj[5][e]=(geom_t)(J[2]*J[3]-J[0]*J[5]);
            m.adj[6][e]=(geom_t)(J[3]*J[7]-J[4]*J[6]); m.adj[7][e]=(geom_t)(J[1]*J[6]-J[0]*J[7]); m.adj[8][e]=(geom_t)(J[0]*J[4]-J[1]*J[3]);
            m.det[e] = (geom_t)D;
        }
    m.px.assign(pts[0].begin(), pts[0].end());
    m.py.assign(pts[1].begin(), pts[1].end());
    m.pz.assign(pts[2].begin(), pts[2].end());
#if NXE == 10
    // One node per distinct edge, keyed on the sorted vertex pair.
    static const int edges[6][2] = {{0,1},{1,2},{0,2},{0,3},{1,3},{2,3}};
    std::unordered_map<uint64_t, idx_t> midpoint;
    midpoint.reserve((size_t)m.nelements * 3);
    idx_t next = (idx_t)nvert;
    for (ptrdiff_t i = 0; i < m.nelements; ++i)
        for (int t = 0; t < 6; ++t) {
            idx_t a = m.ev[edges[t][0]][i], b = m.ev[edges[t][1]][i];
            if (a > b) std::swap(a, b);
            const uint64_t key = ((uint64_t)a << 32) | (uint64_t)b;
            auto it = midpoint.find(key);
            if (it == midpoint.end()) {
                it = midpoint.emplace(key, next++).first;
                m.px.push_back(0.5 * (m.px[a] + m.px[b]));
                m.py.push_back(0.5 * (m.py[a] + m.py[b]));
                m.pz.push_back(0.5 * (m.pz[a] + m.pz[b]));
            }
            m.ev[4 + t][i] = it->second;
        }
    m.nnodes = (ptrdiff_t)next;
#else
    m.nnodes = nvert;
#endif
    for (auto &r : m.ev) m.evp.push_back(r.data());
    return m;
}
#endif

