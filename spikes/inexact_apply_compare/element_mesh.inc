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
#include <numeric>
#include <cstring>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Mesh ordering.  The default is `morton3` because that is what SFEM runs:
// `smesh::SFC::create_from_env()` defaults to it and `drivers/bench/*` reorder
// with it before they measure.  A benchmark on a differently ordered mesh is
// measuring a configuration the library does not use.
//
//   -DMESH_ORDER="morton3"    the default, and SFEM's
//   -DMESH_ORDER="hilbert3"   smesh's other curve
//   -DMESH_ORDER="lex"        as built: nodes (k*nn+j)*nn+i, elements likewise
//   -DMESH_ORDER="random3"    `rand()` as the sort key.  Not a configuration
//                             anything produces -- it is here only to confirm
//                             that these measurements are sensitive to the
//                             ordering at all, and no speed-up should be
//                             quoted against it.
//
// The curves come from smesh's own encoders, so this measures the library's
// ordering rather than a private reimplementation of one.
#ifndef MESH_ORDER
#define MESH_ORDER "morton3"
#endif
#include "smesh_sfc.impl.hpp"

#if defined(ELEMENT_PROTEUS_HEX27)
#define NXE 27
#define ELEMENT_NAME "PROTEUS_HEX27"
#ifndef SIZES
#define SIZES {4, 8, 12, 16, 20}
#endif
#elif defined(ELEMENT_HEX8)
#define NXE 8
#define ELEMENT_NAME "HEX8"
#ifndef SIZES
#define SIZES {8, 16, 24, 32, 40}
#endif
#elif defined(ELEMENT_TET10)
#define NXE 10
#define ELEMENT_NAME "TET10"
#ifndef SIZES
#define SIZES {4, 8, 12, 16, 20}
#endif
#else
#define NXE 4
#define ELEMENT_NAME "TET4"
#ifndef SIZES
#define SIZES {8, 16, 24, 32, 40}
#endif
#endif

struct Mesh {
    ptrdiff_t nelements = 0, nnodes = 0;
    std::vector<std::vector<idx_t>> ev;
    std::vector<idx_t *> evp;
    // The same pointers in the order the templated micro-kernel is written
    // against, which is not always the mesh's own.  See
    // `finalise_element_pointers`.
    std::vector<idx_t *> kevp;
    std::vector<std::vector<geom_t>> adj;
    std::vector<geom_t> det;
    std::vector<double> px, py, pz;  // node positions, for seeding the fields
};

// The element pointers, in the two orders a benchmark needs.
//
// `evp` is the mesh's own, and is what the `extern "C"` symbols take.  `kevp`
// is the order the templated `..._impl` bodies are written against, and for a
// mesh-ordered tensor-product element the two differ: HEX8 publishes no kernel
// of its own and forwards to PROTEUS_HEX8, whose basis is lexicographic.  The
// generated forwarder does exactly this permutation on the way in; a benchmark
// that instantiates the template directly bypasses that forwarder and has to
// do it itself.
//
// Getting this wrong is silent.  It compiles, it runs, and it reports a
// projected apply whose relative deviation from the exact one is 3.6e-01
// instead of 2.3e-04 -- a plausible-looking number for an approximation, three
// orders of magnitude from the truth, and not visibly wrong without the
// recorded value to hand.
//
// `KERNEL_SHAPE_ORDER` comes from `plans/layout.cartesian_twin` by way of
// `kernel_element.sh`.  Absent, the element owns its kernel and the two orders
// are the same.
static void finalise_element_pointers(Mesh &m) {
    m.evp.clear();
    for (auto &row : m.ev) m.evp.push_back(row.data());
    m.kevp = m.evp;
#ifdef KERNEL_SHAPE_ORDER
    static const int kernel_shape_order[] = {KERNEL_SHAPE_ORDER};
    static_assert(sizeof(kernel_shape_order) / sizeof(kernel_shape_order[0]) == NXE,
                  "KERNEL_SHAPE_ORDER must name every node of the element");
    for (int node = 0; node < NXE; ++node)
        m.kevp[node] = m.evp[kernel_shape_order[node]];
#endif
}

// SFEM's algorithm, from `smesh::SFC::reorder`: sort the elements by the curve
// through their barycentres, then renumber the nodes in the order the reordered
// elements first touch them.  Numbering the nodes by their own curve would be
// the obvious alternative and is not what the library does -- first-touch order
// is what makes the gather walk forwards.
static void reorder_mesh(Mesh &m, const char *ordering) {
    if (!ordering || !std::strcmp(ordering, "lex")) return;

    std::vector<geom_t> bx(m.nelements), by(m.nelements), bz(m.nelements);
    for (ptrdiff_t e = 0; e < m.nelements; ++e) {
        double sx = 0, sy = 0, sz = 0;
        for (int c = 0; c < NXE; ++c) {
            const idx_t v = m.ev[c][e];
            sx += m.px[v]; sy += m.py[v]; sz += m.pz[v];
        }
        bx[e] = (geom_t)(sx / NXE); by[e] = (geom_t)(sy / NXE); bz[e] = (geom_t)(sz / NXE);
    }

    std::vector<smesh::u32> code(m.nelements);
    int status = -1;
    if (!std::strcmp(ordering, "morton3"))
        status = smesh::encode_morton3<geom_t>(m.nelements, bx.data(), by.data(), bz.data(), code.data());
    else if (!std::strcmp(ordering, "hilbert3"))
        status = smesh::encode_hilbert3<geom_t>(m.nelements, bx.data(), by.data(), bz.data(), code.data());
    else if (!std::strcmp(ordering, "random3"))
        status = smesh::encode_random3<geom_t>(m.nelements, bx.data(), by.data(), bz.data(), code.data());
    if (status != 0) {
        std::fprintf(stderr, "unknown or failed MESH_ORDER '%s'\n", ordering);
        std::abort();
    }

    std::vector<ptrdiff_t> order(m.nelements);
    std::iota(order.begin(), order.end(), (ptrdiff_t)0);
    // Stable, so two elements sharing a code keep the order they were built in
    // and the mesh is a function of the ordering name alone.
    std::stable_sort(order.begin(), order.end(),
                     [&](ptrdiff_t l, ptrdiff_t r) { return code[l] < code[r]; });

    std::vector<std::vector<idx_t>> ev(NXE, std::vector<idx_t>(m.nelements));
    std::vector<std::vector<geom_t>> adj(9, std::vector<geom_t>(m.nelements));
    std::vector<geom_t> det(m.nelements);
    for (ptrdiff_t e = 0; e < m.nelements; ++e) {
        const ptrdiff_t from = order[e];
        for (int c = 0; c < NXE; ++c) ev[c][e] = m.ev[c][from];
        for (int c = 0; c < 9; ++c) adj[c][e] = m.adj[c][from];
        det[e] = m.det[from];
    }
    m.ev.swap(ev); m.adj.swap(adj); m.det.swap(det);

    std::vector<idx_t> renumber((size_t)m.nnodes, (idx_t)-1);
    idx_t next = 0;
    for (ptrdiff_t e = 0; e < m.nelements; ++e)
        for (int c = 0; c < NXE; ++c) {
            const idx_t v = m.ev[c][e];
            if (renumber[v] < 0) renumber[v] = next++;
        }
    // A node no element reaches keeps a number, so the coordinate arrays stay
    // the length the rest of the harness expects.
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) if (renumber[v] < 0) renumber[v] = next++;

    for (auto &row : m.ev)
        for (ptrdiff_t e = 0; e < m.nelements; ++e) row[e] = renumber[row[e]];

    std::vector<double> px((size_t)m.nnodes), py((size_t)m.nnodes), pz((size_t)m.nnodes);
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
        px[renumber[v]] = m.px[v]; py[renumber[v]] = m.py[v]; pz[renumber[v]] = m.pz[v];
    }
    m.px.swap(px); m.py.swap(py); m.pz.swap(pz);

    finalise_element_pointers(m);
}

#if defined(ELEMENT_PROTEUS_HEX27)
// A uniform grid of triquadratic cubes, nodes in the lexicographic order of the PROTEUS basis:
// local node i + 3 j + 9 k sits at (i, j, k) / 2 of the cell.  The map is affine.
static Mesh build_unordered(int n) {
    Mesh m;
    const int nn = 2 * n + 1;
    const double h = 1.0 / n;
    m.nnodes = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n;
    m.ev.assign(NXE, std::vector<idx_t>(m.nelements));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    ptrdiff_t e = 0;
    for (int c = 0; c < n; ++c) for (int b = 0; b < n; ++b) for (int a = 0; a < n; ++a, ++e)
        for (int k = 0; k < 3; ++k) for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i)
            m.ev[i + 3 * j + 9 * k][e] = nid(2 * a + i, 2 * b + j, 2 * c + k);
    m.adj.assign(9, std::vector<geom_t>(m.nelements, 0));
    m.det.assign(m.nelements, (geom_t)(h*h*h));
    for (ptrdiff_t i = 0; i < m.nelements; ++i) { m.adj[0][i]=(geom_t)(h*h); m.adj[4][i]=(geom_t)(h*h); m.adj[8][i]=(geom_t)(h*h); }
    m.px.resize(m.nnodes); m.py.resize(m.nnodes); m.pz.resize(m.nnodes);
    for (int k = 0; k < nn; ++k) for (int j = 0; j < nn; ++j) for (int i = 0; i < nn; ++i) {
        const ptrdiff_t v = nid(i,j,k);
        m.px[v] = i*h/2; m.py[v] = j*h/2; m.pz[v] = k*h/2;
    }
    finalise_element_pointers(m);
    return m;
}
#elif defined(ELEMENT_HEX8)
static Mesh build_unordered(int n) {
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
    finalise_element_pointers(m);
    return m;
}
#else
// TET4 and TET10 share the Freudenthal split of a cube grid.  TET10 adds the
// six edge midpoints per tetrahedron, in SFEM's edge order
// (0,1) (1,2) (0,2) (0,3) (1,3) (2,3).  The edges are straight, so the geometry
// is still affine and the adjugate is still the tetrahedron's -- what makes
// TET10 a genuine test of the projection is that its reference gradients vary
// over the element, not that its geometry is curved.
// The vertices' adjugate and determinant of every tetrahedron, positively oriented: a negative
// one swaps its second and third vertices.
static void tet_geometry(Mesh &m, const std::vector<std::vector<geom_t>> &pts) {
    m.adj.assign(9, std::vector<geom_t>(m.nelements, 0));
    m.det.assign(m.nelements, 0);
    for (ptrdiff_t e = 0; e < m.nelements; ++e) {
        idx_t v[4];
        for (int c = 0; c < 4; ++c) v[c] = m.ev[c][e];
        double J[9];
        for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
            J[d*3+c] = (double)pts[d][v[c+1]] - (double)pts[d][v[0]];
        double D = J[0]*(J[4]*J[8]-J[5]*J[7]) - J[1]*(J[3]*J[8]-J[5]*J[6]) + J[2]*(J[3]*J[7]-J[4]*J[6]);
        if (D < 0) { std::swap(v[1], v[2]);
            // A TET10 read from disk carries its edge nodes: the swap exchanges edges (0,1) and (0,2),
            // and (1,3) and (2,3).
            if (m.ev.size() == 10) { std::swap(m.ev[4][e], m.ev[6][e]); std::swap(m.ev[8][e], m.ev[9][e]); }
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
}

static Mesh build_unordered(int n) {
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
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
        for (int t = 0; t < 6; ++t, ++e)
            for (int c = 0; c < 4; ++c) {
                const int b = tets[t][c];
                m.ev[c][e] = nid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
            }
    tet_geometry(m, pts);
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
    finalise_element_pointers(m);
    return m;
}
#endif

#if (NXE == 4 || NXE == 10) && !defined(ELEMENT_HEX8) && !defined(ELEMENT_PROTEUS_HEX27)
template <typename T>
static std::vector<T> read_raw(const std::string &path) {
    std::vector<T> v;
    if (FILE *f = std::fopen(path.c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        v.resize((size_t)std::ftell(f) / sizeof(T));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) v.clear();
        std::fclose(f);
    }
    if (v.empty()) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::abort();
    }
    return v;
}

// A TET4 or TET10 mesh in smesh's raw layout (i0..i<NXE-1>.int32, x/y/z.float32), as the Newton study
// reads it; a TET10 one from smesh's mesh_promote.
static Mesh read_unordered(const std::string &dir) {
    Mesh m;
    std::vector<std::vector<geom_t>> pts(3);
    const char *xyz[3] = {"x", "y", "z"};
    for (int d = 0; d < 3; ++d) {
        const std::vector<float> p = read_raw<float>(dir + "/" + xyz[d] + ".float32");
        pts[d].assign(p.begin(), p.end());
    }
    m.nnodes = (ptrdiff_t)pts[0].size();
    for (int c = 0; c < NXE; ++c) {
        const std::vector<int32_t> ic = read_raw<int32_t>(dir + "/i" + std::to_string(c) + ".int32");
        m.ev.emplace_back(ic.begin(), ic.end());
    }
    m.nelements = (ptrdiff_t)m.ev[0].size();
    tet_geometry(m, pts);
    finalise_element_pointers(m);
    return m;
}
#endif

// QP_MESH=<folder> reads a TET4 or TET10 mesh instead of building the cube of n cells.
static Mesh build(int n) {
#if (NXE == 4 || NXE == 10) && !defined(ELEMENT_HEX8) && !defined(ELEMENT_PROTEUS_HEX27)
    if (const char *dir = std::getenv("QP_MESH")) {
        Mesh m = read_unordered(dir);
        reorder_mesh(m, MESH_ORDER);
        return m;
    }
#endif
    Mesh m = build_unordered(n);
    reorder_mesh(m, MESH_ORDER);
    return m;
}
