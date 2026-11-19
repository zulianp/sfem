// Team: the threads' shares are whole chunks, contiguous and starting at multiples of
// team_chunk_align, cover the entries once, and the chunks are the same for any number of
// threads; with reserve_first thread 0 has none. A share of other items is the static schedule's.
// The team's sums (TeamReduction) are the same bits for any number of threads, and OpenMPBLAS's.

#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_BLAS_openmp.hpp"
#include "sfem_Team.hpp"
#include "sfem_TeamBLAS.hpp"
#include "sfem_TeamReduction.hpp"
#include "sfem_test.hpp"

namespace {
    int max_threads() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
}  // namespace

int test_team_shares() {
    for (const ptrdiff_t n : {0, 1, 11, 12, 47, 48, 49, 9408, 150525, 973812, 9000000}) {
        // Chunk boundaries as one thread sees them.
        const sfem::Team       one = sfem::Team::of(n, 0, 1);
        std::vector<ptrdiff_t> bounds;
        for (ptrdiff_t c = 0; c <= one.chunks; ++c) bounds.push_back(one.chunk_first_entry(c));
        SFEM_TEST_ASSERT(bounds.front() == 0 && bounds.back() == n);
        for (size_t c = 1; c < bounds.size(); ++c) {
            SFEM_TEST_ASSERT(bounds[c] >= bounds[c - 1]);
            SFEM_TEST_ASSERT(bounds[c] == n || bounds[c] % sfem::team_chunk_align == 0);
        }
        for (const bool reserve : {false, true}) {
            for (const int threads : {1, 2, 3, 7, 18, 72, 80, 288}) {
                ptrdiff_t next = 0;
                for (int tid = 0; tid < threads; ++tid) {
                    const sfem::Team t = sfem::Team::of(n, tid, threads, reserve);
                    SFEM_TEST_ASSERT(t.chunks == one.chunks);
                    SFEM_TEST_ASSERT(t.first_chunk <= t.end_chunk && t.begin <= t.end);
                    if (reserve && threads > 1 && tid == 0) SFEM_TEST_ASSERT(t.first_chunk == t.end_chunk);
                    if (t.first_chunk == t.end_chunk) {
                        SFEM_TEST_ASSERT(t.begin == t.end);
                        continue;
                    }
                    SFEM_TEST_ASSERT(t.begin == next);
                    SFEM_TEST_ASSERT(t.begin == bounds[t.first_chunk] && t.end == bounds[t.end_chunk]);
                    for (ptrdiff_t c = t.first_chunk; c < t.end_chunk; ++c) {
                        ptrdiff_t b, e;
                        t.chunk_entries(c, &b, &e);
                        SFEM_TEST_ASSERT(b == bounds[c] && e == bounds[c + 1]);
                    }
                    next = t.end;
                }
                SFEM_TEST_ASSERT(next == n);
            }
        }
    }
    return SFEM_TEST_SUCCESS;
}

// share(m, 1) is the iterations `omp for schedule(static)` gives the thread; with a reserved thread
// the others split the items among themselves, and `all` takes the reserved one back in.
int test_share_is_the_static_schedule() {
    for (const ptrdiff_t m : {0, 1, 5, 71, 72, 73, 1000, 123457}) {
        for (const int threads : {1, 2, 7, max_threads()}) {
            std::vector<int> owner_share((size_t)m, -1), owner_static((size_t)m, -1), owner_reserved((size_t)m, -1);
            int              bad = 0;
#pragma omp parallel num_threads(threads)
            {
                const sfem::Team team = sfem::Team::current(m);
                ptrdiff_t        b, e;
                team.share(m, 1, &b, &e);
                for (ptrdiff_t i = b; i < e; ++i) owner_share[(size_t)i] = team.tid;
#pragma omp for schedule(static)
                for (ptrdiff_t i = 0; i < m; ++i) owner_static[(size_t)i] = team.tid;

                const sfem::Team reserved = sfem::Team::current(m, true);
                reserved.share(m, 1, &b, &e);
                if (reserved.threads > 1 && reserved.tid == 0 && e > b) {
#pragma omp atomic
                    bad++;
                }
                for (ptrdiff_t i = b; i < e; ++i) owner_reserved[(size_t)i] = reserved.tid;
                reserved.share(m, 1, &b, &e, true);
                ptrdiff_t b0, e0;
                team.share(m, 1, &b0, &e0);
                if (b != b0 || e != e0) {
#pragma omp atomic
                    bad++;
                }
            }
            SFEM_TEST_ASSERT(bad == 0);
            SFEM_TEST_ASSERT(owner_share == owner_static);
            for (const int o : owner_reserved) SFEM_TEST_ASSERT(o >= 0);
        }
    }
    return SFEM_TEST_SUCCESS;
}

// The team's dot product: the same bits on 1 thread, on any number, serially outside a region,
// and from OpenMPBLAS.
int test_team_sums_do_not_depend_on_threads() {
    for (const ptrdiff_t n : {1, 100, 4095, 4096, 150525, 973812}) {
        std::vector<double> l((size_t)n), r((size_t)n);
        for (ptrdiff_t i = 0; i < n; ++i) {
            l[(size_t)i] = std::sin(0.37 * (double)i) * 1e3;
            r[(size_t)i] = std::cos(1.3 * (double)i) / (1 + (double)(i % 97));
        }
        const ptrdiff_t     chunks = sfem::team_chunk_count(n);
        std::vector<double> buffer((size_t)sfem::TeamReduction<double, 1>::buffer_size(chunks));

        sfem::TeamReduction<double, 1> serial(buffer.data(), chunks);
        const double                   ref = sfem::team_blas::dot(sfem::Team::of(n, 0, 1), serial, l.data(), r.data());

        for (const int threads : {1, 3, 7, max_threads()}) {
            int differ = 0;
#pragma omp parallel num_threads(threads)
            {
                sfem::TeamReduction<double, 1> reduction(buffer.data(), chunks);
                const sfem::Team               team = sfem::Team::current(n);
                // Twice in a row: the halves alternate.
                for (int rep = 0; rep < 2; ++rep) {
                    const double v = sfem::team_blas::dot(team, reduction, l.data(), r.data());
                    if (std::memcmp(&v, &ref, sizeof(double)) != 0) {
#pragma omp atomic
                        differ++;
                    }
                }
            }
            SFEM_TEST_ASSERT(differ == 0);
        }
        sfem::OpenMPBLAS<double> blas;
        const double             v = blas.dot(n, l.data(), r.data());
        SFEM_TEST_ASSERT(std::memcmp(&v, &ref, sizeof(double)) == 0);
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_team_shares);
    SFEM_RUN_TEST(test_share_is_the_static_schedule);
    SFEM_RUN_TEST(test_team_sums_do_not_depend_on_threads);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
