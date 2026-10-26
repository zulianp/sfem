// Team: the threads' shares are whole chunks, contiguous and starting at multiples of
// team_chunk_align, cover the entries once, and the chunks are the same for any number of
// threads; with reserve_first thread 0 has none.

#include <vector>

#include "sfem_Team.hpp"
#include "sfem_test.hpp"

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

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_team_shares);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
