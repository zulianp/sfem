// Team: the threads' shares are whole chunks, contiguous and block-aligned, cover the entries once,
// and the chunks are the same for any number of threads; with reserve_first thread 0 has none.

#include <vector>

#include "sfem_Team.hpp"
#include "sfem_test.hpp"

int test_team_shares() {
    const ptrdiff_t blocks[] = {0, 1, 15, 16, 17, 3136, 50175, 324604, 3000000};
    for (const ptrdiff_t block : {1, 3}) {
        for (const ptrdiff_t n_blocks : blocks) {
            const ptrdiff_t n = n_blocks * block;
            // Chunk boundaries as one thread sees them.
            const sfem::Team     one = sfem::Team::of(n, block, 0, 1);
            std::vector<ptrdiff_t> bounds;
            for (ptrdiff_t c = 0; c <= one.chunks; ++c) bounds.push_back(one.chunk_first_entry(c));
            SFEM_TEST_ASSERT(bounds.front() == 0 && bounds.back() == n);
            for (size_t c = 1; c < bounds.size(); ++c) {
                SFEM_TEST_ASSERT(bounds[c] >= bounds[c - 1] && bounds[c] % block == 0);
            }
            for (const bool reserve : {false, true})
            for (const int threads : {1, 2, 3, 7, 18, 72, 80, 288}) {
                ptrdiff_t next = 0;
                for (int tid = 0; tid < threads; ++tid) {
                    const sfem::Team t = sfem::Team::of(n, block, tid, threads, reserve);
                    if (reserve && threads > 1 && tid == 0) SFEM_TEST_ASSERT(t.begin == t.end);
                    SFEM_TEST_ASSERT(t.chunks == one.chunks);
                    SFEM_TEST_ASSERT(t.first_chunk <= t.end_chunk);
                    SFEM_TEST_ASSERT(t.begin <= t.end);
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
                SFEM_TEST_ASSERT(next == n || (n == 0 && next == 0));
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
