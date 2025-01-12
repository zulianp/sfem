#include "sfem_glob.hpp"

#ifndef _WIN32
#include <glob.h>
#endif

namespace sfem {
    std::vector<std::string> find_files(const std::string &pattern) {
#ifndef _WIN32
        glob_t gl;
        glob(pattern.c_str(), GLOB_MARK, NULL, &gl);

        int                      n_files = gl.gl_pathc;
        std::vector<std::string> ret;
        for (int np = 0; np < n_files; np++) {
            ret.push_back(gl.gl_pathv[np]);
        }

        globfree(&gl);
        return ret;
#else
#error "Not implemented for Windows!"
#endif
    }

    size_t count_files(const char *pattern) {
        // FIXME
        return find_files(pattern).size();
    }
}  // namespace sfem


extern "C" size_t count_files(const char *pattern)
{
    return sfem::count_files(pattern);
}
