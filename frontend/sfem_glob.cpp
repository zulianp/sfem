#include "sfem_glob.hpp"
#include "sfem_base.h"
#ifndef _WIN32
#include <glob.h>
#else
#include <filesystem>
#include <iostream>
#include <cassert>
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
        SFEM_ERROR("IMPLEMENT ME!");
        return {};
#endif
    }

    size_t count_files(const char *pattern) {
        // FIXME
        return find_files(pattern).size();
    }

    int create_directory(const char *path) {
#ifdef _WIN32
        namespace fs = std::filesystem;
        try {
            return fs::create_directory(path) ? SFEM_SUCCESS : SFEM_FAILURE;
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            return SFEM_FAILURE;
        }
#else
        struct stat st = {0};
        if (stat(path, &st) == -1) {
            mkdir(path, 0700);
        }
#endif
    }
}  // namespace sfem

extern "C" size_t count_files(const char *pattern) { return sfem::count_files(pattern); }

extern "C" int create_directory(const char *path) { return sfem::create_directory(path); }
