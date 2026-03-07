#ifndef SFEM_GLOB_HPP
#define SFEM_GLOB_HPP

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

size_t count_files(const char *pattern);
int create_directory(const char *path);

#ifdef __cplusplus
}
#endif

#include <string>
#include <vector>

namespace sfem {
size_t count_files(const char *pattern);
int create_directory(const char *path);
std::vector<std::string> find_files(const std::string &pattern);
}

#endif //SFEM_GLOB_HPP
