#ifndef SFEM_GLOB_HPP
#define SFEM_GLOB_HPP

#include <string>
#include <vector>

namespace sfem {
	std::vector<std::string> find_files(const std::string &pattern);
	size_t count_files(const char *pattern);
	int create_directory(const char *path);
}

 extern "C" size_t count_files(const char *pattern);
extern "C" int create_directory(const char *path);

#endif //SFEM_GLOB_HPP