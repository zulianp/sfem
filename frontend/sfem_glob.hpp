#ifndef SFEM_GLOB_HPP
#define SFEM_GLOB_HPP

#include <string>
#include <vector>

namespace sfem {
	std::vector<std::string> find_files(const std::string &pattern);
	size_t count_files(const std::string &pattern);
}

#endif //SFEM_GLOB_HPP