#include "sfem_glob.hpp"

#include <glob.h>

namespace sfem {
	std::vector<std::string> find_files(const std::string &pattern)
	{
		glob_t gl;
		glob(pattern.c_str(), GLOB_MARK, NULL, &gl);

		int n_files = gl.gl_pathc;
		std::vector<std::string> ret;
		for (int np = 0; np < n_files; np++) {
		    ret.push_back(gl.gl_pathv[np]);
		}

		globfree(&gl);
		return ret;
	}

	size_t count_files(const std::string &pattern)
	{
		return find_files(pattern).size();
	}
}
