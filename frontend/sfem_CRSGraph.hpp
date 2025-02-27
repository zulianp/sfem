#ifndef SFEM_CRS_GRAPH_HPP
#define SFEM_CRS_GRAPH_HPP

// C++ includes
#include "sfem_Buffer.hpp"

#include <memory>

namespace sfem {
	class CRSGraph final {
	public:
	    CRSGraph();
	    ~CRSGraph();

	    CRSGraph(const std::shared_ptr<Buffer<count_t>> &rowptr, const std::shared_ptr<Buffer<idx_t>> &colidx);

	    friend class Mesh;

	    ptrdiff_t                        n_nodes() const;
	    ptrdiff_t                        nnz() const;
	    std::shared_ptr<Buffer<count_t>> rowptr() const;
	    std::shared_ptr<Buffer<idx_t>>   colidx() const;
	    std::shared_ptr<CRSGraph>        block_to_scalar(const int block_size);

	    void print(std::ostream &os) const;

	private:
	    class Impl;
	    std::unique_ptr<Impl> impl_;
	};
}

#endif //SFEM_CRS_GRAPH_HPP
