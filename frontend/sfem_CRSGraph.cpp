#include "sfem_CRSGraph.hpp"

// C includes
#include "crs_graph.h"

namespace sfem {

	class CRSGraph::Impl {
	public:
	    std::shared_ptr<Buffer<count_t>> rowptr;
	    std::shared_ptr<Buffer<idx_t>>   colidx;
	    ~Impl() {}
	};

	CRSGraph::CRSGraph(const std::shared_ptr<Buffer<count_t>> &rowptr, const std::shared_ptr<Buffer<idx_t>> &colidx)
	    : impl_(std::make_unique<Impl>()) {
	    impl_->rowptr = rowptr;
	    impl_->colidx = colidx;
	}

	void CRSGraph::print(std::ostream &os) const {
	    auto            rowptr = this->rowptr()->data();
	    auto            colidx = this->colidx()->data();
	    const ptrdiff_t nnodes = this->n_nodes();
	    const ptrdiff_t nnz    = this->nnz();

	    os << "CRSGraph (" << nnodes << " nodes, " << nnz << " nnz)\n";
	    for (ptrdiff_t i = 0; i < nnodes; i++) {
	        os << i << " (" << (rowptr[i + 1] - rowptr[i]) << "): ";
	        for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
	            assert(j < nnz);
	            os << colidx[j] << " ";
	        }
	        os << "\n";
	    }
	}

	CRSGraph::CRSGraph() : impl_(std::make_unique<Impl>()) {}
	CRSGraph::~CRSGraph() = default;

	ptrdiff_t CRSGraph::n_nodes() const { return rowptr()->size() - 1; }
	ptrdiff_t CRSGraph::nnz() const { return colidx()->size(); }

	std::shared_ptr<Buffer<count_t>> CRSGraph::rowptr() const { return impl_->rowptr; }
	std::shared_ptr<Buffer<idx_t>>   CRSGraph::colidx() const { return impl_->colidx; }

	std::shared_ptr<CRSGraph> CRSGraph::block_to_scalar(const int block_size) {
	    auto rowptr = create_host_buffer<count_t>(this->n_nodes() * block_size + 1);
	    auto colidx = create_host_buffer<idx_t>(this->nnz() * block_size * block_size);

	    crs_graph_block_to_scalar(
	            this->n_nodes(), block_size, this->rowptr()->data(), this->colidx()->data(), rowptr->data(), colidx->data());

	    return std::make_shared<CRSGraph>(rowptr, colidx);
	}

}
