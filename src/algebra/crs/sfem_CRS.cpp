#include "sfem_CRS.hpp"

#include "matrixio_crs.h"
#include "sfem_base.hpp"
#include "smesh_glob.hpp"

namespace sfem {

    int write_crs(const std::string &path, CRSGraph &graph, sfem::Buffer<real_t> &values) {
        smesh::create_directory(path.c_str());
        crs_t crs_out;
        crs_out.rowptr      = (char *)graph.rowptr()->data();
        crs_out.colidx      = (char *)graph.colidx()->data();
        crs_out.values      = (char *)values.data();
        crs_out.grows       = graph.rowptr()->size() - 1;
        crs_out.lrows       = graph.rowptr()->size() - 1;
        crs_out.lnnz        = values.size();
        crs_out.gnnz        = values.size();
        crs_out.start       = 0;
        crs_out.rowoffset   = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = SFEM_MPI_REAL_T;
        return crs_write_folder(MPI_COMM_SELF, path.c_str(), &crs_out);
    }

}  // namespace sfem
