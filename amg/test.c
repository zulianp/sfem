#include <stdio.h>
#include <stdlib.h>
#include "interpolation.h"
#include "mg_builder.h"
#include "sparse.h"

int partition_test();

int main() {
    int failed = partition_test();

    if (failed) {
        printf("Some test(s) failed....\n");
    } else {
        printf("All tests successful!\n");
    }

    return failed;
}

int partition_test() {
    SymmCOOMatrix *symm_coo = (SymmCOOMatrix *)malloc(sizeof(SymmCOOMatrix));
    PWCHierarchy hierarchy;
    int test_result = 0;
    real_t coarsening_factor = 2.0;
    ptrdiff_t nrows = 1492;
    ptrdiff_t ncols = 1492;
    count_t nnz = 18794;

    count_t *row_ptr = (count_t *)malloc((nrows + 1) * sizeof(count_t));
    idx_t *col_indices = (idx_t *)malloc(nnz * sizeof(idx_t));
    real_t *values = (real_t *)malloc(nnz * sizeof(real_t));

    count_t nweights = (nnz - nrows) / 2;
    symm_coo->offdiag_nnz = nweights;
    symm_coo->offdiag_row_indices = (idx_t *)malloc(nweights * sizeof(idx_t));
    symm_coo->offdiag_col_indices = (idx_t *)malloc(nweights * sizeof(idx_t));
    symm_coo->offdiag_values = (real_t *)malloc(nweights * sizeof(real_t));
    symm_coo->dim = nrows;
    symm_coo->diag = (real_t *)malloc(nrows * sizeof(real_t));

    // TODO not sure how to use matrixio or what best way to load matrix is
    load_binary_file("../data/cylinder/laplace_indptr.raw", row_ptr, (nrows + 1) * sizeof(count_t));
    load_binary_file("../data/cylinder/laplace_indices.raw", col_indices, nnz * sizeof(idx_t));
    load_binary_file("../data/cylinder/laplace_values.raw", values, nnz * sizeof(real_t));

    csr_to_symmcoo(nrows, nnz, row_ptr, col_indices, values, symm_coo);
    test_result = builder(coarsening_factor, symm_coo, &hierarchy);

    free(row_ptr);
    free(col_indices);
    free(values);

    for (idx_t level = 0; level < hierarchy.levels; level++) {
        SymmCOOMatrix *mat = hierarchy.matrices[level];
        free(mat->offdiag_row_indices);
        free(mat->offdiag_col_indices);
        free(mat->offdiag_values);
        free(mat->diag);
        free(mat);

        if (level < (hierarchy.levels - 1)) {
            PiecewiseConstantTransfer *p = hierarchy.transer_operators[level];

            free(p->weights);
            free(p->partition);
            free(p);
        }
    }

    free(hierarchy.transer_operators);
    free(hierarchy.matrices);

    return test_result;
}
