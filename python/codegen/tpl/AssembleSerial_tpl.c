////////////////////////////////
// AssembleSerial_tpl.c
////////////////////////////////

#define NNODES      {NNODES};
#define BLOCK_SIZE  {BLOCK_SIZE};
#define SPATIAL_DIM {SPATIAL_DIM}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {{
    int i;
    int v;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {{
        for(v = 0; v < SFEM_VECTOR_SIZE; v++) {
            if (arr[i + v] == target) return i + v;
        }
    }}
    for (; i < size; i++) {{
        if (arr[i] == target) return i;
    }}
    return -1;
}}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {{
    if (lenrow <= 32) {{
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {{
        //     // Hi
        // }}
        // assert(k < lenrow);
        // assert(key == row[k]);
    }} else {{
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }}
}}

static SFEM_INLINE void find_cols(const idx_t *targets, const idx_t *const row, const int lenrow, int *ks) {{
    if (lenrow > 32) {{
        for (int d = 0; d < NNODES; ++d) {{
            ks[d] = find_col(targets[d], row, lenrow);
        }}
    }} else {{
#pragma unroll(NNODES)
        for (int d = 0; d < NNODES; ++d) {{
            ks[d] = 0;
        }}

        for (int i = 0; i < lenrow; ++i) {{
#pragma unroll(NNODES)
            for (int d = 0; d < NNODES; ++d) {{
                ks[d] += row[i] < targets[d];
            }}
        }}
    }}
}}

static const int n_qp = QUADRATURE_NQP;
static const real_t qx[QUADRATURE_NQP] = {{{QUADRATURE_X}}};
static const real_t qy[QUADRATURE_NQP] = {{{QUADRATURE_Y}}};
static const real_t qz[QUADRATURE_NQP] = {{{QUADRATURE_Z}}};
static const real_t qw[QUADRATURE_NQP] = {{{QUADRATURE_W}}};

void {MATERIAL}_assemble_hessian(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const elems,
    geom_t **const xyz,
    {ARG_MATERIAL_PARAMS}
    {ARG_FIELDS}
    idx_t *const rowptr,
    idx_t *const colidx,
    real_t *const values)
{{
    double tick = MPI_Wtime();

    static const int block_size = {BLOCK_SIZE};
    static const int mat_block_size = block_size * block_size;

    idx_t ev[NNODES];
    idx_t ks[NNODES];

    real_t element_node_matrix[(BLOCK_SIZE * BLOCK_SIZE)];

    {DECL_FIELDS_ELEMENT}

    for (ptrdiff_t i = 0; i < nelements; ++i) {{
#pragma unroll(NNODES)
        for (int v = 0; v < NNODES; ++v) {{
            ev[v] = elems[v][i];
        }}

        {READ_FIELDS_ELEMENT}

        {HESSIAN_PRE_KERNEL}

        for (int edof_i = 0; edof_i < NNODES; ++edof_i) {{
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
            const idx_t *row = &colidx[rowptr[dof_i]];
            
            find_cols(ev, row, lenrow, ks);

            real_t *row_blocks = &values[rowptr[dof_i] * mat_block_size];

            for (int edof_j = 0; edof_j < NNODES; ++edof_j) {{
                memset(element_node_matrix, 0, sizeof(real_t) * (BLOCK_SIZE * BLOCK_SIZE));

                for (int q = 0; q < n_qp; ++q) {{
                    const real_t dV = qw[q] * measure;
                    {HESSIAN_MICRO_KERNEL}
                }}

                const idx_t block_k = ks[edof_j] * mat_block_size;
                real_t *block = &row_blocks[block_k];

                // Iterate over dimensions
                for (int bj = 0; bj < block_size; ++bj) {{
                    const idx_t offset_j = bj * block_size;

                    for (int bi = 0; bi < block_size; ++bi) {{
                        const real_t val = element_node_matrix[bi * block_size + bj];

                        assert(val == val);

                        block[offset_j + bi] += val;
                    }}
                }}
            }}
        }}
    }}

    double tock = MPI_Wtime();
    printf("{MATERIAL}.c: assemble_hessian\t%g seconds\n", tock - tick);
}}

void {MATERIAL}_assemble_gradient(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const elems,
    geom_t **const xyz,
    {ARG_MATERIAL_PARAMS}
    {ARG_FIELDS}
    real_t *const values) 
{{
    double tick = MPI_Wtime();

    static const int block_size = BLOCK_SIZE;
    static const int mat_block_size = block_size * block_size;

    idx_t ev[NNODES];
    real_t element_node_vector[BLOCK_SIZE];

    {DECL_FIELDS_ELEMENT}

    for (ptrdiff_t i = 0; i < nelements; ++i) {{
#pragma unroll(NNODES)
        for (int v = 0; v < NNODES; ++v) {{
            ev[v] = elems[v][i];
        }}

        {READ_FIELDS_ELEMENT}

        {GRADIENT_PRE_KERNEL}

        for (int edof_i = 0; edof_i < NNODES; ++edof_i) {{
            const idx_t dof_i = ev[edof_i] * block_size;

            memset(element_node_vector, 0, sizeof(real_t) * (block_size));

            for (int q = 0; q < n_qp; ++q) {{
                const real_t dV = qw[q] * measure;
                {GRADIENT_MICRO_KERNEL}
            }}

            for (int bi = 0; bi < block_size; ++bi) {{
                values[dof_i + bi] += element_node_vector[bi];
            }}
        }}
    }}

    double tock = MPI_Wtime();
    printf("{MATERIAL}.c: assemble_gradient\t%g seconds\n", tock - tick);
}}

void {MATERIAL}_assemble_value(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const elems,
    geom_t **const xyz,
    {ARG_MATERIAL_PARAMS}
    {ARG_FIELDS}
    real_t *const value)
{{
    double tick = MPI_Wtime();

    static const int block_size = BLOCK_SIZE;
    static const int mat_block_size = block_size * block_size;

    idx_t ev[NNODES];

    {DECL_FIELDS_ELEMENT}

    for (ptrdiff_t i = 0; i < nelements; ++i) {{
#pragma unroll(NNODES)
        for (int v = 0; v < NNODES; ++v) {{
            ev[v] = elems[v][i];
        }}

        {READ_FIELDS_ELEMENT}

        real_t element_scalar = 0;

        {VALUE_PRE_KERNEL}

        for (int q = 0; q < n_qp; ++q) {{
            const real_t dV = qw[q] * measure;

            {VALUE_MICRO_KERNEL}
        }}

        *value += element_scalar;
    }}

    double tock = MPI_Wtime();
    printf("{MATERIAL}.c: assemble_value\t%g seconds\n", tock - tick);
}}

#undef NNODES
#undef BLOCK_SIZE
#undef SPATIAL_DIM

////////////////////////////////

