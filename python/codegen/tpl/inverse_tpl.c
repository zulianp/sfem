
static SFEM_INLINE void inverse{SIZE}(
// Input
{INPUT_ARGS} //Output
{OUTPUT_ARGS}
	)
{{
{BODY}
}}

void dinvert{SIZE}(
	const ptrdiff_t nnodes,
	const count_t *const SFEM_RESTRICT rowptr,
	const idx_t *const SFEM_RESTRICT colidx,
	real_t **const SFEM_RESTRICT values,
	real_t **const SFEM_RESTRICT inv_diag
	)
{{

for(ptrdiff_t i = 0; i < nnodes; i++) 
{{

const count_t r_begin = rowptr[i];
const count_t r_end = rowptr[i+1];
const count_t r_extent = r_end - r_begin;
const idx_t *cols = &colidx[r_begin];

count_t diag_idx = -1;
for(count_t k = 0; k < r_extent; k++) 
{{

if(cols[k] == i) 
{{
	diag_idx = k;
	break;
}}

}} // end for

assert(diag_idx != -1);

inverse{SIZE}(
{DINVERT_PASS_ARGS}
);

}}
}}