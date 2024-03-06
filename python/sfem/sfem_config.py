import numpy as np

SFEM_INDEX_BITSIZE=32
SFEM_COUNT_BITSIZE=32

if SFEM_INDEX_BITSIZE == 64:
	idx_t = np.int64
else:
	idx_t = np.int32
	
if SFEM_COUNT_BITSIZE == 64:
	count_t = np.int64
else:
	count_t = np.int32
	
geom_t = np.float32
real_t = np.float64
