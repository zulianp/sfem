export SFEM_INDEX_SIZE=32
export SFEM_COUNT_SIZE=32

if [[ $SFEM_COUNT_SIZE -eq 64 ]]
then
	export py_sfem_count_t="int64"
	export sfem_count_size=8
else
	export py_sfem_count_t="int32"
	export sfem_count_size=4
fi

if [[ $SFEM_INDEX_SIZE -eq 64 ]]
then
	export py_sfem_idx_t="int64"
	export sfem_idx_size=8
else
	export py_sfem_idx_t="int32"
	export sfem_idx_size=4
fi

export py_sfem_real_t="float64"
export sfem_real_size=8

export py_sfem_geom_t="float32"
export sfem_geom_size=4

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
