SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Automatically filled with CMake
export SFEM_GEOM_T="float32"
export SFEM_GEOM_SIZE=4

export SFEM_REAL_T="float64"
export SFEM_REAL_SIZE=8

export SFEM_SCALAR_T="float64"
export SFEM_SCALAR_SIZE=8

export SFEM_JACOBIAN_T="float32"
export SFEM_JACOBIAN_SIZE=4

export SFEM_IDX_T="int32"
export SFEM_IDX_SIZE=4

export SFEM_COUNT_T="int64"
export SFEM_COUNT_SIZE=8

export SFEM_ELEMENT_IDX_T="int32"
export SFEM_ELEMENT_IDX_SIZE=4

export SFEM_LOCAL_IDX_T="int16"
export SFEM_LOCAL_IDX_SIZE=2

export SFEM_VEC_SIZE=4
