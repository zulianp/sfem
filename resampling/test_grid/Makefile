# set compiler

CC = g++-13
# CC = clang++-15
# CC = icpx

ifeq (${VERBOSE}, on)
	VERBOSE_FLAG = -v
endif

CFLAGS = -Wall 
LDFLAGS = -lpthread -lstdc++ -lm

DEBUG ?= 0

TARGET = test_grid_new

# if debug 

CUDA_DEBUG =

if eq(${ARM}, 1)
	VECTOR_WIDTH_OPT = -msve-vector-bits
# print message (STATUS "ARM")
print message (STATUS "ARM")

endif

SIMD ?= 512
CPU_ARCH = native

ifeq ($(DEBUG),1)
	CFLAGS += -g
	CUDA_DEBUG += -g
else
	CFLAGS += -std=c++17 -mtune=${CPU_ARCH} -march=${CPU_ARCH}  ${VECTOR_WIDTH_OPT}=${SIMD} -Wshadow=local -Ofast -fno-signed-zeros -fno-trapping-math -fassociative-math 
endif

CU_OPT_FLAGS = -O3 -use_fast_math -Xptxas=-O3,-v -Xcompiler=-O3,-march=native,-mtune=native
CU_TEST_GRID = test_grid_new_cuda
CU_LIB=lib${CU_TEST_GRID}.a
GPU_ARCH ?= sm_75

CU_VERBOSE_FLAG = 
CU_VERBOSE ?= off

DISABLE_CUDA ?= 0

ifeq (${DISABLE_CUDA}, 1)
	CUDA_FLAGS = 
	CU_LIB =
	CUDA_FLAGS = 
else
	CUDA_LIBS_PATH = /usr/local/cuda-12.3/targets/x86_64-linux/lib
	CUDA_FLAGS = -L${CUDA_LIBS_PATH} -lcudart  -L. -ltest_grid_new_cuda 
endif



#######################################
#######################################

${TARGET}: ${TARGET}.cpp test_grid_new.h ${CU_LIB}
	$(CC) -o ${TARGET} ${TARGET}.cpp  $(CFLAGS) $(LDFLAGS) -I. $(CUDA_FLAGS) -D DISABLE_CUDA=${DISABLE_CUDA} 

#######################################
#######################################

${CU_LIB}: test_grid_new_cuda.o temp.o test_grid_new_cuda.cu
	ar cru ${CU_LIB} test_grid_new_cuda.o temp.o; \
	ranlib ${CU_LIB}

test_grid_new_cuda.o: temp.o
	nvcc --gpu-architecture=${GPU_ARCH} -dlink -o test_grid_new_cuda.o temp.o $(CUDA_DEBUG)

temp.o: test_grid_new_cuda.cu test_grid_new.h
	nvcc ${CU_OPT_FLAGS} ${VERBOSE_FLAG} --gpu-architecture=${GPU_ARCH} -rdc=true -c test_grid_new_cuda.cu -o temp.o $(CUDA_DEBUG)

clean:
	rm -f test_grid_new *.o ${CU_LIB}
