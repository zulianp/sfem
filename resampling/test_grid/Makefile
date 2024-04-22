# set compiler

# # CC = g++-13
# CC = clang++-15
CC = icpx


CU_TEST_GRID = test_grid_new_cuda
CU_LIB=lib${CU_TEST_GRID}.a
GPU_ARCH ?= sm_75

CU_VERBOSE_FLAG = 
CU_VERBOSE ?= off

CU_OPT_FLAGS = -O3 -use_fast_math -Xptxas=-O3

ifeq (${VERBOSE}, on)
	VERBOSE_FLAG = -v
endif



CFLAGS = -Wall -Ofast -std=c++17 -mtune=native -march=native  -mprefer-vector-width=512 -Wshadow=local
LDFLAGS = -lpthread -lstdc++ -lm

DEBUG ?= 0

# if debug 
ifeq ($(DEBUG),1)
	CFLAGS += -g
endif

test_grid_new: test_grid_new.cpp test_grid_new.h ${CU_LIB}
	$(CC) $(CFLAGS) -o test_grid_new test_grid_new.cpp $(LDFLAGS) -I.  -L. -ltest_grid_new_cuda 

#######################################
#######################################

${CU_LIB}: test_grid_new_cuda.o temp.o test_grid_new_cuda.cu
	ar cru ${CU_LIB} test_grid_new_cuda.o temp.o; \
	ranlib ${CU_LIB}

test_grid_new_cuda.o: temp.o
	nvcc --gpu-architecture=${GPU_ARCH} -dlink -o test_grid_new_cuda.o temp.o 

temp.o: test_grid_new_cuda.cu test_grid_new.h
	nvcc ${OPT_FLAGS} ${VERBOSE_FLAG} --gpu-architecture=${GPU_ARCH} -rdc=true -c test_grid_new_cuda.cu -o temp.o

clean:
	rm -f test_grid_new *.o ${CU_LIB}
