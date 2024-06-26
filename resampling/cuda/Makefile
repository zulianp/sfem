LIB=libsfem_resample_field_cuda.a
GPU_ARCH ?= sm_75

VERBOSE_FLAG = 
VERBOSE ?= off

OPT_FLAGS = -O3 -use_fast_math -Xptxas=-O3,-v  -Xcompiler=-O3,-march=native,-mtune=native

ifeq (${VERBOSE}, on)
	VERBOSE_FLAG = -v
endif

${LIB}: sfem_resample_field_cuda.o temp.o sfem_resample_field_cuda.cu tet10_resample_field.o temp_tet10.o tet10_resample_field.cu
	ar cru libsfem_resample_field_cuda.a sfem_resample_field_cuda.o temp.o tet10_resample_field.o temp_tet10.o ;\
	ranlib libsfem_resample_field_cuda.a

sfem_resample_field_cuda.o: temp.o
	nvcc --gpu-architecture=${GPU_ARCH} -dlink -o sfem_resample_field_cuda.o temp.o 

temp.o: sfem_resample_field_cuda.cu quadratures_rule_cuda.h
	nvcc ${OPT_FLAGS} ${VERBOSE_FLAG} --gpu-architecture=${GPU_ARCH} -rdc=true -c sfem_resample_field_cuda.cu -o temp.o

tet10_resample_field.o: temp_tet10.o
	nvcc --gpu-architecture=${GPU_ARCH} -dlink -o tet10_resample_field.o temp_tet10.o

temp_tet10.o: tet10_resample_field.cu quadratures_rule_cuda.h
	nvcc ${OPT_FLAGS} ${VERBOSE_FLAG} --gpu-architecture=${GPU_ARCH} -rdc=true -c tet10_resample_field.cu -o temp_tet10.o -I${PWD}/base

clean:
	rm -f *.o

cleanall:
	rm -f *.o ${LIB}
