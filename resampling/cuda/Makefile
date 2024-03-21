LIB=libsfem_resample_field_cuda.a
GPU_ARCH ?= sm_75

VERBOSE_FLAG = 
VERBOSE ?= off

ifeq (${VERBOSE}, on)
	VERBOSE_FLAG = -v
endif

${LIB}: sfem_resample_field_cuda.o temp.o sfem_resample_field_cuda.cu
	ar cru libsfem_resample_field_cuda.a sfem_resample_field_cuda.o temp.o; \
	ranlib libsfem_resample_field_cuda.a

sfem_resample_field_cuda.o: temp.o
	nvcc --gpu-architecture=${GPU_ARCH} -dlink -o sfem_resample_field_cuda.o temp.o 

temp.o: sfem_resample_field_cuda.cu
	nvcc -O3 ${VERBOSE_FLAG} -use_fast_math -Xptxas=-O3 --gpu-architecture=${GPU_ARCH} -rdc=true -c sfem_resample_field_cuda.cu -o temp.o

clean:
	rm -f *.o

cleanall:
	rm -f *.o ${LIB}
