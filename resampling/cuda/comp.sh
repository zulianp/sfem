
nvcc -O3 -v -use_fast_math -Xptxas=-O3  --gpu-architecture=sm_75 -rdc=true -c sfem_resample_field_cuda.cu -o temp.o
# nvcc -g -G  --gpu-architecture=sm_75 -rdc=true -c sfem_resample_field_cuda.cu -o temp.o

nvcc --gpu-architecture=sm_75 -dlink -o sfem_resample_field_cuda.o temp.o 
ar cru libsfem_resample_field_cuda.a sfem_resample_field_cuda.o temp.o
rm temp.o
ranlib libsfem_resample_field_cuda.a



