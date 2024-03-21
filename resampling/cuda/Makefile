# default target is libsfem_resample_field_cuda.a  
# 

all: libsfem_resample_field_cuda.a

all: sfem_resample_field_cuda.o temp.o
	ar cru libsfem_resample_field_cuda.a sfem_resample_field_cuda.o temp.o
	rm temp.o
	ranlib libsfem_resample_field_cuda.a