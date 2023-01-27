ifeq ($(debug),1)
	CFLAGS += -O0 -g
	CXXFLAGS += -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -O2 -g -DNDEBUG
	CXXFLAGS += -O2 -g -DNDEBUG
else
	CFLAGS += -Ofast -DNDEBUG
	CXXFLAGS += -Ofast -DNDEBUG
endif

ifeq ($(avx2sort), 1)
	CXXFLAGS += -DSFEM_ENABLE_AVX2_SORT -Iexternal

endif

CFLAGS += -pedantic 
# CFLAGS += -std=c99 

CXXFLAGS += -std=c++11
CXXFLAGS += -fno-exceptions -fno-rtti -static
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fPIC

GOALS = assemble assemble3 condense_matrix condense_vector idx_to_indicator remap_vector
DEPS = -L../matrix.io/ -lmatrix.io -lstdc++

LDFLAGS += $(DEPS)

MPICC ?= mpicc
CXX ?= c++
AR ?= ar

all : $(GOALS)

OBJS = \
	sortreduce.o \
	crs_graph.o \
	sortreduce.o \
	read_mesh.o  \
	laplacian.o \
	mass.o \
	neohookean.o

# SIMD experiment (worse perf)
# simd_laplacian.o
# simd_neohookean.o

# Scalar
#laplacian.o


libsfem.a : $(OBJS)
	ar rcs $@ $^

assemble : assemble.o libsfem.a
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

assemble3 : assemble3.o libsfem.a
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_matrix : condense_matrix.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_vector : condense_vector.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

idx_to_indicator : idx_to_indicator.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

remap_vector : remap_vector.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

sortreduce.o: sortreduce.cpp
	$(CXX) $(CXXFLAGS) -c $<

%.o : %.c
	$(MPICC) $(CFLAGS) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o *.a $(GOALS)

.SUFFIXES:

.PHONY: clean all
