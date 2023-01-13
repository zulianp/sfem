ifeq ($(debug),1)
	CFLAGS += -O0 -g
	CXXFLAGS += -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -O2 -g -DNDEBUG
	CXXFLAGS += -O2 -g -DNDEBUG
else
	CFLAGS += -O3 -DNDEBUG
	CXXFLAGS += -O3 -DNDEBUG
endif

ifeq ($(avx2sort), 1)
	CXXFLAGS += -DSFEM_ENABLE_AVX2_SORT -Iexternal

endif

CFLAGS += -std=c99 -pedantic 

CXXFLAGS += -std=c++11
CXXFLAGS += -fno-exceptions -fno-rtti -static
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fPIC

GOALS = assemble condense_matrix condense_vector idx_to_indicator remap_vector
DEPS = -L../matrix.io/ -lmatrix.io -lstdc++

LDFLAGS += $(DEPS)

MPICC ?= mpicc
CXX ?= c++
AR ?= ar

all : $(GOALS)

OBJS = \
	sortreduce.o \
	crs_graph.o \
	laplacian.o \
	mass.o \
	sortreduce.o

libsfem.a : $(OBJS)
	ar rcs $@ $^

assemble : assemble.o libsfem.a
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
