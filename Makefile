ifeq ($(debug),1)
	CFLAGS += -pedantic -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -pedantic -O2 -g
else
	CFLAGS += -pedantic -O3 -DNDEBUG
endif

GOALS = assemble condense_matrix condense_vector idx_to_indicator remap_vector
DEPS = -L../matrix.io/ -lmatrix.io

LDFLAGS += $(DEPS)

MPICC ?= mpicc

all : $(GOALS)

assemble : assemble.o crs_graph.o laplacian.o mass.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_matrix : condense_matrix.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_vector : condense_vector.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

idx_to_indicator : idx_to_indicator.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

remap_vector : remap_vector.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

%.o : %.c
	$(MPICC) $(CFLAGS) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o $(GOALS)

.SUFFIXES:

.PHONY: clean all
