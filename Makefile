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

CC=mpicc

all : $(GOALS)

assemble : assemble.o crs_graph.o laplacian.o mass.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_matrix : condense_matrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_vector : condense_vector.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

idx_to_indicator : idx_to_indicator.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

remap_vector : remap_vector.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

%.o : %.c
	$(CC) $(CFLAGS) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o $(GOALS)

.SUFFIXES:

.PHONY: clean all
