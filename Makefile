ifeq ($(debug),1)
	CFLAGS += -pedantic -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -pedantic -O2 -g
else
	CFLAGS += -pedantic -O3 -DNDEBUG
endif

GOALS = assemble condense_matrix condense_vector idx_to_indicator

CC=mpicc

all : $(GOALS)

assemble : assemble.o ../matrix.io/matrixio_crs.o ../matrix.io/utils.o ../matrix.io/matrixio_array.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_matrix : condense_matrix.o ../matrix.io/matrixio_crs.o ../matrix.io/utils.o ../matrix.io/matrixio_array.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

condense_vector : condense_vector.o ../matrix.io/matrixio_crs.o ../matrix.io/utils.o ../matrix.io/matrixio_array.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

idx_to_indicator : idx_to_indicator.o ../matrix.io/matrixio_crs.o ../matrix.io/utils.o ../matrix.io/matrixio_array.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

%.o : %.c
	$(CC) $(CFLAGS) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o $(GOALS) 

.SUFFIXES:

.PHONY: clean all
