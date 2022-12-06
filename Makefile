ifeq ($(debug),1)
	CFLAGS += -pedantic -O0 -g
else ifeq ($(prof),1)
	CFLAGS += -pedantic -O2 -g
else
	CFLAGS += -pedantic -O3 -DNDEBUG
endif

GOALS = assemble

CC=mpicc

all : $(GOALS)

assemble : assemble.o ../matrix.io/matrixio_crs.o ../matrix.io/utils.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) ; \

%.o : %.c
	$(CC) $(CFLAGS) -c $<

.SUFFIXES :
.PRECIOUS :

clean:
	rm *.o $(GOALS) 

.SUFFIXES:

.PHONY: clean all
