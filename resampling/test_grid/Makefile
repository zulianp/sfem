CC = g++-13
CFLAGS = -Wall -Ofast -std=c++17 -mtune=native -march=native  -mprefer-vector-width=512
LDFLAGS = 

DEBUG ?= 0

# if debug 
ifeq ($(DEBUG),1)
	CFLAGS += -g
endif



test_grid_new: test_grid_new.cpp
	$(CC) $(CFLAGS) -o test_grid_new test_grid_new.cpp $(LDFLAGS)

clean:
	rm -f test_grid_new
