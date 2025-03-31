#ifndef __SFEM_STACK_H__
#define __SFEM_STACK_H__

#include <stddef.h>
#include <stdlib.h>

struct sfem_stack {
    void** stack;
    size_t size;
    size_t capacity;
    size_t delta_capacity;
};

typedef struct sfem_stack sfem_stack_t;

sfem_stack_t* sfem_stack_create(size_t capacity);

size_t sfem_stack_size(sfem_stack_t* stack);

void sfem_stack_destroy(sfem_stack_t* stack);

void* sfem_stack_push(sfem_stack_t* stack, void* item);

void* sfem_stack_pop(sfem_stack_t* stack);

#endif  // __SFEM_STACK_H__