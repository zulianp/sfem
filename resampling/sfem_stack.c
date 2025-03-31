#include "sfem_stack.h"

sfem_stack_t*                         //
sfem_stack_create(size_t capacity) {  //

    sfem_stack_t* stack = (sfem_stack_t*)malloc(sizeof(sfem_stack_t));
    if (stack == NULL) {
        return NULL;
    }
    stack->stack = (void**)malloc(capacity * sizeof(void*));
    if (stack->stack == NULL) {
        free(stack);
        return NULL;
    }
    stack->size     = 0;
    stack->capacity = capacity;
    return stack;
}

size_t                                  //
sfem_stack_size(sfem_stack_t* stack) {  //
    return stack->size;                 //
}

void                                       //
sfem_stack_destroy(sfem_stack_t* stack) {  //

    if (stack != NULL) {
        free(stack->stack);
        free(stack);
    }
}

void*                                 //
sfem_stack_push(sfem_stack_t* stack,  //
                void*         item) {         //

    if (stack->size >= stack->capacity) {
        size_t new_capacity = stack->capacity + stack->delta_capacity;
        void** new_stack    = (void**)realloc(stack->stack, new_capacity * sizeof(void*));
        if (new_stack == NULL) {
            return NULL;  // Memory allocation failed
        }
        stack->stack    = new_stack;
        stack->capacity = new_capacity;
    }

    stack->stack[stack->size++] = item;
    return item;
}

void* sfem_stack_pop(sfem_stack_t* stack) {
    if (stack->size <= 0) {
        return NULL;  // Stack is empty
    }
    return stack->stack[--stack->size];
}