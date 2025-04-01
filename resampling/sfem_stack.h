#ifndef __SFEM_STACK_H__
#define __SFEM_STACK_H__

#include <stddef.h>
#include <stdlib.h>

/**
 * @brief A generic stack data structure
 *
 * The sfem_stack implements a dynamically resizable stack that can
 * store pointers to any type of data.
 */
struct sfem_stack {
    void** stack;          /**< Array of pointers to stack elements */
    size_t size;           /**< Current number of elements in the stack */
    size_t capacity;       /**< Current maximum capacity of the stack */
    size_t delta_capacity; /**< Amount by which to increase capacity when stack is full */
};

/** Typedef for the stack structure */
typedef struct sfem_stack sfem_stack_t;

/**
 * @brief Creates a new stack with specified initial capacity
 *
 * @param capacity Initial capacity of the stack
 * @return sfem_stack_t* Pointer to the newly created stack, or NULL on allocation failure
 */
sfem_stack_t* sfem_stack_create(size_t capacity);

/**
 * @brief Returns the current number of elements in the stack
 *
 * @param stack Pointer to the stack
 * @return size_t Number of elements currently in the stack
 */
size_t sfem_stack_size(sfem_stack_t* stack);

/**
 * @brief Destroys the stack and frees allocated memory
 *
 * @param stack Pointer to the stack to be destroyed
 */
void sfem_stack_destroy(sfem_stack_t* stack);

/**
 * @brief Pushes an item onto the stack
 *
 * If the stack is full, its capacity will be increased by delta_capacity.
 *
 * @param stack Pointer to the stack
 * @param item Pointer to the item to be pushed
 * @return void* Pointer to the pushed item, or NULL on reallocation failure
 */
void* sfem_stack_push(sfem_stack_t* stack, void* item);

/**
 * @brief Pops the top item from the stack
 *
 * @param stack Pointer to the stack
 * @return void* Pointer to the popped item, or NULL if the stack is empty
 */
void* sfem_stack_pop(sfem_stack_t* stack);

#endif  // __SFEM_STACK_H__