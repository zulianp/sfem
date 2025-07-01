#ifndef __SFEM_QUEUE_H__
#define __SFEM_QUEUE_H__

#include <stddef.h>
#include <stdlib.h>

struct sfem_queue {
    void** queue;          /**< Array of pointers to stack elements */
    size_t head;           /**< Index of the next element to enqueue */
    size_t tail;           /**< Index of the next element to dequeue */
    size_t capacity;       /**< Current maximum capacity of the stack */
    size_t delta_capacity; /**< Amount by which to increase capacity when stack is full */
};

typedef struct sfem_queue sfem_queue_t;

sfem_queue_t* sfem_queue_create(const size_t capacity);

void* sfem_queue_destroy(sfem_queue_t* queue);

void sfem_queue_clear(sfem_queue_t* queue);

size_t sfem_queue_size(sfem_queue_t* queue);

void sfem_queue_realloc(sfem_queue_t* queue, const size_t new_capacity);

void sfem_queue_push(sfem_queue_t* queue, void* object);

void* sfem_queue_pop(sfem_queue_t* queue);

void sfem_queue_test(void);

#endif  // __SFEM_QUEUE_H__