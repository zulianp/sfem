#include "sfem_queue.h"
#include <stdio.h>

sfem_queue_t* sfem_queue_create(const size_t capacity) {
    //
    sfem_queue_t* queue = (sfem_queue_t*)malloc(sizeof(sfem_queue_t));

    if (queue == NULL) {
        return NULL;  // Memory allocation failed
    }

    queue->queue = (void**)malloc(capacity * sizeof(void*));
    if (queue->queue == NULL) {
        free(queue);
        return NULL;  // Memory allocation failed
    }

    queue->head           = 0;
    queue->tail           = 0;
    queue->capacity       = capacity;
    queue->delta_capacity = capacity;  // Default delta capacity is the same as initial capacity

    return queue;
}

size_t sfem_queue_size(sfem_queue_t* queue) { return queue->head - queue->tail; }

void* sfem_queue_destroy(sfem_queue_t* queue) {
    if (queue != NULL) {
        free(queue->queue);
        free(queue);
    }
    return NULL;  // Return NULL to indicate the queue has been destroyed
}

void sfem_queue_realloc(sfem_queue_t* queue, const size_t new_capacity) {
    if (queue->head >= queue->capacity) {
        void* new_queue = (void**)calloc(new_capacity, sizeof(void*));

        if (new_queue == NULL) {
            fprintf(stderr,                                               //
                    "Memory allocation failed while resizing queue.\n");  //
            exit(EXIT_FAILURE);                                           // Handle memory allocation failure
        }

        // Copy existing elements to the new queue
        size_t current_size = queue->head - queue->tail;
        for (size_t i = 0; i < current_size; i++) {
            ((void**)new_queue)[i] = queue->queue[queue->tail + i];
        }

        free(queue->queue);              // Free the old queue memory
        queue->queue    = new_queue;     // Update the queue pointer to the new memory
        queue->capacity = new_capacity;  // Update the capacity
        queue->head     = current_size;  // Reset head to the end of copied elements
        queue->tail     = 0;             // Reset tail to the beginning
    }
}

void sfem_queue_push(sfem_queue_t* queue, void* object) {
    if (queue->head >= queue->capacity) {
        sfem_queue_realloc(queue, queue->capacity + queue->delta_capacity);
    }

    queue->queue[queue->head++] = object;
}

void* sfem_queue_pop(sfem_queue_t* queue) {
    if (queue->tail >= queue->head) {
        return NULL;  // Queue is empty
    }

    void* object = queue->queue[queue->tail++];

    // Optional: Reset the tail if it reaches the head to avoid memory leaks
    if (queue->tail >= queue->head) {
        queue->tail = 0;
        queue->head = 0;
    }
    // Compact the queue if more than half the capacity is wasted space
    else if (queue->tail > queue->capacity / 2 && queue->tail > 0) {
        size_t current_size = queue->head - queue->tail;
        // Move remaining elements to the beginning
        for (size_t i = 0; i < current_size; i++) {
            queue->queue[i] = queue->queue[queue->tail + i];
        }
        queue->head = current_size;
        queue->tail = 0;
    }

    return object;
}

void sfem_queue_test(void) {
    sfem_queue_t* queue = sfem_queue_create(5);

    if (queue == NULL) {
        fprintf(stderr, "Failed to create queue\n");
        return;
    }

    for (int i = 0; i < 20; i++) {
        int* item = (int*)malloc(sizeof(int));
        *item     = i;
        sfem_queue_push(queue, item);
    }

    printf("Queue size after pushing 20 items: %zu\n", sfem_queue_size(queue));

    // pop 10 items from the queue
    for (int i = 0; i < 10; i++) {
        int* item = (int*)sfem_queue_pop(queue);
        if (item != NULL) {
            printf("* Popped item: %d\n", *item);
            free(item);
        } else {
            printf("* Queue is empty\n");
        }
    }

    printf("Queue size after popping 100 items: %zu\n", sfem_queue_size(queue));

    // push 10 more items
    int aa = 1111;
    for (int i = 10; i < 44; i++) {
        int* item = (int*)malloc(sizeof(int));
        *item     = aa++;
        sfem_queue_push(queue, item);
    }

    // print queue size
    printf("Queue size after pushing 10 more items: %zu\n", sfem_queue_size(queue));

    while (sfem_queue_size(queue) > 0) {
        int* item = (int*)sfem_queue_pop(queue);
        printf("+ Popped item: %d\n", *item);
        free(item);
    }

    printf("Queue size after popping all items: %zu\n", sfem_queue_size(queue));

    sfem_queue_destroy(queue);
}