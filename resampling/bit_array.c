#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bit_array.h"
#include "sfem_config.h"

/**
 * @brief Number of bits in an unsigned int
 */
#define BITS_PER_INT (sizeof(unsigned int) * 8)

// Function to create a bit array with a given number of bits
BitArray create_bit_array(size_t num_bits) {
    BitArray bit_array;
    bit_array.size  = num_bits;
    bit_array.array = (unsigned int *)calloc((num_bits + BITS_PER_INT - 1) / BITS_PER_INT, sizeof(unsigned int));
    if (bit_array.array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return bit_array;
}

// Function to set all bits in the bit array to 0
void to_zero(BitArray *bit_array) {
    memset(bit_array->array, 0, (bit_array->size + BITS_PER_INT - 1) / BITS_PER_INT * sizeof(unsigned int));
}

// Function to free the memory allocated for the bit array
void free_bit_array(BitArray bit_array) { free(bit_array.array); }

// Function to set a bit at a specific index
void set_bit(BitArray *bit_array, size_t index) {
    if (index >= bit_array->size) {
        printf("Index out of bounds\n");
        return;
    }
    bit_array->array[index / BITS_PER_INT] |= (1 << (index % BITS_PER_INT));
}

// Function to clear a bit at a specific index
void clear_bit(BitArray *bit_array, size_t index) {
    if (index >= bit_array->size) {
        printf("Index out of bounds\n");
        return;
    }
    bit_array->array[index / BITS_PER_INT] &= ~(1 << (index % BITS_PER_INT));
}

// Function to toggle a bit at a specific index
void toggle_bit(BitArray *bit_array, size_t index) {
    if (index >= bit_array->size) {
        printf("Index out of bounds\n");
        return;
    }
    bit_array->array[index / BITS_PER_INT] ^= (1 << (index % BITS_PER_INT));
}

// Function to get the state of a bit at a specific index
int get_bit(BitArray bit_array, size_t index) {
    if (index >= bit_array.size) {
        printf("Index out of bounds\n");
        return -1;
    }
    return (bit_array.array[index / BITS_PER_INT] >> (index % BITS_PER_INT)) & 1;
}

// Function to convert a bit array to a real array
real_t *to_real_array(BitArray bit_array) {
    real_t *real_array = (real_t *)malloc(bit_array.size * sizeof(real_t));
    for (size_t i = 0; i < bit_array.size; i++) {
        real_array[i] = get_bit(bit_array, i);
    }
    return real_array;
}

// int main() {
//     size_t num_bits = 100; // Example: 100 bits
//     BitArray bit_array = create_bit_array(num_bits);

//     // Set bit at index 50
//     set_bit(&bit_array, 50);
//     printf("Bit at index 50: %d\n", get_bit(bit_array, 50)); // Output: 1

//     // Clear bit at index 50
//     clear_bit(&bit_array, 50);
//     printf("Bit at index 50: %d\n", get_bit(bit_array, 50)); // Output: 0

//     // Toggle bit at index 50
//     toggle_bit(&bit_array, 50);
//     printf("Bit at index 50: %d\n", get_bit(bit_array, 50)); // Output: 1

//     // Toggle bit at index 50 again
//     toggle_bit(&bit_array, 50);
//     printf("Bit at index 50: %d\n", get_bit(bit_array, 50)); // Output: 0

//     // Free the bit array
//     free_bit_array(bit_array);

//     return 0;
// }
