/**
 * @file bit_array.h
 * @brief A compact bit array implementation for efficient boolean storage
 *
 * This header provides a BitArray data structure and functions to manipulate
 * individual bits within the array. It's designed for memory-efficient storage
 * of boolean values, where each bit represents a single boolean state.
 */

#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include <stddef.h>

/**
 * @brief Structure representing a bit array
 *
 * This structure stores an array of unsigned integers that are used
 * to represent a collection of bits. Each bit can be individually
 * accessed, set, cleared, or toggled.
 */
typedef struct {
    unsigned int *array; /**< Array of unsigned integers to store the bits */
    size_t        size;  /**< Number of bits in the array */
} BitArray;

/**
 * @brief Creates a bit array with the specified number of bits
 *
 * @param num_bits The number of bits to allocate
 * @return BitArray A newly allocated bit array structure
 */
BitArray create_bit_array(size_t num_bits);

/**
 * @brief Sets all bits in the array to 0
 *
 * @param bit_array
 */
void to_zero(BitArray *bit_array);

/**
 * @brief Frees the memory allocated for a bit array
 *
 * @param bit_array The bit array to free
 */
void free_bit_array(BitArray bit_array);

/**
 * @brief Sets the bit at the specified index to 1
 *
 * @param bit_array Pointer to the bit array
 * @param index Index of the bit to set
 */
void set_bit(BitArray *bit_array, size_t index);

/**
 * @brief Clears the bit at the specified index to 0
 *
 * @param bit_array Pointer to the bit array
 * @param index Index of the bit to clear
 */
void clear_bit(BitArray *bit_array, size_t index);

/**
 * @brief Toggles the bit at the specified index (0->1, 1->0)
 *
 * @param bit_array Pointer to the bit array
 * @param index Index of the bit to toggle
 */
void toggle_bit(BitArray *bit_array, size_t index);

/**
 * @brief Gets the state of the bit at the specified index
 *
 * @param bit_array The bit array
 * @param index Index of the bit to get
 * @return int The bit value (0 or 1), or -1 if index is out of bounds
 */
int get_bit(BitArray bit_array, size_t index);

#endif /* BIT_ARRAY_H */