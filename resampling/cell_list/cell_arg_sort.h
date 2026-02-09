#ifndef __CELL_ARG_SORT_H__
#define __CELL_ARG_SORT_H__

#include <stdlib.h>
#include "precision_types.h"

/**
 * @brief Sort indices array based on comparison function applied to base array
 *
 * @param indices Output array of sorted indices (will be modified in-place)
 * @param base Pointer to the array of elements to compare
 * @param n Number of elements
 * @param size Size of each element in bytes
 * @param cmp Comparison function that returns:
 *            - negative if first argument < second argument
 *            - zero if first argument == second argument
 *            - positive if first argument > second argument
 */
void argsort(int *indices, const void *base, size_t n, size_t size, int (*cmp)(const void *, const void *));

/**
 * @brief Find the first position in a sorted array where target can be inserted
 *        without violating the order.
 *
 * @param base Pointer to the sorted array
 * @param nmemb Number of elements in the array
 * @param size Size of each element in bytes
 * @param target Pointer to the target element to search for
 * @param cmp Comparison function that returns:
 *            - negative if first argument < second argument
 *            - zero if first argument == second argument
 *            - positive if first argument > second argument
 * @return int Index of the first position where target can be inserted
 */
int lower_bound_generic(const void *base, size_t nmemb, size_t size, const void *target, int (*cmp)(const void *, const void *));

/**
 * @brief Find the first position in a sorted array where the element is strictly
 *        greater than the target (upper bound).
 *
 * @param base Pointer to the sorted array
 * @param nmemb Number of elements in the array
 * @param size Size of each element in bytes
 * @param target Pointer to the target element to search for
 * @param cmp Comparison function that returns:
 *            - negative if first argument < second argument
 *            - zero if first argument == second argument
 *            - positive if first argument > second argument
 * @return int Index of the first position where element > target
 */
int upper_bound_generic(const void *base, size_t nmemb, size_t size, const void *target, int (*cmp)(const void *, const void *));

/**
 * @brief Specialized upper_bound function for float32 arrays
 * @param elements_array Pointer to the sorted float array
 * @param nmemb Number of elements in the array
 * @param to_search Target float value to search for
 * @return int Index of the first position where element > to_search
 */
int upper_bound_float32(const float *elements_array, size_t nmemb, float to_search);

/**
 * @brief Specialized lower_bound function for float32 arrays
 * @param elements_array Pointer to the sorted float array
 * @param nmemb Number of elements in the array
 * @param to_search Target float value to search for
 * @return int Index of the first position where element >= to_search
 */
int lower_bound_float32(const float *elements_array, size_t nmemb, float to_search);

/**
 * @brief Specialized upper_bound function for float64 arrays
 * @param elements_array Pointer to the sorted double array
 * @param nmemb Number of elements in the array
 * @param to_search Target double value to search for
 * @return int Index of the first position where element > to_search
 */
int upper_bound_float64(const double *elements_array, size_t nmemb, double to_search);

/**
 * @brief Specialized lower_bound function for float64 arrays
 * @param elements_array Pointer to the sorted double array
 * @param nmemb Number of elements in the array
 * @param to_search Target double value to search for
 * @return int Index of the first position where element >= to_search
 */
int lower_bound_float64(const double *elements_array, size_t nmemb, double to_search);

/**
 * @brief Specialized upper_bound function for float64 arrays
 * @param elements_array Pointer to the sorted double array
 * @param nmemb Number of elements in the array
 * @param to_search Target double value to search for
 * @return int Index of the first position where element > to_search
 */
int upper_bound_float(const real_t *elements_array, size_t nmemb, real_t to_search);

/**
 * @brief Specialized lower_bound function for float64 arrays
 * @param elements_array Pointer to the sorted double array
 * @param nmemb Number of elements in the array
 * @param to_search Target double value to search for
 * @return int Index of the first position where element >= to_search
 */
int lower_bound_float(const real_t *elements_array, size_t nmemb, real_t to_search);

#endif  // __CELL_ARG_SORT_H__
