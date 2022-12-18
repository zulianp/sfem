#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

// Compare two integers for qsort
int compare_int(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}

// Partition the array around the pivot element using AVX2 intrinsics
int partition(int *data, int n, int pivot) {
  // Iterate over the array in blocks of 8 elements
  int i = 0;
  int j = n - 8;
  while (i <= j) {
    printf("%d %d\n",i, j);
    // Load the current block of 8 elements into a vector
    __m256i vec = _mm256_loadu_si256((__m256i*)&data[i]);

    // Compare the elements in the vector to the pivot
    __m256i cmp = _mm256_cmpgt_epi32(vec, _mm256_set1_epi32(pivot));

    // Extract the elements that are greater than the pivot
    int mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      printf("swap\n");

      // If there are any elements greater than the pivot, swap them with the last block of 8 elements
      __m256i vec2 = _mm256_loadu_si256((__m256i*)&data[j]);
      _mm256_storeu_si256((__m256i*)&data[i], vec2);
      _mm256_storeu_si256((__m256i*)&data[j], vec);
      j -= 8;
    } else {
      // If all elements are less than or equal to the pivot, move on to the next block
      i += 8;
    }
  }

  // Use the scalar qsort function to sort the remaining elements
  // qsort(&data[i], n - i, sizeof(int), compare_int);
  return i;
}

// Vectorized quicksort using AVX2 intrinsics
void quicksort(int *data, int n) {
  if (n <= 1) return;
  int pivot = data[n / 2];
  int index = partition(data, n, pivot);

  if(index > 0) {
    quicksort(data, index);
    quicksort(&data[index], n - index);
  }
}

int main(int argc, char **argv) {
  int data[8] = {3, 7, 8, 5, 2, 1, 9, 5};
  int n = sizeof(data) / sizeof(int);
  quicksort(data, n);
  for (int i = 0; i < n; i++) {
    printf("%d ", data[i]);
  }
  printf("\n");
  return 0;
}

// Remove duplicates from an array of integers using AVX2 intrinsics
void remove_duplicates(int *data, int n) {
  // Sort the array
  quicksort(data, n);

  // Iterate over the array in blocks of 8 elements
  for (int i = 0; i < n; i += 8) {
    // Load the current block of 8 elements into a vector
    __m256i vec = _mm256_loadu_si256((__m256i*)&data[i]);

    // Compare the elements in the vector to eliminate duplicates
    __m256i cmp = _mm256_cmpeq_epi32(vec, _mm256_set1_epi32(data[i]));
    for (int j = 1; j < 8; j++) {
      __m256i cmp2 = _mm256_cmpeq_epi32(vec, _mm256_set1_epi32(data[i+j]));
      cmp = _mm256_or_si256(cmp, cmp2);
    }

    // Extract the unique elements from the vector
    int mask = _mm256_movemask_epi8(cmp);
    int count = __builtin_popcount(mask);
    if (count > 1) {
      int index = i + 1;
      for (int j = 0; j < 8; j++) {
        if ((mask & (1 << (j * 4))) == 0) {
          data[index++] = data[i+j];
        }
      }
    }
  }
}

// int main(int argc, char **argv) {
//   int data[16] = {1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 8, 9, 3, 3, 5, 6};
//   int n = sizeof(data) / sizeof(int);
//   remove_duplicates(data, n);
//   for (int i = 0; i < n; i++) {
//     printf("%d ", data[i]);
//   }
//   printf("\n");
//   return 0;
// }

