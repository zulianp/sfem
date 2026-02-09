#include <stdlib.h>

#include "cell_list_3d_map.h"

////////////////////////////////////////////////
// swap
////////////////////////////////////////////////
static void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

////////////////////////////////////////////////
// cmp_idx
////////////////////////////////////////////////
static int cmp_idx(int i, int j, const void *base, size_t size,
                   int (*cmp)(const void *, const void *))
{
    const char *pa = (const char *)base + i * size;
    const char *pb = (const char *)base + j * size;
    return cmp(pa, pb);
}

////////////////////////////////////////////////
// quicksort_rec
////////////////////////////////////////////////
static void quicksort_rec(int *idx, int lo, int hi, const void *base,
                          size_t size, int (*cmp)(const void *, const void *))
{
    if (lo >= hi)
        return;

    // Handle small arrays with insertion sort
    if (hi - lo < 10)
    {
        for (int i = lo + 1; i <= hi; i++)
        {
            int tmp = idx[i];
            int j = i - 1;
            while (j >= lo && cmp_idx(idx[j], tmp, base, size, cmp) > 0)
            {
                idx[j + 1] = idx[j];
                j--;
            }
            idx[j + 1] = tmp;
        }
        return;
    }

    // Median-of-three pivot
    int mid = lo + (hi - lo) / 2;
    if (cmp_idx(idx[mid], idx[lo], base, size, cmp) < 0)
        swap(&idx[mid], &idx[lo]);
    if (cmp_idx(idx[hi], idx[lo], base, size, cmp) < 0)
        swap(&idx[hi], &idx[lo]);
    if (cmp_idx(idx[hi], idx[mid], base, size, cmp) < 0)
        swap(&idx[hi], &idx[mid]);

    int pivot = idx[mid];
    swap(&idx[mid], &idx[hi - 1]);

    int i = lo, j = hi - 1;
    while (1)
    {
        while (cmp_idx(idx[++i], pivot, base, size, cmp) < 0)
            ;
        while (cmp_idx(idx[--j], pivot, base, size, cmp) > 0)
            ;
        if (i >= j)
            break;
        swap(&idx[i], &idx[j]);
    }
    swap(&idx[i], &idx[hi - 1]);

    quicksort_rec(idx, lo, i - 1, base, size, cmp);
    quicksort_rec(idx, i + 1, hi, base, size, cmp);
}

////////////////////////////////////////////////
// argsort
////////////////////////////////////////////////
void argsort(int *indices, const void *base, size_t n, size_t size,
             int (*cmp)(const void *, const void *))
{
    for (size_t i = 0; i < n; i++)
        indices[i] = (int)i;

    if (n > 1)
        quicksort_rec(indices, 0, (int)n - 1, base, size, cmp);
}

////////////////////////////////////////////////
// lower_bound_generic
////////////////////////////////////////////////
int lower_bound_generic(const void *elements_array, size_t nmemb, size_t size,
                        const void *to_search,
                        int (*cmp)(const void *, const void *))
{
    int low = 0, high = (int)nmemb;
    const char *ptr = (const char *)elements_array;

    while (low < high)
    {
        const int mid = low + (high - low) / 2;
        const void *mid_elem = ptr + (mid * size);

        const int cmp_result = cmp(mid_elem, to_search);

        if (cmp_result < 0)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

////////////////////////////////////////////////
// upper_bound_generic
////////////////////////////////////////////////
int upper_bound_generic(const void *elements_array, size_t nmemb, size_t size,
                        const void *to_search,
                        int (*cmp)(const void *, const void *))
{
    int low = 0, high = (int)nmemb;
    const char *ptr = (const char *)elements_array;

    while (low < high)
    {
        const int mid = low + (high - low) / 2;
        const void *mid_elem = ptr + (mid * size);

        // upper_bound: find first element > target
        // so we continue searching right if element <= target

        const int cmp_result = cmp(mid_elem, to_search);

        if (cmp_result <= 0)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

////////////////////////////////////////////////
// upper_bound_float32
////////////////////////////////////////////////
int upper_bound_float32(const float *elements_array, size_t nmemb,
                        float to_search)
{
    int low = 0, high = (int)nmemb;

    while (low < high)
    {
        const int mid = low + (high - low) / 2;

        if (elements_array[mid] <= to_search)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

////////////////////////////////////////////////
// lower_bound_float32
////////////////////////////////////////////////
int lower_bound_float32(const float *elements_array, size_t nmemb,
                        float to_search)
{
    int low = 0, high = (int)nmemb;

    while (low < high)
    {
        int mid = low + (high - low) / 2;

        if (elements_array[mid] < to_search)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

////////////////////////////////////////////////
// upper_bound_float64
////////////////////////////////////////////////
int upper_bound_float64(const double *elements_array, size_t nmemb,
                        double to_search)
{
    int low = 0, high = (int)nmemb;

    while (low < high)
    {
        const int mid = low + (high - low) / 2;

        // Direct float comparison eliminates function pointer overhead
        if (elements_array[mid] <= to_search)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

////////////////////////////////////////////////
// lower_bound_float64
////////////////////////////////////////////////
int lower_bound_float64(const double *elements_array, size_t nmemb,
                        double to_search)
{
    int low = 0, high = (int)nmemb;

    while (low < high)
    {
        int mid = low + (high - low) / 2;

        // Direct float comparison eliminates function pointer overhead
        if (elements_array[mid] < to_search)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

////////////////////////////////////////////////
// upper_bound_float
////////////////////////////////////////////////
int upper_bound_float(const real_t *elements_array, size_t nmemb,
                      real_t to_search)
{
#ifdef USE_SINGLE_PRECISION
    return upper_bound_float32((const float *)elements_array, nmemb, (float)to_search);
#elif defined(USE_DOUBLE_PRECISION)
    return upper_bound_float64(elements_array, nmemb, to_search);
#else
#pragma error "Precision type not defined. Define either USE_SINGLE_PRECISION or USE_DOUBLE_PRECISION."
#endif
}

////////////////////////////////////////////////
// lower_bound_float
////////////////////////////////////////////////
int lower_bound_float(const real_t *elements_array, size_t nmemb,
                      real_t to_search)
{
#ifdef USE_SINGLE_PRECISION
    return lower_bound_float32((const float *)elements_array, nmemb, (float)to_search);
#elif defined(USE_DOUBLE_PRECISION)
    return lower_bound_float64(elements_array, nmemb, to_search);
#else
#pragma error "Precision type not defined. Define either USE_SINGLE_PRECISION or USE_DOUBLE_PRECISION."
#endif
}
