
#include "cell_tet2box.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MY_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MY_MIN(a, b) (((a) < (b)) ? (a) : (b))

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// calculate_bounding_box_statistics
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
bounding_box_statistics_t  //
calculate_bounding_box_statistics(const boxes_t *boxes) {
    bounding_box_statistics_t stats = {0};

    if (boxes == NULL || boxes->num_boxes <= 0) {
        return stats;
    }  // END if (boxes == NULL || boxes->num_boxes <= 0)

    stats.max_box_side_x = 0.0;
    stats.max_box_side_y = 0.0;
    stats.max_box_side_z = 0.0;

    stats.min_box_side_x = INFINITY;
    stats.min_box_side_y = INFINITY;
    stats.min_box_side_z = INFINITY;

    stats.min_x = INFINITY;
    stats.min_y = INFINITY;
    stats.min_z = INFINITY;

    stats.max_x = -INFINITY;
    stats.max_y = -INFINITY;
    stats.max_z = -INFINITY;

    real_t sum_box_side_x = 0.0;
    real_t sum_box_side_y = 0.0;
    real_t sum_box_side_z = 0.0;

    stats.max_volume     = 0.0;
    stats.min_volume     = INFINITY;
    real_t sum_volume    = 0.0;
    stats.max_volume_idx = -1;
    stats.min_volume_idx = -1;

    for (int i = 0; i < boxes->num_boxes; i++) {
        const real_t side_x = boxes->max_x[i] - boxes->min_x[i];
        const real_t side_y = boxes->max_y[i] - boxes->min_y[i];
        const real_t side_z = boxes->max_z[i] - boxes->min_z[i];

        const real_t volume = side_x * side_y * side_z;

        stats.max_box_side_x = MY_MAX(stats.max_box_side_x, side_x);
        stats.max_box_side_y = MY_MAX(stats.max_box_side_y, side_y);
        stats.max_box_side_z = MY_MAX(stats.max_box_side_z, side_z);

        stats.min_box_side_x = MY_MIN(stats.min_box_side_x, side_x);
        stats.min_box_side_y = MY_MIN(stats.min_box_side_y, side_y);
        stats.min_box_side_z = MY_MIN(stats.min_box_side_z, side_z);

        stats.min_x = MY_MIN(stats.min_x, boxes->min_x[i]);
        stats.min_y = MY_MIN(stats.min_y, boxes->min_y[i]);
        stats.min_z = MY_MIN(stats.min_z, boxes->min_z[i]);

        stats.max_x = MY_MAX(stats.max_x, boxes->max_x[i]);
        stats.max_y = MY_MAX(stats.max_y, boxes->max_y[i]);
        stats.max_z = MY_MAX(stats.max_z, boxes->max_z[i]);

        sum_box_side_x += side_x;
        sum_box_side_y += side_y;
        sum_box_side_z += side_z;

        if (volume > stats.max_volume) {
            stats.max_volume        = volume;
            stats.max_volume_idx    = i;
            stats.max_volume_side_x = side_x;
            stats.max_volume_side_y = side_y;
            stats.max_volume_side_z = side_z;
        }  // END if (volume > stats.max_volume)

        if (volume < stats.min_volume) {
            stats.min_volume        = volume;
            stats.min_volume_idx    = i;
            stats.min_volume_side_x = side_x;
            stats.min_volume_side_y = side_y;
            stats.min_volume_side_z = side_z;
        }  // END if (volume < stats.min_volume)

        sum_volume += volume;
    }  // END: for i

    stats.avg_box_side_x = sum_box_side_x / boxes->num_boxes;
    stats.avg_box_side_y = sum_box_side_y / boxes->num_boxes;
    stats.avg_box_side_z = sum_box_side_z / boxes->num_boxes;
    stats.avg_volume     = sum_volume / boxes->num_boxes;
    stats.volume_ratio   = stats.max_volume / stats.min_volume;

    RETURN_FROM_FUNCTION(stats);
}  // END Function: calculate_bounding_box_statistics

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_bounding_box_statistics
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void print_bounding_box_statistics(const bounding_box_statistics_t *stats) {
    if (stats == NULL) {
        return;
    }  // END if (stats == NULL)

    printf("\n== Bounding box statistics: ==\n");

    printf("Largest bounding box sides: dX=%g, dY=%g, dZ=%g\n",
           (double)stats->max_box_side_x,
           (double)stats->max_box_side_y,
           (double)stats->max_box_side_z);

    printf("Smallest bounding box sides: dX=%g, dY=%g, dZ=%g\n",
           (double)stats->min_box_side_x,
           (double)stats->min_box_side_y,
           (double)stats->min_box_side_z);

    printf("Average bounding box sides: dX=%g, dY=%g, dZ=%g\n",
           (double)stats->avg_box_side_x,
           (double)stats->avg_box_side_y,
           (double)stats->avg_box_side_z);

    printf("Minimum coordinates: X=%g, Y=%g, Z=%g\n", (double)stats->min_x, (double)stats->min_y, (double)stats->min_z);

    printf("Maximum coordinates: X=%g, Y=%g, Z=%g\n", (double)stats->max_x, (double)stats->max_y, (double)stats->max_z);

    printf("Domain side lengths: dX=%g, dY=%g, dZ=%g\n",
           (double)(stats->max_x - stats->min_x),
           (double)(stats->max_y - stats->min_y),
           (double)(stats->max_z - stats->min_z));

    printf("Largest bounding box volume: %g\n", (double)stats->max_volume);
    printf("*  Box index: %d, Sides: dX=%g, dY=%g, dZ=%g\n",
           stats->max_volume_idx,
           (double)stats->max_volume_side_x,
           (double)stats->max_volume_side_y,
           (double)stats->max_volume_side_z);
    printf("Smallest bounding box volume: %g\n", (double)stats->min_volume);
    printf("*  Box index: %d, Sides: dX=%g, dY=%g, dZ=%g\n",
           stats->min_volume_idx,
           (double)stats->min_volume_side_x,
           (double)stats->min_volume_side_y,
           (double)stats->min_volume_side_z);
    printf("Average bounding box volume: %g\n", (double)stats->avg_volume);
    printf("Volume ratio (max/min): %g\n", (double)stats->volume_ratio);

    printf("\n== End of Bounding box statistics: ==\n\n");

}  // END Function: print_bounding_box_statistics

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// write_domain_side_lengths
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int write_domain_side_lengths(const bounding_box_statistics_t *stats, const char *output_dir) {
    if (stats == NULL || output_dir == NULL) {
        fprintf(stderr, "ERROR: Invalid arguments to write_domain_side_lengths\n");
        RETURN_FROM_FUNCTION(EXIT_FAILURE);
    }  // END if (stats == NULL || output_dir == NULL)

    // Construct file path with default filename
    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s/domain_side_lengths.csv", output_dir);

    FILE *fp = fopen(filepath, "w");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: Could not open file %s for writing\n", filepath);
        RETURN_FROM_FUNCTION(EXIT_FAILURE);
    }  // END if (fp == NULL)

    // Write header
    fprintf(fp, "x,y,z\n");

    // Calculate and write domain side lengths
    const real_t domain_x = stats->max_x - stats->min_x;
    const real_t domain_y = stats->max_y - stats->min_y;
    const real_t domain_z = stats->max_z - stats->min_z;

    fprintf(fp, "%.15e,%.15e,%.15e\n", (double)domain_x, (double)domain_y, (double)domain_z);

    fclose(fp);

    printf("Domain side lengths written to: %s\n", filepath);

    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}  // END Function: write_domain_side_lengths

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// calculate_side_length_histograms
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
side_length_histograms_t  //
calculate_side_length_histograms(const boxes_t *boxes, const bounding_box_statistics_t *stats, const int num_classes) {
    side_length_histograms_t histograms = {0};

    if (boxes == NULL || stats == NULL || num_classes <= 0) {
        return histograms;
    }  // END if (boxes == NULL || stats == NULL || num_classes <= 0)

    // Initialize histograms for x, y, z dimensions
    histograms.x_histogram.num_classes = num_classes;
    histograms.x_histogram.min_value   = stats->min_box_side_x;
    histograms.x_histogram.max_value   = stats->max_box_side_x;
    histograms.x_histogram.bin_width   = (stats->max_box_side_x - stats->min_box_side_x) / num_classes;
    histograms.x_histogram.counts      = (int *)malloc(num_classes * sizeof(int));
    memset(histograms.x_histogram.counts, 0, num_classes * sizeof(int));

    histograms.y_histogram.num_classes = num_classes;
    histograms.y_histogram.min_value   = stats->min_box_side_y;
    histograms.y_histogram.max_value   = stats->max_box_side_y;
    histograms.y_histogram.bin_width   = (stats->max_box_side_y - stats->min_box_side_y) / num_classes;
    histograms.y_histogram.counts      = (int *)malloc(num_classes * sizeof(int));
    memset(histograms.y_histogram.counts, 0, num_classes * sizeof(int));

    histograms.z_histogram.num_classes = num_classes;
    histograms.z_histogram.min_value   = stats->min_box_side_z;
    histograms.z_histogram.max_value   = stats->max_box_side_z;
    histograms.z_histogram.bin_width   = (stats->max_box_side_z - stats->min_box_side_z) / num_classes;
    histograms.z_histogram.counts      = (int *)malloc(num_classes * sizeof(int));
    memset(histograms.z_histogram.counts, 0, num_classes * sizeof(int));

    // Populate histograms
    for (int i = 0; i < boxes->num_boxes; i++) {
        const real_t side_x = boxes->max_x[i] - boxes->min_x[i];
        const real_t side_y = boxes->max_y[i] - boxes->min_y[i];
        const real_t side_z = boxes->max_z[i] - boxes->min_z[i];

        // Calculate bin indices for x, y, z
        int bin_x = (int)((side_x - histograms.x_histogram.min_value) / histograms.x_histogram.bin_width);
        int bin_y = (int)((side_y - histograms.y_histogram.min_value) / histograms.y_histogram.bin_width);
        int bin_z = (int)((side_z - histograms.z_histogram.min_value) / histograms.z_histogram.bin_width);

        // Clamp bins to valid range (handle edge case of max value)
        bin_x = (bin_x >= num_classes) ? num_classes - 1 : bin_x;
        bin_y = (bin_y >= num_classes) ? num_classes - 1 : bin_y;
        bin_z = (bin_z >= num_classes) ? num_classes - 1 : bin_z;

        if (bin_x >= 0 && bin_x < num_classes) {
            histograms.x_histogram.counts[bin_x]++;
        }  // END if (bin_x >= 0 && bin_x < num_classes)

        if (bin_y >= 0 && bin_y < num_classes) {
            histograms.y_histogram.counts[bin_y]++;
        }  // END if (bin_y >= 0 && bin_y < num_classes)

        if (bin_z >= 0 && bin_z < num_classes) {
            histograms.z_histogram.counts[bin_z]++;
        }  // END if (bin_z >= 0 && bin_z < num_classes)
    }  // END: for i

    RETURN_FROM_FUNCTION(histograms);
}  // END Function: calculate_side_length_histograms

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_side_length_histograms
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void print_side_length_histograms(const side_length_histograms_t *histograms) {
    if (histograms == NULL) {
        return;
    }  // END if (histograms == NULL)

    printf("\n== Side Length Histograms ==\n\n");

    // Calculate totals for each dimension
    int total_x = 0, total_y = 0, total_z = 0;
    for (int i = 0; i < histograms->x_histogram.num_classes; i++) {
        total_x += histograms->x_histogram.counts[i];
        total_y += histograms->y_histogram.counts[i];
        total_z += histograms->z_histogram.counts[i];
    }  // END: for i

    // Print X-dimension histogram
    printf("X-dimension side lengths histogram (%d classes):\n", histograms->x_histogram.num_classes);
    printf("Range: [%12.5e, %12.5e], Bin width: %12.5e\n",
           (double)histograms->x_histogram.min_value,
           (double)histograms->x_histogram.max_value,
           (double)histograms->x_histogram.bin_width);
    printf(" Bin |      Min        |      Max        |    Count  |    %%   | Cumul Count | Cumul %%\n");
    printf("-----+-----------------+-----------------+-----------+---------+-------------+--------\n");
    int cumul_x = 0;
    for (int i = 0; i < histograms->x_histogram.num_classes; i++) {
        real_t bin_start  = histograms->x_histogram.min_value + i * histograms->x_histogram.bin_width;
        real_t bin_end    = bin_start + histograms->x_histogram.bin_width;
        real_t percentage = (total_x > 0) ? (100.0 * histograms->x_histogram.counts[i] / total_x) : 0.0;
        cumul_x += histograms->x_histogram.counts[i];
        real_t cumul_percentage = (total_x > 0) ? (100.0 * cumul_x / total_x) : 0.0;
        printf(" %3d | %15.5e | %15.5e | %9d | %7.2f | %10d | %7.2f\n",
               i,
               (double)bin_start,
               (double)bin_end,
               histograms->x_histogram.counts[i],
               (double)percentage,
               cumul_x,
               (double)cumul_percentage);
    }  // END: for i

    // Print Y-dimension histogram
    printf("\nY-dimension side lengths histogram (%d classes):\n", histograms->y_histogram.num_classes);
    printf("Range: [%12.5e, %12.5e], Bin width: %12.5e\n",
           (double)histograms->y_histogram.min_value,
           (double)histograms->y_histogram.max_value,
           (double)histograms->y_histogram.bin_width);
    printf(" Bin |      Min        |      Max        |    Count  |    %%   | Cumul Count | Cumul %%\n");
    printf("-----+-----------------+-----------------+-----------+---------+-------------+--------\n");
    int cumul_y = 0;
    for (int i = 0; i < histograms->y_histogram.num_classes; i++) {
        real_t bin_start  = histograms->y_histogram.min_value + i * histograms->y_histogram.bin_width;
        real_t bin_end    = bin_start + histograms->y_histogram.bin_width;
        real_t percentage = (total_y > 0) ? (100.0 * histograms->y_histogram.counts[i] / total_y) : 0.0;
        cumul_y += histograms->y_histogram.counts[i];
        real_t cumul_percentage = (total_y > 0) ? (100.0 * cumul_y / total_y) : 0.0;
        printf(" %3d | %15.5e | %15.5e | %9d | %7.2f | %10d | %7.2f\n",
               i,
               (double)bin_start,
               (double)bin_end,
               histograms->y_histogram.counts[i],
               (double)percentage,
               cumul_y,
               (double)cumul_percentage);
    }  // END: for i

    // Print Z-dimension histogram
    printf("\nZ-dimension side lengths histogram (%d classes):\n", histograms->z_histogram.num_classes);
    printf("Range: [%12.5e, %12.5e], Bin width: %12.5e\n",
           (double)histograms->z_histogram.min_value,
           (double)histograms->z_histogram.max_value,
           (double)histograms->z_histogram.bin_width);
    printf(" Bin |      Min        |      Max        |    Count  |    %%   | Cumul Count | Cumul %%\n");
    printf("-----+-----------------+-----------------+-----------+---------+-------------+--------\n");
    int cumul_z = 0;
    for (int i = 0; i < histograms->z_histogram.num_classes; i++) {
        real_t bin_start  = histograms->z_histogram.min_value + i * histograms->z_histogram.bin_width;
        real_t bin_end    = bin_start + histograms->z_histogram.bin_width;
        real_t percentage = (total_z > 0) ? (100.0 * histograms->z_histogram.counts[i] / total_z) : 0.0;
        cumul_z += histograms->z_histogram.counts[i];
        real_t cumul_percentage = (total_z > 0) ? (100.0 * cumul_z / total_z) : 0.0;
        printf(" %3d | %15.5e | %15.5e | %9d | %7.2f | %10d | %7.2f\n",
               i,
               (double)bin_start,
               (double)bin_end,
               histograms->z_histogram.counts[i],
               (double)percentage,
               cumul_z,
               (double)cumul_percentage);
    }  // END: for i

    printf("\n== End of Side Length Histograms ==\n\n");

}  // END Function: print_side_length_histograms

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// free_side_length_histograms
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void free_side_length_histograms(side_length_histograms_t *histograms) {
    if (histograms == NULL) {
        return;
    }  // END if (histograms == NULL)

    if (histograms->x_histogram.counts != NULL) {
        free(histograms->x_histogram.counts);
        histograms->x_histogram.counts = NULL;
    }  // END if (histograms->x_histogram.counts != NULL)

    if (histograms->y_histogram.counts != NULL) {
        free(histograms->y_histogram.counts);
        histograms->y_histogram.counts = NULL;
    }  // END if (histograms->y_histogram.counts != NULL)

    if (histograms->z_histogram.counts != NULL) {
        free(histograms->z_histogram.counts);
        histograms->z_histogram.counts = NULL;
    }  // END if (histograms->z_histogram.counts != NULL)

}  // END Function: free_side_length_histograms

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// write_side_length_histograms
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int write_side_length_histograms(const side_length_histograms_t *histograms, const char *output_dir) {
    if (histograms == NULL || output_dir == NULL) {
        return EXIT_FAILURE;
    }  // END if (histograms == NULL || output_dir == NULL)

    // Calculate totals for each dimension
    int total_x = 0, total_y = 0, total_z = 0;
    for (int i = 0; i < histograms->x_histogram.num_classes; i++) {
        total_x += histograms->x_histogram.counts[i];
        total_y += histograms->y_histogram.counts[i];
        total_z += histograms->z_histogram.counts[i];
    }  // END: for i

    // Write X-dimension histogram
    {
        char filepath[4096];
        snprintf(filepath, sizeof(filepath), "%s/x_histogram.csv", output_dir);
        FILE *fp = fopen(filepath, "w");
        if (fp == NULL) {
            fprintf(stderr, "ERROR: Could not open file %s for writing\n", filepath);
            return EXIT_FAILURE;
        }  // END if (fp == NULL)

        fprintf(fp, "Min,Max,PDF,CDF,Count,Cumul_Count\n");
        int cumul_x = 0;
        for (int i = 0; i < histograms->x_histogram.num_classes; i++) {
            real_t bin_start = histograms->x_histogram.min_value + i * histograms->x_histogram.bin_width;
            real_t bin_end   = bin_start + histograms->x_histogram.bin_width;
            real_t pdf       = (total_x > 0) ? ((real_t)histograms->x_histogram.counts[i] / total_x) : 0.0;
            cumul_x += histograms->x_histogram.counts[i];
            real_t cdf = (total_x > 0) ? ((real_t)cumul_x / total_x) : 0.0;
            fprintf(fp,
                    "%.15e,%.15e,%.15e,%.15e,%d,%d\n",
                    (double)bin_start,
                    (double)bin_end,
                    (double)pdf,
                    (double)cdf,
                    histograms->x_histogram.counts[i],
                    cumul_x);
        }  // END: for i
        fclose(fp);
    }  // END write X-dimension histogram

    // Write Y-dimension histogram
    {
        char filepath[4096];
        snprintf(filepath, sizeof(filepath), "%s/y_histogram.csv", output_dir);
        FILE *fp = fopen(filepath, "w");
        if (fp == NULL) {
            fprintf(stderr, "ERROR: Could not open file %s for writing\n", filepath);
            return EXIT_FAILURE;
        }  // END if (fp == NULL)

        fprintf(fp, "Min,Max,PDF,CDF,Count,Cumul_Count\n");
        int cumul_y = 0;
        for (int i = 0; i < histograms->y_histogram.num_classes; i++) {
            real_t bin_start = histograms->y_histogram.min_value + i * histograms->y_histogram.bin_width;
            real_t bin_end   = bin_start + histograms->y_histogram.bin_width;
            real_t pdf       = (total_y > 0) ? ((real_t)histograms->y_histogram.counts[i] / total_y) : 0.0;
            cumul_y += histograms->y_histogram.counts[i];
            real_t cdf = (total_y > 0) ? ((real_t)cumul_y / total_y) : 0.0;
            fprintf(fp,
                    "%.15e,%.15e,%.15e,%.15e,%d,%d\n",
                    (double)bin_start,
                    (double)bin_end,
                    (double)pdf,
                    (double)cdf,
                    histograms->y_histogram.counts[i],
                    cumul_y);
        }  // END: for i
        fclose(fp);
    }  // END write Y-dimension histogram

    // Write Z-dimension histogram
    {
        char filepath[4096];
        snprintf(filepath, sizeof(filepath), "%s/z_histogram.csv", output_dir);
        FILE *fp = fopen(filepath, "w");
        if (fp == NULL) {
            fprintf(stderr, "ERROR: Could not open file %s for writing\n", filepath);
            return EXIT_FAILURE;
        }  // END if (fp == NULL)

        fprintf(fp, "Min,Max,PDF,CDF,Count,Cumul_Count\n");
        int cumul_z = 0;
        for (int i = 0; i < histograms->z_histogram.num_classes; i++) {
            real_t bin_start = histograms->z_histogram.min_value + i * histograms->z_histogram.bin_width;
            real_t bin_end   = bin_start + histograms->z_histogram.bin_width;
            real_t pdf       = (total_z > 0) ? ((real_t)histograms->z_histogram.counts[i] / total_z) : 0.0;
            cumul_z += histograms->z_histogram.counts[i];
            real_t cdf = (total_z > 0) ? ((real_t)cumul_z / total_z) : 0.0;
            fprintf(fp,
                    "%.15e,%.15e,%.15e,%.15e,%d,%d\n",
                    (double)bin_start,
                    (double)bin_end,
                    (double)pdf,
                    (double)cdf,
                    histograms->z_histogram.counts[i],
                    cumul_z);
        }  // END: for i
        fclose(fp);
    }  // END write Z-dimension histogram

    printf("Histograms written to:\n");
    printf("  - %s/x_histogram.csv\n", output_dir);
    printf("  - %s/y_histogram.csv\n", output_dir);
    printf("  - %s/z_histogram.csv\n", output_dir);

    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}  // END Function: write_side_length_histograms

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// calculate_threshold_for_histogram (static helper)
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static real_t  //
calculate_threshold_for_histogram(const side_length_histogram_t *histogram, real_t cdf_ratio) {
    if (histogram == NULL || histogram->counts == NULL) {
        return 0.0;
    }  // END if (histogram == NULL || histogram->counts == NULL)

    // Calculate total count
    int total = 0;
    for (int i = 0; i < histogram->num_classes; i++) {
        total += histogram->counts[i];
    }  // END: for i

    // Find threshold where CDF exceeds cdf_ratio
    int    cumul     = 0;
    real_t threshold = histogram->max_value;  // Default to max
    for (int i = 0; i < histogram->num_classes; i++) {
        cumul += histogram->counts[i];
        real_t cdf = (total > 0) ? ((real_t)cumul / total) : 0.0;

        if (cdf >= cdf_ratio) {
            real_t bin_start = histogram->min_value + i * histogram->bin_width;
            // Linear interpolation within the bin
            real_t local_ratio =
                    (cdf_ratio - (cdf - (real_t)histogram->counts[i] / total)) / ((real_t)histogram->counts[i] / total);
            local_ratio = (local_ratio < 0.0) ? 0.0 : (local_ratio > 1.0) ? 1.0 : local_ratio;
            threshold   = bin_start + local_ratio * histogram->bin_width;
            break;
        }  // END if (cdf >= cdf_ratio)
    }  // END: for i

    RETURN_FROM_FUNCTION(threshold);
}  // END Function: calculate_threshold_for_histogram

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// calculate_cdf_thresholds
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
side_length_cdf_thresholds_t  //
calculate_cdf_thresholds(const side_length_histograms_t *histograms, real_t cdf_ratio_x, real_t cdf_ratio_y, real_t cdf_ratio_z) {
    side_length_cdf_thresholds_t thresholds = {0};

    if (histograms == NULL) {
        return thresholds;
    }  // END if (histograms == NULL)

    // Clamp ratios to [0, 1] range
    cdf_ratio_x = (cdf_ratio_x < 0.0) ? 0.0 : (cdf_ratio_x > 1.0) ? 1.0 : cdf_ratio_x;
    cdf_ratio_y = (cdf_ratio_y < 0.0) ? 0.0 : (cdf_ratio_y > 1.0) ? 1.0 : cdf_ratio_y;
    cdf_ratio_z = (cdf_ratio_z < 0.0) ? 0.0 : (cdf_ratio_z > 1.0) ? 1.0 : cdf_ratio_z;

    thresholds.cdf_ratio = cdf_ratio_x;  // Store the ratio (assuming same for all dimensions)

    // Calculate thresholds for each dimension
    thresholds.threshold_x = calculate_threshold_for_histogram(&histograms->x_histogram, cdf_ratio_x);
    thresholds.threshold_y = calculate_threshold_for_histogram(&histograms->y_histogram, cdf_ratio_y);
    thresholds.threshold_z = calculate_threshold_for_histogram(&histograms->z_histogram, cdf_ratio_z);

    RETURN_FROM_FUNCTION(thresholds);
}  // END Function: calculate_cdf_thresholds

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_cdf_thresholds
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void print_cdf_thresholds(const side_length_cdf_thresholds_t *thresholds) {
    if (thresholds == NULL) {
        return;
    }  // END if (thresholds == NULL)

    printf("\n== CDF-based Thresholds (CDF ratio: %.4f) ==\n", (double)thresholds->cdf_ratio);
    printf("Threshold X: %g\n", (double)thresholds->threshold_x);
    printf("Threshold Y: %g\n", (double)thresholds->threshold_y);
    printf("Threshold Z: %g\n", (double)thresholds->threshold_z);
    printf("== End of CDF-based Thresholds ==\n\n");

}  // END Function: print_cdf_thresholds

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// make_mesh_tets_boxes
///////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
int                                                                     //
make_mesh_tets_boxes(const ptrdiff_t                    start_element,  //
                     const ptrdiff_t                    end_element,    //
                     const ptrdiff_t                    nnodes,         //
                     const idx_t **const SFEM_RESTRICT  elems,          //
                     const geom_t **const SFEM_RESTRICT xyz,            //
                     boxes_t                          **boxes) {                                 //

    PRINT_CURRENT_FUNCTION;

    const ptrdiff_t num_elements = end_element - start_element;

    // Allocate memory for boxes
    boxes_t *const SFEM_RESTRICT boxes_loc_ptr = allocate_boxes_t((int)num_elements);

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        idx_t ev[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }  // END: for vq

#if SFEM_LOG_LEVEL > 5
        if (element_i % 1000000 == 0) {
            printf("*** Processing element %td / %td \n", element_i, end_element);
        }
#endif

        // Read the coordinates of the vertices of the tetrahedron
        // In the physical space
        const real_t x0_n = xyz[0][ev[0]];
        const real_t x1_n = xyz[0][ev[1]];
        const real_t x2_n = xyz[0][ev[2]];
        const real_t x3_n = xyz[0][ev[3]];

        const real_t y0_n = xyz[1][ev[0]];
        const real_t y1_n = xyz[1][ev[1]];
        const real_t y2_n = xyz[1][ev[2]];
        const real_t y3_n = xyz[1][ev[3]];

        const real_t z0_n = xyz[2][ev[0]];
        const real_t z1_n = xyz[2][ev[1]];
        const real_t z2_n = xyz[2][ev[2]];
        const real_t z3_n = xyz[2][ev[3]];

        const real_t min_x = MY_MIN(MY_MIN(x0_n, x1_n), MY_MIN(x2_n, x3_n));
        const real_t max_x = MY_MAX(MY_MAX(x0_n, x1_n), MY_MAX(x2_n, x3_n));

        const real_t min_y = MY_MIN(MY_MIN(y0_n, y1_n), MY_MIN(y2_n, y3_n));
        const real_t max_y = MY_MAX(MY_MAX(y0_n, y1_n), MY_MAX(y2_n, y3_n));

        const real_t min_z = MY_MIN(MY_MIN(z0_n, z1_n), MY_MIN(z2_n, z3_n));
        const real_t max_z = MY_MAX(MY_MAX(z0_n, z1_n), MY_MAX(z2_n, z3_n));

        const ptrdiff_t box_index = element_i - start_element;

        boxes_loc_ptr->min_x[box_index] = min_x;
        boxes_loc_ptr->max_x[box_index] = max_x;
        boxes_loc_ptr->min_y[box_index] = min_y;
        boxes_loc_ptr->max_y[box_index] = max_y;
        boxes_loc_ptr->min_z[box_index] = min_z;
        boxes_loc_ptr->max_z[box_index] = max_z;

    }  // END: for element_i

    *boxes = (boxes_t *)boxes_loc_ptr;

    bounding_box_statistics_t stats = calculate_bounding_box_statistics(boxes_loc_ptr);
    // print_bounding_box_statistics(&stats);

    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}  // END: Function: make_mesh_tets_boxes