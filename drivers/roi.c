#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

int main(int argc, const char *argv[]) {
    if (argc != 7) {
        printf("usage %s <nx> <ny> <nz> <data.float32.raw> <xslice> <output.float32.raw>\n",
               argv[0]);
        return EXIT_FAILURE;
    }

    ptrdiff_t nx = atol(argv[1]);
    ptrdiff_t ny = atol(argv[2]);
    ptrdiff_t nz = atol(argv[3]);
    ptrdiff_t xslice = atol(argv[5]);
    ptrdiff_t n = nx * ny * nz;

    float *sdt = malloc(n * sizeof(float));

    {
        FILE *f_sdt = fopen(argv[4], "r");
        ptrdiff_t nread = fread(sdt, sizeof(float), n, f_sdt);
        fclose(f_sdt);

        if (nread != n) {
        	free(sdt);
            return EXIT_FAILURE;
        }
    }

    // Code to extract slice
    ptrdiff_t nslice = ny * nz;
    float *slice = malloc(nslice * sizeof(float));

    for (ptrdiff_t z = 0; z < nz; z++) {
        ptrdiff_t z_offset = z * (ny * nx);
        for (ptrdiff_t y = 0; y < ny; y++) {
            ptrdiff_t idx = z_offset + y * nx + xslice;
            slice[z * ny + y] = sdt[idx];
        }
    }

    {
        FILE *f_slice = fopen(argv[6], "w");
        fwrite(slice, sizeof(float), n, f_slice);
        fclose(f_slice);
    }

    free(sdt);
    free(slice);
    return EXIT_SUCCESS;
}
