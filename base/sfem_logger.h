#ifndef SFEM_LOGGER_H
#define SFEM_LOGGER_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FILE *stream;
    char *buffer;
    size_t len;
    // off_t eob;
} logger_t;

void log_write_double(logger_t *l, const double val);
int log_create_memstream(logger_t *l);
int log_create_file(logger_t *l, const char *path, const char *mode);
void log_destroy(logger_t *l);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_LOGGER_H
