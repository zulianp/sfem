#include "sfem_logger.h"

void log_init(logger_t *l) {
    l->stream = 0;
    l->buffer = 0;
    l->len = 0;
}

int log_is_empty(logger_t *l)
{
    return l->stream == 0;
}

int log_create_memstream(logger_t *l) {
    l->buffer = 0;
    l->len = 0;
    l->stream = open_memstream(&l->buffer, &l->len);
    return !l->stream;
}

int log_create_file(logger_t *l, const char *path, const char *mode) {
    l->buffer = 0;
    l->len = 0;
    l->stream = fopen(path, mode);
    return !l->stream;
}

void log_write_double(logger_t *l, const double val) {
    fprintf(l->stream, "%g\n", val);
    fflush(l->stream);
}

void log_destroy(logger_t *l) {
    if (l->stream) {
        fflush(l->stream);
        fclose(l->stream);
        if (l->buffer) {
            free(l->buffer);
        }
    }
}
