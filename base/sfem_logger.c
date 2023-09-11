#include "sfem_logger.h"

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
    fflush(l->stream);
    fclose(l->stream);
    if (l->buffer) {
        free(l->buffer);
    }
}
