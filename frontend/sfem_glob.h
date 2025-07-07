#ifndef SFEM_GLOB_H
#define SFEM_GLOB_H

#ifdef __cplusplus
extern "C" {
#endif

size_t count_files(const char *pattern);
int create_directory(const char *path);

#ifdef __cplusplus
}
#endif

#endif