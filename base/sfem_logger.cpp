#include "sfem_logger.hpp"
#include "sfem_base.hpp"

#include <stdlib.h>

namespace sfem {

    Logger::~Logger() { close(); }

    void Logger::close() {
        if (stream_) {
            fflush(stream_);
            fclose(stream_);
            if (buffer_) {
                free(buffer_);
            }
        }
        stream_ = nullptr;
        buffer_ = nullptr;
        len_    = 0;
    }

    int Logger::open_memstream() {
        close();
#ifdef _WIN32
        SFEM_ERROR("IMPLEMENT ME!");
        return SFEM_FAILURE;
#else
        stream_ = ::open_memstream(&buffer_, &len_);
        return stream_ ? SFEM_SUCCESS : SFEM_FAILURE;
#endif
    }

    int Logger::open_file(const char *path, const char *mode) {
        close();
        stream_ = fopen(path, mode);
        return stream_ ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    void Logger::write(const double val) {
        fprintf(stream_, "%g\n", val);
        fflush(stream_);
    }

}  // namespace sfem
