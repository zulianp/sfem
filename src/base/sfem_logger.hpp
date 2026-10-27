#ifndef SFEM_LOGGER_H
#define SFEM_LOGGER_H

#include <stddef.h>
#include <stdio.h>

namespace sfem {

    /// A log of doubles, one per line, to a file or to memory. The stream closes with the logger.
    class Logger {
    public:
        Logger() = default;
        ~Logger();
        Logger(const Logger &)            = delete;
        Logger &operator=(const Logger &) = delete;

        int  open_file(const char *path, const char *mode);
        int  open_memstream();
        void write(const double val);
        bool empty() const { return stream_ == nullptr; }

    private:
        void close();

        FILE  *stream_{nullptr};
        char  *buffer_{nullptr};
        size_t len_{0};
    };

}  // namespace sfem

#endif  // SFEM_LOGGER_H
