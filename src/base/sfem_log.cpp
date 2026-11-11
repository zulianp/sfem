#include "sfem_log.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace sfem {

    namespace {
        int     levels[LOG_COMPONENTS] = {LOG_NORMAL, LOG_NORMAL, LOG_NORMAL};
        LogSink sink_fn                = nullptr;
        void   *sink_context           = nullptr;

        const char *level_names[]     = {"quiet", "normal", "verbose", "debug"};
        const char *component_names[] = {"setup", "solid", "contact"};
    }  // namespace

    int log_level(const int component) {
        if (component < 0 || component >= LOG_COMPONENTS) return LOG_NORMAL;
        return levels[component];
    }

    void set_log_level(const int component, const int level) {
        if (component < 0 || component >= LOG_COMPONENTS) return;
        levels[component] = level < LOG_QUIET ? LOG_QUIET : (level > LOG_DEBUG ? LOG_DEBUG : level);
    }

    void set_log_levels(const int level) {
        for (int c = 0; c < LOG_COMPONENTS; c++) set_log_level(c, level);
    }

    bool log_enabled(const int component, const int level) { return level <= log_level(component); }

    void set_log_sink(LogSink sink, void *context) {
        sink_fn      = sink;
        sink_context = context;
    }

    void log_printf(const int component, const int level, const int kind, const char *format, ...) {
        char    small[512];
        char   *text = small;
        va_list args;
        va_start(args, format);
        const int n = vsnprintf(small, sizeof(small), format, args);
        va_end(args);
        if (n < 0) return;
        if ((size_t)n >= sizeof(small)) {
            text = (char *)malloc((size_t)n + 1);
            if (!text) return;
            va_start(args, format);
            vsnprintf(text, (size_t)n + 1, format, args);
            va_end(args);
        }

        // One line per message, without its newline: the sink lays it out.
        size_t len = strlen(text);
        while (len > 0 && text[len - 1] == '\n') text[--len] = '\0';

        if (sink_fn) {
            sink_fn(sink_context, component, level, kind, text);
        } else {
            fputs(text, stdout);
            fputc('\n', stdout);
            fflush(stdout);
        }
        if (text != small) free(text);
    }

    int log_level_from_name(const char *name) {
        for (int l = 0; l < 4; l++) {
            if (!strcmp(name, level_names[l])) return l;
        }
        return -1;
    }

    const char *log_level_name(const int level) { return level >= 0 && level < 4 ? level_names[level] : "?"; }

    int log_component_from_name(const char *name) {
        for (int c = 0; c < LOG_COMPONENTS; c++) {
            if (!strcmp(name, component_names[c])) return c;
        }
        return -1;
    }

    const char *log_component_name(const int component) {
        return component >= 0 && component < LOG_COMPONENTS ? component_names[component] : "?";
    }

}  // namespace sfem
