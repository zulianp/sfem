#ifndef SFEM_TRACER_HPP
#define SFEM_TRACER_HPP

#include "sfem_base.h"

#include <memory>
#include <string>

namespace sfem {
    class Tracer {
    public:
        static Tracer &instance();
        void           record_event(const char *name, const double duration);
        void           record_event(std::string &&name, const double duration);
        Tracer();
        ~Tracer();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class ScopedEvent {
    public:
        // const char *name{nullptr};
        std::string name;
        double      elapsed{0};
        ScopedEvent(const char *name);
        ScopedEvent(const char *format, int num);
        ~ScopedEvent();
    };
}  // namespace sfem

#ifdef SFEM_ENABLE_TRACE
#define SFEM_TRACE_SCOPE(name) sfem::ScopedEvent sfem_scoped_trace_event_(name);
#define SFEM_TRACE_SCOPE_VARIANT(format__, num__) sfem::ScopedEvent sfem_scoped_trace_event_(format__, num__);
#else
#define SFEM_TRACE_SCOPE(...)
#define SFEM_TRACE_SCOPE_VARIANT(...)
#endif

#endif  // SFEM_TRACER_HPP
