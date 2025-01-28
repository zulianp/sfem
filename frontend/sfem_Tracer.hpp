#ifndef SFEM_TRACER_HPP
#define SFEM_TRACER_HPP

#include "sfem_base.h"

#include <memory>

namespace sfem {
	class Tracer {
	public:
		static Tracer &instance();
		void record_event(const char *name, const double duration);
		Tracer();
		~Tracer();
	private:
		class Impl;
		std::unique_ptr<Impl> impl_;
	};

	class ScopedEvent {
	public:
		const char *name{nullptr};
		double elapsed{0};
		ScopedEvent(const char *name);
		~ScopedEvent();
	};
}

#ifdef SFEM_ENABLE_TRACE
#define SFEM_TRACE_SCOPE(name) sfem::ScopedEvent sfem_scoped_trace_event_(name);
#else
#define SFEM_TRACE_SCOPE(...)
#endif

#endif //SFEM_TRACER_HPP
