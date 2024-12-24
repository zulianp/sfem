#include "sfem_Tracer.hpp"

#include "sfem_base.h"
#include <mpi.h>

#include <map>
#include <fstream>
#include <cassert>
#include <cstdio>


namespace sfem {
    class Tracer::Impl {
    public:
        std::map<std::string, std::pair<int, double>> events;
    };

    Tracer &Tracer::instance() {
        static Tracer instance_;
        return instance_;
    }

    void Tracer::record_event(const char *name, const double duration) {
        auto &e = impl_->events[name];
        e.first++;
        e.second += duration;
    }

    Tracer::Tracer() : impl_(std::make_unique<Impl>()) {}

    Tracer::~Tracer() {
    	const char * SFEM_TRACE_FILE = "sfem.trace.csv";
    	SFEM_READ_ENV(SFEM_TRACE_FILE, );

        std::ofstream os(SFEM_TRACE_FILE);

        if (!os.good()) {
            SFEM_ERROR("Unable to write trace file!\n");
        }

        os << "name,calls,total,avg;\n";
        for (auto &e : impl_->events) {
            os << e.first << "," << e.second.first << "," << e.second.second << "," << e.second.second/e.second.first <<";\n";
        }

        os.close();
    }

    ScopedEvent::ScopedEvent(const char *name) : name(name), elapsed(MPI_Wtime()) {}
    ScopedEvent::~ScopedEvent() {
        elapsed = MPI_Wtime() - elapsed;
        Tracer::instance().record_event(name, elapsed);
    }

}  // namespace sfem
