#include "sfem_Tracer.hpp"

#include "sfem_base.h"

#include "sfem_API.hpp"

#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <map>

#ifdef SFEM_ENABLE_CUDA
#include <nvToolsExt.h>
#endif

// #define SFEM_ENABLE_BLOCK_KERNELS

namespace sfem {
    class Tracer::Impl {
    public:
        std::map<std::string, std::pair<int, double>> events;

        void dump() {
            const char *SFEM_TRACE_FILE = "sfem.trace.csv";
            SFEM_READ_ENV(SFEM_TRACE_FILE, );

            std::ofstream os(SFEM_TRACE_FILE);

            if (!os.good()) {
                SFEM_ERROR("Unable to write trace file!\n");
            }

            // printf("Writing trace file at %s\n", SFEM_TRACE_FILE);

            os << "name,calls,total,avg\n";
            for (auto &e : events) {
                os << e.first << "," << e.second.first << "," << e.second.second << "," << e.second.second / e.second.first
                   << "\n";
            }

            os << std::flush;
            os.close();
        }
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
        impl_->dump();
        impl_ = nullptr;
    }

    ScopedEvent::ScopedEvent(const char *name) : name(name) {
#ifdef SFEM_ENABLE_BLOCK_KERNELS
        sfem::device_synchronize();
#endif
#ifdef SFEM_ENABLE_CUDA
        nvtxRangePushA(name);
#endif
        elapsed = MPI_Wtime();
    }

    ScopedEvent::~ScopedEvent() {
#ifdef SFEM_ENABLE_BLOCK_KERNELS
        sfem::device_synchronize();
#endif

        elapsed = MPI_Wtime() - elapsed;
#ifdef SFEM_ENABLE_CUDA
        nvtxRangePop();
#endif
        Tracer::instance().record_event(name, elapsed);
    }

}  // namespace sfem
