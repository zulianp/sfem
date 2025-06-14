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
#include <cuda_runtime.h>
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

            os << "name,calls,total,avg\n";
            for (auto &e : events) {
                os << e.first << "," << e.second.first << "," << e.second.second << "," << e.second.second / e.second.first
                   << "\n";
            }

            os << std::flush;
            os.close();
        }

        bool log_mode{false};
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

    void Tracer::record_event(std::string &&name, const double duration) {
        auto &e = impl_->events[name];
        e.first++;
        e.second += duration;

        if (impl_->log_mode) {
#ifdef SFEM_ENABLE_CUDA
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            printf("-- LOG: %s (%g)\n"
                   "   MEMORY: free %g [GB] (total %g [GB])\n",
                   name.c_str(),
                   duration,
                   free * 1e-9,
                   total * 1e-9);

#else
            printf("-- LOG: %s (%g)\n", name.c_str(), duration);
#endif
            fflush(stdout);
        }
    }

    Tracer::Tracer() : impl_(std::make_unique<Impl>()) {
        int SFEM_ENABLE_LOG = 0;
        SFEM_READ_ENV(SFEM_ENABLE_LOG, atoi);
        impl_->log_mode = SFEM_ENABLE_LOG;
    }

    Tracer::~Tracer() {
        impl_->dump();
        impl_ = nullptr;
    }

    ScopedEvent::ScopedEvent(const char *format, int num) {
        char str[1024];
        int  err = snprintf(str, 1024, format, num);
        if (err < 0) SFEM_ERROR("UNABLE TO TRACE %s\n", format);

        name = str;

#ifdef SFEM_ENABLE_BLOCK_KERNELS
        sfem::device_synchronize();
#endif
#ifdef SFEM_ENABLE_CUDA
        nvtxRangePushA(name.c_str());
#endif
        elapsed = MPI_Wtime();
    }

    ScopedEvent::ScopedEvent(const char *name) : name(name) {
#ifdef SFEM_ENABLE_BLOCK_KERNELS
        sfem::device_synchronize();
#endif
#ifdef SFEM_ENABLE_CUDA
        nvtxRangePushA(this->name.c_str());
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
        Tracer::instance().record_event(std::move(name), elapsed);
    }

}  // namespace sfem
