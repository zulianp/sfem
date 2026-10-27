#include "sfem_OpTracer.hpp"
#include "smesh_base.hpp"
#include "sfem_FunctionSpace.hpp"

namespace sfem {

    OpTracer::OpTracer(const std::shared_ptr<FunctionSpace> &space, const std::string &name) : space(space), name(name) {}

    OpTracer::~OpTracer() {
        if (calls) {
            printf("%s called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   name.c_str(),
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    OpTracer::ScopedCapture::ScopedCapture(OpTracer &profiler) : profiler(profiler) { start_time = smesh::time_seconds(); }

    OpTracer::ScopedCapture::~ScopedCapture() {
        double end_time = smesh::time_seconds();
        profiler.calls++;
        profiler.total_time += end_time - start_time;
    }

}  // namespace sfem
