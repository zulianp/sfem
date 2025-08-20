/**
 * @file sfem_OpTracer.hpp
 * @brief Performance tracing utilities for operators
 *
 * This file defines the OpTracer class which provides performance monitoring
 * capabilities for SFEM operators. It tracks call counts, timing, and throughput
 * statistics for operator methods.
 */

#pragma once

#include <mpi.h>
#include <memory>
#include <string>

namespace sfem {

    class FunctionSpace;

    /**
     * @brief Performance tracer for operator methods
     *
     * The OpTracer class provides automatic performance monitoring for operator
     * methods. It tracks:
     * - Number of calls
     * - Total execution time
     * - Average time per call
     * - Throughput in MDOF/s (Million Degrees of Freedom per second)
     *
     * Usage:
     * @code
     * OpTracer tracer(space, "MyOperator::apply");
     * {
     *     OpTracer::ScopedCapture capture(tracer);
     *     // ... operator work ...
     * }
     * // Statistics printed automatically on destruction
     * @endcode
     */
    class OpTracer {
    public:
        /**
         * @brief Constructor
         * @param space Function space for DOF count calculation
         * @param name Name of the operation being traced
         */
        OpTracer(const std::shared_ptr<FunctionSpace> &space, const std::string &name);

        /**
         * @brief Destructor
         *
         * Automatically prints performance statistics if calls > 0
         */
        ~OpTracer();

        /**
         * @brief RAII wrapper for automatic timing capture
         *
         * This class automatically captures timing when constructed and
         * updates the tracer statistics when destroyed.
         */
        class ScopedCapture {
        public:
            /**
             * @brief Constructor - starts timing
             * @param profiler Reference to the OpTracer instance
             */
            ScopedCapture(OpTracer &profiler);

            /**
             * @brief Destructor - stops timing and updates statistics
             */
            ~ScopedCapture();

        private:
            OpTracer &profiler;
            double    start_time;
        };

    private:
        std::shared_ptr<FunctionSpace> space;          ///< Function space for DOF calculations
        std::string                    name;           ///< Operation name for reporting
        long                           calls{0};       ///< Number of calls tracked
        double                         total_time{0};  ///< Total execution time in seconds
    };

#if SFEM_PRINT_THROUGHPUT
#define SFEM_OP_CAPTURE() OpTracer::ScopedCapture __sfem_op_capture(*impl_->op_profiler);
#else
#define SFEM_OP_CAPTURE()
#endif

}  // namespace sfem