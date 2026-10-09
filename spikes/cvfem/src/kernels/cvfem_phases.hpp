#ifndef CVFEM_PHASES_HPP
#define CVFEM_PHASES_HPP

// PHASE INSTRUMENTATION, AND HOW TO MAKE IT COST NOTHING.
//
// The packed sweeps report where their time goes -- local memset, gather, kernel, scatter, ghost
// reduction -- which is what --breakdown prints. That instrumentation used to be written inline,
// so a kernel body read
//
//     double _t = phase_now();
//     ...
//     if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }
//
// between its statements. Two things are wrong with that. It is invasive: the probe is as
// prominent as the computation, in a body whose whole job is the computation. And it is only
// switchable at RUNTIME -- g_breakdown guards the clock reads, but the branch, the accumulator
// and the live `_t` are still in the function, and in a kernel whose limit is what fits in
// registers that is not free.
//
// So the probes are macros that expand to nothing when CVFEM_PHASES is 0. A production build --
// the operator library SFEM links, which has no use for a benchmark breakdown -- compiles them
// out entirely: no clock read, no accumulator, no branch, and no parameter carrying the
// accumulator through the signature.
//
// THIS HEADER INCLUDES NOTHING, deliberately. A macro is text, resolved where it is used, so the
// enabled expansion can name PhaseAcc, g_breakdown, wall_time, g_phase and the PH_ enumerators
// without this header depending on whatever defines them -- which is what lets src/kernels/ keep
// the instrumentation and still carry no library dependency. It is the same contract the kernels
// already use for scalar_t: the includer provides the names. DESIGN.md allows this directory
// "localized macros", and this is what that is for.

#ifndef CVFEM_PHASES
#define CVFEM_PHASES 1
#endif

#if CVFEM_PHASES

// The per-thread accumulator, as a declaration and as a parameter. Separate macros because one
// sweep declares it inside its parallel region and hands it to the kernel, which means the
// parameter and the argument have to disappear together with everything else.
#define CVFEM_PHASE_ACC(name) PhaseAcc name
#define CVFEM_PHASE_ACC_PARAM PhaseAcc &acc,
#define CVFEM_PHASE_ACC_ARG   acc,

// Start a clock. phase_now() already returns 0 when the breakdown is off, so this costs a
// compare when enabled and nothing at all when not.
#define CVFEM_PHASE_CLOCK(clk) double clk = phase_now()

// Close one phase and open the next on the same clock.
//
// The clock parameter is `clk` and not `t`, which is not cosmetic: the accumulator's member is
// also called t, so a parameter of that name rewrites `(acc).t[ph]` into `(acc)._t[ph]` at every
// call site. The compiler says "no member named '_t' in 'PhaseAcc'", which points at the
// accumulator rather than at the macro that renamed its member.
#define CVFEM_PHASE_MARK(acc, clk, ph)                   \
    do {                                                 \
        if (g_breakdown) {                               \
            const double _cvfem_now = wall_time();       \
            (acc).t[ph] += _cvfem_now - (clk);           \
            (clk) = _cvfem_now;                          \
        }                                                \
    } while (0)

// Close the last phase of a pack, where there is no next one to open.
#define CVFEM_PHASE_MARK_LAST(acc, clk, ph)                    \
    do {                                                       \
        if (g_breakdown) (acc).t[ph] += wall_time() - (clk);   \
    } while (0)

#define CVFEM_PHASE_FLUSH(acc) (acc).flush()

// A phase measured once per sweep rather than per pack, so it accumulates straight into the
// global rather than through a thread's accumulator.
#define CVFEM_PHASE_GLOBAL(clk, ph)                          \
    do {                                                     \
        if (g_breakdown) g_phase[ph] += wall_time() - (clk); \
    } while (0)

#else  // CVFEM_PHASES == 0

#define CVFEM_PHASE_ACC(name) ((void)0)
#define CVFEM_PHASE_ACC_PARAM
#define CVFEM_PHASE_ACC_ARG
#define CVFEM_PHASE_CLOCK(clk) ((void)0)
#define CVFEM_PHASE_MARK(acc, clk, ph) ((void)0)
#define CVFEM_PHASE_MARK_LAST(acc, clk, ph) ((void)0)
#define CVFEM_PHASE_FLUSH(acc) ((void)0)
#define CVFEM_PHASE_GLOBAL(clk, ph) ((void)0)

#endif  // CVFEM_PHASES

// ------------------------------------------------- the tracer, if the caller has one
//
// SFEM_TRACE_SCOPE comes from sfem_aliases.hpp, which src/kernels/ may not name. The sweeps had
// one each, and the names they trace are worth keeping, so the kernels get their own spelling
// that is nothing unless the including translation unit has said what it means. The front-end
// family headers define it as SFEM_TRACE_SCOPE before they include any kernel, which is the same
// #ifndef shape cvfem_portability.hpp uses for SFEM_HOST_DEVICE, and the clause DESIGN.md allows
// this directory: "localized macros".
#ifndef CVFEM_TRACE_SCOPE
#define CVFEM_TRACE_SCOPE(name) ((void)0)
#endif

#endif  // CVFEM_PHASES_HPP
