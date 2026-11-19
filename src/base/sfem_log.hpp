#ifndef SFEM_LOG_HPP
#define SFEM_LOG_HPP

namespace sfem {

    /// Runtime messages, by component and level. Each component has its own level (all `normal` to
    /// start with); a message is written when its level is at most its component's. Callers test
    /// log_enabled before formatting, and print on the ranks they always printed on.
    ///
    /// Levels:
    ///   quiet    errors only (they go to stderr as before, outside this log)
    ///   normal   one summary per step, and the events: a failed attempt, a retry, a fallback taken
    ///   verbose  and the progress of each step: Newton's iterations
    ///   debug    and the inner solvers' iterations (MPRGP)
    ///
    /// The messages go to stdout, one line each, unless a program sets a sink (set_log_sink), which
    /// then receives each line without its newline, with its component, level and kind, to lay out
    /// as it likes (ldp_fsi indents a coupling interval's solid messages under its row).
    ///
    /// A case sets the levels in its `log` block (Playground): `log: {level: normal, solid: verbose}`.
    enum LogComponent {
        /// Reading the case, building the operators: sizes, choices, timings.
        LOG_SETUP = 0,
        /// The time steps and their nonlinear solves.
        LOG_SOLID = 1,
        /// The contact's QP (MPRGP) and its fallbacks.
        LOG_CONTACT = 2,
        LOG_COMPONENTS = 3
    };

    enum LogLevel { LOG_QUIET = 0, LOG_NORMAL = 1, LOG_VERBOSE = 2, LOG_DEBUG = 3 };

    /// What a message is: a step's summary, an event (a failure, a retry, a fallback), or progress
    /// within a step (an iteration).
    enum LogKind { LOG_SUMMARY = 0, LOG_EVENT = 1, LOG_PROGRESS = 2 };

    /// Receives each message: its text without the trailing newline.
    typedef void (*LogSink)(void *context, const int component, const int level, const int kind, const char *text);

    int  log_level(const int component);
    void set_log_level(const int component, const int level);
    /// Every component at `level`.
    void set_log_levels(const int level);
    bool log_enabled(const int component, const int level);

    /// null: back to stdout.
    void set_log_sink(LogSink sink, void *context);

    /// Writes the message whatever the levels (the caller tested log_enabled): `level` and `kind`
    /// go to the sink with it.
    void log_printf(const int component, const int level, const int kind, const char *format, ...)
#if defined(__GNUC__) || defined(__clang__)
            __attribute__((format(printf, 4, 5)))
#endif
            ;

    /// "quiet", "normal", "verbose", "debug"; -1 for another name.
    int         log_level_from_name(const char *name);
    const char *log_level_name(const int level);
    /// "setup", "solid", "contact"; -1 for another name.
    int         log_component_from_name(const char *name);
    const char *log_component_name(const int component);

}  // namespace sfem

#endif  // SFEM_LOG_HPP
