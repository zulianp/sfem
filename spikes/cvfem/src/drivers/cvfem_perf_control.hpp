#ifndef CVFEM_PERF_CONTROL_HPP
#define CVFEM_PERF_CONTROL_HPP
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <poll.h>
#include <unistd.h>

class CounterPhase {
    using Clock = std::chrono::steady_clock;
    std::string selected;
    struct Channel { int control_fd, ack_fd; };
    std::vector<Channel> channels;
    int transitions = 0;
    Clock::time_point enable_start, enable_done, disable_start, disable_done;

    void command(const std::string &action) {
        const std::string message = action + "\n";
        for (const auto &channel : channels) {
            ssize_t written;
            do { written = write(channel.control_fd, message.data(), message.size()); } while (written < 0 && errno == EINTR);
            if (written != static_cast<ssize_t>(message.size())) throw std::runtime_error("perf control write failed");
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        // Both controllers must finish switching before the timed region proceeds.
        for (const auto &channel : channels) {
            // perf writes sizeof("ack\n"), including the terminating NUL.
            constexpr char expected[] = "ack\n";
            char reply[sizeof(expected)];
            size_t received = 0;
            while (received < sizeof(reply)) {
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();
                if (remaining <= 0) throw std::runtime_error("perf acknowledgement timed out");
                pollfd descriptor{channel.ack_fd, POLLIN, 0};
                const int status = poll(&descriptor, 1, static_cast<int>(remaining));
                if (status < 0 && errno == EINTR) continue;
                if (status <= 0 || !(descriptor.revents & POLLIN)) throw std::runtime_error("perf acknowledgement missing");
                const auto count = read(channel.ack_fd, reply+received, sizeof(reply)-received);
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) throw std::runtime_error("perf acknowledgement read failed");
                received += static_cast<size_t>(count);
            }
            if (std::memcmp(reply, expected, sizeof(expected)) != 0) throw std::runtime_error("invalid perf acknowledgement");
            ++transitions;
        }
    }

public:
    static int descriptor(const std::string &value) {
        size_t used = 0;
        const int fd = std::stoi(value, &used);
        if (used != value.size() || fd < 0) throw std::runtime_error("invalid perf descriptor");
        return fd;
    }
    CounterPhase(const std::string &phase, const std::vector<int> &fds) : selected(phase) {
        if (selected.empty()) {
            if (!fds.empty()) throw std::runtime_error("descriptors without counter phase");
            return;
        }
        if (selected != "setup" && selected != "warmup" && selected != "sweep" &&
            selected != "check" && selected != "teardown" && selected != "empty" && selected != "sweep-empty")
            throw std::runtime_error("unknown counter phase");
        if (fds.size() != 2 && fds.size() != 4) throw std::runtime_error("invalid descriptor count");
        for (size_t i = 0; i < fds.size(); i += 2) {
            const Channel channel{fds[i], fds[i+1]};
            if (channel.control_fd < 0 || channel.ack_fd < 0 || channel.control_fd == channel.ack_fd)
                throw std::runtime_error("invalid perf control descriptors");
            for (const auto &old : channels)
                if (old.control_fd == channel.control_fd || old.ack_fd == channel.control_fd ||
                    old.control_fd == channel.ack_fd || old.ack_fd == channel.ack_fd)
                    throw std::runtime_error("duplicate perf control descriptors");
            channels.push_back(channel);
        }
        std::signal(SIGPIPE, SIG_IGN);
    }
    void begin(const char *phase) {
        if (selected != phase) return;
        enable_start = Clock::now();
        command("enable");
        enable_done = Clock::now();
    }
    void end(const char *phase) {
        if (selected != phase) return;
        disable_start = Clock::now();
        command("disable");
        disable_done = Clock::now();
    }
    void finish() const {
        if (selected.empty()) return;
        if (transitions != 2*static_cast<int>(channels.size()))
            throw std::runtime_error("counter phase did not complete all acknowledgements");
        if (channels.size() == 1)
            std::printf("PERF_PHASE phase=%s transitions=2 PASS\n", selected.c_str());
        else
            std::printf("PERF_PHASE phase=%s controllers=2 transitions=4 PASS\n", selected.c_str());
        // Report request/acknowledgement intervals after counting has stopped.
        std::printf("PERF_CONTROL phase=%s controllers=%zu enable_seconds=%.12g disable_seconds=%.12g inner_seconds=%.12g outer_seconds=%.12g\n",
                    selected.c_str(), channels.size(),
                    std::chrono::duration<double>(enable_done-enable_start).count(),
                    std::chrono::duration<double>(disable_done-disable_start).count(),
                    std::chrono::duration<double>(disable_start-enable_done).count(),
                    std::chrono::duration<double>(disable_done-enable_start).count());
    }
};

#endif
