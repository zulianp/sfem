#include "sfem_RelaxationGroups.hpp"
#include "sfem_HistoryDependent.hpp"
#include "sfem_StateFiles.hpp"
#include "sfem_prony_buffers.hpp"

#include <cassert>
#include <utility>

#include "sfem_blas_factory.hpp"
#include "sfem_logger.hpp"

#ifdef SFEM_ENABLE_CUDA
#include <cuda_runtime_api.h>

#include "sfem_prony_kernels_cuda.hpp"
#endif

namespace sfem {

    RelaxationGroups::RelaxationGroups(std::vector<RelaxationGroupDeclaration> declarations,
                                       const ptrdiff_t                         n_dofs,
                                       const ExecutionSpace                    es)
        : declarations_(std::move(declarations)), n_dofs_(n_dofs), es_(es) {}

    int RelaxationGroups::group_index(const std::string &group) const {
        for (size_t i = 0; i < declarations_.size(); i++) {
            if (declarations_[i].group == group) return static_cast<int>(i);
        }
        SFEM_ERROR("RelaxationGroups: the material declares no relaxation group \"%s\"\n", group.c_str());
        return -1;
    }

    int RelaxationGroups::history_index(const std::string &history) const {
        for (size_t i = 0; i < declarations_.size(); i++) {
            if (declarations_[i].history == history) return static_cast<int>(i);
        }
        SFEM_ERROR("RelaxationGroups: no relaxation group has the history \"%s\"\n", history.c_str());
        return -1;
    }

    int RelaxationGroups::stress_index(const std::string &stress) const {
        for (size_t i = 0; i < declarations_.size(); i++) {
            if (declarations_[i].stress == stress) return static_cast<int>(i);
        }
        SFEM_ERROR("RelaxationGroups: no relaxation group has the stress output \"%s\"\n", stress.c_str());
        return -1;
    }

    RelaxationGroups::Block &RelaxationGroups::block(const std::string &name) {
        auto it = blocks_.find(name);
        if (it == blocks_.end()) {
            SFEM_ERROR("RelaxationGroups: unknown block \"%s\"\n", name.c_str());
        }
        return it->second;
    }

    const RelaxationGroups::Block &RelaxationGroups::block(const std::string &name) const {
        auto it = blocks_.find(name);
        if (it == blocks_.end()) {
            SFEM_ERROR("RelaxationGroups: unknown block \"%s\"\n", name.c_str());
        }
        return it->second;
    }

    void RelaxationGroups::add_block(const std::string &name, const ptrdiff_t n_elements, const int n_qp) {
        Block b;
        b.n_elements = n_elements;
        b.n_qp       = n_qp;
        PronyRelaxation::Options options;
        options.execution_space = es_;
        options.stream          = stream_;
        for (const auto &d : declarations_) {
            b.groups.push_back(std::make_shared<PronyRelaxation>(PronySeries(), d.n_components, n_qp, n_elements, options));
        }
        blocks_[name] = std::move(b);
    }

    void RelaxationGroups::set_series(const std::string              &block_name,
                                      const std::string              &group,
                                      const PronySeries              &series,
                                      const PronyRelaxation::Options &options) {
        const int g = group_index(group);
        // The state lives where the operator's vectors do, whatever the caller asked.
        PronyRelaxation::Options placed = options;
        placed.execution_space          = es_;
        placed.stream                   = stream_;
        for (auto &entry : blocks_) {
            if (!block_name.empty() && entry.first != block_name) continue;
            Block &b = entry.second;
            // Released first, so the old state and the new are never held together.
            b.groups[g].reset();
            b.groups[g] = std::make_shared<PronyRelaxation>(
                    series, declarations_[g].n_components, b.n_qp, b.n_elements, placed);
        }
        if (!block_name.empty()) block(block_name);  // an unknown name is an error, not a no-op
        norms_current_ = false;
        // The new series holds nothing yet; another group or block may still hold a term on the
        // last step's increment, which reads the displacement that step started from.
        bool held = false;
        for (const auto &entry : blocks_) {
            for (const auto &other : entry.second.groups) held = held || other->holds_coefficients();
        }
        if (!held) previous_.reset();
    }

    void RelaxationGroups::set_temperature(const std::string &block_name, const std::string &group, const real_t temperature) {
        const int g = group_index(group);
        for (auto &entry : blocks_) {
            if (!block_name.empty() && entry.first != block_name) continue;
            entry.second.groups[g]->set_temperature(temperature);
        }
    }

    PronyRelaxation &RelaxationGroups::relaxation(const std::string &block_name, const std::string &group) {
        return *block(block_name).groups[group_index(group)];
    }

    const PronyRelaxation &RelaxationGroups::relaxation(const std::string &block_name, const std::string &group) const {
        return *block(block_name).groups[group_index(group)];
    }

    real_t RelaxationGroups::gamma(const std::string &block_name, const std::string &group) const {
        return relaxation(block_name, group).gamma();
    }

    const real_t *RelaxationGroups::history(const std::string &block_name, const std::string &history_name) const {
        return block(block_name).groups[history_index(history_name)]->history();
    }

    void RelaxationGroups::set_stream(void *const stream) {
        stream_ = stream;
        for (auto &entry : blocks_) {
            for (auto &group : entry.second.groups) group->set_stream(stream);
        }
    }

    bool RelaxationGroups::narrow(const Block &b, const size_t group) const {
#if SFEM_REAL_T_IS_FLOAT64
        return b.groups[group]->options().stress_storage == HistoryStorage::FLOAT32;
#else
        // real_t is float already: a FLOAT32 slot is a real_t one.
        (void)b;
        (void)group;
        return false;
#endif
    }

    void RelaxationGroups::evaluate(const real_t *const x, const int slot) {
        if (!evaluate_) {
            SFEM_ERROR("RelaxationGroups: no stress evaluation; the operator sets it (set_stress_evaluation)\n");
        }
        // The slot, for every block and group, and the float32 groups' real_t scratch, for this
        // evaluation only.
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            if (b.stress.empty()) b.stress.assign(2, std::vector<Block::Slot>(declarations_.size()));
            b.scratch.assign(declarations_.size(), nullptr);
            for (size_t g = 0; g < declarations_.size(); g++) {
                const ptrdiff_t n = static_cast<ptrdiff_t>(declarations_[g].n_components) * b.n_qp * b.n_elements;
                if (narrow(b, g)) {
                    b.stress[slot][g].narrow = prony::transient_buffer<float>(n, es_, stream_);
                    b.scratch[g]             = prony::transient_buffer<real_t>(n, es_, stream_);
                } else {
                    b.stress[slot][g].full = prony::transient_buffer<real_t>(n, es_, stream_);
                }
            }
        }
        if (evaluate_(x, slot) != SFEM_SUCCESS) {
            SFEM_ERROR("RelaxationGroups: evaluating the relaxed stress failed\n");
        }
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) {
                if (!narrow(b, g)) continue;
                const ptrdiff_t     n   = b.scratch[g]->size();
                const real_t *const in  = b.scratch[g]->data();
                float *const        out = b.stress[slot][g].narrow->data();
#ifdef SFEM_ENABLE_CUDA
                if (es_ == EXECUTION_SPACE_DEVICE) {
                    if (cu_prony_narrow(n, in, out, stream_) != SFEM_SUCCESS) {
                        SFEM_ERROR("RelaxationGroups: rounding the stress to float32 failed on the device\n");
                    }
                } else
#endif
                {
#pragma omp parallel for
                    for (ptrdiff_t k = 0; k < n; k++) out[k] = static_cast<float>(in[k]);
                }
            }
            b.scratch.clear();
        }
    }

    void RelaxationGroups::release_stress() {
        for (auto &entry : blocks_) entry.second.stress.clear();
        prony::release_transients(es_, stream_);
    }

    real_t *RelaxationGroups::stress_scratch(const std::string &block_name, const std::string &stress_name, const int s) {
        if (s < 0 || s > 1) {
            SFEM_ERROR("RelaxationGroups: there is no stress slot %d (0 and 1 exist)\n", s);
        }
        Block    &b = block(block_name);
        const int g = stress_index(stress_name);
        if (b.stress.empty() || (narrow(b, g) ? !b.scratch.size() || !b.scratch[g] : !b.stress[s][g].full)) {
            SFEM_ERROR("RelaxationGroups: a stress slot is written only while the stress is evaluated\n");
        }
        return narrow(b, g) ? b.scratch[g]->data() : b.stress[s][g].full->data();
    }

    const real_t *RelaxationGroups::start(const real_t *const x_n, std::shared_ptr<Buffer<real_t>> &zeros) const {
        if (!start_reference_) return x_n;
        // Before any step or reset: the reference configuration.
        zeros = prony::transient_buffer<real_t>(n_dofs_, es_, stream_);
        return zeros->data();
    }

    void RelaxationGroups::keep_previous(const real_t *const x_start) {
        bool needed = false;
        for (const auto &entry : blocks_) {
            for (const auto &group : entry.second.groups) needed = needed || group->holds_coefficients();
        }
        if (!needed || !x_start) {
            previous_.reset();
            return;
        }
        if (!previous_) previous_ = create_buffer<real_t>(n_dofs_, es_);
        sfem::blas<real_t>(es_)->copy(n_dofs_, x_start, previous_->data());
    }

    const real_t *RelaxationGroups::previous() const {
        if (!previous_) {
            SFEM_ERROR("RelaxationGroups: a term held on the last step's increment needs the displacement that step "
                       "started from, and none is held\n");
        }
        return previous_->data();
    }

    namespace {
        /// The number of components of a symmetric tensor of `n_components` stored on the
        /// diagonal: the rest count twice in its norm.
        int diagonal_components(const int n_components) { return n_components == 6 ? 3 : n_components == 3 ? 2 : n_components; }
    }  // namespace

    void RelaxationGroups::begin_step(const real_t t, const real_t dt, const real_t *const x_n) {
        if (!relaxes()) {
            for (auto &entry : blocks_) {
                for (auto &group : entry.second.groups) group->begin_step(t, dt, static_cast<const real_t *>(nullptr));
            }
            return;
        }
        std::shared_ptr<Buffer<real_t>> zeros;
        evaluate(start(x_n, zeros), 0);
        bool needs_previous = false;
        for (const auto &entry : blocks_) {
            for (const auto &group : entry.second.groups) needs_previous = needs_previous || group->needs_previous_stress(dt);
        }
        // Slot 1: the stress the last committed step started from, only for a term a shorter
        // step brings back.
        if (needs_previous) evaluate(previous(), 1);
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) {
                PronyRelaxation &r = *b.groups[g];
                if (narrow(b, g)) {
#if SFEM_REAL_T_IS_FLOAT64
                    r.begin_step(t, dt, b.stress[0][g].narrow->data(), needs_previous ? b.stress[1][g].narrow->data() : nullptr);
#endif
                } else {
                    r.begin_step(t, dt, b.stress[0][g].full->data(), needs_previous ? b.stress[1][g].full->data() : nullptr);
                }
            }
        }
        release_stress();
    }

    void RelaxationGroups::advance(const real_t *const x_n, const real_t *const x) {
        if (!relaxes()) {
            for (auto &entry : blocks_) {
                for (auto &group : entry.second.groups) {
                    group->advance(static_cast<const real_t *>(nullptr), static_cast<const real_t *>(nullptr));
                }
            }
            start_reference_ = false;
            norms_current_   = false;
            keep_previous(nullptr);
            return;
        }
        std::shared_ptr<Buffer<real_t>> zeros;
        const real_t *const             x_start = start(x_n, zeros);
        evaluate(x_start, 0);
        evaluate(x, 1);
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) {
                PronyRelaxation &r = *b.groups[g];
                if (narrow(b, g)) {
#if SFEM_REAL_T_IS_FLOAT64
                    r.advance(b.stress[0][g].narrow->data(), b.stress[1][g].narrow->data());
#endif
                } else {
                    r.advance(b.stress[0][g].full->data(), b.stress[1][g].full->data());
                }
            }
        }
        // The committed history's norms, while its stresses are at hand: the next step starts
        // from slot 1, and the last one started from slot 0.
        if (track_norms_) {
            for (auto &entry : blocks_) {
                Block &b = entry.second;
                b.norms.assign(b.groups.size(), {});
                for (size_t g = 0; g < b.groups.size(); g++) {
                    b.norms[g].resize(b.groups[g]->series().n_terms());
                    group_norms(b, g, 1, 0, b.norms[g].data());
                }
            }
        }
        release_stress();
        norms_current_   = track_norms_;
        start_reference_ = false;
        keep_previous(x_start);
    }

    void RelaxationGroups::reset(const real_t *const) {
        for (auto &entry : blocks_) {
            for (auto &group : entry.second.groups) group->reset();
        }
        // The next step starts from the displacement given here, which its caller holds.
        start_reference_ = false;
        norms_current_   = false;
        keep_previous(nullptr);
    }

    void RelaxationGroups::set_default_remainder_horizon(const real_t horizon) {
        for (auto &entry : blocks_) {
            for (auto &group : entry.second.groups) {
                if (group->options().remainder_horizon < 0) group->set_remainder_horizon(horizon);
            }
        }
    }

    namespace {
        /// `n` values of `src`, in `es`, as real_t on the host.
        template <typename T>
        std::vector<real_t> slot_on_host(const T *const src, const ptrdiff_t n, const ExecutionSpace es) {
            std::vector<T> copy;
            const T       *host = src;
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                copy.resize(n);
                if (cudaMemcpy(copy.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    SFEM_ERROR("RelaxationGroups: copying a stress slot from the device failed\n");
                }
                host = copy.data();
            }
#else
            (void)es;
#endif
            return std::vector<real_t>(host, host + n);
        }
    }  // namespace

    void RelaxationGroups::decode_history(const std::string &block_name,
                                          const std::string &group,
                                          const int          i,
                                          const real_t *const x_n,
                                          real_t *const      values) {
        Block                 &b = block(block_name);
        const size_t           g = group_index(group);
        const PronyRelaxation &r = *b.groups[g];
        const ptrdiff_t        n = static_cast<ptrdiff_t>(r.n_components()) * r.n_qp() * r.n_elements();
        if (!relaxes()) {
            const std::vector<real_t> none(n, real_t(0));
            r.decode_history(i, values, none.data(), none.data());
            return;
        }
        std::shared_ptr<Buffer<real_t>> zeros;
        evaluate(start(x_n, zeros), 0);
        // A term held on the last increment reads the stress that step started from.
        const bool with_previous = !r.stored(i) && r.holds_coefficients();
        if (with_previous) evaluate(previous(), 1);
        const auto slot = [&](const int s) {
            return narrow(b, g) ? slot_on_host(b.stress[s][g].narrow->data(), n, es_) : slot_on_host(b.stress[s][g].full->data(), n, es_);
        };
        const std::vector<real_t> S_n = slot(0), S_prev = with_previous ? slot(1) : S_n;
        release_stress();
        r.decode_history(i, values, S_n.data(), S_prev.data());
    }

    void RelaxationGroups::decode_stored_histories(
            const real_t *const                                                                          x_n,
            const std::function<void(const std::string &, const std::string &, int, const real_t *)> &emit) {
        std::shared_ptr<Buffer<real_t>> zeros;
        const bool                      relaxing = relaxes();
        if (relaxing) evaluate(start(x_n, zeros), 0);
        std::vector<real_t> values;
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) {
                const PronyRelaxation &r = *b.groups[g];
                const ptrdiff_t        n = static_cast<ptrdiff_t>(r.n_components()) * r.n_qp() * r.n_elements();
                std::vector<real_t>    S_n;
                if (!relaxing) {
                    S_n.assign(n, real_t(0));
                } else if (narrow(b, g)) {
                    S_n = slot_on_host(b.stress[0][g].narrow->data(), n, es_);
                } else {
                    S_n = slot_on_host(b.stress[0][g].full->data(), n, es_);
                }
                values.resize(n);
                for (int i = 0; i < r.series().n_terms(); ++i) {
                    if (!r.stored(i)) continue;
                    r.decode_history(i, values.data(), S_n.data(), S_n.data());
                    emit(entry.first, declarations_[g].group, i, values.data());
                }
            }
        }
        if (relaxing) release_stress();
    }

    void RelaxationGroups::group_norms(const Block &b, const size_t g, const int slot_n, const int slot_prev, real_t *const out) const {
        const PronyRelaxation &r          = *b.groups[g];
        const int              n_diagonal = diagonal_components(declarations_[g].n_components);
        const bool             prev       = !b.stress[slot_prev].empty() && (b.stress[slot_prev][g].full || b.stress[slot_prev][g].narrow);
        if (narrow(b, g)) {
#if SFEM_REAL_T_IS_FLOAT64
            r.squared_norms(b.stress[slot_n][g].narrow->data(), prev ? b.stress[slot_prev][g].narrow->data() : nullptr, n_diagonal, out);
#endif
        } else {
            r.squared_norms(b.stress[slot_n][g].full->data(), prev ? b.stress[slot_prev][g].full->data() : nullptr, n_diagonal, out);
        }
    }

    void RelaxationGroups::squared_norms(const std::string &block_name,
                                         const std::string &group,
                                         const int          n_diagonal,
                                         const real_t *const x_n,
                                         real_t *const      out) {
        Block                 &b = block(block_name);
        const size_t           g = group_index(group);
        const PronyRelaxation &r = *b.groups[g];
        if (n_diagonal != diagonal_components(declarations_[g].n_components)) {
            SFEM_ERROR("RelaxationGroups::squared_norms: %d diagonal components for %d stored ones\n", n_diagonal,
                       declarations_[g].n_components);
        }
        if (norms_current_) {
            std::copy(b.norms[g].begin(), b.norms[g].end(), out);
            return;
        }
        if (!relaxes()) {
            r.squared_norms(static_cast<const real_t *>(nullptr), static_cast<const real_t *>(nullptr), n_diagonal, out);
            return;
        }
        std::shared_ptr<Buffer<real_t>> zeros;
        evaluate(start(x_n, zeros), 0);
        if (r.holds_coefficients()) evaluate(previous(), 1);
        group_norms(b, g, 0, 1, out);
        release_stress();
    }

    std::vector<std::string> RelaxationGroups::block_names() const {
        std::vector<std::string> names;
        for (const auto &entry : blocks_) names.push_back(entry.first);
        return names;
    }

    bool RelaxationGroups::relaxes() const {
        for (const auto &entry : blocks_) {
            for (const auto &group : entry.second.groups) {
                if (group->series().n_terms() > 0) return true;
            }
        }
        return false;
    }

    std::size_t RelaxationGroups::bytes() const { return bytes_breakdown().total(); }

    RelaxationGroups::Bytes RelaxationGroups::bytes_breakdown() const {
        Bytes ret;
        for (const auto &entry : blocks_) {
            for (const auto &group : entry.second.groups) {
                ret.history += group->history_bytes();
                ret.history_packed += group->history_packed_bytes();
                ret.reference += group->reference_bytes();
                ret.frozen += group->frozen_bytes();
            }
        }
        if (previous_) ret.previous = previous_->size() * sizeof(real_t);
        return ret;
    }

#ifdef SFEM_ENABLE_RYAML
    namespace {
        std::string read_string(const ryml::ConstNodeRef &node) {
            const auto value = node.val();
            return std::string(value.str, value.len);
        }

        real_t read_real(const ryml::ConstNodeRef &node, const char *const key) {
            if (!node.has_child(ryml::to_csubstr(key))) {
                SFEM_ERROR("relaxation: missing \"%s\"\n", key);
            }
            real_t value = 0;
            node[ryml::to_csubstr(key)].load(&value);
            return value;
        }
    }  // namespace

    void RelaxationGroups::configure_from_yaml(const std::string &block_name, const ryml::ConstNodeRef &node) {
        if (!node.has_child("relaxation")) return;

        static const char *const keys[] = {
                "prony", "integrator", "wlf", "temperature", "history_storage", "history_elements_per_mean", "stress_storage",
                "remainder_horizon", "history_tolerance", "history_rounding"};

        for (auto group_node : node["relaxation"].children()) {
            const auto        key   = group_node.key();
            const std::string group = std::string(key.str, key.len);
            group_index(group);  // an undeclared group is an error, not a no-op

            for (auto entry : group_node.children()) {
                const auto        k     = entry.key();
                const std::string name  = std::string(k.str, k.len);
                bool              known = false;
                for (const char *const allowed : keys) known = known || name == allowed;
                if (!known) {
                    SFEM_ERROR("relaxation.%s: unknown key \"%s\" (expected prony, integrator, wlf, temperature, "
                               "history_storage, history_elements_per_mean, stress_storage, remainder_horizon, "
                               "history_tolerance, history_rounding)\n",
                               group.c_str(),
                               name.c_str());
                }
            }

            // An entry overrides what it names and keeps the rest: a block
            // that only sets a temperature keeps the spectrum, the shift and
            // the storage it inherited from the top level.
            for (auto &entry : blocks_) {
                if (!block_name.empty() && entry.first != block_name) continue;
                const PronyRelaxation   &current = *entry.second.groups[group_index(group)];
                PronySeries              series  = current.series();
                PronyRelaxation::Options options = current.options();
                options.temperature              = current.temperature();

                if (group_node.has_child("prony")) {
                    std::vector<real_t> g, tau;
                    for (auto term : group_node["prony"].children()) {
                        g.push_back(read_real(term, "g"));
                        tau.push_back(read_real(term, "tau"));
                    }
                    series = PronySeries(std::move(g), std::move(tau));
                }
                if (group_node.has_child("integrator")) {
                    options.integrator = create_relaxation_integrator(read_string(group_node["integrator"]));
                }
                if (group_node.has_child("wlf")) {
                    const auto wlf = group_node["wlf"];
                    options.shift  = std::make_shared<WLFShift>(read_real(wlf, "C1"), read_real(wlf, "C2"), read_real(wlf, "T_ref"));
                }
                if (group_node.has_child("temperature")) {
                    options.temperature = read_real(group_node, "temperature");
                }
                if (group_node.has_child("history_storage")) {
                    options.history_storage = history_storage_from_string(read_string(group_node["history_storage"]));
                }
                if (group_node.has_child("stress_storage")) {
                    const auto storage = history_storage_from_string(read_string(group_node["stress_storage"]));
                    if (storage != HistoryStorage::FULL && storage != HistoryStorage::FLOAT32) {
                        SFEM_ERROR("relaxation.%s.stress_storage: full or float32, not %s\n", group.c_str(), to_string(storage));
                    }
                    options.stress_storage = storage;
                }
                if (group_node.has_child("remainder_horizon")) {
                    options.remainder_horizon = read_real(group_node, "remainder_horizon");
                }
                if (group_node.has_child("history_tolerance")) {
                    options.history_tolerance = read_real(group_node, "history_tolerance");
                }
                if (group_node.has_child("history_rounding")) {
                    options.history_rounding = history_rounding_from_string(read_string(group_node["history_rounding"]));
                }
                if (group_node.has_child("history_elements_per_mean")) {
                    options.history_elements_per_mean = static_cast<int>(read_real(group_node, "history_elements_per_mean"));
                }

                set_series(entry.first, group, series, options);
            }
            if (!block_name.empty()) block(block_name);  // an unknown block is an error, not a no-op
        }
    }
#endif  // SFEM_ENABLE_RYAML

    int RelaxationGroups::write_state(const std::string &dir) {
        const smesh::Path d(dir);
        if (write_state_numbers(d / "relaxation.txt",
                                {{"start_reference", start_reference_ ? 1.0 : 0.0},
                                 {"previous", previous_ ? 1.0 : 0.0},
                                 {"n_blocks", (double)blocks_.size()},
                                 {"n_groups", (double)declarations_.size()}})) {
            return SFEM_FAILURE;
        }
        for (auto &entry : blocks_) {
            const std::string name = entry.first;
            Block            &b    = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) {
                if (b.groups[g] && b.groups[g]->write_state(dir + "/prony_" + name + "_" + std::to_string(g))) return SFEM_FAILURE;
            }
        }
        if (previous_ && write_state_buffer(d / "previous_displacement.raw", previous_)) return SFEM_FAILURE;
        return SFEM_SUCCESS;
    }

    int RelaxationGroups::read_state(const std::string &dir) {
        std::map<std::string, double> numbers;
        double                        reference = 0, previous = 0, n_blocks = 0, n_groups = 0;
        if (read_state_numbers(smesh::Path(dir) / "relaxation.txt", numbers)) return SFEM_FAILURE;
        if (numbers.count("start_stress_current")) {
            fprintf(stderr,
                    "RelaxationGroups::read_state: %s holds relaxed-stress slots, which are no longer kept; it cannot be "
                    "restarted from\n",
                    dir.c_str());
            return SFEM_FAILURE;
        }
        if (state_number(numbers, "start_reference", &reference) || state_number(numbers, "previous", &previous) ||
            state_number(numbers, "n_blocks", &n_blocks) || state_number(numbers, "n_groups", &n_groups)) {
            return SFEM_FAILURE;
        }
        if ((size_t)n_blocks != blocks_.size() || (size_t)n_groups != declarations_.size()) {
            fprintf(stderr, "RelaxationGroups::read_state: %s was saved with other blocks or groups\n", dir.c_str());
            return SFEM_FAILURE;
        }
        for (auto &entry : blocks_) {
            const std::string name = entry.first;
            Block            &b    = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) {
                if (b.groups[g] && b.groups[g]->read_state(dir + "/prony_" + name + "_" + std::to_string(g))) return SFEM_FAILURE;
            }
        }
        previous_.reset();
        if (previous != 0) {
            previous_ = create_buffer<real_t>(n_dofs_, es_);
            if (read_state_buffer(smesh::Path(dir) / "previous_displacement.raw", previous_)) return SFEM_FAILURE;
        }
        start_reference_ = reference != 0;
        norms_current_   = false;
        return SFEM_SUCCESS;
    }

    int HistoryDependent::write_history(const std::string &dir) {
        auto *groups = relaxation_state();
        if (!groups) {
            fprintf(stderr, "HistoryDependent::write_history: this history has no saved form\n");
            return SFEM_FAILURE;
        }
        return groups->write_state(dir);
    }

    int HistoryDependent::read_history(const std::string &dir) {
        auto *groups = relaxation_state();
        if (!groups) {
            fprintf(stderr, "HistoryDependent::read_history: this history has no saved form\n");
            return SFEM_FAILURE;
        }
        return groups->read_state(dir);
    }

}  // namespace sfem
