#include "sfem_RelaxationGroups.hpp"

#include <cassert>
#include <utility>

#include "sfem_blas_factory.hpp"
#include "sfem_logger.hpp"

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

    void RelaxationGroups::allocate_stress() {
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            if (!b.stress.empty()) continue;
            b.stress.assign(2, {});
            for (int s = 0; s < 2; s++) {
                for (const auto &d : declarations_) {
                    b.stress[s].push_back(
                            create_buffer<real_t>(static_cast<ptrdiff_t>(d.n_components) * b.n_qp * b.n_elements, es_));
                }
            }
        }
    }

    real_t *RelaxationGroups::stress_scratch(const std::string &block_name, const std::string &stress_name, const int s) {
        if (s < 0 || s > 1) {
            SFEM_ERROR("RelaxationGroups: there is no stress slot %d (0 and 1 exist)\n", s);
        }
        allocate_stress();
        return block(block_name).stress[s][stress_index(stress_name)]->data();
    }

    const real_t *RelaxationGroups::slot(const Block &b, const int s, const size_t group) const {
        return b.stress.empty() ? nullptr : b.stress[s][group]->data();
    }

    void RelaxationGroups::evaluate(const real_t *const x, const int slot, const StressEvaluation &evaluate) {
        allocate_stress();
        if (evaluate(x, slot) != SFEM_SUCCESS) {
            SFEM_ERROR("RelaxationGroups: evaluating the relaxed stress failed\n");
        }
    }

    void RelaxationGroups::begin_step(const real_t t, const real_t dt, const StressEvaluation &evaluate) {
        if (relaxes() && !start_stress_current_) {
            if (start_unknown_) {
                // The step starts from a state committed while nothing relaxed,
                // so its stress was never evaluated and its displacement is gone.
                SFEM_ERROR("RelaxationGroups: a series was configured after a step or reset; call reset(x) first\n");
            }
            auto reference = create_buffer<real_t>(n_dofs_, es_);
            sfem::blas<real_t>(es_)->zeros(n_dofs_, reference->data());
            this->evaluate(reference->data(), 0, evaluate);
            start_stress_current_ = true;
        }
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) b.groups[g]->begin_step(t, dt, slot(b, 0, g));
        }
    }

    void RelaxationGroups::advance(const real_t *const x, const StressEvaluation &evaluate) {
        const bool relaxing = relaxes();
        if (relaxing) this->evaluate(x, 1, evaluate);
        for (auto &entry : blocks_) {
            Block &b = entry.second;
            for (size_t g = 0; g < b.groups.size(); g++) b.groups[g]->advance(slot(b, 0, g), slot(b, 1, g));
            // The stress the step ended at is the one the next step starts from.
            if (!b.stress.empty()) std::swap(b.stress[0], b.stress[1]);
        }
        start_stress_current_ = relaxing;
        start_unknown_        = !relaxing;
    }

    void RelaxationGroups::reset(const real_t *const x, const StressEvaluation &evaluate) {
        for (auto &entry : blocks_) {
            for (auto &group : entry.second.groups) group->reset();
        }
        const bool relaxing = relaxes();
        if (relaxing) this->evaluate(x, 0, evaluate);
        start_stress_current_ = relaxing;
        start_unknown_        = !relaxing;
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

    std::size_t RelaxationGroups::bytes() const {
        std::size_t ret = 0;
        for (const auto &entry : blocks_) {
            for (const auto &group : entry.second.groups) ret += group->bytes();
        }
        for (const auto &entry : blocks_) {
            for (const auto &slots : entry.second.stress) {
                for (const auto &buffer : slots) ret += buffer->size() * sizeof(real_t);
            }
        }
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

        static const char *const keys[] = {"prony", "integrator", "wlf", "temperature", "history_storage"};

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
                               "history_storage)\n",
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

                set_series(entry.first, group, series, options);
            }
            if (!block_name.empty()) block(block_name);  // an unknown block is an error, not a no-op
        }
    }
#endif  // SFEM_ENABLE_RYAML

}  // namespace sfem
