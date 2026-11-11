#include "sfem_Observers.hpp"
#include "sfem_log.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <sys/resource.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "sfem_MemoryLedger.hpp"
#include "smesh_write.hpp"
#include "sfem_Function.hpp"
#include "sfem_blas_factory.hpp"
#include "sfem_prony_diagnostics.hpp"
#include "smesh_glob.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_BLAS_cuda.hpp"
#endif

namespace sfem {

    namespace {
        /// A host view of n values that live in `es`: the values themselves on the host, a copy
        /// from the device. Called at export time only.
        SharedBuffer<real_t> on_host(const ExecutionSpace es, const real_t *const values, const ptrdiff_t n) {
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                return smesh::to_host(Buffer<real_t>::wrap(n, const_cast<real_t *>(values), MEMORY_SPACE_DEVICE));
            }
#else
            SFEM_UNUSED(es);
#endif
            return Buffer<real_t>::wrap(n, const_cast<real_t *>(values));
        }

        /// n values that live in `es` copied into the host array `out`.
        void copy_to_host(const ExecutionSpace es, const real_t *const values, const ptrdiff_t n, real_t *const out) {
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                device_to_host(n, values, out);
                return;
            }
#else
            SFEM_UNUSED(es);
#endif
            std::copy(values, values + n, out);
        }
    }  // namespace

    // ---------------------------------------------------------------------------------------------

    class FieldExport::Impl {
    public:
        struct Field {
            std::string                                                name;
            std::function<int(const StepRecord &, real_t *const)> fill;
        };

        std::shared_ptr<Function> f;
        smesh::Path               path;
        int                       export_freq{1};
        std::shared_ptr<Output>   output;  // null on the ranks that do not write
        std::vector<Field>        fields;
        SharedBuffer<real_t>      scratch;
        size_t                    index{0};

        // gather_to: the serial mesh (rank 0), each rank's owned values per gathered node, and
        // where each gathered node goes in the serial numbering (rank 0).
        bool                          gathered{false};
        std::shared_ptr<Mesh>         serial_mesh;
        std::shared_ptr<Communicator> comm;
        std::vector<int>              counts, displs;  // values, block_size per node
        std::vector<ptrdiff_t>        place;
        std::vector<real_t>           all;
        ptrdiff_t                     n_serial_values{0};

        // The files are written by a thread of their own, in step order, from host copies of an
        // export's values taken at the step: the solver goes on meanwhile. At most `depth`
        // exports are in flight; a failed write is returned by the next export.
        struct Export {
            size_t                            index{0};
            real_t                            t{0};
            std::vector<SharedBuffer<real_t>> values;  // disp, then the fields
        };
        static constexpr int                           depth = 2;
        std::thread                                    writer;
        std::mutex                                     mutex;
        std::condition_variable                        cv;
        std::deque<Export>                             queue;
        std::vector<std::vector<SharedBuffer<real_t>>> idle;  // the copies of exports written
        std::vector<SharedBuffer<real_t>>              staged;  // the export being taken
        int                                            n_copies{0}, pending{0};
        bool                                           stop{false};
        int                                            error{SFEM_SUCCESS};

        ~Impl() {
            drain();
            if (writer.joinable()) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    stop = true;
                }
                cv.notify_all();
                writer.join();
            }
        }

        const char *name(const size_t k) const { return k == 0 ? "disp" : fields[k - 1].name.c_str(); }

        void run() {
#ifdef __linux__
            // Not on the core of the thread that started it, which OMP_PROC_BIND binds to one.
            cpu_set_t any;
            CPU_ZERO(&any);
            for (int c = 0; c < CPU_SETSIZE; ++c) CPU_SET(c, &any);
            pthread_setaffinity_np(pthread_self(), sizeof(any), &any);
#endif
            std::unique_lock<std::mutex> lock(mutex);
            for (;;) {
                cv.wait(lock, [this] { return stop || !queue.empty(); });
                if (queue.empty()) return;
                Export e = std::move(queue.front());
                queue.pop_front();
                lock.unlock();
                int rc = SFEM_SUCCESS;
                {
                    // Off the solver's thread: not traced unless every event takes a mutex.
                    SFEM_TRACE_SCOPE_MUTEX("FieldExport::write_files");
                    for (size_t k = 0; k < e.values.size() && rc == SFEM_SUCCESS; ++k) {
                        rc = output->write_step(name(k), e.index, e.values[k]->data());
                    }
                    if (rc == SFEM_SUCCESS) output->log_time(e.t);
                }
                lock.lock();
                if (rc != SFEM_SUCCESS && error == SFEM_SUCCESS) error = rc;
                idle.push_back(std::move(e.values));
                pending--;
                cv.notify_all();
            }
        }

        /// Waits for the exports in flight to be written; the first failure among them.
        int drain() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [this] { return pending == 0; });
            return error;
        }

        /// Host arrays for an export, once one of the `depth` in flight is written.
        int acquire(std::vector<SharedBuffer<real_t>> &values) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return !idle.empty() || n_copies < depth || error != SFEM_SUCCESS; });
                if (error != SFEM_SUCCESS) return error;
                if (!idle.empty()) {
                    values = std::move(idle.back());
                    idle.pop_back();
                    return SFEM_SUCCESS;
                }
                n_copies++;
            }
            const ptrdiff_t n = gathered ? n_serial_values : f->space()->n_dofs();
            values.resize(fields.size() + 1);
            for (auto &v : values) v = create_host_buffer<real_t>(n);
            return SFEM_SUCCESS;
        }

        /// `values` (n_dofs in `es`) into `out` (null on the ranks that do not write): a copy, or
        /// gathered to rank 0 in the serial numbering.
        void stage(const ExecutionSpace es, const real_t *const values, real_t *const out) {
            const ptrdiff_t ndofs = f->space()->n_dofs();
            if (!gathered) {
                copy_to_host(es, values, ndofs, out);
                return;
            }
            const int bs = f->space()->block_size();
            comm->allgatherv(on_host(es, values, ndofs)->data(), (int)f->space()->n_dofs_owned(), all.data(), counts.data(), displs.data());
            if (!out) return;
            for (size_t k = 0; k < place.size(); ++k) {
                for (int d = 0; d < bs; ++d) out[place[k] * bs + d] = all[k * bs + d];
            }
        }

        int write(const StepRecord &record) {
            SFEM_TRACE_SCOPE("FieldExport::write");
            const ExecutionSpace es = f->execution_space();
            if (output && acquire(staged) != SFEM_SUCCESS) return SFEM_FAILURE;

            stage(es, record.x, output ? staged[0]->data() : nullptr);
            for (size_t k = 0; k < fields.size(); ++k) {
                if (fields[k].fill(record, scratch->data()) != SFEM_SUCCESS) {
                    if (output) {
                        std::lock_guard<std::mutex> lock(mutex);
                        idle.push_back(std::move(staged));
                    }
                    return SFEM_FAILURE;
                }
                stage(es, scratch->data(), output ? staged[k + 1]->data() : nullptr);
            }

            if (output) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    queue.push_back(Export{index, record.t, std::move(staged)});
                    pending++;
                }
                cv.notify_all();
                if (!writer.joinable()) writer = std::thread([this] { run(); });
            }
            index++;
            return record.last ? drain() : SFEM_SUCCESS;
        }
    };

    FieldExport::FieldExport(const std::shared_ptr<Function> &f, const smesh::Path &path, const int export_freq)
        : impl_(std::make_unique<Impl>()) {
        impl_->f           = f;
        impl_->path        = path;
        impl_->export_freq = std::max(1, export_freq);
    }

    FieldExport::~FieldExport() = default;

    void FieldExport::add_field(const std::string &name, const std::function<int(const StepRecord &, real_t *const)> &fill) {
        if (!impl_->scratch) impl_->scratch = create_buffer<real_t>(impl_->f->space()->n_dofs(), impl_->f->execution_space());
        impl_->fields.push_back({name, fill});
    }

    int FieldExport::gather_to(const std::shared_ptr<Mesh> &serial_mesh) {
        auto &impl = *impl_;
        auto  fs   = impl.f->space();
        auto  mesh = fs->mesh();
        impl.comm  = mesh->comm();
        if (impl.comm->size() == 1 || !mesh->is_distributed()) {
            SFEM_ERROR("FieldExport::gather_to: the function's mesh is not distributed\n");
            return SFEM_FAILURE;
        }
        const bool root = impl.comm->rank() == 0;
        if (root != (serial_mesh != nullptr)) {
            SFEM_ERROR("FieldExport::gather_to: the serial mesh is rank 0's\n");
            return SFEM_FAILURE;
        }

        // Each rank's owned nodes, by their ids in the file.
        const int       size    = impl.comm->size();
        const int       bs      = fs->block_size();
        const ptrdiff_t n_owned = fs->n_dofs_owned() / bs;
        std::vector<int> node_counts(size), node_displs(size + 1, 0);
        const int        mine = (int)n_owned;
        impl.comm->allgather(&mine, node_counts.data(), 1);
        for (int r = 0; r < size; ++r) node_displs[r + 1] = node_displs[r] + node_counts[r];
        std::vector<smesh::large_idx_t> ids(node_displs[size]);
        impl.comm->allgatherv(mesh->distributed()->node_mapping()->data(), mine, ids.data(), node_counts.data(), node_displs.data());

        impl.counts.resize(size);
        impl.displs.resize(size);
        for (int r = 0; r < size; ++r) {
            impl.counts[r] = node_counts[r] * bs;
            impl.displs[r] = node_displs[r] * bs;
        }
        impl.all.resize((size_t)node_displs[size] * bs);

        if (root) {
            // The serial mesh's node i is node node_mapping[i] of the file; a mesh read whole has
            // none, and gets the identity, written with it: the output says which node of the file
            // each is, whatever an earlier run left in the folder.
            const ptrdiff_t n_serial = serial_mesh->n_nodes();
            if (!serial_mesh->node_mapping()) {
                auto identity = create_host_buffer<idx_t>(n_serial);
                for (ptrdiff_t i = 0; i < n_serial; ++i) identity->data()[i] = (idx_t)i;
                serial_mesh->set_node_mapping(identity);
            }
            auto                                      file_ids = serial_mesh->node_mapping();
            std::unordered_map<smesh::large_idx_t, ptrdiff_t> at;
            at.reserve(n_serial);
            for (ptrdiff_t i = 0; i < n_serial; ++i) at[file_ids ? (smesh::large_idx_t)file_ids->data()[i] : (smesh::large_idx_t)i] = i;
            if ((ptrdiff_t)ids.size() != n_serial) {
                SFEM_ERROR("FieldExport::gather_to: %zu nodes owned over the ranks, %td in the serial mesh\n", ids.size(), n_serial);
                return SFEM_FAILURE;
            }
            impl.place.resize(ids.size());
            for (size_t k = 0; k < ids.size(); ++k) {
                auto found = at.find(ids[k]);
                if (found == at.end()) {
                    SFEM_ERROR("FieldExport::gather_to: node %lld of the file is not in the serial mesh\n", (long long)ids[k]);
                    return SFEM_FAILURE;
                }
                impl.place[k] = found->second;
            }
            impl.n_serial_values = n_serial * bs;
        }
        impl.serial_mesh = serial_mesh;
        impl.gathered    = true;
        return SFEM_SUCCESS;
    }

    int FieldExport::initial(const StepRecord &record) {
        auto &impl = *impl_;
        auto  fs   = impl.f->space();
        if (impl.gathered) {
            if (impl.serial_mesh) {
                smesh::create_directory(impl.path);
                if (impl.serial_mesh->write(impl.path / "mesh") != SFEM_SUCCESS) return SFEM_FAILURE;
                impl.output = std::make_shared<Output>(FunctionSpace::create(impl.serial_mesh, fs->block_size()));
                impl.output->set_output_dir(impl.path / "out");
                impl.output->enable_AoS_to_SoA(fs->block_size() > 1);
            }
            return impl.write(record);
        }

        smesh::create_directory(impl.path);
        if (fs->has_semi_structured_mesh()) {
            smesh::semistructured_export_as_standard(fs->mesh(), impl.path / "mesh");
        } else {
            fs->mesh()->write(impl.path / "mesh");
        }

        impl.output = impl.f->output();
        impl.output->set_output_dir(impl.path / "out");
        impl.output->enable_AoS_to_SoA(fs->block_size() > 1);
        return impl.write(record);
    }

    int FieldExport::committed(const StepRecord &record) {
        if (record.step % impl_->export_freq != 0 && !record.last) return SFEM_SUCCESS;
        return impl_->write(record);
    }

    int FieldExport::failed(const StepRecord &) { return impl_->drain(); }

    // ---------------------------------------------------------------------------------------------

    class Quantities::Impl {
    public:
        std::shared_ptr<Function>            f;
        std::shared_ptr<Op>                  material;
        std::shared_ptr<DirichletConditions> dirichlet;
        smesh::Path                          path;
        FILE                                *file{nullptr};
        SharedBuffer<real_t>                 reaction;
        std::vector<mask_t>                  mask;

        ~Impl() {
            if (file) fclose(file);
        }
    };

    Quantities::Quantities(const std::shared_ptr<Function>            &f,
                           const std::shared_ptr<Op>                  &material,
                           const std::shared_ptr<DirichletConditions> &dirichlet,
                           const smesh::Path                          &path)
        : impl_(std::make_unique<Impl>()) {
        impl_->f         = f;
        impl_->material  = material;
        impl_->dirichlet = dirichlet;
        impl_->path      = path;
        impl_->reaction  = create_buffer<real_t>(f->space()->n_dofs(), f->execution_space());
    }

    Quantities::~Quantities() = default;

    int Quantities::initial(const StepRecord &) {
        auto &impl = *impl_;
        smesh::create_directory(impl.path);
        impl.file = fopen((impl.path / "quantities.yaml").c_str(), "w");
        if (!impl.file) {
            fprintf(stderr, "Quantities: unable to write %s\n", (impl.path / "quantities.yaml").c_str());
            return SFEM_FAILURE;
        }
        fprintf(impl.file, "material_objective_history:\n");
        fflush(impl.file);

        const ptrdiff_t ndofs = impl.f->space()->n_dofs();
        impl.mask.assign(mask_count(ndofs), 0);
        if (impl.dirichlet) impl.dirichlet->mask(impl.mask.data());
        return SFEM_SUCCESS;
    }

    int Quantities::committed(const StepRecord &record) {
        SFEM_TRACE_SCOPE("Quantities::committed");

        auto           &impl  = *impl_;
        const ptrdiff_t ndofs = impl.f->space()->n_dofs();
        const int       bs    = impl.f->space()->block_size();
        auto            blas  = sfem::blas<real_t>(impl.f->execution_space());

        blas->zeros(ndofs, impl.reaction->data());
        if (impl.material->gradient(record.x, impl.reaction->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
        real_t    objective = 0;
        const int status    = impl.material->value(record.x, &objective);

        auto                reaction = on_host(impl.f->execution_space(), impl.reaction->data(), ndofs);
        const real_t *const r        = reaction->data();
        std::vector<real_t> resultant(bs, 0);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            if (mask_get(i, impl.mask.data())) resultant[i % bs] += r[i];
        }

        FILE *out = impl.file;
        fprintf(out, "  - time: %.17g\n", (double)record.t);
        if (status == SFEM_SUCCESS) {
            fprintf(out, "    value: %.17g\n", (double)objective);
        } else {
            fprintf(out, "    value: null\n");
        }
        fprintf(out, "    constrained_reaction_resultant: [");
        for (int d = 0; d < bs; ++d) fprintf(out, "%s%.17g", d ? ", " : "", (double)resultant[d]);
        fprintf(out, "]\n");
        fprintf(out, "    constraint_reactions:\n");
        if (impl.dirichlet) {
            const auto &conditions = impl.dirichlet->conditions();
            for (size_t c = 0; c < conditions.size(); ++c) {
                const auto &condition = conditions[c];
                real_t      sum       = 0;
                for (size_t i = 0; i < condition.nodeset->size(); ++i) {
                    sum += r[(ptrdiff_t)condition.nodeset->data()[i] * bs + condition.component];
                }
                fprintf(out, "      - condition: %zu\n", c);
                fprintf(out, "        component: %d\n", condition.component);
                fprintf(out, "        resultant: %.17g\n", (double)sum);
            }
        }
        fflush(out);
        return SFEM_SUCCESS;
    }

    // ---------------------------------------------------------------------------------------------

    class CarriedRates {
    public:
        explicit CarriedRates(const std::shared_ptr<Function> &f) : f_(f) {}

        /// The record's velocity and acceleration (null without a scheme; the acceleration null
        /// when the scheme carries none), evaluated at the first call for a record.
        void of(const StepRecord &record, const real_t **velocity, const real_t **acceleration) {
            *velocity = *acceleration = nullptr;
            if (!record.scheme) return;
            const bool with_acceleration = record.scheme->has_acceleration();
            if (record.scheme.get() != scheme_ || record.step != step_ || record.t != t_) {
                if (!velocity_) {
                    const ptrdiff_t ndofs = f_->space()->n_dofs();
                    velocity_             = create_buffer<real_t>(ndofs, f_->execution_space());
                    acceleration_         = create_buffer<real_t>(ndofs, f_->execution_space());
                }
                record.scheme->carried_state(velocity_->data(), with_acceleration ? acceleration_->data() : nullptr);
                scheme_ = record.scheme.get();
                step_   = record.step;
                t_      = record.t;
            }
            *velocity = velocity_->data();
            if (with_acceleration) *acceleration = acceleration_->data();
        }

    private:
        std::shared_ptr<Function> f_;
        SharedBuffer<real_t>      velocity_, acceleration_;
        const TimeScheme         *scheme_{nullptr};
        int                       step_{-1};
        real_t                    t_{0};
    };

    std::shared_ptr<CarriedRates> create_carried_rates(const std::shared_ptr<Function> &f) {
        return std::make_shared<CarriedRates>(f);
    }

    namespace {
        class ControlPoint final : public Probe {
        public:
            ControlPoint(const std::shared_ptr<Function>     &f,
                         const std::vector<real_t>           &point,
                         const bool                           with_rates,
                         const std::shared_ptr<CarriedRates> &rates)
                : f_(f), rates_(rates ? rates : create_carried_rates(f)), with_rates_(with_rates) {
                auto      mesh = f->space()->mesh();
                const int dim  = mesh->spatial_dimension();
                auto      xyz  = mesh->points()->data();
                // Distributed, the nearest of the owned nodes over the ranks, sampled by its owner
                // (the lowest rank on a tie) and summed: the history is written by rank 0, which
                // may hold none of the body the point is in.
                comm_                  = mesh->comm();
                const ptrdiff_t nnodes = f->space()->n_dofs_owned() / f->space()->block_size();

                double best = std::numeric_limits<double>::max();
                for (ptrdiff_t i = 0; i < nnodes; ++i) {
                    double d2 = 0;
                    for (int d = 0; d < dim && d < (int)point.size(); ++d) {
                        const double diff = (double)xyz[d][i] - (double)point[d];
                        d2 += diff * diff;
                    }
                    if (d2 < best) {
                        best  = d2;
                        node_ = i;
                    }
                }
                if (comm_->size() > 1) {
                    const double nearest = -comm_->max(-best);
                    const int    rank    = best == nearest ? comm_->rank() : comm_->size();
                    owner_               = comm_->rank() == -comm_->max(-rank);
                }
            }

            std::vector<std::string> columns() const override {
                const int                bs    = f_->space()->block_size();
                const char              *axes  = "xyz";
                std::vector<std::string> names;
                for (const char *field : {"u", "v", "a"}) {
                    if (field[0] != 'u' && !with_rates_) break;
                    for (int d = 0; d < bs; ++d) names.push_back(std::string(field) + (bs <= 3 ? std::string(1, axes[d]) : std::to_string(d)));
                }
                return names;
            }

            int sample(const StepRecord &record, real_t *const values) override {
                SFEM_TRACE_SCOPE("ControlPoint::sample");
                const ExecutionSpace es = f_->execution_space();
                const int            bs = f_->space()->block_size();

                // The node's bs values of each field, not the fields.
                auto read = [&](const real_t *const field, real_t *const out) {
                    if (!field) {
                        std::fill(out, out + bs, real_t(0));
                        return;
                    }
                    auto node = on_host(es, field + node_ * bs, bs);
                    std::copy(node->data(), node->data() + bs, out);
                };
                const int n_values = (with_rates_ ? 3 : 1) * bs;
                read(record.x, values);
                if (!with_rates_) return reduce(values, n_values);

                const real_t *velocity, *acceleration;
                rates_->of(record, &velocity, &acceleration);
                read(velocity, values + bs);
                read(acceleration, values + 2 * bs);
                return reduce(values, n_values);
            }

        private:
            /// The owner's values on every rank.
            int reduce(real_t *const values, const int n) const {
                if (comm_->size() == 1) return SFEM_SUCCESS;
                if (!owner_) std::fill(values, values + n, real_t(0));
                comm_->sum(values, n, smesh::TypeToEnum<real_t>::value());
                return SFEM_SUCCESS;
            }

            std::shared_ptr<Function>     f_;
            std::shared_ptr<CarriedRates> rates_;
            std::shared_ptr<Communicator> comm_;
            bool                          with_rates_;
            bool                          owner_{true};
            ptrdiff_t                     node_{0};
        };
    }  // namespace

    std::shared_ptr<Probe> create_control_point_probe(const std::shared_ptr<Function>     &f,
                                                      const std::vector<real_t>           &point,
                                                      const bool                           with_rates,
                                                      const std::shared_ptr<CarriedRates> &rates) {
        return std::make_shared<ControlPoint>(f, point, with_rates, rates);
    }

    namespace {
        class PronyProbe final : public Probe {
        public:
            PronyProbe(RelaxationGroups &relaxations, const std::shared_ptr<Communicator> &comm)
                : relaxations_(relaxations), comm_(comm), blocks_(relaxations.block_names()) {
                if (blocks_.empty()) SFEM_ERROR("Prony probe: the material has no blocks\n");
                for (const auto &d : relaxations.declarations()) {
                    const auto &series = relaxations.relaxation(blocks_[0], d.group).series();
                    for (const auto &b : blocks_) {
                        if (relaxations.relaxation(b, d.group).series().n_terms() != series.n_terms()) {
                            SFEM_ERROR("Prony probe: blocks %s and %s hold different series in %s\n",
                                       blocks_[0].c_str(),
                                       b.c_str(),
                                       d.group.c_str());
                        }
                    }
                    for (int k = 0; k < series.n_terms(); ++k) {
                        if (comm->rank() == 0 && log_enabled(LOG_SETUP, LOG_VERBOSE)) {
                            log_printf(LOG_SETUP,
                                       LOG_VERBOSE,
                                       LOG_SUMMARY,
                                       "Prony probe: %s_H%d is the term g = %g, tau = %g",
                                       d.group.c_str(),
                                       k,
                                       (double)series.g(k),
                                       (double)series.tau(k));
                        }
                    }
                }
            }

            std::vector<std::string> columns() const override {
                std::vector<std::string> names;
                for (const auto &d : relaxations_.declarations()) {
                    names.push_back(d.group + "_gamma");
                    const int n = relaxations_.relaxation(blocks_[0], d.group).series().n_terms();
                    for (int k = 0; k < n; ++k) names.push_back(d.group + "_H" + std::to_string(k));
                    names.push_back(d.group + "_stored");
                }
                // Persistent bytes over every rank, by what holds them (RelaxationGroups::Bytes).
                for (const char *what : {"bytes_history", "bytes_reference", "bytes_frozen", "bytes_stress"}) names.push_back(what);
#ifdef SFEM_ENABLE_PRONY_DIAGNOSTICS
                // Cumulative over the run, host loops only (sfem_prony_diagnostics.hpp).
                names.push_back("updates");
                names.push_back("updates_lost");
#endif
                return names;
            }

            int sample(const StepRecord &, real_t *const values) override {
                SFEM_TRACE_SCOPE("PronyProbe::sample");
                int c = 0;
                for (const auto &d : relaxations_.declarations()) {
                    values[c++] = relaxations_.relaxation(blocks_[0], d.group).gamma();
                    const int n = relaxations_.relaxation(blocks_[0], d.group).series().n_terms();
                    // A symmetric tensor's off-diagonal components (after the diagonal ones, 3 of 6 or
                    // 1 of 3) count twice in its Frobenius norm.
                    const int n_diagonal = d.n_components == 6 ? 3 : d.n_components == 3 ? 2 : d.n_components;
                    // Per term the sum of |H|^2 over the points, computed where the history lives,
                    // then the number of points; summed over the blocks and the ranks.
                    sums_.assign(n + 1, real_t(0));
                    block_.resize(n);
                    for (const auto &b : blocks_) {
                        const auto &r = relaxations_.relaxation(b, d.group);
                        relaxations_.squared_norms(b, d.group, n_diagonal, block_.data());
                        for (int k = 0; k < n; ++k) sums_[k] += block_[k];
                        sums_[n] += static_cast<real_t>(r.n_qp()) * r.n_elements();
                    }
                    if (comm_->size() > 1) comm_->sum(sums_.data(), n + 1, smesh::TypeToEnum<real_t>::value());
                    for (int k = 0; k < n; ++k) values[c++] = sums_[n] > 0 ? std::sqrt(sums_[k] / sums_[n]) : 0;
                    // The terms holding a tensor: the same on every block and rank, the weights being.
                    values[c++] = relaxations_.relaxation(blocks_[0], d.group).n_stored_terms();
                }
                const auto bytes   = relaxations_.bytes_breakdown();
                real_t     held[4] = {real_t(bytes.history), real_t(bytes.reference), real_t(bytes.frozen), real_t(bytes.stress)};
                if (comm_->size() > 1) comm_->sum(held, 4, smesh::TypeToEnum<real_t>::value());
                for (int k = 0; k < 4; ++k) values[c++] = held[k];
#ifdef SFEM_ENABLE_PRONY_DIAGNOSTICS
                real_t counts[2] = {real_t(prony::update_counts().updated), real_t(prony::update_counts().lost)};
                if (comm_->size() > 1) comm_->sum(counts, 2, smesh::TypeToEnum<real_t>::value());
                values[c++] = counts[0];
                values[c++] = counts[1];
#endif
                return SFEM_SUCCESS;
            }

        private:
            RelaxationGroups             &relaxations_;
            std::shared_ptr<Communicator> comm_;
            std::vector<std::string>      blocks_;
            // Per term the sum of |H|^2 (and the points last), and one block's share.
            std::vector<real_t>           sums_, block_;
        };

        class MemoryProbe final : public Probe {
        public:
            MemoryProbe(const std::shared_ptr<Communicator> &comm, const std::function<void(MemoryLedger &)> &account)
                : comm_(comm), account_(account) {}

            std::vector<std::string> columns() const override {
                // device_used_* are sampled at step ends; device_pool_peak is the stream-ordered pool's
                // high-water mark, buffers an update held while it ran (a stored tangent's window).
                std::vector<std::string> names{"host_peak_rss", "device_used_peak", "device_used_now", "device_pool_peak"};
                if (!account_) return names;
                for (int c = 0; c < MemoryLedger::N_CATEGORIES; ++c) {
                    const std::string category = MemoryLedger::name(MemoryLedger::Category(c));
                    names.push_back("bytes_" + category);
                    names.push_back("bytes_" + category + "_device");
                }
                return names;
            }

            int sample(const StepRecord &, real_t *const values) override {
                struct rusage usage;
                getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
                const real_t rss = real_t(usage.ru_maxrss);  // bytes
#else
                const real_t rss = real_t(usage.ru_maxrss) * 1024;  // kilobytes
#endif
                real_t device_now = 0, pool_peak = 0;
#ifdef SFEM_ENABLE_CUDA
                device_now   = real_t(d_memory_in_use());
                device_peak_ = std::max(device_peak_, device_now);
                pool_peak    = real_t(d_memory_pool_peak());
#endif
                sums_.assign({rss, device_peak_, device_now, pool_peak});
                if (account_) {
                    MemoryLedger ledger;
                    account_(ledger);
                    for (int c = 0; c < MemoryLedger::N_CATEGORIES; ++c) {
                        sums_.push_back(real_t(ledger.host(MemoryLedger::Category(c))));
                        sums_.push_back(real_t(ledger.device(MemoryLedger::Category(c))));
                    }
                }
                if (comm_->size() > 1) comm_->sum(sums_.data(), (int)sums_.size(), smesh::TypeToEnum<real_t>::value());
                std::copy(sums_.begin(), sums_.end(), values);
                return SFEM_SUCCESS;
            }

        private:
            std::shared_ptr<Communicator>       comm_;
            std::function<void(MemoryLedger &)> account_;
            real_t                              device_peak_{0};
            std::vector<real_t>                 sums_;
        };

        class NamedProbe final : public Probe {
        public:
            NamedProbe(const std::string &name, const std::shared_ptr<Probe> &probe) : name_(name), probe_(probe) {}

            std::vector<std::string> columns() const override {
                auto names = probe_->columns();
                for (auto &n : names) n = name_ + "_" + n;
                return names;
            }
            int converged(const StepRecord &record) override { return probe_->converged(record); }
            int sample(const StepRecord &record, real_t *const values) override { return probe_->sample(record, values); }

        private:
            std::string            name_;
            std::shared_ptr<Probe> probe_;
        };
    }  // namespace

    std::shared_ptr<Probe> create_prony_probe(RelaxationGroups &relaxations, const std::shared_ptr<Communicator> &comm) {
        return std::make_shared<PronyProbe>(relaxations, comm);
    }

    namespace {
        class HistorySnapshots final : public StepObserver {
        public:
            HistorySnapshots(RelaxationGroups &relaxations, const std::vector<real_t> &times, const smesh::Path &dir, const ExecutionSpace es)
                : relaxations_(relaxations), times_(times), dir_(dir), es_(es) {}

            int committed(const StepRecord &record) override {
                for (size_t k = 0; k < times_.size(); ++k) {
                    // The grid's times are reached exactly (a cut-back finishes its interval).
                    if (std::abs(record.t - times_[k]) > real_t(1e-6) * record.dt) continue;
                    if (write(k) != SFEM_SUCCESS) return SFEM_FAILURE;
                }
                return SFEM_SUCCESS;
            }

        private:
            RelaxationGroups   &relaxations_;
            std::vector<real_t> times_;
            smesh::Path         dir_;
            ExecutionSpace      es_;
            std::vector<real_t> values_, mean_;  // [c][q][e], [c][e]

            // values_ averaged over each element's points into mean_, then written.
            int write_mean(const smesh::Path &path, const int nc, const int nq, const ptrdiff_t ne) {
                mean_.assign((size_t)nc * ne, real_t(0));
                for (int c = 0; c < nc; ++c)
                    for (int q = 0; q < nq; ++q)
                        for (ptrdiff_t e = 0; e < ne; ++e) mean_[(size_t)c * ne + e] += values_[((size_t)c * nq + q) * ne + e] / nq;
                return smesh::array_write(path, mean_.data(), (ptrdiff_t)mean_.size());
            }

            int write(const size_t k) {
                SFEM_TRACE_SCOPE("HistorySnapshots::write");
                const smesh::Path dir = dir_ / std::to_string(k);
                if (smesh::create_directory(dir_) != SFEM_SUCCESS || smesh::create_directory(dir) != SFEM_SUCCESS) return SFEM_FAILURE;
                for (const auto &block : relaxations_.block_names()) {
                    for (const auto &group : relaxations_.declarations()) {
                        const auto     &r  = relaxations_.relaxation(block, group.group);
                        const int       nc = r.n_components(), nq = r.n_qp();
                        const ptrdiff_t ne = r.n_elements(), n = (ptrdiff_t)nc * nq * ne;
                        const std::string stem = block + "." + group.group + ".";
                        values_.resize(n);
                        copy_to_host(es_, relaxations_.history(block, group.history), n, values_.data());
                        if (write_mean(dir / (stem + "Shist.float64"), nc, nq, ne) != SFEM_SUCCESS) return SFEM_FAILURE;
                        for (int i = 0; i < r.series().n_terms(); ++i) {
                            if (!r.stored(i)) continue;
                            relaxations_.decode_history(block, group.group, i, values_.data());
                            if (write_mean(dir / (stem + "H" + std::to_string(i) + ".float64"), nc, nq, ne) != SFEM_SUCCESS) return SFEM_FAILURE;
                        }
                    }
                }
                return SFEM_SUCCESS;
            }
        };
    }  // namespace

    std::shared_ptr<StepObserver> create_history_snapshots(RelaxationGroups          &relaxations,
                                                           const std::vector<real_t> &times,
                                                           const smesh::Path         &dir,
                                                           const ExecutionSpace       es) {
        return std::make_shared<HistorySnapshots>(relaxations, times, dir, es);
    }

    std::shared_ptr<Probe> create_memory_probe(const std::shared_ptr<Communicator>      &comm,
                                               const std::function<void(MemoryLedger &)> &account) {
        return std::make_shared<MemoryProbe>(comm, account);
    }

    std::shared_ptr<Probe> create_named_probe(const std::string &name, const std::shared_ptr<Probe> &probe) {
        return std::make_shared<NamedProbe>(name, probe);
    }

    // ---------------------------------------------------------------------------------------------

    class StepHistory::Impl {
    public:
        FILE                               *csv{nullptr};
        bool                                verbose{true};
        std::vector<std::shared_ptr<Probe>> probes;
        std::vector<real_t>                 values;

        ~Impl() {
            if (csv) fclose(csv);
        }

        int row(const StepRecord &record, const char *status) {
            SFEM_TRACE_SCOPE("StepHistory::row");
            if (verbose && log_enabled(LOG_SOLID, LOG_NORMAL)) {
                log_printf(LOG_SOLID,
                           LOG_NORMAL,
                           LOG_SUMMARY,
                           "step %d t=%g dt=%g: %s, %d Newton iterations, %d linear, |r|/|r0| = %.3e",
                           record.step,
                           (double)record.t,
                           (double)record.dt,
                           status,
                           record.report.iterations,
                           record.report.linear_iterations,
                           record.report.reference > 0 ? (double)(record.report.final_measure / record.report.reference) : 0.0);
            }
            // Every rank samples (a probe may reduce over the ranks); the one with the file writes.
            std::vector<real_t> sampled;
            for (auto &p : probes) {
                values.assign(p->columns().size(), 0);
                if (p->sample(record, values.data()) != SFEM_SUCCESS) return SFEM_FAILURE;
                sampled.insert(sampled.end(), values.begin(), values.end());
            }
            if (!csv) return SFEM_SUCCESS;

            fprintf(csv,
                    "%d,%.17g,%.17g,%s,%d,%d,%d,%d,%.17g,%.17g,%.17g",
                    record.step,
                    (double)record.t,
                    (double)record.dt,
                    status,
                    record.attempts,
                    record.report.iterations,
                    record.report.linear_iterations,
                    record.report.capped_linear_solves,
                    (double)record.report.reference,
                    (double)record.report.final_measure,
                    (double)record.report.min_alpha);
            for (real_t v : sampled) fprintf(csv, ",%.17g", (double)v);
            fprintf(csv, "\n");
            fflush(csv);
            return SFEM_SUCCESS;
        }
    };

    StepHistory::StepHistory(const std::string &csv_path, const bool verbose) : impl_(std::make_unique<Impl>()) {
        impl_->verbose = verbose;
        if (!csv_path.empty()) {
            const smesh::Path path(csv_path);
            if (!path.parent().to_string().empty()) smesh::create_directory(path.parent());
            impl_->csv = fopen(csv_path.c_str(), "w");
            if (!impl_->csv) SFEM_ERROR("StepHistory: unable to open %s\n", csv_path.c_str());
        }
    }

    StepHistory::~StepHistory() = default;

    void StepHistory::add_probe(const std::shared_ptr<Probe> &probe) { impl_->probes.push_back(probe); }

    int StepHistory::initial(const StepRecord &record) {
        auto &impl = *impl_;
        if (impl.csv) {
            fprintf(impl.csv, "step,t,dt,status,attempts,newton,linear,capped,reference,final,min_alpha");
            for (auto &p : impl.probes) {
                for (auto &c : p->columns()) fprintf(impl.csv, ",%s", c.c_str());
            }
            fprintf(impl.csv, "\n");
        }

        StepRecord at_start    = record;
        at_start.report        = NonlinearReport();
        at_start.report.status = NonlinearReport::Status::CONVERGED;
        const bool verbose     = impl.verbose;
        impl.verbose           = false;
        const int ret          = impl.row(at_start, "initial");
        impl.verbose           = verbose;
        return ret;
    }

    int StepHistory::converged(const StepRecord &record) {
        for (auto &p : impl_->probes) {
            if (p->converged(record) != SFEM_SUCCESS) return SFEM_FAILURE;
        }
        return SFEM_SUCCESS;
    }

    int StepHistory::committed(const StepRecord &record) { return impl_->row(record, to_string(record.report.status)); }

    int StepHistory::failed(const StepRecord &record) {
        if (!impl_->verbose || !log_enabled(LOG_SOLID, LOG_NORMAL)) return SFEM_SUCCESS;
        log_printf(LOG_SOLID,
                   LOG_NORMAL,
                   LOG_EVENT,
                   "step %d t=%g dt=%g: attempt failed, %s after %d Newton iterations, |r| = %.3e (reference %.3e)",
                   record.step,
                   (double)record.t,
                   (double)record.dt,
                   to_string(record.report.status),
                   record.report.iterations,
                   (double)record.report.final_measure,
                   (double)record.report.reference);
        return SFEM_SUCCESS;
    }

}  // namespace sfem
