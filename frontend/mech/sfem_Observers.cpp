#include "sfem_Observers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "sfem_Function.hpp"
#include "sfem_blas_factory.hpp"
#include "smesh_glob.hpp"

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
        std::vector<real_t>           all, serial;

        /// `values` (n_dofs, on the host) written as `name`, by this rank or gathered to rank 0.
        int emit(const char *const name, const real_t *const values) {
            if (!gathered) return output->write_step(name, index, values);

            const int bs = f->space()->block_size();
            comm->allgatherv(values, (int)f->space()->n_owned_dofs(), all.data(), counts.data(), displs.data());
            if (!output) return SFEM_SUCCESS;
            for (size_t k = 0; k < place.size(); ++k) {
                for (int d = 0; d < bs; ++d) serial[place[k] * bs + d] = all[k * bs + d];
            }
            return output->write_step(name, index, serial.data());
        }

        int write(const StepRecord &record) {
            SFEM_TRACE_SCOPE("FieldExport::write");

            const ExecutionSpace es    = f->execution_space();
            const ptrdiff_t      ndofs = f->space()->n_dofs();

            if (emit("disp", on_host(es, record.x, ndofs)->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            for (auto &field : fields) {
                if (field.fill(record, scratch->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
                if (emit(field.name.c_str(), on_host(es, scratch->data(), ndofs)->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
            }

            if (output) output->log_time(record.t);
            index++;
            return SFEM_SUCCESS;
        }
    };

    FieldExport::FieldExport(const std::shared_ptr<Function> &f, const smesh::Path &path, const int export_freq)
        : impl_(std::make_unique<Impl>()) {
        impl_->f           = f;
        impl_->path        = path;
        impl_->export_freq = std::max(1, export_freq);

        const ptrdiff_t ndofs = f->space()->n_dofs();
        impl_->scratch        = create_buffer<real_t>(ndofs, f->execution_space());
    }

    FieldExport::~FieldExport() = default;

    void FieldExport::add_field(const std::string &name, const std::function<int(const StepRecord &, real_t *const)> &fill) {
        impl_->fields.push_back({name, fill});
    }

    int FieldExport::gather_to(const std::shared_ptr<Mesh> &serial_mesh) {
        auto &impl = *impl_;
        auto  fs   = impl.f->space();
        auto  mesh = fs->mesh_ptr();
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
        const ptrdiff_t n_owned = fs->n_owned_dofs() / bs;
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
            impl.serial.assign((size_t)n_serial * bs, 0);
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
            smesh::semistructured_export_as_standard(fs->mesh_ptr(), impl.path / "mesh");
        } else {
            fs->mesh_ptr()->write(impl.path / "mesh");
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

    namespace {
        class ControlPoint final : public Probe {
        public:
            ControlPoint(const std::shared_ptr<Function> &f, const std::vector<real_t> &point, const bool with_rates)
                : f_(f), with_rates_(with_rates) {
                auto      mesh = f->space()->mesh_ptr();
                const int dim  = mesh->spatial_dimension();
                auto      xyz  = mesh->points()->data();
                // Distributed, the nearest of the owned nodes over the ranks, sampled by its owner
                // (the lowest rank on a tie) and summed: the history is written by rank 0, which
                // may hold none of the body the point is in.
                comm_                  = mesh->comm();
                const ptrdiff_t nnodes = f->space()->n_owned_dofs() / f->space()->block_size();

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

                const ptrdiff_t ndofs = f->space()->n_dofs();
                velocity_             = create_buffer<real_t>(ndofs, f->execution_space());
                acceleration_         = create_buffer<real_t>(ndofs, f->execution_space());
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
                const ExecutionSpace es    = f_->execution_space();
                const ptrdiff_t      ndofs = f_->space()->n_dofs();
                const int            bs    = f_->space()->block_size();

                const int n_values = (with_rates_ ? 3 : 1) * bs;
                auto      u        = on_host(es, record.x, ndofs);
                for (int d = 0; d < bs; ++d) values[d] = u->data()[node_ * bs + d];
                if (!with_rates_) return reduce(values, n_values);

                const bool with_acceleration = record.scheme && record.scheme->has_acceleration();
                if (record.scheme) {
                    record.scheme->carried_state(velocity_->data(), with_acceleration ? acceleration_->data() : nullptr);
                }
                auto v = on_host(es, velocity_->data(), ndofs);
                auto a = on_host(es, acceleration_->data(), ndofs);
                for (int d = 0; d < bs; ++d) {
                    values[bs + d]     = record.scheme ? v->data()[node_ * bs + d] : real_t(0);
                    values[2 * bs + d] = with_acceleration ? a->data()[node_ * bs + d] : real_t(0);
                }
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
            std::shared_ptr<Communicator> comm_;
            bool                          with_rates_;
            bool                          owner_{true};
            ptrdiff_t                     node_{0};
            SharedBuffer<real_t>      velocity_, acceleration_;
        };
    }  // namespace

    std::shared_ptr<Probe> create_control_point_probe(const std::shared_ptr<Function> &f,
                                                      const std::vector<real_t>       &point,
                                                      const bool                       with_rates) {
        return std::make_shared<ControlPoint>(f, point, with_rates);
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
                        if (comm->rank() == 0) {
                            printf("Prony probe: %s_H%d is the term g = %g, tau = %g\n", d.group.c_str(), k, (double)series.g(k), (double)series.tau(k));
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
                }
                return names;
            }

            int sample(const StepRecord &, real_t *const values) override {
                int c = 0;
                std::vector<real_t> H;
                for (const auto &d : relaxations_.declarations()) {
                    values[c++] = relaxations_.relaxation(blocks_[0], d.group).gamma();
                    const int n = relaxations_.relaxation(blocks_[0], d.group).series().n_terms();
                    // A symmetric tensor's off-diagonal components (after the diagonal ones, 3 of 6 or
                    // 1 of 3) count twice in its Frobenius norm.
                    const int n_diagonal = d.n_components == 6 ? 3 : d.n_components == 3 ? 2 : d.n_components;
                    for (int k = 0; k < n; ++k) {
                        real_t sums[2] = {0, 0};  // sum of |H|^2 over the points, how many points
                        for (const auto &b : blocks_) {
                            const auto     &r      = relaxations_.relaxation(b, d.group);
                            const ptrdiff_t points = static_cast<ptrdiff_t>(r.n_qp()) * r.n_elements();
                            H.resize(static_cast<size_t>(r.n_components()) * points);
                            r.decode_history(k, H.data());
                            for (int comp = 0; comp < r.n_components(); ++comp) {
                                const real_t weight = comp < n_diagonal ? 1 : 2;
                                const real_t *const h = &H[comp * points];
                                real_t              s = 0;
#pragma omp parallel for reduction(+ : s)
                                for (ptrdiff_t q = 0; q < points; ++q) s += h[q] * h[q];
                                sums[0] += weight * s;
                            }
                            sums[1] += points;
                        }
                        if (comm_->size() > 1) comm_->sum(sums, 2, smesh::TypeToEnum<real_t>::value());
                        values[c++] = sums[1] > 0 ? std::sqrt(sums[0] / sums[1]) : 0;
                    }
                }
                return SFEM_SUCCESS;
            }

        private:
            RelaxationGroups             &relaxations_;
            std::shared_ptr<Communicator> comm_;
            std::vector<std::string>      blocks_;
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
            if (verbose) {
                printf("step %d t=%g dt=%g: %s, %d Newton iterations, %d linear, |r|/|r0| = %.3e\n",
                       record.step,
                       (double)record.t,
                       (double)record.dt,
                       status,
                       record.report.iterations,
                       record.report.linear_iterations,
                       record.report.reference > 0 ? (double)(record.report.final_measure / record.report.reference) : 0.0);
                fflush(stdout);
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
        at_start.report.status = NonlinearReport::CONVERGED;
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
        if (!impl_->verbose) return SFEM_SUCCESS;
        printf("step %d t=%g dt=%g: attempt failed, %s after %d Newton iterations\n",
               record.step,
               (double)record.t,
               (double)record.dt,
               to_string(record.report.status),
               record.report.iterations);
        fflush(stdout);
        return SFEM_SUCCESS;
    }

}  // namespace sfem
