#ifndef SFEM_OBSERVERS_HPP
#define SFEM_OBSERVERS_HPP

#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sfem_RelaxationGroups.hpp"
#include "sfem_Stepper.hpp"
#include "smesh_path.hpp"

namespace sfem {

    class Function;
    class Op;
    class DirichletConditions;

    /// Writes the solution and its derived fields every `export_freq` committed steps, the last
    /// step, and the initial state: `<path>/mesh`, and `<path>/out/<field>.<component>.<index>.raw` with
    /// `<path>/out/time.txt`, one index per export shared by every field.
    class FieldExport final : public StepObserver {
    public:
        FieldExport(const std::shared_ptr<Function> &f, const smesh::Path &path, const int export_freq);
        ~FieldExport() override;

        /// A field filled at each export, n_dofs entries in the function's execution space.
        void add_field(const std::string &name, const std::function<int(const StepRecord &, real_t *const)> &fill);

        /// Under MPI: every rank's owned values written by rank 0, in the numbering of
        /// `serial_mesh`, the case's mesh read there on its own (null on the other ranks). Each
        /// node is placed by its id in the file, which the distributed mesh's node_mapping and
        /// the serial mesh's carry alike, so the output is the one a serial run writes.
        /// Collective.
        int gather_to(const std::shared_ptr<Mesh> &serial_mesh);

        int initial(const StepRecord &record) override;
        int committed(const StepRecord &record) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// `<path>/quantities.yaml`: at every committed step the material's objective, the resultant
    /// of its reaction over the constrained dofs and over each Dirichlet condition.
    ///
    /// ```yaml
    /// material_objective_history:
    ///   - time: 1
    ///     value: 0.25
    ///     constrained_reaction_resultant: [1, 0, 0]
    ///     constraint_reactions:
    ///       - {condition: 0, component: 0, resultant: -1}
    /// ```
    class Quantities final : public StepObserver {
    public:
        /// `dirichlet` on the host, or null.
        Quantities(const std::shared_ptr<Function>            &f,
                   const std::shared_ptr<Op>                  &material,
                   const std::shared_ptr<DirichletConditions> &dirichlet,
                   const smesh::Path                          &path);
        ~Quantities() override;

        int initial(const StepRecord &record) override;
        int committed(const StepRecord &record) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Scalars sampled at every committed step, as extra columns of the step history.
    class Probe {
    public:
        virtual ~Probe()                                                                       = default;
        virtual std::vector<std::string> columns() const                                       = 0;
        /// The step's solution before it is committed (StepObserver::converged), for what has to
        /// be read while the step's history and scheme are still open.
        virtual int                      converged(const StepRecord &) { return SFEM_SUCCESS; }
        virtual int                      sample(const StepRecord &record, real_t *const values) = 0;
    };

    /// The displacement at the node nearest to a point, and with a time scheme its carried
    /// velocity and acceleration.
    std::shared_ptr<Probe> create_control_point_probe(const std::shared_ptr<Function> &f,
                                                      const std::vector<real_t>       &point,
                                                      const bool                       with_rates);

    /// The state of a Prony series, group by group (`relaxations`, the material's): the step's
    /// relaxation factor gamma (<group>_gamma) and, for each term k of the series, the RMS over the
    /// integration points of every block of |H_k|, the overstress that term carries (<group>_H<k>,
    /// the stress's unit; the Frobenius norm of the symmetric tensor). Sampled once the step is
    /// committed. Prints which g and tau each k is. Every block must hold the same series.
    std::shared_ptr<Probe> create_prony_probe(RelaxationGroups &relaxations, const std::shared_ptr<Communicator> &comm);

    /// The same probe with its columns prefixed by `name` and an underscore, so that several of a
    /// kind can share one history.
    std::shared_ptr<Probe> create_named_probe(const std::string &name, const std::shared_ptr<Probe> &probe);

    /// One CSV row per committed step: the step, its time, how its nonlinear solve went, and the
    /// probes' columns. Also prints the row's summary.
    class StepHistory final : public StepObserver {
    public:
        /// An empty path writes no file.
        explicit StepHistory(const std::string &csv_path, const bool verbose = true);
        ~StepHistory() override;

        void add_probe(const std::shared_ptr<Probe> &probe);

        int initial(const StepRecord &record) override;
        int converged(const StepRecord &record) override;
        int committed(const StepRecord &record) override;
        int failed(const StepRecord &record) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_OBSERVERS_HPP
