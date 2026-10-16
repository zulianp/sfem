#ifndef SFEM_OBSERVERS_HPP
#define SFEM_OBSERVERS_HPP

#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
        virtual int                      sample(const StepRecord &record, real_t *const values) = 0;
    };

    /// The displacement at the node nearest to a point, and with a time scheme its carried
    /// velocity and acceleration.
    std::shared_ptr<Probe> create_control_point_probe(const std::shared_ptr<Function> &f,
                                                      const std::vector<real_t>       &point,
                                                      const bool                       with_rates);

    /// One CSV row per committed step: the step, its time, how its nonlinear solve went, and the
    /// probes' columns. Also prints the row's summary.
    class StepHistory final : public StepObserver {
    public:
        /// An empty path writes no file.
        explicit StepHistory(const std::string &csv_path, const bool verbose = true);
        ~StepHistory() override;

        void add_probe(const std::shared_ptr<Probe> &probe);

        int initial(const StepRecord &record) override;
        int committed(const StepRecord &record) override;
        int failed(const StepRecord &record) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_OBSERVERS_HPP
