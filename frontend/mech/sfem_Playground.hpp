#ifndef SFEM_PLAYGROUND_HPP
#define SFEM_PLAYGROUND_HPP

#include <memory>

#include "sfem_ElasticityCase.hpp"
#include "sfem_NewtonSolver.hpp"
#include "sfem_Stepper.hpp"
#include "sfem_aliases.hpp"

#ifdef SFEM_ENABLE_RYAML

namespace sfem {

    class Function;
    class Op;

    /// An elasticity problem assembled from an `ElasticityCase`: the mesh and its named sidesets,
    /// the material and its loads, the boundary conditions, the time scheme the material holds,
    /// Newton with its linearised step, the stepper and its observers. The steady, quasi-static
    /// and dynamic regimes are the same objects with or without a scheme and with one step or
    /// many.
    class Playground {
    public:
        /// Null, with the reason on stderr, when the case cannot be assembled.
        static std::unique_ptr<Playground> create(const std::shared_ptr<Communicator> &comm, const ElasticityCase &config);
        ~Playground();

        /// Every step of the case. SFEM_FAILURE when a step could not be solved.
        int run();

        std::shared_ptr<Function>   function() const;
        std::shared_ptr<Op>         material() const;
        std::shared_ptr<TimeScheme> scheme() const;
        NewtonSolver               &newton();
        Stepper                    &stepper();
        /// The state: the initial one before `run`, the last solution after it.
        SharedBuffer<real_t>        solution() const;

    private:
        Playground();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_ENABLE_RYAML
#endif  // SFEM_PLAYGROUND_HPP
