#include "sfem_DerefinedTimeScheme.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_InertiaPotential.hpp"
#include "sfem_logger.hpp"

namespace sfem {

    namespace {
        class DerefinedTimeScheme final : public TimeScheme {
        public:
            DerefinedTimeScheme(const std::shared_ptr<TimeScheme> &fine, const std::shared_ptr<InertiaPotential> &inertia)
                : fine_(fine), inertia_(inertia) {}

            real_t        shift() const override { return fine_->shift(); }
            const real_t *history() const override { return nullptr; }
            real_t        weight(const char *name) const override { return fine_->weight(name); }
            bool          has_acceleration() const override { return fine_->has_acceleration(); }

            std::shared_ptr<Op> inertia_op() const override {
                if (!inertia_) return nullptr;
                inertia_->set_alpha(fine_->weight("alpha_a"));
                return inertia_;
            }

            void begin_step(real_t, real_t) override { refuse("begin_step"); }
            void advance(const real_t *const) override { refuse("advance"); }
            void reconstruct(const real_t *const, real_t *const, real_t *const) const override { refuse("reconstruct"); }
            void start(real_t *const) const override { refuse("start"); }
            void carried_state(real_t *const, real_t *const) const override { refuse("carried_state"); }

        private:
            static void refuse(const char *what) {
                SFEM_ERROR("a derefined time scheme is not stepped (%s): the fine level steps\n", what);
            }

            std::shared_ptr<TimeScheme>       fine_;
            std::shared_ptr<InertiaPotential> inertia_;
        };
    }  // namespace

    std::shared_ptr<TimeScheme> derefine_time_scheme(const std::shared_ptr<TimeScheme>    &fine,
                                                     const std::shared_ptr<FunctionSpace> &coarse) {
        if (!fine) return nullptr;
        std::shared_ptr<InertiaPotential> inertia;
        if (auto fine_inertia = std::dynamic_pointer_cast<InertiaPotential>(fine->inertia_op())) {
            inertia = std::dynamic_pointer_cast<InertiaPotential>(fine_inertia->derefine_op(coarse));
            if (!inertia) return nullptr;
        } else if (fine->inertia_op()) {
            SFEM_ERROR("derefine_time_scheme: the scheme's inertia is not an InertiaPotential\n");
            return nullptr;
        }
        return std::make_shared<DerefinedTimeScheme>(fine, inertia);
    }

}  // namespace sfem
