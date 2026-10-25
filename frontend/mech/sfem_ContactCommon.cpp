#include "sfem_ContactCommon.hpp"

#include <algorithm>

namespace sfem {

    namespace {
        /// The buffer itself on the host, a host copy of it on the device.
        SharedBuffer<real_t> on_host(const SharedBuffer<real_t> &buffer) {
#ifdef SFEM_ENABLE_CUDA
            if (buffer->mem_space() == MEMORY_SPACE_DEVICE) return smesh::to_host(buffer);
#endif
            return buffer;
        }

        class ContactProbe final : public Probe {
        public:
            ContactProbe(const std::function<void(real_t *const)> &reaction,
                         const std::function<real_t()>             &min_gap,
                         const ptrdiff_t                           ndofs,
                         const ExecutionSpace                      es,
                         const real_t                              touch_gap,
                         const real_t                              force_scale,
                         const std::shared_ptr<Communicator>      &comm)
                : reaction_(reaction),
                  min_gap_(min_gap),
                  force_(create_buffer<real_t>(ndofs, es)),
                  touch_gap_(touch_gap),
                  force_scale_(force_scale),
                  comm_(comm) {}

            std::vector<std::string> columns() const override { return {"reaction_x", "reaction_y", "reaction_z", "touched"}; }

            int converged(const StepRecord &) override {
                const ptrdiff_t n = force_->size();
                reaction_(force_->data());
                auto                force = on_host(force_);
                const real_t *const r     = force->data();
                real_t              x = 0, y = 0, z = 0;
#pragma omp parallel for reduction(+ : x, y, z)
                for (ptrdiff_t i = 0; i < n / 3; ++i) {
                    x += r[i * 3];
                    y += r[i * 3 + 1];
                    z += r[i * 3 + 2];
                }
                // The reaction lives on the contact nodes, each owned by one rank.
                real_t sums[3] = {x, y, z};
                if (comm_->size() > 1) comm_->sum(sums, 3, smesh::TypeToEnum<real_t>::value());
                resultant_[0] = force_scale_ * sums[0];
                resultant_[1] = force_scale_ * sums[1];
                resultant_[2] = force_scale_ * sums[2];
                touched_      = min_gap_() <= touch_gap_;
                return SFEM_SUCCESS;
            }

            int sample(const StepRecord &record, real_t *const values) override {
                const bool measured = record.step > 0;
                for (int d = 0; d < 3; ++d) values[d] = measured ? resultant_[d] : real_t(0);
                values[3] = measured && touched_ ? 1 : 0;
                return SFEM_SUCCESS;
            }

        private:
            std::function<void(real_t *const)> reaction_;
            std::function<real_t()>             min_gap_;
            SharedBuffer<real_t>                force_;
            real_t                              touch_gap_;
            real_t                              force_scale_;
            std::shared_ptr<Communicator>       comm_;
            real_t                              resultant_[3]{0, 0, 0};
            bool                                touched_{false};
        };
    }  // namespace

    std::shared_ptr<Probe> create_contact_probe(const std::function<void(real_t *const)> &reaction,
                                                const std::function<real_t()>             &min_gap,
                                                const ptrdiff_t                           ndofs,
                                                const ExecutionSpace                      es,
                                                const real_t                              touch_gap,
                                                const real_t                              force_scale,
                                                const std::shared_ptr<Communicator>      &comm) {
        return std::make_shared<ContactProbe>(reaction, min_gap, ndofs, es, touch_gap, force_scale, comm);
    }

}  // namespace sfem
