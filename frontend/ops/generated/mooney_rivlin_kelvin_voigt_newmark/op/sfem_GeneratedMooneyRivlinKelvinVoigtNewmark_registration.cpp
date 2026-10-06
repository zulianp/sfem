#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark.hpp"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void register_GeneratedMooneyRivlinKelvinVoigtNewmark_generated_op() {
        Factory::register_op("GeneratedMooneyRivlinKelvinVoigtNewmark", GeneratedMooneyRivlinKelvinVoigtNewmark::create);
        Factory::register_op("ss:GeneratedMooneyRivlinKelvinVoigtNewmark", GeneratedMooneyRivlinKelvinVoigtNewmark::create);
    }
}  // namespace sfem
