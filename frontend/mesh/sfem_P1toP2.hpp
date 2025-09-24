#ifndef SFEM_P1TOP2_HPP
#define SFEM_P1TOP2_HPP

#include <memory>

namespace sfem {

    class Mesh;

    std::shared_ptr<Mesh> convert_p1_mesh_to_p2(const std::shared_ptr<Mesh> &p1_mesh);

}  // namespace sfem

#endif  // SFEM_P1TOP2_HPP
