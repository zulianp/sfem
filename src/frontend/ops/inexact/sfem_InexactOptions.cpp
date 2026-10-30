#include "sfem_InexactOptions.hpp"

#include "sfem_base.hpp"

namespace sfem {

    namespace {
        InexactOptions::Store store_from_string(const std::string &name) {
            if (name == "points") return InexactOptions::Store::POINTS;
            if (name == "element_matrix") return InexactOptions::Store::ELEMENT_MATRIX;
            if (name == "bsr") return InexactOptions::Store::BSR;
            SFEM_ERROR("inexact: unknown store \"%s\" (points, element_matrix, bsr)\n", name.c_str());
            return InexactOptions::Store::POINTS;
        }

        InexactOptions::Encoding encoding_from_string(const std::string &name) {
            if (name == "f64") return InexactOptions::Encoding::F64;
            if (name == "f32") return InexactOptions::Encoding::F32;
            if (name == "f16") return InexactOptions::Encoding::F16;
            if (name == "i8md") return InexactOptions::Encoding::I8_MEAN_DEVIATION;
            if (name == "i8pair") return InexactOptions::Encoding::I8_PAIR_SCALE;
            if (name == "i8") return InexactOptions::Encoding::I8_BLOCK;
            SFEM_ERROR("inexact: unknown encoding \"%s\" (f64, f32, f16, i8md, i8pair, i8)\n", name.c_str());
            return InexactOptions::Encoding::F32;
        }

        InexactOptions::Switch switch_from_string(const std::string &name) {
            if (name == "auto") return InexactOptions::Switch::AUTO;
            if (name == "on") return InexactOptions::Switch::ON;
            if (name == "off") return InexactOptions::Switch::OFF;
            SFEM_ERROR("inexact: unknown fold \"%s\" (auto, on, off)\n", name.c_str());
            return InexactOptions::Switch::AUTO;
        }

        InexactOptions::Traversal traversal_from_string(const std::string &name) {
            if (name == "auto") return InexactOptions::Traversal::AUTO;
            if (name == "packed") return InexactOptions::Traversal::PACKED;
            if (name == "atomic") return InexactOptions::Traversal::ATOMIC;
            SFEM_ERROR("inexact: unknown traversal \"%s\" (auto, packed, atomic)\n", name.c_str());
            return InexactOptions::Traversal::AUTO;
        }

        const char *to_string(const InexactOptions::Store store) {
            switch (store) {
                case InexactOptions::Store::POINTS:
                    return "points";
                case InexactOptions::Store::ELEMENT_MATRIX:
                    return "element_matrix";
                case InexactOptions::Store::BSR:
                    return "bsr";
            }
            return "?";
        }

        const char *to_string(const InexactOptions::Encoding encoding) {
            switch (encoding) {
                case InexactOptions::Encoding::F64:
                    return "f64";
                case InexactOptions::Encoding::F32:
                    return "f32";
                case InexactOptions::Encoding::F16:
                    return "f16";
                case InexactOptions::Encoding::I8_MEAN_DEVIATION:
                    return "i8md";
                case InexactOptions::Encoding::I8_PAIR_SCALE:
                    return "i8pair";
                case InexactOptions::Encoding::I8_BLOCK:
                    return "i8";
            }
            return "?";
        }

        const char *to_string(const InexactOptions::Switch value) {
            switch (value) {
                case InexactOptions::Switch::AUTO:
                    return "auto";
                case InexactOptions::Switch::ON:
                    return "on";
                case InexactOptions::Switch::OFF:
                    return "off";
            }
            return "?";
        }

        const char *to_string(const InexactOptions::Traversal value) {
            switch (value) {
                case InexactOptions::Traversal::AUTO:
                    return "auto";
                case InexactOptions::Traversal::PACKED:
                    return "packed";
                case InexactOptions::Traversal::ATOMIC:
                    return "atomic";
            }
            return "?";
        }
    }  // namespace

    void InexactOptions::validate() const {
        // What each store offers; anything else is refused rather than mapped to a neighbour.
        bool offered = false;
        switch (store) {
            case Store::POINTS:
                offered = encoding == Encoding::F64 || encoding == Encoding::F32 || encoding == Encoding::F16 ||
                          encoding == Encoding::I8_MEAN_DEVIATION;
                break;
            case Store::ELEMENT_MATRIX:
                offered = encoding == Encoding::F64 || encoding == Encoding::F32 || encoding == Encoding::F16 ||
                          encoding == Encoding::I8_PAIR_SCALE;
                break;
            case Store::BSR:
                offered = encoding == Encoding::F16 || encoding == Encoding::I8_BLOCK;
                break;
        }
        if (!offered) {
            SFEM_ERROR("inexact: store %s takes no encoding %s (points: f64 f32 f16 i8md; element_matrix: f64 f32 "
                       "f16 i8pair; bsr: f16 i8)\n",
                       to_string(store),
                       to_string(encoding));
        }
        if (store != Store::POINTS && fold == Switch::ON) {
            SFEM_ERROR("inexact: the fold is a property of the point store\n");
        }
    }

    InexactOptions InexactOptions::from_strings(const std::string &store,
                                                const std::string &encoding,
                                                const std::string &fold,
                                                const std::string &traversal) {
        InexactOptions ret;
        ret.store     = store_from_string(store);
        ret.encoding  = encoding_from_string(encoding);
        ret.fold      = switch_from_string(fold);
        ret.traversal = traversal_from_string(traversal);
        ret.validate();
        return ret;
    }

#ifdef SFEM_ENABLE_RYAML
    InexactOptions inexact_options_from_yaml(const ryml::ConstNodeRef &node) {
        const auto read = [&](const char *const key, const char *const fallback) {
            if (!node.has_child(ryml::to_csubstr(key))) return std::string(fallback);
            const auto value = node[ryml::to_csubstr(key)].val();
            return std::string(value.str, value.len);
        };
        InexactOptions defaults;
        return InexactOptions::from_strings(read("store", to_string(defaults.store)),
                            read("encoding", to_string(defaults.encoding)),
                            read("fold", to_string(defaults.fold)),
                            read("traversal", to_string(defaults.traversal)));
    }
#endif

    std::string InexactOptions::describe() const {
        return std::string(to_string(store)) + " " + to_string(encoding) + " fold=" + to_string(fold) +
               " traversal=" + to_string(traversal);
    }

}  // namespace sfem
