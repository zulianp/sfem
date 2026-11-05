#include "sfem_HistoryStorage.hpp"

namespace sfem {

    HistoryStorage history_storage_from_string(const std::string &name) {
        if (name == "full") return HistoryStorage::FULL;
        if (name == "float32") return HistoryStorage::FLOAT32;
        if (name == "float16") return HistoryStorage::FLOAT16;
        if (name == "float16_scaled") return HistoryStorage::FLOAT16_SCALED;
        if (name == "int16_scaled") return HistoryStorage::INT16_SCALED;
        if (name == "int8_scaled") return HistoryStorage::INT8_SCALED;
        if (name == "float16_md") return HistoryStorage::FLOAT16_MD;
        if (name == "int8_md") return HistoryStorage::INT8_MD;

        SFEM_ERROR(
                "Unknown history storage \"%s\" (expected full, float32, float16, float16_scaled, int16_scaled, "
                "int8_scaled, float16_md or int8_md)\n",
                name.c_str());
        return HistoryStorage::FULL;
    }

    const char *to_string(const HistoryStorage storage) {
        switch (storage) {
            case HistoryStorage::FULL:
                return "full";
            case HistoryStorage::FLOAT32:
                return "float32";
            case HistoryStorage::FLOAT16:
                return "float16";
            case HistoryStorage::FLOAT16_SCALED:
                return "float16_scaled";
            case HistoryStorage::INT16_SCALED:
                return "int16_scaled";
            case HistoryStorage::INT8_SCALED:
                return "int8_scaled";
            case HistoryStorage::FLOAT16_MD:
                return "float16_md";
            case HistoryStorage::INT8_MD:
                return "int8_md";
        }
        return "unknown";
    }

    std::size_t bytes_per_value(const HistoryStorage storage) {
        return dispatch_history_codec(storage, [](auto codec) { return sizeof(typename decltype(codec)::storage_t); });
    }

    bool is_scaled(const HistoryStorage storage) {
        return dispatch_history_codec(storage, [](auto codec) { return decltype(codec)::scaled; });
    }

    bool is_mean_deviation(const HistoryStorage storage) {
        return dispatch_history_codec(storage, [](auto codec) { return decltype(codec)::mean_deviation; });
    }

}  // namespace sfem
