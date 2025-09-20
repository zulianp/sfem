#include "sfem_Env.hpp"
#include "sfem_defs.h"
// FIXME
#include "sfem_Buffer.hpp"
#include <type_traits>

namespace sfem {
    template <typename T>
    struct FromString {};
    
    template <>
    struct FromString<std::string> {
        static const std::string& parse(const std::string& str_value) { return str_value; }
    };

    template <>
    struct FromString<bool> {
        static bool parse(const std::string& str_value) { return str_value == "true" || str_value == "1"; }
    };

    template <>
    struct FromString<int64_t> {
        static int64_t parse(const std::string& str_value) { return std::stoll(str_value); }
    };

    template <>
    struct FromString<uint64_t> {
        static uint64_t parse(const std::string& str_value) { return std::stoull(str_value); }
    };

    template <>
    struct FromString<int32_t> {
        static int32_t parse(const std::string& str_value) { return static_cast<int32_t>(std::stoll(str_value)); }
    };
    
    template <>
    struct FromString<uint32_t> {
        static uint32_t parse(const std::string& str_value) { return static_cast<uint32_t>(std::stoull(str_value)); }
    };

    template <>
    struct FromString<int16_t> {
        static int16_t parse(const std::string& str_value) { return static_cast<int16_t>(std::stoll(str_value)); }
    };

    template <> 
    struct FromString<int8_t> {
        static int8_t parse(const std::string& str_value) { return static_cast<int8_t>(std::stoi(str_value)); }
    };

    template <>
    struct FromString<float> {
        static float parse(const std::string& str_value) { return static_cast<float>(std::stod(str_value)); }
    };

    template <>
    struct FromString<double> {
        static double parse(const std::string& str_value) { return std::stod(str_value); }
    };

    template <>
    struct FromString<enum ExecutionSpace> {
        static enum ExecutionSpace parse(const std::string& str_value) { return execution_space_from_string(str_value); }
    };

    template <>
    struct FromString<enum ElemType> {
        static enum ElemType parse(const std::string& str_value) { return type_from_string(str_value.c_str()); }
    };

    template <typename T>
    T from_string(const std::string& str_value) {
        return FromString<T>::parse(str_value);
    }

    // Explicit instantiation of from_string for all types
    template double from_string<double>(const std::string& str_value);
    template float from_string<float>(const std::string& str_value);
    template int64_t from_string<int64_t>(const std::string& str_value);
    template uint64_t from_string<uint64_t>(const std::string& str_value);
    template int32_t from_string<int32_t>(const std::string& str_value);
    template uint32_t from_string<uint32_t>(const std::string& str_value);
    template int16_t from_string<int16_t>(const std::string& str_value);
    template int8_t from_string<int8_t>(const std::string& str_value);
    template bool from_string<bool>(const std::string& str_value);
    template ExecutionSpace from_string<enum ExecutionSpace>(const std::string& str_value);
    template enum ElemType from_string<enum ElemType>(const std::string& str_value);
    template std::string from_string<std::string>(const std::string& str_value);


}  // namespace sfem
