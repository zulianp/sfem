#ifndef SFEM_ENV_HPP
#define SFEM_ENV_HPP

#include <cstdlib>
#include <string>

namespace sfem {

    template <typename T>
    T from_string(const std::string& str_value);

    class Env {
    public:
        template <typename T>
        static T read(const std::string& var_name, const T default_value) {
            const char* value = std::getenv(var_name.c_str());
            if (value) {
                return from_string<T>(value);
            }
            return default_value;
        }
        
        static std::string read_string(const std::string& var_name, const std::string& default_value) {
            const char* value = std::getenv(var_name.c_str());
            if (value) {
                return from_string<std::string>(value);
            }
            return default_value;
        }
    };

}  // namespace sfem

#endif