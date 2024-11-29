#include "sfem_Input.hpp"

#include "sfem_base.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include <mpi.h>

namespace sfem {
    class YAMLNoIndent::Impl {
    public:
        static int convert(const std::string &str, std::string &value) {
            value = str;
            return SFEM_SUCCESS;
        }

        static int convert(const std::string &str, int &value) {
            value = atoi(str.c_str());
            return SFEM_SUCCESS;
        }

        static long convert(const std::string &str, long &value) {
            value = atol(str.c_str());
            return SFEM_SUCCESS;
        }

        static float convert(const std::string &str, float &value) {
            value = atof(str.c_str());
            return SFEM_SUCCESS;
        }

        static double convert(const std::string &str, double &value) {
            value = atof(str.c_str());
            return SFEM_SUCCESS;
        }

        static void error_not_found(const std::string &str) {
            fprintf(stderr, "[Error] YAMLNoIndent: Unable to find required key %s!\n", str.c_str());
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        template <typename V>
        int get(const std::string &key, V &value) {
            auto iter = kv.find(key);
            if (iter == kv.end()) return 0;
            return convert(iter->second, value);
        }

        template <typename V>
        int require(const std::string &key, V &value) {
            auto iter = kv.find(key);
            if (iter == kv.end()) {
                error_not_found(key);
            }

            return convert(iter->second, value);
        }

        std::map<std::string, std::string> kv;
    };

    std::unique_ptr<YAMLNoIndent> YAMLNoIndent::create_from_file(const std::string &path) {
        std::ifstream file(path.c_str());
        if (!file.good()) SFEM_ERROR("Unable to open file %s", path.c_str());

        auto ret = std::make_unique<YAMLNoIndent>();
        ret->parse(file);
        file.close();
        return ret;
    }

    int YAMLNoIndent::add_setting_aux(const std::string &key, const std::string &value) {
        impl_->kv[key] = value;
        return SFEM_SUCCESS;
    }

    void YAMLNoIndent::print(std::ostream &os) const {
        for (auto kv : impl_->kv) {
            os << kv.first << ": " << kv.second << "\n";
        }
    }

    YAMLNoIndent::YAMLNoIndent() : impl_(std::make_unique<Impl>()) {}
    YAMLNoIndent::~YAMLNoIndent() = default;

    int YAMLNoIndent::parse(const std::string &input) {
        std::stringstream ss(input);
        return parse(ss);
    }

    int YAMLNoIndent::parse(std::istream &input) {
        char key[1024];
        char value[1024];
        while (input.good()) {
            std::string line;
            std::getline(input, line, '\n');

            const int n = sscanf(line.c_str(), "%s: %s", key, value);
            printf("%s: %s\n", key, value);
            impl_->kv[key] = value;
        }

        return SFEM_FAILURE;
    }

    int YAMLNoIndent::get(const std::string &key, ptrdiff_t &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, int &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, float &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, double &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, std::string &val) { return impl_->get(key, val); }
    int YAMLNoIndent::require(const std::string &key, ptrdiff_t &val) {
        return impl_->require(key, val);
    }
    int YAMLNoIndent::require(const std::string &key, int &val) { return impl_->require(key, val); }
    int YAMLNoIndent::require(const std::string &key, float &val) {
        return impl_->require(key, val);
    }
    int YAMLNoIndent::require(const std::string &key, double &val) {
        return impl_->require(key, val);
    }
    int YAMLNoIndent::require(const std::string &key, std::string &val) {
        return impl_->require(key, val);
    }

}  // namespace sfem
