#include "sfem_Input.hpp"

#include "sfem_base.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include <mpi.h>

static const char *ws = " \t\n\r\f\v";

// trim from end of string (right)
inline static std::string &rtrim(std::string &s, const char *t = ws) {
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline static std::string &ltrim(std::string &s, const char *t = ws) {
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline static std::string &trim(std::string &s, const char *t = ws) { return ltrim(rtrim(s, t), t); }

namespace sfem {
    class YAMLNoIndent::Impl {
    public:
        class Node {
        public:
            virtual ~Node()                      = default;
            virtual void print(std::ostream &os) = 0;
        };

        class StringNode : public Node {
        public:
            std::string value;
            void        print(std::ostream &os) override { os << value; }
        };

        class MapNode : public Node {
        public:
            std::map<std::string, std::string> values;

            void print(std::ostream &os) override {
                for (auto p : values) {
                    os << "- " << p.first << ": " << p.second << "\n";
                }
            }
        };

        static int convert(const std::string &str, std::string &value) {
            value = str;
            return SFEM_SUCCESS;
        }

        static int convert(const std::string &str, int &value) {
            value = atoi(str.c_str());
            return SFEM_SUCCESS;
        }

         static unsigned int convert(const std::string &str, unsigned int &value) {
            // FIXME
            value = atoi(str.c_str());
            return SFEM_SUCCESS;
        }

#ifdef _WIN32
     static ptrdiff_t convert(const std::string &str, ptrdiff_t &value) {
            value = atoll(str.c_str());
            return SFEM_SUCCESS;
        }
#endif

        static int convert(const std::string &str, bool &value) {
            value = str == "true" || atoi(str.c_str());
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
            SFEM_ERROR("[Error] YAMLNoIndent: Unable to find required key \"%s\"!\n", str.c_str());
        }

        static void error_invalid_format(const std::string &str) {
            SFEM_ERROR("[Error] YAMLNoIndent: Unable to convert required key \"%s\"!\n", str.c_str());
        }

        template <typename V>
        int get(const std::string &key, V &value) {
            auto iter = kv.find(key);
            if (iter == kv.end()) return 0;

            auto node = std::dynamic_pointer_cast<StringNode>(iter->second);
            if (!node) {
                error_invalid_format(key);
                return SFEM_FAILURE;
            }

            return convert(node->value, value);
        }

        template <typename V>
        int require(const std::string &key, V &value) {
            auto iter = kv.find(key);
            if (iter == kv.end()) {
                error_not_found(key);
            }

            auto node = std::dynamic_pointer_cast<StringNode>(iter->second);
            if (!node) {
                error_invalid_format(key);
                return SFEM_FAILURE;
            }

            return convert(node->value, value);
        }

        template <typename V>
        int array_require_keys(const std::string &key, V &value) {
            auto iter = kv.find(key);
            if (iter == kv.end()) {
                error_not_found(key);
            }

            // return convert(iter->second, value);
            return SFEM_SUCCESS;
        }

        template <typename V>
        int array_require_values(const std::string &key, V &value) {
            auto iter = kv.find(key);
            if (iter == kv.end()) {
                error_not_found(key);
            }

            // return convert(iter->second, value);
            return SFEM_SUCCESS;
        }

        bool key_exists(const std::string &key) const { return kv.find(key) != kv.end(); }

        std::map<std::string, std::shared_ptr<Node>> kv;
        bool                                         debug{false};
    };

    int YAMLNoIndent::array_require_keys(const std::string &key, std::vector<std::string> &ret) {
        return impl_->array_require_keys(key, ret);
    }

    int YAMLNoIndent::array_require_values(const std::string &key, std::vector<std::string> &ret) {
        return impl_->array_require_values(key, ret);
    }

    std::unique_ptr<YAMLNoIndent> YAMLNoIndent::create_from_file(const std::string &path) {
        std::ifstream file(path.c_str());
        if (!file.good()) SFEM_ERROR("Unable to open file %s\n", path.c_str());

        auto ret = std::make_unique<YAMLNoIndent>();
        ret->parse(file);
        file.close();
        return ret;
    }

    int YAMLNoIndent::add_setting_aux(const std::string &key, const std::string &value) {
        auto node   = std::make_shared<Impl::StringNode>();
        node->value = value;

        impl_->kv[key] = node;
        return SFEM_SUCCESS;
    }

    void YAMLNoIndent::print(std::ostream &os) const {
        for (auto kv : impl_->kv) {
            os << kv.first << ": ";
            kv.second->print(os);
            os << "\n";
        }
    }

    YAMLNoIndent::YAMLNoIndent() : impl_(std::make_unique<Impl>()) {}
    YAMLNoIndent::~YAMLNoIndent() = default;

    int YAMLNoIndent::parse(const std::string &input) {
        std::stringstream ss(input);
        return parse(ss);
    }

#ifdef _WIN32
   static char *strsep(char **stringp, const char *delim) {
        char *start = *stringp;
        char *p;

        p = (start != NULL) ? strpbrk(start, delim) : NULL;

        if (p == NULL) {
            *stringp = NULL;
        } else {
            *p       = '\0';
            *stringp = p + 1;
        }

        return start;
    }
#endif

    static int parse_key_and_value(const std::string &line, std::string &key, std::string &value) {
        char *token, *string, *tofree;
        tofree = string = strdup(line.c_str());
        assert(string != NULL);

        token = strsep(&string, ":");
        assert(token);
        key   = token;
        key   = trim(key);
        token = strsep(&string, ":");

        int comp = 1;
        if (!token) {
            value = "";
        } else {
            value = token;
            value = trim(value);
            comp++;
        }

        free(tofree);
        return comp;
    }

    int YAMLNoIndent::parse(std::istream &input) {
        std::string key;
        std::string value;

        while (input.good()) {
            std::string line;
            std::getline(input, line, '\n');
            line = trim(line);
            if (line.empty()) continue;
            if (line[0] == '#') continue;

            const int ncomp = parse_key_and_value(line, key, value);
            if (ncomp == 1) {
                // Assuming relational array of - key: val

                if (impl_->debug) printf("Reading list %s\n", key.c_str());

                auto map_node  = std::make_shared<Impl::MapNode>();
                impl_->kv[key] = map_node;

                bool in_list = false;
                do {
                    std::getline(input, line, '\n');
                    line = trim(line);
                    if (line.empty() || line[0] == '#') continue;

                    // Check for malformed input
                    if (!in_list) {
                        assert(line[0] == '-');
                        if (line[0] != '-') {
                            SFEM_ERROR("Malfofmed YAML file at key %s (expected value or list)\nline: %s",
                                       key.c_str(),
                                       line.c_str());
                        }

                        in_list = true;
                    }

                    if (line[0] == '-') {
                        int ncomp = parse_key_and_value(line.substr(1, line.size() - 1), key, value);
                        assert(ncomp == 2);
                        if (ncomp < 2) {
                            SFEM_ERROR("Malfofmed YAML file at key (expected value)%s\n", key.c_str());
                        }

                        map_node->values[key] = value;
                    } else {
                        in_list = false;

                        if (impl_->debug) printf("Exited list and encouted %s: %s\n", key.c_str(), value.c_str());
                        auto node      = std::make_shared<Impl::StringNode>();
                        node->value    = value;
                        impl_->kv[key] = node;
                    }

                } while (in_list);

            } else {
                if (impl_->debug) printf("%s: %s\n", key.c_str(), value.c_str());
                auto node      = std::make_shared<Impl::StringNode>();
                node->value    = value;
                impl_->kv[key] = node;
            }
        }

        return SFEM_FAILURE;
    }

    bool YAMLNoIndent::key_exists(const std::string &key) const { return impl_->key_exists(key); }

    int YAMLNoIndent::get(const std::string &key, ptrdiff_t &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, int &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, float &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, double &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, std::string &val) { return impl_->get(key, val); }
    int YAMLNoIndent::get(const std::string &key, bool &val) { return impl_->get(key, val); }

    int YAMLNoIndent::require(const std::string &key, ptrdiff_t &val) { return impl_->require(key, val); }
    int YAMLNoIndent::require(const std::string &key, int &val) { return impl_->require(key, val); }
    int YAMLNoIndent::require(const std::string &key, float &val) { return impl_->require(key, val); }
    int YAMLNoIndent::require(const std::string &key, double &val) { return impl_->require(key, val); }
    int YAMLNoIndent::require(const std::string &key, std::string &val) { return impl_->require(key, val); }

    int YAMLNoIndent::require(const std::string &key, bool &val) { return impl_->require(key, val); }

}  // namespace sfem
