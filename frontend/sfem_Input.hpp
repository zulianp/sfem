#ifndef SFEM_INPUT_HPP
#define SFEM_INPUT_HPP

#include <istream>
#include <ostream>
#include <string>
#include <memory>

namespace sfem {
    class Input {
    public:
        virtual ~Input() = default;
        virtual int get(const std::string &key, ptrdiff_t &val) = 0;
        virtual int get(const std::string &key, int &val) = 0;
        virtual int get(const std::string &key, float &val) = 0;
        virtual int get(const std::string &key, double &val) = 0;
        virtual int get(const std::string &key, std::string &val) = 0;

        virtual int require(const std::string &key, ptrdiff_t &val) = 0;
        virtual int require(const std::string &key, int &val) = 0;
        virtual int require(const std::string &key, float &val) = 0;
        virtual int require(const std::string &key, double &val) = 0;
        virtual int require(const std::string &key, std::string &val) = 0;
    };
    /**
     * @brief A class that reads YAML input without indentation.
     */
    class YAMLNoIndent final : public Input {
    public:
        /// Default constructor
        YAMLNoIndent();
        /// Default destructor
        ~YAMLNoIndent();

        static std::unique_ptr<YAMLNoIndent> create_from_file(const std::string &path);

        /// Parse YAML from a string input
        /// @param input String containing YAML content
        /// @return 0 on success, non-zero on failure
        int parse(const std::string &input);
        /// Parse YAML from an input stream
        /// @param input Input stream containing YAML content
        /// @return 0 on success, non-zero on failure
        int parse(std::istream &input);

        /// Get a ptrdiff_t value for a key if it exists
        /// @param key The key to look up
        /// @param val Reference to store the value if found
        /// @return 0 if found, non-zero if not found
        int get(const std::string &key, ptrdiff_t &val) override;
        /// Get an int value for a key if it exists
        /// @param key The key to look up
        /// @param val Reference to store the value if found
        /// @return 0 if found, non-zero if not found
        int get(const std::string &key, int &val) override;
        /// Get a float value for a key if it exists
        /// @param key The key to look up
        /// @param val Reference to store the value if found
        /// @return 0 if found, non-zero if not found
        int get(const std::string &key, float &val) override;
        /// Get a double value for a key if it exists
        /// @param key The key to look up
        /// @param val Reference to store the value if found
        /// @return 0 if found, non-zero if not found
        int get(const std::string &key, double &val) override;
        /// Get a string value for a key if it exists
        /// @param key The key to look up
        /// @param val Reference to store the value if found
        /// @return 0 if found, non-zero if not found
        int get(const std::string &key, std::string &val) override;

        /// Get a required ptrdiff_t value for a key
        /// @param key The key to look up
        /// @param val Reference to store the value
        /// @return 0 on success, non-zero if key not found
        int require(const std::string &key, ptrdiff_t &val) override;
        /// Get a required int value for a key
        /// @param key The key to look up
        /// @param val Reference to store the value
        /// @return 0 on success, non-zero if key not found
        int require(const std::string &key, int &val) override;
        /// Get a required float value for a key
        /// @param key The key to look up
        /// @param val Reference to store the value
        /// @return 0 on success, non-zero if key not found
        int require(const std::string &key, float &val) override;
        /// Get a required double value for a key
        /// @param key The key to look up
        /// @param val Reference to store the value
        /// @return 0 on success, non-zero if key not found
        int require(const std::string &key, double &val) override;
        /// Get a required string value for a key
        /// @param key The key to look up
        /// @param val Reference to store the value
        /// @return 0 on success, non-zero if key not found
        int require(const std::string &key, std::string &val) override;

        /// Add a new key-value setting
        /// @tparam T Type of the value to add
        /// @param key The key for the setting
        /// @param value The value to store
        /// @return 0 on success, non-zero on failure
        template <typename T>
        int add_setting(const std::string &key, const T &value) {
            return add_setting_aux(key, std::to_string(value));
        }

        /// Stream output operator for YAMLNoIndent
        /// @param os Output stream
        /// @param that YAMLNoIndent instance to output
        /// @return Reference to the output stream
        friend std::ostream &operator<<(std::ostream &os, const YAMLNoIndent &that) {
            that.print(os);
            return os;
        }

        /// Print the YAML content to an output stream
        /// @param os Output stream to print to
        void print(std::ostream &os) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
        int add_setting_aux(const std::string &key, const std::string &value);
    };
}  // namespace sfem

#endif  // SFEM_INPUT_HPP