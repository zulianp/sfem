#ifndef SFEM_PLAYGROUND_TEST_HISTORY_HPP
#define SFEM_PLAYGROUND_TEST_HISTORY_HPP

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sfem {
    namespace test {
        /// The rows of a step history CSV, each column by name.
        inline std::vector<std::map<std::string, double>> read_history(const std::string &path) {
            std::ifstream                              is(path);
            std::string                                line, cell;
            std::vector<std::string>                   names;
            std::vector<std::map<std::string, double>> rows;
            std::getline(is, line);
            for (std::stringstream ss(line); std::getline(ss, cell, ',');) names.push_back(cell);
            while (std::getline(is, line)) {
                std::map<std::string, double> row;
                std::stringstream             ss(line);
                for (const auto &name : names) {
                    std::getline(ss, cell, ',');
                    row[name] = name == "status" ? (cell == "initial" ? -1 : 0) : std::stod(cell);
                }
                rows.push_back(row);
            }
            return rows;
        }
    }  // namespace test
}  // namespace sfem

#endif  // SFEM_PLAYGROUND_TEST_HISTORY_HPP
