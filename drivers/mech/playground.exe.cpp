// The elasticity playground: one YAML case, any regime.
//
//   playground case.yaml [--set key.path=value]... [--dry-run]
//
// The case format is documented in frontend/mech/sfem_ElasticityCase.hpp. --set overrides a key
// of the case, --dry-run prints the case as resolved and stops.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Playground.hpp"

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);

    std::string              path;
    std::vector<std::string> overrides;
    bool                     dry_run = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            overrides.push_back(argv[++i]);
        } else if (!strcmp(argv[i], "--dry-run")) {
            dry_run = true;
        } else if (path.empty()) {
            path = argv[i];
        } else {
            fprintf(stderr, "usage: %s case.yaml [--set key.path=value]... [--dry-run]\n", argv[0]);
            return SFEM_FAILURE;
        }
    }
    if (path.empty()) {
        fprintf(stderr, "usage: %s case.yaml [--set key.path=value]... [--dry-run]\n", argv[0]);
        return SFEM_FAILURE;
    }

    auto config = sfem::ElasticityCase::read(path, overrides);
    if (!config) return SFEM_FAILURE;
    if (dry_run) {
        printf("%s", config->emit().c_str());
        return SFEM_SUCCESS;
    }

    auto playground = sfem::Playground::create(ctx->communicator(), *config);
    if (!playground) return SFEM_FAILURE;
    return playground->run();
}
