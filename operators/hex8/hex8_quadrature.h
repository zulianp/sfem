#ifndef HEX8_QUADRATURE_H
#define HEX8_QUADRATURE_H

#define q6_n 6

static const scalar_t q6_w[q6_n] = {0.16666666666666666666666666666667,
                                    0.16666666666666666666666666666667,
                                    0.16666666666666666666666666666667,
                                    0.16666666666666666666666666666667,
                                    0.16666666666666666666666666666667,
                                    0.16666666666666666666666666666667};

static const scalar_t q6_x[q6_n] = {0.0, 0.5, 0.5, 0.5, 0.5, 1.0};
static const scalar_t q6_y[q6_n] = {0.5, 0.0, 0.5, 0.5, 1.0, 0.5};
static const scalar_t q6_z[q6_n] = {0.5, 0.5, 0.0, 1.0, 0.5, 0.5};

#define q8_n 8

static const scalar_t q8_w[q8_n] = {1. / 8, 1. / 8, 1. / 8, 1. / 8, 1. / 8, 1. / 8};

static const scalar_t q8_x[q8_n] = {0.2113248654,
                                    0.7886751346,
                                    0.2113248654,
                                    0.7886751346,
                                    0.2113248654,
                                    0.7886751346,
                                    0.2113248654,
                                    0.7886751346};
static const scalar_t q8_y[q8_n] = {0.2113248654,
                                    0.2113248654,
                                    0.7886751346,
                                    0.7886751346,
                                    0.2113248654,
                                    0.2113248654,
                                    0.7886751346,
                                    0.7886751346};
static const scalar_t q8_z[q8_n] = {0.2113248654,
                                    0.2113248654,
                                    0.2113248654,
                                    0.2113248654,
                                    0.7886751346,
                                    0.7886751346,
                                    0.7886751346,
                                    0.7886751346};

#define q27_n 27

static const scalar_t q27_w[q27_n] = {
        0.021433470507545, 0.034293552812071, 0.021433470507545, 0.034293552812071,
        0.054869684499314, 0.034293552812071, 0.021433470507545, 0.034293552812071,
        0.021433470507545, 0.034293552812071, 0.054869684499314, 0.034293552812071,
        0.054869684499314, 0.087791495198903, 0.054869684499314, 0.034293552812071,
        0.054869684499314, 0.034293552812071, 0.021433470507545, 0.034293552812071,
        0.021433470507545, 0.034293552812071, 0.054869684499314, 0.034293552812071,
        0.021433470507545, 0.034293552812071, 0.021433470507545};

static const scalar_t q27_x[q27_n] = {
        0.112701665379258, 0.500000000000000, 0.887298334620742, 0.112701665379258,
        0.500000000000000, 0.887298334620742, 0.112701665379258, 0.500000000000000,
        0.887298334620742, 0.112701665379258, 0.500000000000000, 0.887298334620742,
        0.112701665379258, 0.500000000000000, 0.887298334620742, 0.112701665379258,
        0.500000000000000, 0.887298334620742, 0.112701665379258, 0.500000000000000,
        0.887298334620742, 0.112701665379258, 0.500000000000000, 0.887298334620742,
        0.112701665379258, 0.500000000000000, 0.887298334620742};

static const scalar_t q27_y[q27_n] = {
        0.112701665379258, 0.112701665379258, 0.112701665379258, 0.500000000000000,
        0.500000000000000, 0.500000000000000, 0.887298334620742, 0.887298334620742,
        0.887298334620742, 0.112701665379258, 0.112701665379258, 0.112701665379258,
        0.500000000000000, 0.500000000000000, 0.500000000000000, 0.887298334620742,
        0.887298334620742, 0.887298334620742, 0.112701665379258, 0.112701665379258,
        0.112701665379258, 0.500000000000000, 0.500000000000000, 0.500000000000000,
        0.887298334620742, 0.887298334620742, 0.887298334620742};

static const scalar_t q27_z[q27_n] = {
        0.112701665379258, 0.112701665379258, 0.112701665379258, 0.112701665379258,
        0.112701665379258, 0.112701665379258, 0.112701665379258, 0.112701665379258,
        0.112701665379258, 0.500000000000000, 0.500000000000000, 0.500000000000000,
        0.500000000000000, 0.500000000000000, 0.500000000000000, 0.500000000000000,
        0.500000000000000, 0.500000000000000, 0.887298334620742, 0.887298334620742,
        0.887298334620742, 0.887298334620742, 0.887298334620742, 0.887298334620742,
        0.887298334620742, 0.887298334620742, 0.887298334620742

};

#define q58_n 58

static const scalar_t q58_x[q58_n] = {0.19315926520414550308255755512972,
                                      0.5,
                                      0.5,
                                      0.5,
                                      0.5,
                                      0.80684073479585449691744244487028,
                                      0.061156438371160856756612120502834,
                                      0.061156438371160856756612120502834,
                                      0.061156438371160856756612120502834,
                                      0.061156438371160856756612120502834,
                                      0.5,
                                      0.5,
                                      0.5,
                                      0.5,
                                      0.93884356162883914324338787949717,
                                      0.93884356162883914324338787949717,
                                      0.93884356162883914324338787949717,
                                      0.93884356162883914324338787949717,
                                      0.21794459648998497286669050066846,
                                      0.21794459648998497286669050066846,
                                      0.21794459648998497286669050066846,
                                      0.21794459648998497286669050066846,
                                      0.78205540351001502713330949933154,
                                      0.78205540351001502713330949933154,
                                      0.78205540351001502713330949933154,
                                      0.78205540351001502713330949933154,
                                      0.064950107669012041192468095568038,
                                      0.064950107669012041192468095568038,
                                      0.064950107669012041192468095568038,
                                      0.064950107669012041192468095568038,
                                      0.93504989233098795880753190443196,
                                      0.93504989233098795880753190443196,
                                      0.93504989233098795880753190443196,
                                      0.93504989233098795880753190443196,
                                      0.030734789067664127335511565198461,
                                      0.030734789067664127335511565198461,
                                      0.030734789067664127335511565198461,
                                      0.030734789067664127335511565198461,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.96926521093233587266448843480154,
                                      0.96926521093233587266448843480154,
                                      0.96926521093233587266448843480154,
                                      0.96926521093233587266448843480154};

static const scalar_t q58_y[q58_n] = {0.5,
                                      0.19315926520414550308255755512972,
                                      0.5,
                                      0.5,
                                      0.80684073479585449691744244487028,
                                      0.5,
                                      0.061156438371160856756612120502834,
                                      0.5,
                                      0.5,
                                      0.93884356162883914324338787949717,
                                      0.061156438371160856756612120502834,
                                      0.061156438371160856756612120502834,
                                      0.93884356162883914324338787949717,
                                      0.93884356162883914324338787949717,
                                      0.061156438371160856756612120502834,
                                      0.5,
                                      0.5,
                                      0.93884356162883914324338787949717,
                                      0.21794459648998497286669050066846,
                                      0.21794459648998497286669050066846,
                                      0.78205540351001502713330949933154,
                                      0.78205540351001502713330949933154,
                                      0.21794459648998497286669050066846,
                                      0.21794459648998497286669050066846,
                                      0.78205540351001502713330949933154,
                                      0.78205540351001502713330949933154,
                                      0.064950107669012041192468095568038,
                                      0.064950107669012041192468095568038,
                                      0.93504989233098795880753190443196,
                                      0.93504989233098795880753190443196,
                                      0.064950107669012041192468095568038,
                                      0.064950107669012041192468095568038,
                                      0.93504989233098795880753190443196,
                                      0.93504989233098795880753190443196,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.030734789067664127335511565198461,
                                      0.030734789067664127335511565198461,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.96926521093233587266448843480154,
                                      0.96926521093233587266448843480154,
                                      0.030734789067664127335511565198461,
                                      0.030734789067664127335511565198461,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847,
                                      0.96926521093233587266448843480154,
                                      0.96926521093233587266448843480154,
                                      0.28386604868456891779198756924153,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.71613395131543108220801243075847};

static const scalar_t q58_z[q58_n] = {0.5,
                                      0.5,
                                      0.19315926520414550308255755512972,
                                      0.80684073479585449691744244487028,
                                      0.5,
                                      0.5,
                                      0.5,
                                      0.061156438371160856756612120502834,
                                      0.93884356162883914324338787949717,
                                      0.5,
                                      0.061156438371160856756612120502834,
                                      0.93884356162883914324338787949717,
                                      0.061156438371160856756612120502834,
                                      0.93884356162883914324338787949717,
                                      0.5,
                                      0.061156438371160856756612120502834,
                                      0.93884356162883914324338787949717,
                                      0.5,
                                      0.21794459648998497286669050066846,
                                      0.78205540351001502713330949933154,
                                      0.21794459648998497286669050066846,
                                      0.78205540351001502713330949933154,
                                      0.21794459648998497286669050066846,
                                      0.78205540351001502713330949933154,
                                      0.21794459648998497286669050066846,
                                      0.78205540351001502713330949933154,
                                      0.064950107669012041192468095568038,
                                      0.93504989233098795880753190443196,
                                      0.064950107669012041192468095568038,
                                      0.93504989233098795880753190443196,
                                      0.064950107669012041192468095568038,
                                      0.93504989233098795880753190443196,
                                      0.064950107669012041192468095568038,
                                      0.93504989233098795880753190443196,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.030734789067664127335511565198461,
                                      0.96926521093233587266448843480154,
                                      0.030734789067664127335511565198461,
                                      0.96926521093233587266448843480154,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.030734789067664127335511565198461,
                                      0.96926521093233587266448843480154,
                                      0.030734789067664127335511565198461,
                                      0.96926521093233587266448843480154,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847,
                                      0.28386604868456891779198756924153,
                                      0.71613395131543108220801243075847};

static const scalar_t q58_w[q58_n] = {
        0.05415937446870681787622884914929,   0.05415937446870681787622884914929,
        0.05415937446870681787622884914929,   0.05415937446870681787622884914929,
        0.05415937446870681787622884914929,   0.05415937446870681787622884914929,
        0.011473725767022205271405573614956,  0.011473725767022205271405573614956,
        0.011473725767022205271405573614956,  0.011473725767022205271405573614956,
        0.011473725767022205271405573614956,  0.011473725767022205271405573614956,
        0.011473725767022205271405573614956,  0.011473725767022205271405573614956,
        0.011473725767022205271405573614956,  0.011473725767022205271405573614956,
        0.011473725767022205271405573614956,  0.011473725767022205271405573614956,
        0.024857479768002937540108589823201,  0.024857479768002937540108589823201,
        0.024857479768002937540108589823201,  0.024857479768002937540108589823201,
        0.024857479768002937540108589823201,  0.024857479768002937540108589823201,
        0.024857479768002937540108589823201,  0.024857479768002937540108589823201,
        0.0062685994124186287334314359655827, 0.0062685994124186287334314359655827,
        0.0062685994124186287334314359655827, 0.0062685994124186287334314359655827,
        0.0062685994124186287334314359655827, 0.0062685994124186287334314359655827,
        0.0062685994124186287334314359655827, 0.0062685994124186287334314359655827,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938,
        0.012014600439171670804059992308938,  0.012014600439171670804059992308938};

#endif
