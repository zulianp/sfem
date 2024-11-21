#ifndef SFEM_LINE_QUADRATURE_H
#define SFEM_LINE_QUADRATURE_H

#include "sfem_base.h"

#define line_q2_n 2
static const scalar_t line_q2_x[line_q2_n] = {0.2113248654, 0.7886751346};
static const scalar_t line_q2_w[line_q2_n] = {1. / 2, 1. / 2};

#define line_q3_n 3
static const scalar_t line_q3_x[line_q3_n] = {0.1127016654, 1. / 2, 0.8872983346};
static const scalar_t line_q3_w[line_q3_n] = {0.2777777778, 0.4444444444, 0.2777777778};

#define line_q5_n 5
static const scalar_t line_q5_x[line_q5_n] = {0.04691007703, 0.2307653449, 1/2, 0.7692346551, 0.9530899230};
static const scalar_t line_q5_w[line_q5_n] = {0.1184634425, 0.2393143352, 0.2844444444, 0.2393143352, 0.1184634425};

#define line_q6_n 6
static const scalar_t line_q6_x[line_q6_n] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
static const scalar_t line_q6_w[line_q6_n] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

#endif  // SFEM_LINE_QUADRATURE_H
