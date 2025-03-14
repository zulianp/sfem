#ifndef SFEM_LINE_QUADRATURE_H
#define SFEM_LINE_QUADRATURE_H

#include "sfem_base.h"

#define line_q2_n 2
static const scalar_t line_q2_x[line_q2_n] = {0.2113248654, 0.7886751346};
static const scalar_t line_q2_w[line_q2_n] = {1. / 2, 1. / 2};

#define line_q3_n 3
static const scalar_t line_q3_x[line_q3_n] = {0.1127016654, 1. / 2, 0.8872983346};
static const scalar_t line_q3_w[line_q3_n] = {0.2777777778, 0.4444444444, 0.2777777778};

// Gauss-Legendre
#define line_q5_n 5
static const scalar_t line_q5_x[line_q5_n] = {0.04691007703, 0.2307653449, 1/2, 0.7692346551, 0.9530899230};
static const scalar_t line_q5_w[line_q5_n] = {0.1184634425, 0.2393143352, 0.2844444444, 0.2393143352, 0.1184634425};

// Gauss-Lobatto
// #define line_q5_n 5
// static const scalar_t line_q5_x[line_q5_n] = {0.0, 0.1726731646, 1/2, 0.8273268354, 1.0};
// static const scalar_t line_q5_w[line_q5_n] = {0.05000000000, 0.2722222222, 0.3555555556, 0.2722222222, 0.05000000000};

#define line_q6_n 6
static const scalar_t line_q6_x[line_q6_n] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
static const scalar_t line_q6_w[line_q6_n] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

#define line_q8_n 8
static const scalar_t line_q8_x[line_q8_n] = {0.01985507175, 0.1016667613, 0.2372337950, 0.4082826788, 0.5917173213, 0.7627662050, 0.8983332387, 0.9801449283};
static const scalar_t line_q8_w[line_q8_n] = {0.05061426814, 0.1111905172, 0.1568533229, 0.1813418917, 0.1813418917, 0.1568533229, 0.1111905172, 0.05061426814};

#define line_q9_n 9
static const scalar_t line_q9_x[line_q9_n] = {0.01591988025, 0.08198444634, 0.1933142836, 0.3378732883, 1/2, 0.6621267117, 0.8066857164, 0.9180155537, 0.9840801198};
static const scalar_t line_q9_w[line_q9_n] = {0.04063719418, 0.09032408035, 0.1303053482, 0.1561735385, 0.1651196775, 0.1561735385, 0.1303053482, 0.09032408035, 0.04063719418};

#define line_q16_n 16
static const scalar_t line_q16_x[line_q16_n] = {0.005299532506, 0.02771248846, 0.06718439881, 0.1222977958, 0.1910618778, 0.2709916112, 0.3591982246, 0.4524937451, 0.5475062549, 0.6408017754, 0.7290083888, 0.8089381222, 0.8777022042, 0.9328156012, 0.9722875115, 0.9947004675};
static const scalar_t line_q16_w[line_q16_n] = {0.01357622971, 0.03112676197, 0.04757925584, 0.06231448563, 0.07479799441, 0.08457825970, 0.09130170752, 0.09472530523, 0.09472530523, 0.09130170752, 0.08457825970, 0.07479799441, 0.06231448563, 0.04757925584, 0.03112676197, 0.01357622971};

#define line_q17_n 17
static const scalar_t line_q17_x[line_q17_n] = {0.004712262344, 0.02466223912, 0.05988042314, 0.1092429981, 0.1711644204, 0.2436547315, 0.3243841183, 0.4107579093, 1/2, 0.5892420907, 0.6756158817, 0.7563452685, 0.8288355796, 0.8907570020, 0.9401195769, 0.9753377609, 0.9952877377};
static const scalar_t line_q17_w[line_q17_n] = {0.01207415143, 0.02772976469, 0.04251807416, 0.05594192360, 0.06756818423, 0.07702288054, 0.08400205108, 0.08828135268, 0.08972323518, 0.08828135268, 0.08400205108, 0.07702288054, 0.06756818423, 0.05594192360, 0.04251807416, 0.02772976469, 0.01207415143};

#endif  // SFEM_LINE_QUADRATURE_H
