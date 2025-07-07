#ifndef POINT_TRIANGLE_DISTANCE_H
#define POINT_TRIANGLE_DISTANCE_H

#include "sfem_base.h"

static const short ENTITY_VERTEX_0 = 0;
static const short ENTITY_VERTEX_1 = 1;
static const short ENTITY_VERTEX_2 = 2;
static const short ENTITY_EDGE_0   = 3;
static const short ENTITY_EDGE_1   = 4;
static const short ENTITY_EDGE_2   = 5;
static const short ENTITY_FACE     = 6;

static SFEM_INLINE int isedge(const short entity) {
    return entity == ENTITY_EDGE_0 || entity == ENTITY_EDGE_1 || entity == ENTITY_EDGE_2;
}

static SFEM_INLINE int isface(const short entity) { return entity == ENTITY_FACE; }

static SFEM_INLINE int isvertex(const short entity) {
    return entity == ENTITY_VERTEX_0 || entity == ENTITY_VERTEX_1 || entity == ENTITY_VERTEX_2;
}

static SFEM_INLINE geom_t dot3(const geom_t *const SFEM_RESTRICT a, const geom_t *const SFEM_RESTRICT b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

typedef struct {
    geom_t point[3];
    geom_t s, t;
    short  entity;
} point_triangle_distance_result_t;

static SFEM_INLINE void                                                                    //
point_triangle_distance(const geom_t *const SFEM_RESTRICT                     point,       //
                        const geom_t *const SFEM_RESTRICT                     x_triangle,  //
                        const geom_t *const SFEM_RESTRICT                     y_triangle,  //
                        const geom_t *const SFEM_RESTRICT                     z_triangle,  //
                        point_triangle_distance_result_t *const SFEM_RESTRICT result) {    //
                                                                                           //
    const geom_t diff[3] = {x_triangle[0] - point[0],                                      //
                            y_triangle[0] - point[1],                                      //
                            z_triangle[0] - point[2]};

    const geom_t edge0[3] = {x_triangle[1] - x_triangle[0], y_triangle[1] - y_triangle[0], z_triangle[1] - z_triangle[0]};

    const geom_t edge1[3] = {x_triangle[2] - x_triangle[0], y_triangle[2] - y_triangle[0], z_triangle[2] - z_triangle[0]};

    const geom_t a00 = dot3(edge0, edge0);
    const geom_t a01 = dot3(edge0, edge1);
    const geom_t a11 = dot3(edge1, edge1);
    const geom_t b0  = dot3(diff, edge0);
    const geom_t b1  = dot3(diff, edge1);
    const geom_t det = a00 * a11 - a01 * a01;

    geom_t s      = a01 * b1 - a11 * b0;
    geom_t t      = a01 * b0 - a00 * b1;
    short  entity = -1;

    if (s + t <= det) {
        if (s < 0) {
            if (t < 0) {  // region 4:
                if (b0 < 0) {
                    t = 0;
                    if (-b0 >= a00) {
                        s      = 1;
                        entity = ENTITY_VERTEX_1;
                    } else {
                        s      = -b0 / a00;
                        entity = ENTITY_EDGE_0;
                    }
                } else {
                    s = 0;
                    if (b1 >= 0) {
                        t      = 0;
                        entity = ENTITY_VERTEX_0;
                    } else if (-b1 >= a11) {
                        t      = 1;
                        entity = ENTITY_VERTEX_2;
                    } else {
                        t      = -b1 / a11;
                        entity = ENTITY_EDGE_2;
                    }
                }
            } else {  // region 3:
                s = 0;
                if (b1 >= 0) {
                    t      = 0;
                    entity = ENTITY_VERTEX_0;
                } else if (-b1 >= a11) {
                    t      = 1;
                    entity = ENTITY_VERTEX_2;
                } else {
                    t      = -b1 / a11;
                    entity = ENTITY_EDGE_2;
                }
            }
        } else if (t < 0) {  // region 5
            t = 0;
            if (b0 >= 0) {
                s      = 0;
                entity = ENTITY_VERTEX_0;
            } else if (-b0 >= a00) {
                s      = 1;
                entity = ENTITY_VERTEX_1;
            } else {
                s      = -b0 / a00;
                entity = ENTITY_EDGE_0;
            }
        } else {  // region 0:
            // minimum at interior point
            s /= det;
            t /= det;
            entity = ENTITY_FACE;
        }
    } else {
        if (s < 0) {  // region 2:
            geom_t tmp0 = a01 + b0;
            geom_t tmp1 = a11 + b1;
            if (tmp1 > tmp0) {
                geom_t numer = tmp1 - tmp0;
                geom_t denom = a00 - 2 * a01 + a11;
                if (numer >= denom) {
                    s      = 1;
                    t      = 0;
                    entity = ENTITY_VERTEX_1;
                } else {
                    s      = numer / denom;
                    t      = 1 - s;
                    entity = ENTITY_EDGE_1;
                }
            } else {
                s = 0;
                if (tmp1 <= 0) {
                    t      = 1;
                    entity = ENTITY_VERTEX_2;
                } else if (b1 >= 0) {
                    t      = 0;
                    entity = ENTITY_VERTEX_0;
                } else {
                    t      = -b1 / a11;
                    entity = ENTITY_EDGE_2;
                }
            }
        } else if (t < 0) {  // region 6
            geom_t tmp0 = a01 + b1;
            geom_t tmp1 = a00 + b0;
            if (tmp1 > tmp0) {
                geom_t numer = tmp1 - tmp0;
                geom_t denom = a00 - 2 * a01 + a11;
                if (numer >= denom) {
                    t      = 1;
                    s      = 0;
                    entity = ENTITY_VERTEX_2;
                } else {
                    t      = numer / denom;
                    s      = 1 - t;
                    entity = ENTITY_EDGE_1;
                }
            } else {
                t = 0;
                if (tmp1 <= 0) {
                    s      = 1;
                    entity = ENTITY_VERTEX_1;
                } else if (b0 >= 0) {
                    s      = 0;
                    entity = ENTITY_VERTEX_0;
                } else {
                    s      = -b0 / a00;
                    entity = ENTITY_EDGE_0;
                }
            }
        } else {  // region 1:
            geom_t numer = a11 + b1 - a01 - b0;
            if (numer <= 0) {
                s      = 0;
                t      = 1;
                entity = ENTITY_VERTEX_2;
            } else {
                geom_t denom = a00 - 2 * a01 + a11;
                if (numer >= denom) {
                    s      = 1;
                    t      = 0;
                    entity = ENTITY_VERTEX_1;
                } else {
                    s      = numer / denom;
                    t      = 1 - s;
                    entity = ENTITY_EDGE_1;
                }
            }
        }
    }

    result->point[0] = x_triangle[0] + s * edge0[0] + t * edge1[0];
    result->point[1] = y_triangle[0] + s * edge0[1] + t * edge1[1];
    result->point[2] = z_triangle[0] + s * edge0[2] + t * edge1[2];
    result->s        = s;
    result->t        = t;
    result->entity   = entity;
}

#endif  // POINT_TRIANGLE_DISTANCE_H
