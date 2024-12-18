
// We read a quad mesh stored with files x.float32, y.float32, z.float32, i0.int32, i1.int32,  i2.int32,
// i3.int32 and a coordinate displacements disp.0.float32 disp.1.float32, and disp.2.float32 and visualize it using OpenGL and
// glut. Implement mouse movements for translation (with right click drag), rotations (with left click drag) and zoom (mouse
// wheel).

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
/* 3D view manipulation demo
 * Written by John Tsiombikas <nuclear@member.fsf.org>
 *
 * Demonstrates how to use freeglut callbacks to manipulate a 3D view, similarly
 * to how a modelling program or a model viewer would operate.
 *
 * Rotate: drag with the left mouse button.
 * Scale: drag up/down with the right mouse button.
 * Pan: drag with the middle mouse button.
 *
 * Press space to animate the scene and update the display continuously, press
 *   again to return to updating only when the view needs to change.
 * Press escape or q to exit.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _MSC_VER
#pragma warning(disable : 4305 4244)
#endif

// Convert HSV values (in range [0.0, 1.0]) to RGB
void hsv_to_rgb(float hue, float saturation, float value, float& r, float& g, float& b) {
    int hi = static_cast<int>(floor(hue * 6)) % 6; // sector 0 to 5
    float f = (hue * 6) - floor(hue * 6);
    float p = value * (1 - saturation);
    float q = value * (1 - f * saturation);
    float t = value * (1 - (1 - f) * saturation);

    switch (hi) {
        case 0: r = value, g = t, b = p; break;
        case 1: r = q, g = value, b = p; break;
        case 2: r = p, g = value, b = t; break;
        case 3: r = p, g = q, b = value; break;
        case 4: r = t, g = p, b = value; break;
        case 5: r = value, g = p, b = q; break;
    }
}

// Generate RGB color from intensity
void color_from_intensity(float intensity) {
    float hue = fmod(intensity * 3.0f, 1.0f); // Cycle through hues (R -> G -> B)
    float saturation = 1.0f;                 // Full saturation for vivid colors
    float value = intensity;                 // Intensity directly affects brightness

    float r, g, b;
    hsv_to_rgb(hue, saturation, value, r, g, b);

    glColor3f(r, g, b); // Set OpenGL color to the calculated RGB values
}

static const char *helpprompt[] = {"Press F1 for help", 0};
static const char *helptext[]   = {"Rotate: left mouse drag",
                                   " Scale: right mouse drag up/down",
                                   "   Pan: middle mouse drag",
                                   "",
                                   "Toggle fullscreen: f",
                                   "Toggle animation: space",
                                   "Quit: escape",
                                   0};

void idle(void);
void display(void);
void print_help(void);
void reshape(int x, int y);
void keypress(unsigned char key, int x, int y);
void skeypress(int key, int x, int y);
void mouse(int bn, int st, int x, int y);
void motion(int x, int y);

int   win_width, win_height;
float cam_theta, cam_phi = 25, cam_dist = 15;
float cam_pan[3];
int   mouse_x, mouse_y;
int   bnstate[8];
int   anim, help;
long  anim_start;
long  nframes;

#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8db9
#endif

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809d
#endif

template <typename T>
int read_array(const std::string &path, T **array, ptrdiff_t *size) {
    FILE *fp;
    fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", path.c_str());
        return -1;
    }

    fseek(fp, 0L, SEEK_END);
    *size = ftell(fp);
    *size /= sizeof(T);
    fseek(fp, 0L, SEEK_SET);

    *array                  = (T *)malloc(*size * sizeof(T));
    ptrdiff_t read_elements = fread(*array, sizeof(T), *size, fp);
    assert(read_elements == *size);

    printf("Read %s, len %ld\n", path.c_str(), *size);

    fclose(fp);
    return 0;
}

void evalf(const float x, const float y, float *f) {
    f[0] = (1 - x) * (1 - y);
    f[1] = (x) * (1 - y);
    f[2] = (x) * (y);
    f[3] = (1 - x) * (y);
}

void evalgx(const float x, const float y, float *gx) {
    gx[0] = y - 1;
    gx[1] = 1 - y;
    gx[2] = y;
    gx[3] = -y;
}

void evalgy(const float x, const float y, float *gy) {
    gy[0] = x - 1;
    gy[1] = -x;
    gy[2] = x;
    gy[3] = 1 - x;
}

void normalize(float *n) {
    const float length = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

    n[0] = n[0] / length;
    n[1] = n[1] / length;
    n[2] = n[2] / length;
}

void eval_normal(const float x, const float y, float *p0, float *p1, float *p2, float *p3, float *normal) {
    float gx[4];
    float gy[4];

    evalgx(x, y, gx);
    evalgy(x, y, gy);

    float u[3] = {0, 0, 0};
    float v[3] = {0, 0, 0};

    for (int d = 0; d < 4; d++) {
        u[d] += gx[0] * p0[d] + gx[1] * p1[d] + gx[2] * p2[d] + gx[3] * p3[d];
        v[d] += gy[0] * p0[d] + gy[1] * p1[d] + gy[2] * p2[d] + gy[3] * p3[d];
    }

    normalize(u);
    normalize(v);

    float len_u = sqrtf(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    float len_v = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    float cross[3] = {u[1] * v[2] - v[1] * u[2], u[2] * v[0] - v[2] * u[0], u[0] * v[1] - v[0] * u[1]};
    normalize(cross);

    normal[0] = cross[0];
    normal[1] = cross[1];
    normal[2] = cross[2];
}

struct viz {
    std::vector<float *> points;
    ptrdiff_t            n_nodes;

    std::vector<int *> elements;
    ptrdiff_t          n_elements;
    int               *node_mapping;

    float barycenter[3];
    float bbmin[3], bbmax[3];

    std::vector<float *>  normals;
    std::vector<double *> disp;

    bool enable_displacement{true};
    bool enable_pseudo_normals{false};
    bool enable_wireframe{false};
    int  levels{1};

    ~viz() {
        for (auto p : points) {
            free(p);
        }

        for (auto e : elements) {
            free(e);
        }

        for (auto d : disp) {
            free(d);
        }

        for (auto n : normals) {
            free(n);
        }

        free(node_mapping);
    }

    void setup_lights() {
        // Enable lighting in general
        glEnable(GL_LIGHTING);

        // Set global ambient light (affects all objects equally)
        GLfloat globalAmbientLight[] = {0.2f, 0.2f, 0.2f, 1.0f};  // Low-intensity ambient light
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, globalAmbientLight);

        // Configure and enable light 0 (e.g., a directional light)
        GLfloat light0Position[] = {1.0f, 1.0f, 1.0f, 0.0f};  // Directional light
        GLfloat light0Ambient[]  = {0.01f, 0.01f, 0.01f, 1.0f};
        GLfloat light0Diffuse[]  = {0.6f, 0.6f, 0.6f, 1.0f};
        GLfloat light0Specular[] = {1.f, 1.f, 1.f, 1.0f};
        GLfloat light0Direction[] = {0.0f, 0.0f, -1.0f, 0.0f};  // Direction of the light

        glLightfv(GL_LIGHT0, GL_POSITION, light0Position);
        glLightfv(GL_LIGHT0, GL_AMBIENT, light0Ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light0Diffuse);
        glLightfv(GL_LIGHT0, GL_SPECULAR, light0Specular);

        glEnable(GL_LIGHT0);  // Enable the first light

        // Configure and enable light 1 (e.g., another directional light)
        GLfloat light1Position[] = {-1.0f, -1.0f, 1.0f, 0.0f};  // Another directional light
        GLfloat light1Ambient[]  = {0.01f, 0.01f, 0.01f, 1.0f};
        GLfloat light1Diffuse[]  = {0.5f, 0.5f, 0.5f, 1.0f};
        GLfloat light1Specular[] = {1.0f, 1.0f, 1.0f, 1.0f};

        glLightfv(GL_LIGHT1, GL_POSITION, light1Position);
        glLightfv(GL_LIGHT1, GL_AMBIENT, light1Ambient);
        glLightfv(GL_LIGHT1, GL_DIFFUSE, light1Diffuse);
        glLightfv(GL_LIGHT1, GL_SPECULAR, light1Specular);

        glEnable(GL_LIGHT1);  // Enable the second light
    }

    void compute_bounding_box() {
        for (int d = 0; d < 3; d++) {
            bbmin[d] = points[d][0];
            bbmax[d] = points[d][0];

            for (ptrdiff_t i = 1; i < n_nodes; i++) {
                bbmax[d] = std::max(bbmax[d], points[d][i]);
                bbmin[d] = std::min(bbmin[d], points[d][i]);
            }
        }
    }

    void center_mesh() {
        compute_barycenter();

        for (int d = 0; d < 3; d++) {
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                points[d][i] -= barycenter[d];
            }

            barycenter[d] = 0;
        }
    }

    void compute_barycenter() {
        for (int d = 0; d < 3; d++) {
            barycenter[d] = 0;
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                barycenter[d] += points[d][i];
            }

            barycenter[d] /= n_nodes;
        }
    }

    void compute_pseudo_normals() {
        if (!enable_pseudo_normals) return;

        normals.resize(3);
        for (int d = 0; d < 3; d++) {
            normals[d] = (float *)calloc(n_nodes, sizeof(float));
        }

        for (ptrdiff_t e = 0; e < n_elements; e++) {
            int i0 = elements[0][e];
            int i1 = elements[1][e];
            int i2 = elements[2][e];
            int i3 = elements[3][e];

            float p0[3] = {points[0][i0], points[1][i0], points[2][i0]};
            float p1[3] = {points[0][i1], points[1][i1], points[2][i1]};
            float p2[3] = {points[0][i2], points[1][i2], points[2][i2]};
            float p3[3] = {points[0][i3], points[1][i3], points[2][i3]};

            if (node_mapping && enable_displacement) {
                int g0 = node_mapping[i0];
                int g1 = node_mapping[i1];
                int g2 = node_mapping[i2];
                int g3 = node_mapping[i3];

                for (int d = 0; d < 3; d++) {
                    p0[d] += disp[d][g0];
                    p1[d] += disp[d][g1];
                    p2[d] += disp[d][g2];
                    p3[d] += disp[d][g3];
                }
            }

            float normal[3];
            eval_normal(0, 0, p0, p1, p2, p3, normal);
            for (int d = 0; d < 3; d++) {
                normals[d][i0] += normal[d];
            }

            eval_normal(1, 0, p0, p1, p2, p3, normal);
            for (int d = 0; d < 3; d++) {
                normals[d][i1] += normal[d];
            }

            eval_normal(1, 1, p0, p1, p2, p3, normal);
            for (int d = 0; d < 3; d++) {
                normals[d][i2] += normal[d];
            }

            eval_normal(0, 1, p0, p1, p2, p3, normal);
            for (int d = 0; d < 3; d++) {
                normals[d][i3] += normal[d];
            }
        }

        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            float nx = normals[0][i];
            float ny = normals[1][i];
            float nz = normals[2][i];

            float len = sqrtf(nx * nx + ny * ny + nz * nz);

            normals[0][i] /= len;
            normals[1][i] /= len;
            normals[2][i] /= len;
        }
    }

    void load(const std::string &folder,
              const std::string &path_dispx,
              const std::string &path_dispy,
              const std::string &path_dispz) {
        points.resize(3);
        read_array<float>(folder + "/x.raw", &points[0], &n_nodes);
        read_array<float>(folder + "/y.raw", &points[1], &n_nodes);
        read_array<float>(folder + "/z.raw", &points[2], &n_nodes);

        elements.resize(4);
        read_array<int>(folder + "/i0.raw", &elements[0], &n_elements);
        read_array<int>(folder + "/i1.raw", &elements[1], &n_elements);
        read_array<int>(folder + "/i2.raw", &elements[2], &n_elements);
        read_array<int>(folder + "/i3.raw", &elements[3], &n_elements);
        read_array<int>(folder + "/node_mapping.raw", &node_mapping, &n_nodes);

        ptrdiff_t n_disp;
        disp.resize(3);
        read_array<double>(path_dispx, &disp[0], &n_disp);
        read_array<double>(path_dispy, &disp[1], &n_disp);
        read_array<double>(path_dispz, &disp[2], &n_disp);

        printf("#elements %ld #nodes %ld #disp %ld\n", n_elements, n_nodes, n_disp);

        center_mesh();
        compute_pseudo_normals();
        compute_bounding_box();
    }
};

static struct viz v;

// c++ -o viz quad4_viz.exe.cpp  -framework GLUT -framework OpenGL -DGL_SILENCE_DEPRECATION -g -O0 -std=c++11
int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <mesh> <dispx> <dispy> <dispz>\n", argv[0]);
        return 1;
    }

    std::string folder = argv[1];
    std::string dispx  = argv[2];
    std::string dispy  = argv[3];
    std::string dispz  = argv[4];

    v.load(folder, dispx, dispy, dispz);

    glutInit(&argc, argv);
    glutInitWindowSize(800, 600);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutCreateWindow("QUAD4 VIZ");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keypress);
    glutSpecialFunc(skeypress);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    // glEnable(GL_LIGHT1);
    // glEnable(GL_LIGHT2);

    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.2);
    // glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, 0.2);
    // glEnable(GL_LIGHT3);
    // glEnable(GL_LIGHT4);

    glutMainLoop();
    return 0;
}

void idle(void) { glutPostRedisplay(); }

void draw_node(const float  x,
               const float  y,
               const float *p0,
               const float *p1,
               const float *p2,
               const float *p3,
               const float *n0,
               const float *n1,
               const float *n2,
               const float *n3) {
    float f[4];
    evalf(x, y, f);

    

    float p[3] = {f[0] * p0[0] + f[1] * p1[0] + f[2] * p2[0] + f[3] * p3[0],
                  f[0] * p0[1] + f[1] * p1[1] + f[2] * p2[1] + f[3] * p3[1],
                  f[0] * p0[2] + f[1] * p1[2] + f[2] * p2[2] + f[3] * p3[2]};

    float n[3] = {f[0] * n0[0] + f[1] * n1[0] + f[2] * n2[0] + f[3] * n3[0],
                  f[0] * n0[1] + f[1] * n1[1] + f[2] * n2[1] + f[3] * n3[1],
                  f[0] * n0[2] + f[1] * n1[2] + f[2] * n2[2] + f[3] * n3[2]};


    // color_from_intensity(n[0]);

     glColor3f((1+n[0])/2, (1+n[1])/2, (1+n[2])/2);

    normalize(n);

    // printf("%f %f %f -> %f %f %f\n", p[0], p[1], p[2], n[0], n[1], n[2]);

    glNormal3f(n[0], n[1], n[2]);
    glVertex3f(p[0], p[1], p[2]);
}

void display(void) {
    long tm;
    // float l0pos[] = {-10, 10, 10, 0};
    // float l1pos[] = {10, 10, -10, 0};
    // float l2pos[] = {10, 0, 0, 0};
    // float l3pos[] = {-10, 0, 0, 0};
    // float l4pos[] = {0, -10, 0, 0};

    float ldist = 8;

    // float l2pos[] = {v.bbmin[0] - ldist, v.bbmin[1] - ldist, v.bbmin[2] - ldist, 0};
    // float l1pos[] = {v.bbmax[0] + ldist, v.bbmax[1] + ldist, v.bbmax[2] + ldist, 0};
    // float l0pos[] = {v.barycenter[0], v.barycenter[0], v.barycenter[0] + v.bbmax[2] + ldist, 0};
    

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-v.barycenter[0], -v.barycenter[1], -v.barycenter[2] - cam_dist);
    glRotatef(cam_phi, 1, 0, 0);
    glRotatef(cam_theta, 0, 1, 0);
    glTranslatef(cam_pan[0], cam_pan[1], cam_pan[2]);

    // glLightfv(GL_LIGHT0, GL_POSITION, l0pos);
    // glLightfv(GL_LIGHT1, GL_POSITION, l1pos);
    // glLightfv(GL_LIGHT2, GL_POSITION, l2pos);
    // glLightfv(GL_LIGHT3, GL_POSITION, l3pos);
    // glLightfv(GL_LIGHT4, GL_POSITION, l4pos);

    v.setup_lights();

    glPushMatrix();
    if (anim) {
        tm = glutGet(GLUT_ELAPSED_TIME) - anim_start;
        glRotatef(tm / 10.0f, 1, 0, 0);
        glRotatef(tm / 10.0f, 0, 1, 0);
    }

    if (v.enable_wireframe) {
        glColor3f(0.3, 0.3, 0.3);
        glLineWidth(1.5);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBegin(GL_LINES);
        for (ptrdiff_t i = 0; i < v.n_elements; i++) {
            int i0 = v.elements[0][i];
            int i1 = v.elements[1][i];
            int i2 = v.elements[2][i];
            int i3 = v.elements[3][i];

            float p0[3] = {v.points[0][i0], v.points[1][i0], v.points[2][i0]};
            float p1[3] = {v.points[0][i1], v.points[1][i1], v.points[2][i1]};
            float p2[3] = {v.points[0][i2], v.points[1][i2], v.points[2][i2]};
            float p3[3] = {v.points[0][i3], v.points[1][i3], v.points[2][i3]};

            if (v.node_mapping && v.enable_displacement) {
                int g0 = v.node_mapping[i0];
                int g1 = v.node_mapping[i1];
                int g2 = v.node_mapping[i2];
                int g3 = v.node_mapping[i3];

                for (int d = 0; d < 3; d++) {
                    p0[d] += v.disp[d][g0];
                    p1[d] += v.disp[d][g1];
                    p2[d] += v.disp[d][g2];
                    p3[d] += v.disp[d][g3];
                }
            }

            float n0[3];
            float n1[3];
            float n2[3];
            float n3[3];

            if (v.enable_pseudo_normals) {
                for (int d = 0; d < 3; d++) {
                    n0[d] = v.normals[d][i0];
                    n1[d] = v.normals[d][i1];
                    n2[d] = v.normals[d][i2];
                    n3[d] = v.normals[d][i3];
                }
            } else {
                eval_normal(0, 0, p0, p1, p2, p3, n0);
                eval_normal(1, 0, p0, p1, p2, p3, n1);
                eval_normal(1, 1, p0, p1, p2, p3, n2);
                eval_normal(0, 1, p0, p1, p2, p3, n3);
            }

            for (int xi = 0; xi < v.levels; xi++) {
                const float l = float(xi) / (v.levels);
                const float r = float(xi + 1) / (v.levels);
                draw_node(l, 0, p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(r, 0, p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(l, 1, p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(r, 1, p0, p1, p2, p3, n0, n1, n2, n3);
            }

            for (int yi = 0; yi < v.levels; yi++) {
                const float l = float(yi) / (v.levels);
                const float r = float(yi + 1) / (v.levels);
                draw_node(0, l, p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(0, r, p0, p1, p2, p3, n0, n1, n2, n3);

                draw_node(1, l, p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(1, r, p0, p1, p2, p3, n0, n1, n2, n3);
            }
        }

        glEnd();
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glColor3f(0.9, 0.9, 0.9);
    glBegin(GL_QUADS);

    for (ptrdiff_t i = 0; i < v.n_elements; i++) {
        int i0 = v.elements[0][i];
        int i1 = v.elements[1][i];
        int i2 = v.elements[2][i];
        int i3 = v.elements[3][i];

        float p0[3] = {v.points[0][i0], v.points[1][i0], v.points[2][i0]};
        float p1[3] = {v.points[0][i1], v.points[1][i1], v.points[2][i1]};
        float p2[3] = {v.points[0][i2], v.points[1][i2], v.points[2][i2]};
        float p3[3] = {v.points[0][i3], v.points[1][i3], v.points[2][i3]};

        if (v.node_mapping && v.enable_displacement) {
            int g0 = v.node_mapping[i0];
            int g1 = v.node_mapping[i1];
            int g2 = v.node_mapping[i2];
            int g3 = v.node_mapping[i3];

            for (int d = 0; d < 3; d++) {
                p0[d] += v.disp[d][g0];
                p1[d] += v.disp[d][g1];
                p2[d] += v.disp[d][g2];
                p3[d] += v.disp[d][g3];
            }
        }

        float n0[3];
        float n1[3];
        float n2[3];
        float n3[3];

        if (v.enable_pseudo_normals) {
            for (int d = 0; d < 3; d++) {
                n0[d] = v.normals[d][i0];
                n1[d] = v.normals[d][i1];
                n2[d] = v.normals[d][i2];
                n3[d] = v.normals[d][i3];
            }
        } else {
            eval_normal(0, 0, p0, p1, p2, p3, n0);
            eval_normal(1, 0, p0, p1, p2, p3, n1);
            eval_normal(1, 1, p0, p1, p2, p3, n2);
            eval_normal(0, 1, p0, p1, p2, p3, n3);
        }

        for (int yi = 0; yi < v.levels; yi++) {
            for (int xi = 0; xi < v.levels; xi++) {
                const float l[2] = {float(xi) / (v.levels), float(yi) / (v.levels)};
                const float r[2] = {float(xi + 1) / (v.levels), float(yi + 1) / (v.levels)};
                draw_node(l[0], l[1], p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(r[0], l[1], p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(r[0], r[1], p0, p1, p2, p3, n0, n1, n2, n3);
                draw_node(l[0], r[1], p0, p1, p2, p3, n0, n1, n2, n3);
            }
        }
    }

    glEnd();
    glPopMatrix();

    print_help();

    glutSwapBuffers();
    nframes++;
}

void print_help(void) {
    int         i;
    const char *s, **text;

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, win_width, 0, win_height, -1, 1);

    text = help ? helptext : helpprompt;

    for (i = 0; text[i]; i++) {
        glColor3f(0, 0.1, 0);
        glRasterPos2f(7, win_height - (i + 1) * 20 - 2);
        s = text[i];
        while (*s) {
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *s++);
        }
        glColor3f(0, 0.9, 0);
        glRasterPos2f(5, win_height - (i + 1) * 20);
        s = text[i];
        while (*s) {
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *s++);
        }
    }

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glPopAttrib();
}

#define ZNEAR 0.1f
void reshape(int x, int y) {
    float vsz, aspect = (float)x / (float)y;
    win_width  = x;
    win_height = y;

    glViewport(0, 0, x, y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    vsz = 0.4663f * ZNEAR;
    glFrustum(-aspect * vsz, aspect * vsz, -vsz, vsz, 0.5, 500.0);
}

void keypress(unsigned char key, int x, int y) {
    static int fullscr;
    static int prev_xsz, prev_ysz;

    switch (key) {
        case 27:
        case 'w':
            v.enable_wireframe = !v.enable_wireframe;
            glutPostRedisplay();
            break;
        case 'q':
            exit(0);
            break;
        case '+':
            v.levels++;
            printf("levels %d\n", v.levels);
            glutPostRedisplay();
            break;
        case 'd':
            v.enable_displacement = !v.enable_displacement;
            v.compute_pseudo_normals();
            glutPostRedisplay();
            break;
        case 'p':
            v.enable_pseudo_normals = !v.enable_pseudo_normals;
            v.compute_pseudo_normals();
            glutPostRedisplay();
            break;
        case '-':
            v.levels = std::max(1, v.levels - 1);
            printf("levels %d\n", v.levels);
            glutPostRedisplay();
            break;
        case ' ':
            anim ^= 1;
            glutIdleFunc(anim ? idle : 0);
            glutPostRedisplay();

            if (anim) {
                anim_start = glutGet(GLUT_ELAPSED_TIME);
                nframes    = 0;
            } else {
                long tm  = glutGet(GLUT_ELAPSED_TIME) - anim_start;
                long fps = (nframes * 100000) / tm;
                printf("framerate: %ld.%ld fps\n", fps / 100, fps % 100);
            }
            break;

        case '\n':
        case '\r':
            if (!(glutGetModifiers() & GLUT_ACTIVE_ALT)) {
                break;
            }
        case 'f':
            fullscr ^= 1;
            if (fullscr) {
                prev_xsz = glutGet(GLUT_WINDOW_WIDTH);
                prev_ysz = glutGet(GLUT_WINDOW_HEIGHT);
                glutFullScreen();
            } else {
                glutReshapeWindow(prev_xsz, prev_ysz);
            }
            break;
    }
}

void skeypress(int key, int x, int y) {
    switch (key) {
        case GLUT_KEY_F1:
            help ^= 1;
            glutPostRedisplay();

        default:
            break;
    }
}

void mouse(int bn, int st, int x, int y) {
    int bidx      = bn - GLUT_LEFT_BUTTON;
    bnstate[bidx] = st == GLUT_DOWN;
    mouse_x       = x;
    mouse_y       = y;
}

void motion(int x, int y) {
    int dx  = x - mouse_x;
    int dy  = y - mouse_y;
    mouse_x = x;
    mouse_y = y;

    if (!(dx | dy)) return;

    if (bnstate[0]) {
        cam_theta += dx * 0.5;
        cam_phi += dy * 0.5;
        if (cam_phi < -90) cam_phi = -90;
        if (cam_phi > 90) cam_phi = 90;
        glutPostRedisplay();
    }
    if (bnstate[1]) {
        float up[3], right[3];
        float theta = cam_theta * M_PI / 180.0f;
        float phi   = cam_phi * M_PI / 180.0f;

        up[0]    = -sin(theta) * sin(phi);
        up[1]    = -cos(phi);
        up[2]    = cos(theta) * sin(phi);
        right[0] = cos(theta);
        right[1] = 0;
        right[2] = sin(theta);

        cam_pan[0] += (right[0] * dx + up[0] * dy) * 0.01;
        cam_pan[1] += up[1] * dy * 0.01;
        cam_pan[2] += (right[2] * dx + up[2] * dy) * 0.01;
        glutPostRedisplay();
    }
    if (bnstate[2]) {
        cam_dist += dy * 0.1;
        if (cam_dist < 0) cam_dist = 0;
        glutPostRedisplay();
    }
}