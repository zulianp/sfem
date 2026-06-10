#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, const char * argv[]) {
	double a = atof(argv[1]);
	double b = atof(argv[2]);
	double c = atof(argv[3]);
	double d = atof(argv[4]);
	double e = atof(argv[5]);

    double t0 = b * c;
    double t1 = a * b;
    double t2 = d * e;
    double t3 = c + t1;
    double t4 = fma(t1, t3 * (c + t2), c);

    double v[8];
    v[0]      = a + t0;
    v[1]      = fma(2, t1, fma(3, pow(a, 2) * pow(c, 2), (1.0 / 3.0) / (a * pow(b, 2) * c)));
    v[2]      = fma(4, t1, fma(6, d * t1, e));
    v[3]      = fma(3, t1, c);
    v[4]      = fma(-2, a * t0, d);
    v[5]      = t2 + t3;
    v[6]      = t4;
    v[7]      = t4 + log(c);

    for(int i = 0; i < 8; i++)
    	printf("%g\n", v[i]);
    return 0;
}