#ifndef SFEM_MACROS_H
#define SFEM_MACROS_H

#include <stddef.h>

#ifndef MIN
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif

#ifndef POW2
#define POW2(a) ((a) * (a))
#endif

#ifndef POW3
#define POW3(a) ((a) * (a) * (a))
#endif

#ifndef POW4
#define POW4(a) ((a) * (a) * (a) * (a))
#endif

#endif  // SFEM_MACROS_H
