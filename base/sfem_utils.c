#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// read_env_as_string
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
const char *read_env_as_string(const char *env_var_name, const char *default_value, const char *description) {
    const char *env_val = getenv(env_var_name);

    if (env_val == NULL) {
        if (description != NULL && description[0] != '\0') {
            printf("%s not set. Using default value: %s\n", description, default_value);
            printf("To set it, use: export %s=<value>\n", env_var_name);
        } // END if (description)
        
        return default_value;
    } // END if (env_val == NULL)

    if (description != NULL && description[0] != '\0') {
        printf("Using %s from environment: %s\n", description, env_val);
    } // END if (description)

    return env_val;
} // END Function: read_env_as_string



////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// read_env_as_long
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
long read_env_as_long(const char *env_var_name, const long default_value, const char *description) {
    const char *env_val = getenv(env_var_name);

    if (env_val == NULL) {
        if (description != NULL && description[0] != '\0') {
            printf("%s not set. Using default value: %ld\n", description, default_value);
            printf("To set it, use: export %s=<value>\n", env_var_name);
        } // END if (description)
        return default_value;
    } // END if (env_val == NULL)

    char *endptr;
    errno = 0; // To distinguish success/failure after call
    long value = strtol(env_val, &endptr, 10);

    // Check for various possible errors
    if ((errno == ERANGE && (value == LONG_MAX || value == LONG_MIN)) || (errno != 0 && value == 0)) {
        fprintf(stderr, "Error: Could not convert %s to long. Out of range. Using default value: %ld\n", env_var_name, default_value);
        return default_value;
    } // END if (errno == ERANGE)

    if (endptr == env_val) {
        fprintf(stderr, "Error: %s is not a valid number: %s. No digits were found. Using default value: %ld\n", env_var_name, env_val, default_value);
        return default_value;
    } // END if (endptr == env_val)

    // If we have trailing characters, it's not a valid integer
    if (*endptr != '\0') {
        fprintf(stderr, "Error: %s has invalid trailing characters: %s. Using default value: %ld\n", env_var_name, env_val, default_value);
        return default_value;
    } // END if (*endptr != '\0')

    if (description != NULL && description[0] != '\0') {
        printf("Using %s from environment: %ld\n", description, value);
    } // END if (description)

    return value;
} // END Function: read_env_as_long

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// read_env_as_int
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int read_env_as_int(const char *env_var_name, const int default_value, const char *description) {
    long value = read_env_as_long(env_var_name, default_value, NULL); // Use long reader, supress its prints

    if (value > INT_MAX || value < INT_MIN) {
        fprintf(stderr, "Error: Value for %s is out of int range. Using default: %d\n", env_var_name, default_value);
        return default_value;
    } // END if (value > INT_MAX || value < INT_MIN)

    // Descriptions are handled here to avoid double printing from read_env_as_long
    const char *env_val = getenv(env_var_name);
    if (env_val == NULL) {
         if (description != NULL && description[0] != '\0') {
            printf("%s not set. Using default value: %d\n", description, default_value);
            printf("To set it, use: export %s=<value>\n", env_var_name);
        } // END if (description)
    } else {
        if (description != NULL && description[0] != '\0') {
            printf("Using %s from environment: %d\n", description, (int)value);
        } // END if (description)
    } // END if (env_val == NULL)

    return (int)value;
} // END Function: read_env_as_int

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// read_env_as_double
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
double read_env_as_double(const char *env_var_name, const double default_value, const char *description) {
    const char *env_val = getenv(env_var_name);

    if (env_val == NULL) {
        if (description != NULL && description[0] != '\0') {
            printf("%s not set. Using default value: %g\n", description, default_value);
            printf("To set it, use: export %s=<value>\n", env_var_name);
        } // END if (description)
        return default_value;
    } // END if (env_val == NULL)

    char *endptr;
    errno = 0; // To distinguish success/failure after call
    double value = strtod(env_val, &endptr);

    if (errno == ERANGE) {
        fprintf(stderr, "Error: Could not convert %s to double. Out of range. Using default value: %g\n", env_var_name, default_value);
        return default_value;
    } // END if (errno == ERANGE)

    if (endptr == env_val) {
        fprintf(stderr, "Error: %s is not a valid number: %s. No digits were found. Using default value: %g\n", env_var_name, env_val, default_value);
        return default_value;
    } // END if (endptr == env_val)

    if (*endptr != '\0') {
        fprintf(stderr, "Error: %s has invalid trailing characters: %s. Using default value: %g\n", env_var_name, env_val, default_value);
        return default_value;
    } // END if (*endptr != '\0')

    if (description != NULL && description[0] != '\0') {
        printf("Using %s from environment: %g\n", description, value);
    } // END if (description)

    return value;
} // END Function: read_env_as_double
