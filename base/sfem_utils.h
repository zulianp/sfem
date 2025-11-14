#ifndef SFEM_UTILS_H
#define SFEM_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////
/// @brief Read an environment variable as a string
/// @param env_var_name The name of the environment variable
/// @param default_value The default value to return if the variable is not set
/// @param description Optional description of the variable for logging
/// @return The value from the environment variable or the default value
const char *read_env_as_string(const char *env_var_name, const char *default_value, const char *description);

////////////////////////////////////////////////////////////////////////////
/// @brief Read an environment variable and convert it to a long integer
/// @param env_var_name The name of the environment variable
/// @param default_value The default value to return if the variable is not set or invalid
/// @param description Optional description of the variable for logging
/// @return The value from the environment variable or the default value
long read_env_as_long(const char *env_var_name, long default_value, const char *description);

////////////////////////////////////////////////////////////////////////////
/// @brief Read an environment variable and convert it to an integer
/// @param env_var_name The name of the environment variable
/// @param default_value The default value to return if the variable is not set or invalid
/// @param description Optional description of the variable for logging
/// @return The value from the environment variable or the default value
int read_env_as_int(const char *env_var_name, int default_value, const char *description);

////////////////////////////////////////////////////////////////////////////
/// @brief Read an environment variable and convert it to a double
/// @param env_var_name The name of the environment variable
/// @param default_value The default value to return if the variable is not set or invalid
/// @param description Optional description of the variable for logging
/// @return The value from the environment variable or the default value
double read_env_as_double(const char *env_var_name, double default_value, const char *description);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_UTILS_H